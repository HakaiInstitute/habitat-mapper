"""Benchmark tiled inference throughput of a habitat-mapper model on GPU (CUDA) vs CPU.

Each (device, batch size) configuration runs in its own subprocess so that ONNX Runtime's
CUDA memory arena starts empty and the peak GPU memory reading belongs to that configuration only.

Two measurements are made:

1. Synthetic throughput (always): random uint8 tiles are pushed through ``model._predict``
   (normalisation + ONNX Runtime session, including host<->device copies). This gives tiles/s.
   ``sess.run`` alone is also timed to show how much of the time is CPU-side preprocessing.
   Seconds per km² are derived from tiles/s using the pipeline's 50% tile overlap
   (stride = tile_size / 2, so each tile contributes stride² new pixels) at each ``--gsd``.

2. End-to-end (optional, ``--image``): the full ``model.process`` pipeline (raster I/O,
   uniform-tile skipping, Hann-window stitching, writing output) on a real orthomosaic.
   Note that uniform/NoData tiles are skipped by the pipeline, so this depends on the image.

Usage:
    uv run python scripts/benchmark_inference.py
    uv run python scripts/benchmark_inference.py --gsd 0.03 --image ortho.tif --gpu-batch-sizes 1 4 8 16 32

Requirements: nvidia-smi on PATH for GPU name/VRAM and peak GPU memory.
"""

# ruff: noqa: T201, S404, S603, S607, D103, DOC201, DOC501, TC003
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

RESULT_PREFIX = "BENCHMARK_RESULT:"


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------


def _nvidia_smi(*args: str) -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=10, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return [line.strip() for line in out.strip().splitlines() if line.strip()]


def gpu_info(gpu_id: int) -> dict:
    """Return name, VRAM and driver version of the GPU via nvidia-smi."""
    lines = _nvidia_smi(
        f"--id={gpu_id}", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"
    )
    if not lines:
        return {}
    name, mem_mib, driver = (s.strip() for s in lines[0].split(","))
    return {"name": name, "vram_gb": round(int(mem_mib) / 1024, 1), "driver": driver}


def cpu_info() -> dict:
    """Return CPU model name and core counts."""
    name = platform.processor() or platform.machine()
    physical = None
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
        # Count unique (physical id, core id) pairs for physical cores
        cores, phys_id = set(), None
        for line in text.splitlines():
            if line.startswith("physical id"):
                phys_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                cores.add((phys_id, line.split(":", 1)[1].strip()))
        physical = len(cores) or None
    except OSError:
        pass
    return {"name": name, "physical_cores": physical, "logical_cores": os.cpu_count()}


# ---------------------------------------------------------------------------
# GPU memory monitor (runs inside the worker)
# ---------------------------------------------------------------------------


class GpuMemMonitor(threading.Thread):
    """Poll nvidia-smi for this process's GPU memory and the device's total used memory."""

    def __init__(self, gpu_id: int, interval: float = 0.1) -> None:  # noqa: D107
        super().__init__(daemon=True)
        self.gpu_id = gpu_id
        self.interval = interval
        self.pid = os.getpid()
        self._halt = threading.Event()
        self.baseline_device_mib = self._device_used()
        self.peak_process_mib: int | None = None
        self.peak_device_mib: int | None = None

    def _device_used(self) -> int | None:
        lines = _nvidia_smi(f"--id={self.gpu_id}", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
        return int(lines[0]) if lines else None

    def _process_used(self) -> int | None:
        lines = _nvidia_smi(
            f"--id={self.gpu_id}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
        )
        for line in lines:
            pid, mib = (s.strip() for s in line.split(","))
            if pid.isdigit() and int(pid) == self.pid and mib.isdigit():
                return int(mib)
        return None  # e.g. inside a container where PIDs don't match the host

    def run(self) -> None:  # noqa: D102
        while not self._halt.is_set():
            proc, dev = self._process_used(), self._device_used()
            if proc is not None:
                self.peak_process_mib = max(self.peak_process_mib or 0, proc)
            if dev is not None:
                self.peak_device_mib = max(self.peak_device_mib or 0, dev)
            self._halt.wait(self.interval)

    def stop(self) -> dict:
        """Stop polling and return the peak memory reading."""
        self._halt.set()
        self.join()
        delta = None
        if self.peak_device_mib is not None and self.baseline_device_mib is not None:
            delta = self.peak_device_mib - self.baseline_device_mib
        # Prefer the per-process number; fall back to the device-wide increase over baseline
        peak = self.peak_process_mib if self.peak_process_mib is not None else delta
        return {
            "peak_gpu_mem_gb": None if peak is None else round(peak / 1024, 2),
            "peak_gpu_mem_source": "process" if self.peak_process_mib is not None else "device_delta",
        }


# ---------------------------------------------------------------------------
# Worker: one (device, batch_size) configuration
# ---------------------------------------------------------------------------


def _timed_loop(fn: Callable[[], object], *, warmup: int, min_iters: int, min_seconds: float) -> list[float]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    start = time.perf_counter()
    while len(times) < min_iters or time.perf_counter() - start < min_seconds:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def run_worker(cfg: dict) -> dict:
    """Benchmark a single (device, batch size) configuration in this process."""
    import numpy as np
    import onnxruntime as ort

    from habitat_mapper import model_registry

    device, batch_size, gpu_id = cfg["device"], cfg["batch_size"], cfg["gpu_id"]

    monitor = GpuMemMonitor(gpu_id) if device == "cuda" else None
    if monitor:
        monitor.start()

    # Fresh model instance so the registry's shared (possibly already-loaded) instance isn't reused
    base = model_registry[cfg["model"], cfg["revision"]]
    model = type(base)(base.cfg.model_copy(deep=True))
    model_path = model.cfg.get_local_model_path(quiet=True)

    # Build the session ourselves so the execution provider is forced rather than auto-selected
    so = ort.SessionOptions()
    if device == "cpu":
        providers = ["CPUExecutionProvider"]
        if cfg["cpu_threads"]:
            so.intra_op_num_threads = cfg["cpu_threads"]
    else:
        providers = [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]

    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    load_s = time.perf_counter() - t0
    active = sess.get_providers()
    if device == "cuda" and active[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDAExecutionProvider not active (got {active}). Check onnxruntime-gpu/CUDA install.")
    model._ONNXModel__ort_sess = sess  # type: ignore[attr-defined]  # inject into the model's private session cache

    tile = model.input_size or cfg["tile_size"]
    channels = model.cfg.input_channels

    rng = np.random.default_rng(0)
    batch = rng.integers(0, 256, size=(batch_size, channels, tile, tile), dtype=np.uint8)
    pre = model._preprocess(batch).astype(np.float32)
    input_name = sess.get_inputs()[0].name

    loop_kw = {"warmup": cfg["warmup"], "min_iters": cfg["min_iters"], "min_seconds": cfg["min_seconds"]}

    # Full predict path: normalisation (numpy, CPU) + session run
    predict_times = _timed_loop(lambda: model._predict(batch), **loop_kw)
    # Session run only (input already normalised float32)
    run_times = _timed_loop(lambda: sess.run(None, {input_name: pre}), **loop_kw)

    result = {
        **cfg,
        "tile_size": tile,
        "providers": active,
        "session_load_s": round(load_s, 3),
        "iters": len(predict_times),
        "tiles_per_s": batch_size * len(predict_times) / sum(predict_times),
        "tiles_per_s_run_only": batch_size * len(run_times) / sum(run_times),
        "median_batch_latency_s": float(np.median(predict_times)),
        "intra_op_threads": so.intra_op_num_threads or "default",
    }

    if cfg.get("image"):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.tif"
            t0 = time.perf_counter()
            model.process(cfg["image"], out, batch_size=batch_size, crop_size=tile, blur_kernel_size=0, quiet=True)
            result["e2e_seconds"] = time.perf_counter() - t0

    if monitor:
        result.update(monitor.stop())
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def launch(cfg: dict) -> dict:
    """Run one configuration in a fresh subprocess and return its parsed result."""
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}  # match nvidia-smi device numbering
    if cfg["device"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    label = f"{cfg['device']} bs={cfg['batch_size']}"
    print(f"→ running {label} ...", flush=True)
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", json.dumps(cfg)], env=env, capture_output=True, text=True
    )
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            res = json.loads(line[len(RESULT_PREFIX) :])
            print(f"  {res['tiles_per_s']:.2f} tiles/s", flush=True)
            return res
    err = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
    print(f"  FAILED ({label}):\n    " + "\n    ".join(err), flush=True)
    return {**cfg, "error": "\n".join(err)}


def image_area(image: str, gsd_override: float | None) -> tuple[float, float]:
    """Return (gsd_m, area_km2) for a raster."""
    import rasterio

    with rasterio.open(image) as src:
        gsd = gsd_override
        if gsd is None:
            if src.crs is None or not src.crs.is_projected:
                raise SystemExit("Image CRS is not projected; pass --image-gsd to give its pixel size in metres.")
            gsd = abs(src.transform.a)
        return gsd, src.height * src.width * gsd**2 / 1e6


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="kelp-rgb")
    p.add_argument("--revision", default="20240722")
    p.add_argument("--tile-size", type=int, default=1024, help="Ignored if the model has a fixed input size.")
    p.add_argument("--gpu-batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--cpu-batch-sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--cpu-threads", type=int, default=0, help="ORT intra-op threads on CPU (0 = ORT default).")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--gsd", type=float, nargs="+", default=[0.03, 0.05, 0.10], help="Metres/pixel for s/km².")
    p.add_argument("--image", help="Optional GeoTIFF for an end-to-end `model.process` timing.")
    p.add_argument("--image-gsd", type=float, help="Pixel size (m) of --image if its CRS isn't projected.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--min-iters", type=int, default=5)
    p.add_argument("--gpu-seconds", type=float, default=20.0, help="Minimum timed seconds per GPU config.")
    p.add_argument("--cpu-seconds", type=float, default=60.0, help="Minimum timed seconds per CPU config.")
    p.add_argument("--skip-cpu", action="store_true")
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--out", default="benchmark_results.json")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        print(RESULT_PREFIX + json.dumps(run_worker(json.loads(args.worker))), flush=True)
        return

    import onnxruntime as ort

    common = {
        "model": args.model,
        "revision": args.revision,
        "tile_size": args.tile_size,
        "gpu_id": args.gpu_id,
        "cpu_threads": args.cpu_threads,
        "warmup": args.warmup,
        "min_iters": args.min_iters,
        "image": args.image,
    }
    runs: list[dict] = []
    if not args.skip_gpu:
        for bs in args.gpu_batch_sizes:
            runs.append(launch({**common, "device": "cuda", "batch_size": bs, "min_seconds": args.gpu_seconds}))
    if not args.skip_cpu:
        for bs in args.cpu_batch_sizes:
            runs.append(launch({**common, "device": "cpu", "batch_size": bs, "min_seconds": args.cpu_seconds}))

    ok = [r for r in runs if "error" not in r]
    if not ok:
        raise SystemExit("All benchmark runs failed.")

    tile = ok[0]["tile_size"]
    stride = tile // 2  # matches ProcessingConfig.stride (50% overlap)
    overlap_pct = round(100 * (tile - stride) / tile)

    def per_km2(tiles_per_s: float, gsd: float) -> float:
        tiles_per_km2 = 1e6 / (stride * gsd) ** 2
        return tiles_per_km2 / tiles_per_s

    img = image_area(args.image, args.image_gsd) if args.image else None

    for r in ok:
        r["s_per_km2"] = {str(g): per_km2(r["tiles_per_s"], g) for g in args.gsd}
        if img and "e2e_seconds" in r:
            r["e2e_s_per_km2"] = r["e2e_seconds"] / img[1]

    best = {}
    for dev in ("cuda", "cpu"):
        dev_runs = [r for r in ok if r["device"] == dev]
        if dev_runs:
            best[dev] = max(dev_runs, key=lambda r: r["tiles_per_s"])

    hw = {"gpu": gpu_info(args.gpu_id), "cpu": cpu_info(), "onnxruntime": ort.__version__}
    report = {
        "hardware": hw,
        "model": f"{args.model}@{args.revision}",
        "tile_size": tile,
        "stride": stride,
        "overlap_pct": overlap_pct,
        "image": {"path": args.image, "gsd_m": img[0], "area_km2": img[1]} if img else None,
        "runs": runs,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    # ---- Summary ----
    print("\n" + "=" * 100)
    print(f"Model {args.model}@{args.revision} | tile {tile}x{tile} px | stride {stride} px ({overlap_pct}% overlap)")
    print(f"GPU: {hw['gpu'] or 'n/a'}")
    print(f"CPU: {hw['cpu']} | onnxruntime {hw['onnxruntime']}")
    if img:
        print(f"Image: {args.image} | GSD {img[0]:.4f} m | area {img[1]:.4f} km²")
    print("-" * 100)
    hdr = f"{'device':<6}{'bs':>4}{'tiles/s':>10}{'run-only':>10}{'lat/batch s':>13}{'peak GPU GB':>13}"
    hdr += "".join(f"{'s/km² @' + str(g) + 'm':>16}" for g in args.gsd)
    hdr += f"{'e2e s':>10}{'e2e s/km²':>12}" if img else ""
    print(hdr)
    for r in runs:
        if "error" in r:
            print(f"{r['device']:<6}{r['batch_size']:>4}  FAILED")
            continue
        mem = r.get("peak_gpu_mem_gb")
        line = (
            f"{r['device']:<6}{r['batch_size']:>4}{r['tiles_per_s']:>10.2f}{r['tiles_per_s_run_only']:>10.2f}"
            f"{r['median_batch_latency_s']:>13.3f}{(f'{mem:.2f}' if mem is not None else '-'):>13}"
        )
        line += "".join(f"{r['s_per_km2'][str(g)]:>16.1f}" for g in args.gsd)
        if img:
            line += f"{r.get('e2e_seconds', float('nan')):>10.1f}{r.get('e2e_s_per_km2', float('nan')):>12.1f}"
        print(line)
    print("=" * 100)

    # ---- Paragraph ----
    if "cuda" in best and "cpu" in best:
        g, c = best["cuda"], best["cpu"]
        gsd = args.gsd[0]
        if img and "e2e_s_per_km2" in g and "e2e_s_per_km2" in c:
            res_txt, g_km2, c_km2 = f"{img[0] * 100:.1f} cm GSD", g["e2e_s_per_km2"], c["e2e_s_per_km2"]
        else:
            res_txt, g_km2, c_km2 = f"{gsd * 100:.0f} cm GSD", g["s_per_km2"][str(gsd)], c["s_per_km2"][str(gsd)]
        gpu = hw["gpu"]
        gpu_txt = f"an NVIDIA {gpu['name']} ({gpu['vram_gb']:.0f} GB VRAM)" if gpu else "[GPU]"
        cpu = hw["cpu"]
        cores = f", {cpu['physical_cores']} cores" if cpu.get("physical_cores") else ""
        mem = g.get("peak_gpu_mem_gb")
        print(
            "\nCompute requirements. The model supports both CPU and CUDA-enabled GPU inference, with GPU inference "
            "substantially faster due to the convolutional architecture's parallelizability. "
            f"Benchmarked on {gpu_txt} and {cpu['name']}{cores}, tiled inference (tile size {tile}×{tile} px, "
            f"{overlap_pct}% overlap) achieved {g['tiles_per_s']:.1f} tiles/s on GPU and {c['tiles_per_s']:.2f} "
            f"tiles/s on CPU (a {g['tiles_per_s'] / c['tiles_per_s']:.0f}× speedup), corresponding to approximately "
            f"{g_km2:.0f} s per km² of coverage at {res_txt} on GPU versus {c_km2 / 60:.0f} min on CPU. "
            f"Peak GPU memory usage was {mem if mem is not None else '[E]'} GB at a batch size of {g['batch_size']}."
        )
        print(
            "\nNotes: tiles/s includes numpy normalisation + host<->device copies (see 'run-only' for session time"
            " alone). Peak GPU memory includes the CUDA context and ORT's arena over-allocation (per-process when"
            " nvidia-smi exposes PIDs, otherwise device-wide increase over baseline). Synthetic s/km² assumes"
            " interior tiles; the end-to-end figure includes I/O and stitching, and skips uniform/NoData tiles."
        )
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
