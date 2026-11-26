"""Tests for the `hab models`, `revisions`, `devices` and `clean` commands, whose output people and scripts read."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from habitat_mapper import cache as cache_module, main
from habitat_mapper.cache import ModelCache
from habitat_mapper.config import ModelConfig
from habitat_mapper.registry import ModelRegistry

if TYPE_CHECKING:
    from pathlib import Path


def _config(name: str, revision: str, dependency: str, **changes: object) -> ModelConfig:
    return ModelConfig(
        name=name,
        revision=revision,
        dependencies=[dependency],
        model_filename=dependency.rsplit("/", 1)[-1],
        description=f"{name} {revision}",
        **changes,  # type: ignore[arg-type]
    )


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModelCache:
    """A model cache in a temporary folder, with the folder Kelp-O-Matic left models in moved there too.

    Returns:
        The cache.
    """
    cache = ModelCache(tmp_path / "cache")
    monkeypatch.setattr(main, "model_cache", cache)
    # files() also lists Kelp-O-Matic's old cache, so it must not point at the real one
    legacy_root = tmp_path / "legacy"
    monkeypatch.setattr(cache_module.platformdirs, "user_cache_dir", lambda **kwargs: str(legacy_root))
    return cache


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache: ModelCache) -> ModelRegistry:
    """Models in each state the commands show: local, cached, not yet downloaded, and beta.

    Returns:
        The registry.
    """
    local = tmp_path / "local.onnx"
    local.touch()
    registry = ModelRegistry()
    for cfg in [
        _config("local", "20250101", str(local)),
        _config("cached", "20250101", "https://example.com/cached/20250101/model.onnx"),
        _config("remote", "20240101", "https://example.com/remote/20240101/model.onnx"),
        _config("remote", "20250101", "https://example.com/remote/20250101/model.onnx"),
        _config("remote", "20260101", "https://example.com/remote/20260101/model.onnx", beta=True),
        _config("trial", "20250101", "https://example.com/trial/20250101/model.onnx", beta=True),
    ]:
        registry.register_model(cfg)
    cached = registry.config("cached")
    path = cache.paths(cached)["model.onnx"]
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(main, "model_registry", registry)
    return registry


@pytest.fixture
def console(monkeypatch: pytest.MonkeyPatch) -> Console:
    console = Console(record=True, width=200, force_terminal=False)
    monkeypatch.setattr(main, "console", console)
    return console


def _hab(*args: str) -> int:
    with pytest.raises(SystemExit) as exited:
        main.app(list(args))
    return int(exited.value.code or 0)


def _rows(console: Console) -> dict[str, str]:
    """Rows of the printed table, by their first cell.

    Returns:
        Each row's text, with runs of spaces collapsed.
    """
    rows = {}
    for line in console.export_text(clear=False).splitlines():
        cells = [cell.strip() for cell in line.strip("│┃ ").split("│")]
        if cells and cells[0]:
            rows[cells[0]] = " | ".join(cells)
    return rows


@pytest.mark.usefixtures("registry")
def test_models_lists_each_latest_revision_and_whether_it_is_downloaded(console: Console) -> None:
    assert _hab("models") == 0

    rows = _rows(console)
    assert rows["local"] == "local | 20250101 | local 20250101 | Local"
    assert rows["cached"] == "cached | 20250101 | cached 20250101 | Cached"
    # The newest revision that isn't a beta
    assert rows["remote"] == "remote | 20250101 | remote 20250101 | Available"
    assert rows["trial (beta)"] == "trial (beta) | 20250101 | trial 20250101 | Available"


@pytest.mark.usefixtures("registry")
def test_revisions_lists_newest_first_and_marks_the_latest(console: Console) -> None:
    assert _hab("revisions", "--model", "remote") == 0

    rows = [row for name, row in _rows(console).items() if name.isdigit()]
    assert rows == [
        "20260101 |  | remote 20260101 | Available",
        "20250101 | ✓ | remote 20250101 | Available",
        "20240101 |  | remote 20240101 | Available",
    ]


@pytest.mark.usefixtures("registry", "console")
def test_unknown_model_is_an_error() -> None:
    assert _hab("revisions", "--model", "nope") != 0


def test_devices_lists_hardware_in_the_order_auto_tries_it(console: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = SimpleNamespace(device="cpu", hardware="Test CPU", ep_name="CPUExecutionProvider")
    gpu = SimpleNamespace(device="webgpu", hardware="Test GPU", ep_name="WebGpuExecutionProvider")
    cuda = SimpleNamespace(device="cuda", hardware="Test GPU", ep_name="CUDAExecutionProvider")
    monkeypatch.setattr(main, "find_accelerators", lambda: [cuda, gpu, cpu])
    monkeypatch.setattr(main, "select_accelerators", lambda device: [gpu, cpu])
    monkeypatch.setenv(main.DEVICE_ENV_VAR, "webgpu")

    assert _hab("devices") == 0

    rows = _rows(console)
    assert rows["webgpu"].endswith("| 1")
    assert rows["cpu"].endswith("| 2")
    assert rows["cuda"].endswith("only with --device")
    assert f"{main.DEVICE_ENV_VAR}=webgpu" in console.export_text(clear=False)


def _cached_files(cache: ModelCache, tmp_path: Path, count: int) -> list[Path]:
    files = [cache.root / "model" / f"{i}" / "model.onnx" for i in range(count - 1)]
    files.append(tmp_path / "legacy" / "models" / "kelp.onnx")
    for size, path in enumerate(files, start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    return files


@pytest.mark.parametrize("confirmed", [True, False])
def test_clean_deletes_every_cached_file_once_confirmed(
    cache: ModelCache, console: Console, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, confirmed: bool
) -> None:
    files = _cached_files(cache, tmp_path, 3)
    monkeypatch.setattr(main.Confirm, "ask", lambda *args, **kwargs: confirmed)

    assert _hab("clean") == 0

    assert [path.exists() for path in files] == [not confirmed] * 3
    assert "kelp.onnx" in console.export_text(clear=False)
    assert ("Cleared model cache" in console.export_text(clear=False)) is confirmed


def test_clean_shows_the_largest_files_and_counts_the_rest(
    cache: ModelCache, console: Console, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _cached_files(cache, tmp_path, 20)
    monkeypatch.setattr(main.Confirm, "ask", lambda *args, **kwargs: False)

    _hab("clean")

    text = console.export_text(clear=False)
    assert "... and 5 more files" in text
    # Largest first, so the legacy file, written last and largest, is listed
    assert "kelp.onnx" in text


@pytest.mark.usefixtures("cache")
def test_clean_with_an_empty_cache_says_so(console: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.Confirm, "ask", lambda *args, **kwargs: pytest.fail("asked to confirm"))

    assert _hab("clean") == 0

    assert "already empty" in console.export_text(clear=False)
