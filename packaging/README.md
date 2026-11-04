# Desktop builds

The `hab` command line tool is frozen with PyInstaller into a folder that runs without Python installed. Frozen builds
only support the WebGPU and CPU devices; CUDA users install with pip instead.

```bash
uv sync --group freeze
```

On Windows and Linux, swap `onnxruntime-gpu` for the much smaller CPU package of the same version. WebGPU comes from
the `onnxruntime-ep-webgpu` plugin, which works with either:

```bash
ORT_VERSION=$(uv run --no-sync python -c "from importlib.metadata import version; print(version('onnxruntime-gpu'))")
uv pip uninstall onnxruntime-gpu
uv pip install "onnxruntime==$ORT_VERSION"
```

Then build and check it:

```bash
uv run --no-sync pyinstaller packaging/habitat-mapper.spec --noconfirm
dist/habitat-mapper/hab devices
```
