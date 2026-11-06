# Desktop builds

The desktop app and `hab` command line tool are frozen with PyInstaller into a folder that runs without Python
installed. Frozen builds only support the WebGPU and CPU devices; CUDA users install with pip instead.

CI builds these on every pull request that touches the code, and attaches the installers to each published GitHub
release (`.github/workflows/desktop-build.yml`). To build locally:

```bash
uv sync --group freeze --extra gui
```

On Windows, swap `onnxruntime-gpu` for the much smaller CPU package of the same version. WebGPU comes from the
`onnxruntime-ep-webgpu` plugin, which works with either:

```bash
ORT_VERSION=$(uv run --no-sync python -c "from importlib.metadata import version; print(version('onnxruntime-gpu'))")
uv pip uninstall onnxruntime-gpu
uv pip install "onnxruntime==$ORT_VERSION"
```

Then build and check it:

```bash
uv run --no-sync pyinstaller packaging/habitat-mapper.spec --noconfirm
uv run --no-sync python packaging/smoke_test.py dist/habitat-mapper/hab dist/habitat-mapper/Habitat-Mapper
```

This makes `dist/habitat-mapper/`, and `dist/Habitat-Mapper.app` on macOS. To make the installers:

- Windows: `iscc /DAppVersion=<version> packaging\windows\habitat-mapper.iss`, using [Inno Setup](https://jrsoftware.org/isinfo.php).
- macOS: put `Habitat-Mapper.app` and a link to `/Applications` in a folder, then
  `hdiutil create -volname Habitat-Mapper -srcfolder <folder> -format UDZO Habitat-Mapper.dmg`.

The installers aren't code signed, so Windows SmartScreen and macOS Gatekeeper warn about them the first time they run.
