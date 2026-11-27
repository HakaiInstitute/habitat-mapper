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

The macOS disk image isn't code signed, so Gatekeeper warns about it the first time it runs.

## Windows code signing

Release builds of the Windows executables and installer are signed by [SignPath Foundation](https://signpath.org),
which signs open-source projects for free. Windows shows "SignPath Foundation" as the publisher. The
[desktop build workflow](../.github/workflows/desktop-build.yml) uploads `hab.exe` and `Habitat-Mapper.exe` for
signing, builds the installer from the signed copies, then signs the installer.

- Release tags use the `release-signing` policy. A maintainer approves each request in SignPath, and the job waits up
  to an hour for that.
- Manual runs of the workflow use `test-signing`, which needs no approval. Windows doesn't trust test signatures, so
  use these runs to check the signing steps work, not to publish.
- Pull requests aren't signed.

The workflow needs the `SIGNPATH_API_TOKEN` secret and the `SIGNPATH_ORGANIZATION_ID` variable. The SignPath project
is `habitat-mapper`, with the artifact configurations in [`windows/signpath/`](windows/signpath).

Inno Setup's uninstaller (`unins000.exe`) isn't signed. Signing it would need Inno Setup to call the signing service
while it builds.

Code signing policy: the maintainers listed in [CODEOWNERS](../CODEOWNERS) review changes and approve each release
signing request. Apart from reading images from URLs the user gives it, the app only goes online to download models
and check GitHub for new releases. It sends no other data.

## App icon

`packaging/icons/` holds the icon as an `.icns` for the macOS app bundle and an `.ico` for the Windows executables and
installer. The app sets its window icon from `src/habitat_mapper/gui/icon.png`.
