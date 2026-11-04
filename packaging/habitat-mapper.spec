# PyInstaller spec for the frozen `hab` command line tool. Build with:
#   uv run --group freeze pyinstaller packaging/habitat-mapper.spec --noconfirm
from importlib.metadata import PackageNotFoundError

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

# rasterio imports some of its modules from compiled code and ships GDAL and PROJ data files
datas = collect_data_files("habitat_mapper") + collect_data_files("rasterio")
# Package versions are read at runtime, for --version and `hab devices`
for package in ("habitat-mapper", "onnxruntime", "onnxruntime-gpu", "onnxruntime-ep-webgpu"):
    try:
        datas += copy_metadata(package)
    except PackageNotFoundError:
        pass
hiddenimports = collect_submodules("rasterio") + ["onnxruntime_ep_webgpu"]
# The WebGPU plugin is loaded by path at runtime, so PyInstaller can't find it by following imports
binaries = collect_dynamic_libs("onnxruntime_ep_webgpu")

a = Analysis(
    ["hab_cli.py"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "segmentation_models_pytorch", "onnxscript", "matplotlib", "tkinter", "IPython"],
)
pyz = PYZ(a.pure)

hab = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="hab",
    console=True,
    upx=False,
)

COLLECT(hab, a.binaries, a.datas, name="habitat-mapper", upx=False)
