# PyInstaller spec for the Habitat-Mapper desktop app and `hab` command line tool, which share one folder. Build with:
#   uv run --group freeze --extra gui pyinstaller packaging/habitat-mapper.spec --noconfirm
import os
import sys
from importlib.metadata import PackageNotFoundError, version

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
# App icon for the executables and macOS app bundle
ICON = os.path.join(SPECPATH, "icons", "habitat-mapper")
excludes = ["torch", "segmentation_models_pytorch", "onnxscript", "matplotlib", "tkinter", "IPython"]


def analyse(script):
    return Analysis([script], binaries=binaries, datas=datas, hiddenimports=hiddenimports, excludes=excludes)


cli = analyse("hab_cli.py")
cli_exe = EXE(
    PYZ(cli.pure), cli.scripts, exclude_binaries=True, name="hab", console=True, upx=False, icon=f"{ICON}.ico"
)

gui = analyse("hab_gui.py")
gui_exe = EXE(
    PYZ(gui.pure),
    gui.scripts,
    exclude_binaries=True,
    name="Habitat-Mapper",
    console=False,
    upx=False,
    icon=f"{ICON}.ico",
)

# The app comes first, so it is what the macOS app bundle runs
coll = COLLECT(
    gui_exe, gui.binaries, gui.datas, cli_exe, cli.binaries, cli.datas, name="habitat-mapper", upx=False
)

if sys.platform == "darwin":
    app_version = version("habitat-mapper")
    BUNDLE(
        coll,
        name="Habitat-Mapper.app",
        icon=f"{ICON}.icns",
        bundle_identifier="org.hakai.habitat-mapper",
        version=app_version,
        info_plist={
            "CFBundleShortVersionString": app_version,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "14.0",
            # BUNDLE takes console=True from the hab CLI, the last executable in the collection, and marks the
            # app background only, which hides it from the Dock, the app switcher and the menu bar
            "LSBackgroundOnly": False,
        },
    )
