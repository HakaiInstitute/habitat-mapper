# Desktop App

The Habitat-Mapper desktop app runs the same models as the `hab` command, without a terminal or a Python install.
Download it, install it like any other program, and pick your images and a model in a window.

!!! info "Desktop app or command line?"
    The desktop app runs models on your GPU through WebGPU, which works with most NVIDIA, AMD, Intel and Apple GPUs, or
    on your CPU if there's no GPU it can use. To use CUDA on an NVIDIA GPU, or to script Habitat-Mapper, install the
    [command line version](install.md) instead.

## 1. Download

Download the installer for your computer from the
[latest release](https://github.com/HakaiInstitute/habitat-mapper/releases/latest), under **Assets**:

| Computer | File |
|---|---|
| Windows 10 or 11 (64-bit) | `Habitat-Mapper-<version>-windows-x64-setup.exe` |
| Mac with Apple silicon (M1 or newer), macOS 14 or newer | `Habitat-Mapper-<version>-macos-arm64.dmg` |

## 2. Install

=== "Windows"

    1. Double click the downloaded `.exe`.
    2. Windows may show **"Windows protected your PC"**, because the installer isn't signed yet. Click
       **More info**, then **Run anyway**.
    3. Follow the installer. It installs for your user only, so it doesn't need administrator rights.

    Habitat-Mapper is then in the Start menu.

    !!! warning "If it doesn't start"
        Habitat-Mapper needs the Microsoft Visual C++ Redistributable, which most computers already have. If the app
        closes straight away, install it from <https://aka.ms/vs/17/release/vc_redist.x64.exe> and try again.

=== "macOS"

    1. Double click the downloaded `.dmg`, then drag **Habitat-Mapper** into **Applications**.
    2. Open Habitat-Mapper from Applications. macOS says it can't check the app for malicious software, because the
       app isn't signed yet. Click **Done**.
    3. Open **System Settings → Privacy & Security**, scroll down to the message about Habitat-Mapper, and click
       **Open Anyway**. Enter your password if asked.

    You only need to do this the first time.

## 3. Segment your images

1. Choose a **Model**. See [Models](../models/index.md) to pick the right one for your imagery.
2. Click **Add images…** to add GeoTIFFs, or **Add SAFE folder…** to add a Sentinel-2 L2A `.SAFE` folder. You can
   add as many as you like.
3. Choose an **Output folder**. Each result is saved there as `<image name>_<model>.tif`.
4. Click **Run**.

The model downloads the first time you use it, which needs an internet connection. To work offline later, download
models ahead of time from **Tools → Models…**.

While it runs, the bar at the bottom shows progress through the current image and the hardware it's running on. Click
**Cancel** to stop after the current tiles. If an image fails, the others still run; hover over its status to see why.

Select an image in the list to preview it, with its result on top once it's done. Scroll to zoom, drag to pan, and
double click to fit the image again. The **Result opacity** slider fades the result in and out.

**Advanced options** has the same settings as [`hab segment`](../reference/cli.md), such as band order and the
[processing options](../guides/tune-processing.md). Leave them alone unless you need them.

Then [view the results in GIS](view-results.md).

## Updates

The app tells you when a new version is available. Download and install it the same way; it replaces the old one.

## Getting help

If something doesn't work, choose **Tools → Copy diagnostics** and paste the result into an email or a
[GitHub issue](https://github.com/HakaiInstitute/habitat-mapper/issues). It has the app version, your system and GPU,
and the end of the log. **Tools → Open log folder** shows the full log files.

!!! note "Running on: CPU (slower)"
    This means the app couldn't use your GPU. Updating your graphics driver usually fixes it. On a virtual machine or
    remote desktop without GPU access, the CPU is the only option.
