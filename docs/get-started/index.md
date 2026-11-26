# Get Started

This guide takes you from installing Habitat-Mapper to a finished habitat map you can open in ArcGIS Pro or QGIS.
You don't need to know Python. There are two ways to run Habitat-Mapper. Both use the same models and give the same
results.

<div class="grid cards" markdown="1">

-   :material-application-outline: **Desktop app**

    ---
    For Windows and macOS. Install it like any other program, then pick your images and a model in a window. No
    terminal needed.

    [:material-arrow-right: Use the desktop app](desktop-app.md)

-   :material-console: **Command line**

    ---
    For Windows, macOS and Linux. Use it to script or batch your processing, to run on a server, or to use CUDA on
    an NVIDIA GPU. The steps below assume no command line experience.

    [:material-arrow-right: Start with step 1](terminal.md)

</div>

## Command line steps

Follow these in order; each one ends with a link to the next.

<div class="grid cards" markdown="1" style="grid-template-columns: 1fr;">

-   :material-console: **1. Using the Terminal**

    ---
    Open a terminal and learn the handful of commands you'll need. Skip this if you already use one.

    [:material-arrow-right: Learn the basics](terminal.md)

-   :material-download: **2. Install**

    ---
    Install `uv`, create an environment and install Habitat-Mapper.

    [:material-arrow-right: Install](install.md)

-   :material-play: **3. Your First Segmentation**

    ---
    Pick a model and run it on your image with the `hab` command.

    [:material-arrow-right: Run it](first-segmentation.md)

</div>

## Then view your results

Whichever way you ran the model, the last step is
[View Results in GIS](view-results.md): open the output in ArcGIS Pro or QGIS and make the classes visible.

After that, the [guides](../guides/post-process-kelp-drone.md) show how to clean up results and handle other
imagery.

!!! tip "Need help?"
    See [Troubleshooting & FAQ](../help/faq.md), or
    [open an issue](https://github.com/HakaiInstitute/habitat-mapper/issues).
