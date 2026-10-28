---
title: Habitat-Mapper
hide:
  - navigation
  - toc
---

# Habitat-Mapper {: style="display: none;" }

<div align="center" markdown="1">

<a href="https://github.com/HakaiInstitute/habitat-mapper/actions/workflows/test.yml">
    <img alt="Tests" src="https://github.com/HakaiInstitute/habitat-mapper/actions/workflows/test.yml/badge.svg?style=flat-square" />
</a>
<a href="https://pypi.org/project/habitat-mapper">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/habitat-mapper?style=flat-square" />
</a>
<a href="https://github.com/HakaiInstitute/habitat-mapper/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/pypi/l/habitat-mapper?style=flat-square" />
</a>
<a href="https://zenodo.org/badge/latestdoi/462897183">
    <img alt="DOI" src="https://zenodo.org/badge/462897183.svg?style=flat-square" />
</a>

![Kelp Animation](images/habitat_mapper_optimized.gif){ width=600 }

## **Habitat Segmentation with AI**

*(The artist formerly known as "Kelp-O-Matic")*

[Get Started :material-arrow-right:](get-started/index.md){ .md-button .md-button--primary }
[View on GitHub :material-github:](https://github.com/HakaiInstitute/habitat-mapper){ .md-button }

</div>

---

## What is Habitat-Mapper?

Habitat-Mapper is an open-source tool from the [Hakai Institute](https://hakai.org) that maps coastal habitats in
drone, aerial and satellite imagery using deep learning. Give it a GeoTIFF and it returns a map of where each
species was found, ready to open in ArcGIS Pro or QGIS.

<div class="grid cards" markdown>

-   :material-leaf: **[Kelp from drones](models/kelp-drone.md)**

    ---
    *Macrocystis* and *Nereocystis* canopy in RGB and RGB+NIR drone imagery.

-   :material-satellite-variant: **[Kelp from satellites](models/kelp-satellite.md)**

    ---
    Kelp canopy presence in PlanetScope 8-band and Sentinel-2 imagery.

-   :material-waves: **[Mussels & barnacles](models/intertidal.md)**

    ---
    Mussels and gooseneck barnacles in high-resolution RGB drone imagery.

-   :material-grass: **[Eelgrass](models/eelgrass.md)**

    ---
    Eelgrass (*Zostera marina*) in RGB drone imagery.

</div>

---

## Choose your path

<div class="grid cards" markdown>

-   :material-school: **New to the terminal or Python?**

    ---
    Our step-by-step guide assumes no prior experience. It covers opening a terminal, installing the
    software, processing your first image and viewing the results in GIS.

    [:material-arrow-right: Start the Get Started guide](get-started/index.md)

-   :material-console: **Comfortable with Python or the CLI?**

    ---
    ```bash
    pip install habitat-mapper
    hab models
    hab segment kelp-rgb input.tif output.tif
    ```

    [CLI reference](reference/cli.md) · [Python API](reference/python-api.md) ·
    [Cheat sheet](reference/cheat-sheet.md) · [Models](models/index.md)

</div>

---

!!! quote "Using Habitat-Mapper in your research?"
    Please **[cite the software](cite.md)** and the papers behind the models you used. Habitat-Mapper is built on
    data and models from many contributors. See the **[credits](credits.md)**.

**Questions or feedback?** [Email us](mailto:habitat.mapper.support@hakai.org) or
[open an issue](https://github.com/HakaiInstitute/habitat-mapper/issues).
Habitat-Mapper is released under the
[MIT license](https://github.com/HakaiInstitute/habitat-mapper/blob/main/LICENSE).
