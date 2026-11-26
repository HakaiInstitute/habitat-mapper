"""Plain descriptions of models and inputs for the desktop app, kept free of Qt so they can be tested on their own."""

from __future__ import annotations

from typing import TYPE_CHECKING

from habitat_mapper.reader import SAFEReader, is_safe_zip

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pydantic import ValidationError

    from habitat_mapper.config import ModelConfig

GEOTIFF_SUFFIXES = (".tif", ".tiff")


def reads_safe(cfg: ModelConfig) -> bool:
    """Check whether a model reads Sentinel-2 .SAFE folders, or their .zip downloads, rather than GeoTIFFs.

    Returns:
        True if the model's inputs are .SAFE folders or zips.
    """
    return cfg.reader_cls is SAFEReader


def input_description(cfg: ModelConfig) -> str:
    """Describe the imagery a model needs, e.g. "RGB GeoTIFF".

    Returns:
        The description.
    """
    if reads_safe(cfg):
        return "Sentinel-2 L2A .SAFE.zip or .SAFE folder"
    return {
        3: "RGB GeoTIFF",
        4: "RGB + near-infrared GeoTIFF",
        8: "8-band PlanetScope GeoTIFF",
    }.get(cfg.input_channels, f"{cfg.input_channels}-band GeoTIFF")


def default_band_order(cfg: ModelConfig) -> str:
    """Band order the model uses when none is given, as typed in the band order box.

    Returns:
        Band numbers separated by commas, e.g. "1, 2, 3".
    """
    return ", ".join(str(band) for band in range(1, cfg.input_channels + 1))


def accepted_inputs(paths: Iterable[Path], safe: bool) -> list[Path]:
    """Pick the paths a model can read out of files and folders dropped on the app.

    Args:
        paths: Dropped paths.
        safe: Whether the model reads Sentinel-2 .SAFE folders and zips. Otherwise it reads GeoTIFFs.

    Returns:
        The usable paths, in the order given.
    """
    if safe:
        return [path for path in paths if (path.is_dir() and path.suffix.upper() == ".SAFE") or is_safe_zip(path)]
    return [path for path in paths if path.is_file() and path.suffix.lower() in GEOTIFF_SUFFIXES]


# Names of the advanced options, as the window labels them
_OPTION_LABELS = {
    "crop_size": "Tile size",
    "blur_kernel_size": "Median blur",
    "morph_kernel_size": "Morphological filter",
    "band_order": "Band order",
}


def describe_option_errors(error: ValidationError) -> str:
    """Describe what is wrong with the advanced options, one line per problem, naming each option.

    Returns:
        The description, e.g. "Median blur: Must be an odd number, or 0 to turn the filter off."
    """
    lines = []
    for problem in error.errors():
        field = str(problem["loc"][0]) if problem["loc"] else ""
        label = _OPTION_LABELS.get(field, field or "Options")
        lines.append(f"{label}: {problem['msg'].removeprefix('Value error, ')}")
    return "\n".join(lines)
