"""Plain descriptions of models and inputs for the desktop app, kept free of Qt so they can be tested on their own."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import ValidationError

from habitat_mapper.config import SegmentOptions
from habitat_mapper.reader import SAFEReader, is_safe_zip

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

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


def parse_band_order(text: str) -> list[int] | None:
    """Read a band order as typed, e.g. "3, 2, 1" or "3 2 1".

    Returns:
        Band numbers, starting at 1, or None if nothing was typed, to use the model default.

    Raises:
        ValueError: If the text isn't band numbers, with a message for the user.
    """
    text = text.strip()
    if not text:
        return None
    try:
        return [int(band) for band in re.split(r"[\s,]+", text)]
    except ValueError:
        raise ValueError("Band order must be band numbers separated by commas, e.g. 3, 2, 1.") from None


def parse_options(*, crop_size: int, blur: int, morph: int, band_order: str) -> SegmentOptions:
    """Read the advanced options as the window shows them.

    Args:
        crop_size: Tile size, or 0 for the model's.
        blur: Median blur kernel size, or 0 for none.
        morph: Morphological filter kernel size, or 0 for none.
        band_order: Band order as typed. See `parse_band_order`.

    Returns:
        The options.

    Raises:
        ValueError: If an option is invalid, with a message for the user naming each problem.
    """
    try:
        return SegmentOptions(
            crop_size=crop_size or None,
            # Batches larger than 1 were never faster in testing
            batch_size=1,
            blur_kernel_size=blur,
            morph_kernel_size=morph,
            band_order=parse_band_order(band_order),
        )
    except ValidationError as e:
        raise ValueError(describe_option_errors(e)) from None
