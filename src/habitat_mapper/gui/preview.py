"""Downsampled previews of input images and segmentation outputs, kept free of Qt so they can be tested on their own."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.enums import Resampling

if TYPE_CHECKING:
    from pathlib import Path

# Colours for class values 1-10 of a segmentation. Class 0 and nodata are transparent
CLASS_COLOURS = np.array(
    [
        (0, 0, 0),
        (255, 215, 0),
        (255, 64, 129),
        (0, 229, 255),
        (118, 255, 3),
        (213, 0, 249),
        (255, 145, 0),
        (41, 121, 255),
        (255, 23, 68),
        (29, 233, 182),
        (255, 255, 255),
    ],
    dtype=np.uint8,
)


def _preview_shape(height: int, width: int, max_size: int) -> tuple[int, int]:
    scale = min(1.0, max_size / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _stretch(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Stretch a band to 0-255 between its 2nd and 98th percentiles.

    Returns:
        The stretched band as uint8.
    """
    values = band[valid]
    if values.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    low, high = np.percentile(values, (2, 98))
    if high <= low:
        high = low + 1
    return (np.clip((band.astype(np.float32) - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def _true_colour_image(safe_dir: Path) -> Path:
    try:
        return next(safe_dir.glob("GRANULE/**/IMG_DATA/**/*_TCI_10m.jp2"))
    except StopIteration:
        raise ValueError(f"Could not find a true colour image in {safe_dir}") from None


def read_input_preview(path: Path, max_size: int = 2048, band_order: list[int] | None = None) -> np.ndarray:
    """Read an RGB preview of an input image, using its overviews when it has them.

    Args:
        path: GeoTIFF, or Sentinel-2 .SAFE directory, in which case its true colour image is used.
        max_size: Largest dimension of the preview, in pixels.
        band_order: Band order the model is run with. Its first three bands are shown as red, green and blue.

    Returns:
        A (height, width, 3) uint8 array.
    """
    if path.is_dir():
        path, band_order = _true_colour_image(path), [1, 2, 3]

    with rasterio.open(path) as src:
        bands = (band_order or [1, 2, 3])[:3] if src.count >= 3 else [1, 1, 1]
        height, width = _preview_shape(src.height, src.width, max_size)
        data = src.read(bands, out_shape=(3, height, width), resampling=Resampling.nearest)
        valid = src.read_masks(1, out_shape=(height, width), resampling=Resampling.nearest) > 0

    if data.dtype != np.uint8:
        data = np.stack([_stretch(band, valid) for band in data])
    return np.ascontiguousarray(np.moveaxis(data, 0, -1))


def read_output_preview(path: Path, max_size: int = 2048) -> np.ndarray:
    """Read a segmentation output as a colour overlay.

    Args:
        path: Segmentation GeoTIFF written by Habitat-Mapper.
        max_size: Largest dimension of the preview, in pixels.

    Returns:
        A (height, width, 4) uint8 RGBA array, transparent where the class is 0 or nodata.
    """
    with rasterio.open(path) as src:
        height, width = _preview_shape(src.height, src.width, max_size)
        classes = src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        nodata = src.nodata

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = CLASS_COLOURS[np.clip(classes, 0, len(CLASS_COLOURS) - 1)]
    visible = classes != 0
    if nodata is not None:
        visible &= classes != nodata
    rgba[..., 3] = np.where(visible, 255, 0)
    return rgba
