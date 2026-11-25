"""Downsampled previews of input images and segmentation outputs, kept free of Qt so they can be tested on their own."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

from habitat_mapper.reader import SafeProduct, is_safe_zip

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rasterio.windows import Window

    from habitat_mapper.config import OutputClass

# Largest dimension of previews, in pixels
PREVIEW_SIZE = 2048
# Haze over the parts of a live preview that the model has seen but that aren't finished yet
PENDING_RGBA = np.array([255, 255, 255, 90], dtype=np.uint8)

# Colours for class values 1-10 of a segmentation, for models that don't name their own. Class 0 and nodata are
# transparent
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


def class_palette(classes: Sequence[OutputClass] = ()) -> np.ndarray:
    """Look-up table of colours for every class value a uint8 segmentation can hold.

    Args:
        classes: Classes the model names, whose colours replace the defaults for their values.

    Returns:
        A (256, 3) uint8 array of RGB colours, indexed by class value.
    """
    palette = np.full((256, 3), 255, dtype=np.uint8)
    palette[: len(CLASS_COLOURS)] = CLASS_COLOURS
    for output_class in classes:
        if output_class.value < len(palette):
            hex_colour = output_class.colour.removeprefix("#")
            palette[output_class.value] = [int(hex_colour[i : i + 2], 16) for i in (0, 2, 4)]
    return palette


DEFAULT_PALETTE = class_palette()


def preview_shape(height: int, width: int, max_size: int = PREVIEW_SIZE) -> tuple[int, int]:
    """Size of the preview of an image, scaled down to fit in max_size.

    Returns:
        Preview height and width in pixels.
    """
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


def _true_colour_image(safe_path: Path) -> str:
    try:
        return SafeProduct(safe_path).find("GRANULE/*/IMG_DATA/*_TCI_10m.jp2")
    except FileNotFoundError:
        raise ValueError(f"Could not find a true colour image in {safe_path}") from None


def read_input_preview(path: Path, max_size: int = PREVIEW_SIZE, band_order: list[int] | None = None) -> np.ndarray:
    """Read an RGB preview of an input image, using its overviews when it has them.

    Args:
        path: GeoTIFF, or Sentinel-2 .SAFE directory or .zip, in which case its true colour image is used.
        max_size: Largest dimension of the preview, in pixels.
        band_order: Band order the model is run with. Its first three bands are shown as red, green and blue.

    Returns:
        A (height, width, 3) uint8 array.
    """
    source: Path | str = path
    if path.is_dir() or is_safe_zip(path):
        source, band_order = _true_colour_image(path), [1, 2, 3]

    with rasterio.open(source) as src:
        bands = (band_order or [1, 2, 3])[:3] if src.count >= 3 else [1, 1, 1]
        height, width = preview_shape(src.height, src.width, max_size)
        data = src.read(bands, out_shape=(3, height, width), resampling=Resampling.nearest)
        valid = src.read_masks(1, out_shape=(height, width), resampling=Resampling.nearest) > 0

    if data.dtype != np.uint8:
        data = np.stack([_stretch(band, valid) for band in data])
    return np.ascontiguousarray(np.moveaxis(data, 0, -1))


def colourise(classes: np.ndarray, nodata: float | None, palette: np.ndarray = DEFAULT_PALETTE) -> np.ndarray:
    """Colour a segmentation for drawing over its image.

    Args:
        classes: (height, width) array of class values.
        nodata: Value for pixels outside the image, if any.
        palette: Colour of each class value, from class_palette.

    Returns:
        A (height, width, 4) uint8 RGBA array, transparent where the class is 0 or nodata.
    """
    rgba = np.zeros((*classes.shape, 4), dtype=np.uint8)
    rgba[..., :3] = palette[np.clip(classes, 0, len(palette) - 1)]
    visible = classes != 0
    if nodata is not None:
        visible &= classes != nodata
    rgba[..., 3] = np.where(visible, 255, 0)
    return rgba


def read_output_preview(path: Path, max_size: int = PREVIEW_SIZE, palette: np.ndarray = DEFAULT_PALETTE) -> np.ndarray:
    """Read a segmentation output as a colour overlay.

    Args:
        path: Segmentation GeoTIFF written by Habitat-Mapper.
        max_size: Largest dimension of the preview, in pixels.
        palette: Colour of each class value, from class_palette.

    Returns:
        A (height, width, 4) uint8 RGBA array, transparent where the class is 0 or nodata.
    """
    with rasterio.open(path) as src:
        height, width = preview_shape(src.height, src.width, max_size)
        classes = src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        nodata = src.nodata
    return colourise(classes, nodata, palette)


class LiveOverlay:
    """Preview of a result that fills in as each part of it is written.

    Tiles overlap, and the model's predictions where they overlap are blended, so most of a tile isn't finished until
    the tiles to its right and below have been processed too. Those parts are covered in a haze until they are written.

    Parts are drawn by the processing thread and read by the UI thread. Reading a part-drawn tile only shows it a
    moment early, so there's no locking.

    Attributes:
        rgba: (height, width, 4) uint8 preview, the same size as the preview of the input image.
    """

    def __init__(
        self,
        height: int,
        width: int,
        nodata: float | None,
        max_size: int = PREVIEW_SIZE,
        palette: np.ndarray = DEFAULT_PALETTE,
    ) -> None:
        """Start an empty preview.

        Args:
            height: Height of the image being processed, in pixels.
            width: Width of the image being processed, in pixels.
            nodata: Value the result uses for pixels outside the image.
            max_size: Largest dimension of the preview, in pixels.
            palette: Colour of each class value, from class_palette.
        """
        preview_height, preview_width = preview_shape(height, width, max_size)
        self._scale_y = preview_height / height
        self._scale_x = preview_width / width
        self._nodata = nodata
        self._palette = palette
        self.rgba = np.zeros((preview_height, preview_width, 4), dtype=np.uint8)
        self._written = np.zeros((preview_height, preview_width), dtype=bool)

    def tile_to_preview(self, window: Window) -> tuple[int, int, int, int]:
        """Scale a window of the image to the preview, including any part past the edge of the image.

        Returns:
            Left, top, right and bottom of the window in preview pixels.
        """
        left = math.floor(window.col_off * self._scale_x)
        top = math.floor(window.row_off * self._scale_y)
        right = max(math.ceil((window.col_off + window.width) * self._scale_x), left + 1)
        bottom = max(math.ceil((window.row_off + window.height) * self._scale_y), top + 1)
        return left, top, right, bottom

    def to_preview(self, window: Window) -> tuple[int, int, int, int]:
        """Scale the part of a window inside the image to the preview, covering at least one preview pixel.

        Returns:
            Left, top, right and bottom of the window in preview pixels.
        """
        height, width = self.rgba.shape[:2]
        left, top, right, bottom = self.tile_to_preview(window)
        left, top = min(left, width - 1), min(top, height - 1)
        return left, top, max(min(right, width), left + 1), max(min(bottom, height), top + 1)

    def mark_pending(self, window: Window) -> None:
        """Cover the parts of a tile that aren't finished in a haze.

        Args:
            window: A tile the model is about to process.
        """
        left, top, right, bottom = self.to_preview(window)
        region = self.rgba[top:bottom, left:right]
        region[~self._written[top:bottom, left:right]] = PENDING_RGBA

    def paste(self, window: Window, classes: np.ndarray) -> None:
        """Draw a part of the result.

        Args:
            window: Where the part is in the image.
            classes: Class values of the part, the same size as the window.
        """
        left, top, right, bottom = self.to_preview(window)
        small = cv2.resize(classes, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
        self.rgba[top:bottom, left:right] = colourise(small, self._nodata, self._palette)
        self._written[top:bottom, left:right] = True
