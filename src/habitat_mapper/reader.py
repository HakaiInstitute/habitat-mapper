"""Abstract image reader interface and implementations for multiple file formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from defusedxml.ElementTree import parse as parse_xml
from loguru import logger
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

if TYPE_CHECKING:
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.windows import Window

_FLOAT32_MAX = float(np.finfo(np.float32).max)


class ImageReader(ABC):
    """Abstract base class for reading image data in tiles."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Height of the image in pixels."""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Width of the image in pixels."""
        pass

    @property
    @abstractmethod
    def num_bands(self) -> int:
        """Number of bands in the image."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> str:
        """Data type of the image (e.g., 'uint8', 'uint16')."""
        pass

    @property
    def crs(self) -> object:
        """Coordinate Reference System of the image.

        Returns:
            CRS object or None for non-georeferenced data.
        """
        return None

    @property
    def transform(self) -> object:
        """Geospatial affine transform.

        Returns:
            Affine transform object or None for non-georeferenced data.
        """
        return None

    @abstractmethod
    def read_window(
        self,
        window: Window,
        band_order: list[int] | None = None,
        boundless: bool = True,
        fill_value: int = 0,
    ) -> np.ndarray:
        """Read a rectangular window of data from the image.

        Args:
            window: Rasterio Window object specifying the region to read
            band_order: List of band indices (1-based) to read. If None, reads all bands in order
            boundless: If True, allows reading outside image bounds (padded with fill_value)
            fill_value: Value used for out-of-bounds pixels when boundless=True

        Returns:
            Numpy array of shape [bands, height, width] containing the image data
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the reader and release resources."""
        pass

    def __enter__(self) -> ImageReader:
        """Context manager entry.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Context manager exit.

        Args:
            exc_type: Exception type if raised in context.
            exc_val: Exception value if raised in context.
            exc_tb: Exception traceback if raised in context.
        """
        self.close()


class TIFFReader(ImageReader):
    """Reader for GeoTIFF and standard TIFF files using Rasterio."""

    def __init__(
        self,
        file_path: str | Path,
        **kwargs: object,
    ) -> None:
        """Initialize TIFFReader.

        Args:
            file_path: Path to the TIFF file.
            **kwargs: Additional keyword arguments (ignored for compatibility).

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {self.file_path}")

        self._src = rasterio.open(self.file_path)

    @property
    def height(self) -> int:
        """Height of the raster in pixels."""
        return self._src.height

    @property
    def width(self) -> int:
        """Width of the raster in pixels."""
        return self._src.width

    @property
    def num_bands(self) -> int:
        """Number of bands in the raster."""
        return self._src.count

    @property
    def dtype(self) -> str:
        """Data type of the raster."""
        return self._src.dtypes[0]

    @property
    def crs(self) -> object:
        """Coordinate Reference System of the raster.

        Returns:
            CRS object from rasterio dataset.
        """
        return self._src.crs

    @property
    def transform(self) -> object:
        """Geospatial affine transform of the raster.

        Returns:
            Affine transform object from rasterio dataset.
        """
        return self._src.transform

    @property
    def profile(self) -> dict[str, object]:
        """Full rasterio profile for the dataset."""
        return self._src.profile

    def read_window(
        self,
        window: Window,
        band_order: list[int] | None = None,
        boundless: bool = True,
        fill_value: int = 0,
    ) -> np.ndarray:
        """Read a window of data from the TIFF file.

        Args:
            window: Rasterio Window object specifying the region to read.
            band_order: List of band indices (1-based) to read. If None, reads all bands.
            boundless: If True, allows reading outside image bounds.
            fill_value: Value used for out-of-bounds pixels.

        Returns:
            Numpy array of shape [bands, height, width].

        Raises:
            ValueError: If band_order has a band index the image doesn't have.
        """
        if band_order is None:
            # Read all bands
            data = self._src.read(window=window, boundless=boundless, fill_value=fill_value)
        else:
            # Validate band indices
            if any(b < 1 or b > self.num_bands for b in band_order):
                raise ValueError(
                    f"Band order {band_order} is invalid for image with {self.num_bands} bands. "
                    "Please specify band indices between 1 and the number of bands in the image (GDAL indexing)."
                )

            # Read selected bands in the specified order
            data = self._src.read(indexes=band_order, window=window, boundless=boundless, fill_value=fill_value)

        return data

    def close(self) -> None:
        """Close the rasterio dataset."""
        if self._src is not None:
            self._src.close()


class SAFEReader(ImageReader):
    """Reader for Sentinel-2 SAFE directory format.

    Reads L2A Sentinel-2 data directly from SAFE directories without requiring
    pre-conversion to TIFF. Handles band resampling, data offset, and band selection.
    """

    def __init__(
        self,
        safe_dir_path: str | Path,
        **kwargs: object,
    ) -> None:
        """Initialize SAFEReader.

        Args:
            safe_dir_path: Path to the .SAFE directory containing Sentinel-2 L2A data.
            **kwargs: Additional keyword arguments (ignored for compatibility).

        Raises:
            ValueError: If SAFE directory structure is invalid.
        """
        self.safe_dir_path = Path(safe_dir_path)

        if not (self.safe_dir_path / "GRANULE").exists():
            raise ValueError(f"GRANULE directory does not exist in {self.safe_dir_path}. Please check your path.")

        # Determine data offset from metadata
        self._offset = self._get_data_offset()

        # Load band data
        self._stacked: np.ndarray | None = None
        self._crs: CRS | None = None
        self._transform: Affine | None = None
        self._load_band_data()

    def _get_data_offset(self) -> int:
        """Determine the data offset from SAFE metadata.

        Returns:
            Data offset value (0 or 1000 depending on processing baseline)
        """
        metadata_path = self.safe_dir_path / "MTD_MSIL2A.xml"
        offset = 1000

        if not metadata_path.exists():
            logger.warning("No MTD_MSIL2A.xml file found. Assuming data offset of 1000.")
            return offset

        try:
            tree = parse_xml(metadata_path)
            root = tree.getroot()
            pb = root.findtext(".//PROCESSING_BASELINE")
            if pb and float(pb) < 4:
                offset = 0
        except Exception as e:
            logger.warning(f"Could not parse metadata file: {e}. Assuming data offset of 1000.")

        return offset

    def _load_band_data(self) -> None:
        """Load Sentinel-2 bands and reproject to common resolution.

        Raises:
            ValueError: If required, Sentinel-2 bands cannot be found.
        """
        try:
            # Find band files (B02, B03, B04, B08 @ 10m, B05 @ 20m)
            band_02 = next(self.safe_dir_path.glob("GRANULE/**/IMG_DATA/**/*_B02_10m.jp2"))
            band_03 = next(self.safe_dir_path.glob("GRANULE/**/IMG_DATA/**/*_B03_10m.jp2"))
            band_04 = next(self.safe_dir_path.glob("GRANULE/**/IMG_DATA/**/*_B04_10m.jp2"))
            band_08 = next(self.safe_dir_path.glob("GRANULE/**/IMG_DATA/**/*_B08_10m.jp2"))
            band_05 = next(self.safe_dir_path.glob("GRANULE/**/IMG_DATA/**/*_B05_20m.jp2"))

            band_data: list[np.ndarray] = []
            for band_file in (band_02, band_03, band_04, band_08):
                with rasterio.open(band_file) as src:
                    band_data.append(src.read(1))
                    self._crs, self._transform = src.crs, src.transform
                    height, width = src.height, src.width

            # Resample 20m band (B05) onto the 10m grid
            with (
                rasterio.open(band_05) as src,
                WarpedVRT(
                    src,
                    crs=self._crs,
                    transform=self._transform,
                    width=width,
                    height=height,
                    resampling=Resampling.nearest,
                ) as vrt,
            ):
                band_data.append(vrt.read(1))

            stacked = np.stack(band_data)
            del band_data

            # Apply data offset
            if self._offset > 0:
                stacked = (stacked.astype(np.int32) - self._offset).clip(0).astype(np.uint16)

            self._stacked = stacked
        except StopIteration as e:
            raise ValueError(f"Could not find required Sentinel-2 bands in {self.safe_dir_path}") from e

    @property
    def height(self) -> int:
        """Height of the image in pixels.

        Raises:
            RuntimeError: If band data has not been loaded.

        Returns:
            Height in pixels.
        """
        if self._stacked is None:
            raise RuntimeError("Band data not loaded")
        return int(self._stacked.shape[1])

    @property
    def width(self) -> int:
        """Width of the image in pixels.

        Raises:
            RuntimeError: If band data has not been loaded.

        Returns:
            Width in pixels.
        """
        if self._stacked is None:
            raise RuntimeError("Band data not loaded")
        return int(self._stacked.shape[2])

    @property
    def num_bands(self) -> int:
        """Number of bands.

        Raises:
            RuntimeError: If band data has not been loaded.

        Returns:
            Number of bands.
        """
        if self._stacked is None:
            raise RuntimeError("Band data not loaded")
        return int(self._stacked.shape[0])

    @property
    def dtype(self) -> str:
        """Data type of the image.

        Raises:
            RuntimeError: If band data has not been loaded.

        Returns:
            Data type string.
        """
        if self._stacked is None:
            raise RuntimeError("Band data not loaded")
        return str(self._stacked.dtype)

    @property
    def crs(self) -> object:
        """Coordinate Reference System of the image.

        Raises:
            RuntimeError: If band data has not been loaded.

        Returns:
            CRS object.
        """
        if self._stacked is None:
            raise RuntimeError("Band data not loaded")
        return self._crs

    @property
    def transform(self) -> object:
        """Geospatial affine transform.

        Raises:
            RuntimeError: If band data has not been loaded.

        Returns:
            Affine transform object.
        """
        if self._stacked is None:
            raise RuntimeError("Band data not loaded")
        return self._transform

    def read_window(
        self,
        window: Window,
        band_order: list[int] | None = None,
        boundless: bool = True,
        fill_value: int = 0,
    ) -> np.ndarray:
        """Read a window of data from the stacked SAFE bands.

        Args:
            window: Rasterio Window object specifying the region to read.
            band_order: List of band indices (1-based) to read.
            boundless: If True, pads with fill_value for out-of-bounds regions.
            fill_value: Value used for padding.

        Returns:
            Numpy array of shape [bands, height, width].

        Raises:
            RuntimeError: If band data has not been loaded.
            IndexError: If window is outside image bounds when boundless=False.
            ValueError: If band_order has a band index the image doesn't have.
        """
        if self._stacked is None:
            raise RuntimeError("Band data not loaded")

        # Extract the requested window from the stacked bands
        row_start = window.row_off
        row_end = window.row_off + window.height
        col_start = window.col_off
        col_end = window.col_off + window.width

        # Handle boundless reading by padding if necessary
        img_height = self.height
        img_width = self.width

        if boundless:
            # Clip to image bounds
            row_start_clipped = max(0, row_start)
            row_end_clipped = min(img_height, row_end)
            col_start_clipped = max(0, col_start)
            col_end_clipped = min(img_width, col_end)

            # Read the clipped region
            if row_start_clipped < row_end_clipped and col_start_clipped < col_end_clipped:
                data = self._stacked[:, row_start_clipped:row_end_clipped, col_start_clipped:col_end_clipped]
            else:
                # Completely outside bounds
                data = np.full((self.num_bands, 0, 0), fill_value, dtype=self.dtype)

            # Pad to requested window size
            pad_top = max(0, -row_start)
            pad_bottom = max(0, row_end - img_height)
            pad_left = max(0, -col_start)
            pad_right = max(0, col_end - img_width)

            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                data = np.pad(
                    data,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=fill_value,
                )
        else:
            # Strict bounds checking
            if row_start < 0 or row_end > img_height or col_start < 0 or col_end > img_width:
                raise IndexError(f"Window {window} is outside image bounds ({img_height}x{img_width})")
            data = self._stacked[:, row_start:row_end, col_start:col_end]

        # Handle band ordering
        if band_order is not None:
            # Validate band indices
            if any(b < 1 or b > self.num_bands for b in band_order):
                raise ValueError(
                    f"Band order {band_order} is invalid for image with {self.num_bands} bands. "
                    "Please specify band indices between 1 and the number of bands in the image (GDAL indexing)."
                )
            # Reorder bands (convert from 1-based to 0-based indexing)
            band_indices = [b - 1 for b in band_order]
            data = data[band_indices, :, :]

        return data

    def close(self) -> None:
        """Close the reader and release resources."""
        self._stacked = None
