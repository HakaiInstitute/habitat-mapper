"""Abstract image reader interface and implementations for multiple file formats."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
from defusedxml.ElementTree import parse as parse_xml
from loguru import logger
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject as warp_reproject

if TYPE_CHECKING:
    from rasterio.windows import Window


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
        """
        if band_order is None:
            # Read all bands
            data = self._src.read(window=window, boundless=boundless, fill_value=fill_value)
        else:
            # Validate band indices
            if any(b < 1 or b > self.num_bands for b in band_order):
                logger.error(
                    f"Band order {band_order} is invalid for image with {self.num_bands} bands.\n "
                    "Please specify band indices between 1 and the number of bands in the image (GDAL indexing)."
                )
                sys.exit(1)

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
        self._stacked: xr.DataArray | None = None
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

            # Open bands as rioxarray DataArrays
            band_files = [band_02, band_03, band_04, band_08, band_05]
            band_data: list[xr.DataArray] = [rxr.open_rasterio(b) for b in band_files]  # type: ignore[misc]

            # Resample 20m band (B05) to match 10m resolution.
            # Save the original reference so we can close its file handle after reprojection —
            # accessing .rio creates a circular reference that prevents CPython from immediately
            # freeing the DataArray and its underlying rasterio file descriptor via refcounting.
            b05_raw = band_data[-1]
            band_data[-1] = b05_raw.rio.reproject_match(band_data[0], resampling=Resampling.nearest)
            b05_raw.close()

            # Stack bands along the band dimension
            stacked = xr.concat(band_data, dim="band")  # type: ignore[misc]

            # Close source DataArrays now that their data is loaded into stacked.
            for bd in band_data:
                bd.close()
            del band_data

            # Apply data offset
            if self._offset > 0:
                stacked = stacked.astype(np.int32) - self._offset
                stacked = stacked.clip(0).astype(np.uint16)

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
        return int(self._stacked.rio.height)

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
        return int(self._stacked.rio.width)

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
        return int(self._stacked.sizes["band"])

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
            CRS object from xarray dataset.
        """
        if self._stacked is None:
            raise RuntimeError("Band data not loaded")
        return self._stacked.rio.crs

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
        return self._stacked.rio.transform()

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
        """
        if self._stacked is None:
            raise RuntimeError("Band data not loaded")

        # Extract the requested window from the xarray
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
                data = self._stacked.isel(
                    y=slice(row_start_clipped, row_end_clipped),
                    x=slice(col_start_clipped, col_end_clipped),
                ).values
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
            data = self._stacked.isel(
                y=slice(row_start, row_end),
                x=slice(col_start, col_end),
            ).values

        # Handle band ordering
        if band_order is not None:
            # Validate band indices
            if any(b < 1 or b > self.num_bands for b in band_order):
                logger.error(
                    f"Band order {band_order} is invalid for image with {self.num_bands} bands.\n "
                    "Please specify band indices between 1 and the number of bands in the image (GDAL indexing)."
                )
                sys.exit(1)
            # Reorder bands (convert from 1-based to 0-based indexing)
            band_indices = [b - 1 for b in band_order]
            data = data[band_indices, :, :]

        return data

    def close(self) -> None:
        """Close the reader and release resources."""
        if self._stacked is not None:
            self._stacked.close()
            self._stacked = None


class SkemaFullSAFEReader(SAFEReader):
    """SAFEReader that appends substrate, bathymetry, and slope from an auxiliary directory.

    The substrate file to use is specified explicitly via ``substrate_filename`` so that
    the model config controls which substrate variant is loaded (e.g. BoPs vs standard).

    Bathymetry and slope are always taken from bathymetry_10m_cog.tif and
    slope_10m_cog.tif respectively.

    Auxiliary files are read lazily: only the pixels needed for each inference tile are
    warped at read time, avoiding a full-scene reproject of the (potentially large) BC-wide
    COGs during startup.
    """

    def __init__(
        self,
        safe_dir_path: str | Path,
        aux_dir_path: str | Path,
        substrate_filename: str,
        **kwargs: object,
    ) -> None:
        """Initialize SkemaFullSAFEReader.

        Args:
            safe_dir_path: Path to the .SAFE directory containing Sentinel-2 L2A data.
            aux_dir_path: Directory containing auxiliary rasters (substrate, bathymetry, slope).
            substrate_filename: Filename of the substrate raster inside ``aux_dir_path``.
            **kwargs: Additional keyword arguments passed to SAFEReader.

        Raises:
            FileNotFoundError: If any required auxiliary file is missing.
        """
        self.aux_dir_path = Path(aux_dir_path).expanduser()

        self._substrate_path = self.aux_dir_path / substrate_filename
        self._bathymetry_path = self.aux_dir_path / "bathymetry_10m_cog.tif"
        self._slope_path = self.aux_dir_path / "slope_10m_cog.tif"

        for p in (self._substrate_path, self._bathymetry_path, self._slope_path):
            if not p.exists():
                raise FileNotFoundError(f"Auxiliary file not found: {p}")

        self._aux_datasets: list[rasterio.DatasetReader] = []
        super().__init__(safe_dir_path, **kwargs)

    def _load_band_data(self) -> None:
        """Load S2 bands and open aux file handles (data read lazily per window)."""
        super()._load_band_data()
        self._aux_datasets = [rasterio.open(p) for p in (self._substrate_path, self._bathymetry_path, self._slope_path)]

    @property
    def num_bands(self) -> int:
        """Number of bands (S2 bands plus one per auxiliary raster)."""
        return super().num_bands + len(self._aux_datasets)

    def _read_aux_window(self, ds: rasterio.DatasetReader, window: Window, fill_value: int) -> np.ndarray:
        """Warp a single window from an aux COG into the S2 tile's coordinate space.

        Args:
            ds: Open rasterio dataset for the auxiliary raster.
            window: Rasterio Window specifying the region in S2 pixel space.
            fill_value: Value used for pixels outside the aux raster extent.

        Returns:
            Float32 array of shape [1, height, width].

        Raises:
            RuntimeError: If band data has not been loaded.
        """
        if self._stacked is None:
            raise RuntimeError("Band data not loaded")

        s2_transform = self._stacked.rio.transform()
        s2_crs = self._stacked.rio.crs

        col_off, row_off = window.col_off, window.row_off
        w, h = int(window.width), int(window.height)

        # Geographic bounds of the requested window (may extend beyond S2 image for boundless reads)
        west, north = s2_transform * (col_off, row_off)
        east, south = s2_transform * (col_off + w, row_off + h)

        dst_transform = transform_from_bounds(west, south, east, north, w, h)
        dst = np.full((1, h, w), fill_value, dtype=np.float32)

        warp_reproject(
            source=rasterio.band(ds, 1),
            destination=dst,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=dst_transform,
            dst_crs=s2_crs,
            resampling=Resampling.bilinear,
            dst_nodata=fill_value,
        )

        return dst

    def read_window(
        self,
        window: Window,
        band_order: list[int] | None = None,
        boundless: bool = True,
        fill_value: int = 0,
    ) -> np.ndarray:
        """Read a window of S2 and auxiliary band data, reprojecting aux bands lazily.

        Args:
            window: Rasterio Window object specifying the region to read.
            band_order: List of band indices (1-based) to read. If None, reads all bands.
            boundless: If True, pads with fill_value for out-of-bounds regions.
            fill_value: Value used for padding and missing aux data.

        Returns:
            Float32 array of shape [bands, height, width].
        """
        s2_data = super().read_window(window, band_order=None, boundless=boundless, fill_value=fill_value)
        aux_parts = [self._read_aux_window(ds, window, fill_value) for ds in self._aux_datasets]
        data = np.concatenate([s2_data.astype(np.float32)] + aux_parts, axis=0)

        if band_order is not None:
            if any(b < 1 or b > self.num_bands for b in band_order):
                logger.error(
                    f"Band order {band_order} is invalid for image with {self.num_bands} bands.\n "
                    "Please specify band indices between 1 and the number of bands in the image (GDAL indexing)."
                )
                sys.exit(1)
            data = data[[b - 1 for b in band_order]]

        return data

    def close(self) -> None:
        """Close auxiliary rasterio datasets and the underlying SAFEReader."""
        for ds in self._aux_datasets:
            ds.close()
        self._aux_datasets = []
        super().close()
