"""Tests for SkemaFullSAFEReader auxiliary window warping.

Covers the NoData handling of ``_read_aux_window``: genuine zeros in the aux rasters must
survive the warp untouched (issue #193 -- a ``dst_nodata`` colliding with real values makes
GDAL nudge them off by one ULP and log a warning per tile per band), while pixels with no aux
coverage must report the sentinel the model expects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from habitat_mapper.reader import SkemaFullSAFEReader

if TYPE_CHECKING:
    from pathlib import Path

    from affine import Affine

S2_CRS = CRS.from_epsg(32609)  # UTM zone 9N, as used by BC Sentinel-2 tiles
ALBERS_CRS = CRS.from_epsg(3005)  # BC Albers, a plausible aux COG projection
S2_ORIGIN = (500000.0, 5500000.0)
S2_RES = 10.0
TILE = 30
# Half a pixel offset from the aux raster grid, so every warped pixel is interpolated.
S2_TRANSFORM = from_origin(S2_ORIGIN[0] + S2_RES / 2, S2_ORIGIN[1] - S2_RES / 2, S2_RES, S2_RES)
FULL_WINDOW = Window(col_off=0, row_off=0, width=TILE, height=TILE)
NUDGED_ZERO = np.float32(1.4012984643e-45)  # smallest float32 denormal


def _write_raster(
    path: Path,
    data: np.ndarray,
    dtype: str,
    nodata: float | None,
    crs: CRS = S2_CRS,
    origin: tuple[float, float] = S2_ORIGIN,
    res: float = S2_RES,
) -> Path:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": from_origin(origin[0], origin[1], res, res),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data.astype(dtype), 1)
    return path


def _aux_pattern(nodata: float | None, valid_value: float = 7.0) -> np.ndarray:
    """Build a 20x20 aux raster pattern.

    Args:
        nodata: NoData value to write into the bottom-left quadrant, or None to leave it valid.
        valid_value: Value for the right half of the raster.

    Returns:
        Array with genuine zeros in the left half, valid_value in the right half, and a NoData
        block in the bottom-left quadrant.
    """
    data = np.zeros((20, 20), dtype=np.float64)
    data[:, 10:] = valid_value
    if nodata is not None:
        data[10:, :10] = nodata
    return data


def _fake_stacked(transform: Affine, height: int, width: int) -> xr.DataArray:
    """Build a minimal georeferenced stand-in for SAFEReader._stacked.

    Args:
        transform: Affine transform of the fake S2 grid.
        height: Grid height in pixels.
        width: Grid width in pixels.

    Returns:
        A single-band DataArray carrying the S2 CRS and transform.
    """
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    stacked = xr.DataArray(
        np.zeros((1, height, width), dtype=np.uint16),
        dims=("band", "y", "x"),
        coords={"band": [1], "y": ys, "x": xs},
    )
    return stacked.rio.write_crs(S2_CRS)


def _reader() -> SkemaFullSAFEReader:
    """Build a reader with only the state _read_aux_window needs (no .SAFE dir required).

    Returns:
        A SkemaFullSAFEReader whose _stacked attribute is a fake 30x30 S2 grid.
    """
    reader = SkemaFullSAFEReader.__new__(SkemaFullSAFEReader)
    reader._stacked = _fake_stacked(S2_TRANSFORM, TILE, TILE)
    return reader


@pytest.fixture
def gdal_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Collect GDAL messages that rasterio forwards to the logging module.

    Args:
        caplog: pytest log capture fixture.

    Returns:
        The live list of captured log messages.
    """
    caplog.set_level(logging.DEBUG)
    return caplog.messages


class TestAuxWindowNoData:
    """NoData behaviour of the per-window aux raster warp."""

    @pytest.mark.parametrize(
        ("dtype", "nodata"),
        [
            ("float32", -2000.0),  # bathymetry_10m_cog.tif
            ("float32", 85.0),  # slope_10m_cog.tif
            ("int16", None),  # substrate_20m_cog.tif
            ("int16", 0.0),  # bops_substrate_10m_cog.tif
        ],
    )
    def test_genuine_zeros_are_not_nudged(
        self,
        tmp_path: Path,
        gdal_warnings: list[str],
        dtype: str,
        nodata: float | None,
    ) -> None:
        """Zero is a real measurement in every aux raster and must warp through exactly."""
        path = _write_raster(tmp_path / "aux.tif", _aux_pattern(nodata), dtype, nodata)
        with rasterio.open(path) as ds:
            out = _reader()._read_aux_window(ds, FULL_WINDOW, 0)

        zeros = out[0, 1:9, 1:9]  # well inside the genuine-zero quadrant
        assert np.all(zeros == 0.0), f"genuine zeros were altered: {np.unique(zeros)}"
        assert not np.any(zeros == NUDGED_ZERO)
        assert not any("treated as NoData" in message for message in gdal_warnings), gdal_warnings

    def test_missing_pixels_use_source_nodata_sentinel(self, tmp_path: Path) -> None:
        """Out-of-extent and source-NoData pixels keep the aux raster's own NoData value."""
        path = _write_raster(tmp_path / "bathymetry.tif", _aux_pattern(-2000.0), "float32", -2000.0)
        with rasterio.open(path) as ds:
            out = _reader()._read_aux_window(ds, FULL_WINDOW, 0)

        assert np.all(out[0, 12:19, 1:9] == -2000.0)  # NoData block inside the aux extent
        assert np.all(out[0, 22:, 22:] == -2000.0)  # beyond the aux extent
        assert np.all(out[0, 1:9, 12:19] == 7.0)  # valid data is untouched

    def test_missing_pixels_use_fill_value_without_source_nodata(self, tmp_path: Path) -> None:
        """Aux rasters with no NoData value fall back to the caller's fill_value."""
        path = _write_raster(tmp_path / "substrate.tif", _aux_pattern(None, valid_value=4.0), "int16", None)
        with rasterio.open(path) as ds:
            out = _reader()._read_aux_window(ds, FULL_WINDOW, 0)
            out_filled = _reader()._read_aux_window(ds, FULL_WINDOW, 9)

        assert np.all(out[0, 22:, 22:] == 0.0)
        assert np.all(out_filled[0, 22:, 22:] == 9.0)
        assert np.all(out_filled[0, 1:9, 1:9] == 0.0)  # genuine zeros stay 0 even when fill_value is 9

    def test_window_entirely_outside_aux_extent(self, tmp_path: Path) -> None:
        """A tile with no aux coverage at all is all sentinel, with no error."""
        path = _write_raster(tmp_path / "bathymetry.tif", _aux_pattern(-2000.0), "float32", -2000.0)
        with rasterio.open(path) as ds:
            out = _reader()._read_aux_window(ds, Window(col_off=100_000, row_off=100_000, width=TILE, height=TILE), 0)

        assert out.shape == (1, 30, 30)
        assert np.all(out == -2000.0)

    def test_nan_values_and_nan_nodata_never_reach_the_model(self, tmp_path: Path) -> None:
        """NaNs in the aux raster are collapsed onto a finite sentinel."""
        with_nans = _aux_pattern(-2000.0)
        with_nans[2:6, 2:6] = np.nan
        nan_valued = _write_raster(tmp_path / "nan_values.tif", with_nans, "float32", -2000.0)

        nan_nodata_data = _aux_pattern(None)
        nan_nodata_data[10:, :10] = np.nan
        nan_nodata = _write_raster(tmp_path / "nan_nodata.tif", nan_nodata_data, "float32", float("nan"))

        for path, expected_sentinel in ((nan_valued, -2000.0), (nan_nodata, 0.0)):
            with rasterio.open(path) as ds:
                out = _reader()._read_aux_window(ds, FULL_WINDOW, 0)

            assert out.dtype == np.float32
            assert np.isfinite(out).all(), f"non-finite values in output for {path.name}"
            assert np.all(out[0, 22:, 22:] == expected_sentinel)

    def test_reprojected_aux_raster(self, tmp_path: Path, gdal_warnings: list[str]) -> None:
        """Cross-CRS warps (BC Albers aux COG -> UTM S2 grid) keep zeros and sentinels intact."""
        west, north = S2_TRANSFORM * (0, 0)
        east, south = S2_TRANSFORM * (TILE, TILE)
        # Aux raster in BC Albers covering only the western part of the S2 window, so the
        # output contains both warped values and uncovered pixels.
        a_west, a_south, a_east, a_north = transform_bounds(S2_CRS, ALBERS_CRS, west, south, east, north)
        res = 20.0
        width = int((a_east - a_west) / 2 / res)
        height = int((a_north - a_south) / res) + 10
        path = _write_raster(
            tmp_path / "albers.tif",
            np.zeros((height, width)),  # every valid pixel is a genuine zero
            "float32",
            -2000.0,
            crs=ALBERS_CRS,
            origin=(a_west, a_north + 5 * res),
            res=res,
        )
        with rasterio.open(path) as ds:
            out = _reader()._read_aux_window(ds, FULL_WINDOW, 0)

        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        # Covered pixels stay exactly zero; uncovered pixels report the sentinel. Nothing else.
        assert set(np.unique(out).tolist()) == {0.0, -2000.0}, np.unique(out)
        assert not any("treated as NoData" in message for message in gdal_warnings), gdal_warnings
