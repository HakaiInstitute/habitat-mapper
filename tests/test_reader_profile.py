"""Tests for the output profile readers give, which sets the output's size and location."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from habitat_mapper.reader import ImageReader

if TYPE_CHECKING:
    from rasterio.windows import Window


class _ArrayReader(ImageReader):
    def __init__(self, data: np.ndarray, crs: CRS | None = None, transform: Affine | None = None) -> None:
        self._data, self._crs, self._transform = data, crs, transform

    height = property(lambda self: self._data.shape[1])
    width = property(lambda self: self._data.shape[2])
    num_bands = property(lambda self: self._data.shape[0])
    dtype = property(lambda self: str(self._data.dtype))
    crs = property(lambda self: self._crs)
    transform = property(lambda self: self._transform)

    def read_window(
        self, window: Window, band_order: list[int] | None = None, boundless: bool = True, fill_value: int = 0
    ) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


def test_profile_has_the_image_size_and_location() -> None:
    crs, transform = CRS.from_epsg(32609), Affine(10, 0, 500000, 0, -10, 6000000)
    reader = _ArrayReader(np.zeros((5, 30, 40), dtype=np.uint16), crs, transform)
    assert reader.profile == {"driver": "GTiff", "height": 30, "width": 40, "crs": crs, "transform": transform}


def test_profile_leaves_out_a_missing_location() -> None:
    assert _ArrayReader(np.zeros((3, 2, 2), dtype=np.uint8)).profile == {"driver": "GTiff", "height": 2, "width": 2}
