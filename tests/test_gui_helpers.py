"""Tests for the parts of the desktop app that don't need Qt."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
import requests

from habitat_mapper.gui.batch import BatchItem, existing_outputs, output_path_for
from habitat_mapper.gui.preview import read_input_preview, read_output_preview
from habitat_mapper.gui.updates import newer_release
from tests.test_blank_tile_skipping import _write_raster


def test_output_path_for_file_and_safe_folder() -> None:
    out = Path("/out")
    assert (
        output_path_for(Path("/data/survey.tif"), out, "kelp-rgb", "20240722") == out / "survey_kelp_rgb_20240722.tif"
    )
    assert (
        output_path_for(Path("/data/S2B_MSIL2A_X.SAFE"), out, "kelp-skema", "20261014")
        == out / "S2B_MSIL2A_X_kelp_skema_20261014.tif"
    )


def test_existing_outputs(tmp_path: Path) -> None:
    (tmp_path / "a_kelp_rgb_20240722.tif").touch()
    items = [BatchItem(Path("/data/a.tif")), BatchItem(Path("/data/b.tif"))]
    assert existing_outputs(items, tmp_path, "kelp-rgb", "20240722") == [tmp_path / "a_kelp_rgb_20240722.tif"]


def test_input_preview_is_downsampled_rgb(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    path = tmp_path / "input.tif"
    _write_raster(path, rng.integers(0, 4000, size=(4, 300, 600)).astype(np.uint16))

    preview = read_input_preview(path, max_size=100, band_order=[3, 2, 1, 4])

    assert preview.shape == (50, 100, 3)
    assert preview.dtype == np.uint8
    assert preview.max() == 255


def test_output_preview_hides_background_and_nodata(tmp_path: Path) -> None:
    classes = np.array([[[0, 1, 2, 255]]], dtype=np.uint8)
    path = tmp_path / "output.tif"
    _write_raster(path, classes)
    with rasterio.open(path, "r+") as dst:
        dst.nodata = 255

    overlay = read_output_preview(path)

    assert overlay.shape == (1, 4, 4)
    assert list(overlay[0, :, 3]) == [0, 255, 255, 0]


def _release_response(tag: str) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"tag_name": tag, "html_url": f"https://example.com/{tag}"}
    return response


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("v0.18.0", ("0.18.0", "https://example.com/v0.18.0")), ("v0.17.0", None), ("v0.16.2", None)],
)
def test_newer_release(tag: str, expected: tuple[str, str] | None) -> None:
    with patch("habitat_mapper.gui.updates.requests.get", return_value=_release_response(tag)):
        assert newer_release("0.17.0") == expected


def test_newer_release_ignores_network_errors() -> None:
    with patch("habitat_mapper.gui.updates.requests.get", side_effect=requests.ConnectionError):
        assert newer_release("0.17.0") is None
