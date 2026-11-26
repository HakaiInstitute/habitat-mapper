"""Tests for the parts of the desktop app that don't need Qt."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
import requests
from rasterio.windows import Window

from habitat_mapper.config import OutputClass
from habitat_mapper.gui.batch import BatchItem, existing_outputs, output_path_for, output_stem
from habitat_mapper.gui.model_info import accepted_inputs, default_band_order, input_description
from habitat_mapper.gui.preview import (
    DEFAULT_PALETTE,
    PENDING_RGBA,
    LiveOverlay,
    class_palette,
    colourise,
    read_input_preview,
    read_output_preview,
)
from habitat_mapper.gui.updates import newer_release
from habitat_mapper.registry import model_registry
from tests.test_blank_tile_skipping import _write_raster
from tests.test_safe_reader import _make_safe, _zip


def test_output_path_for_file_and_safe_folder() -> None:
    out = Path("/out")
    assert (
        output_path_for(Path("/data/survey.tif"), out, "kelp-rgb", "20240722") == out / "survey_kelp_rgb_20240722.tif"
    )
    assert (
        output_path_for(Path("/data/S2B_MSIL2A_X.SAFE"), out, "kelp-skema", "20261014")
        == out / "S2B_MSIL2A_X_kelp_skema_20261014.tif"
    )
    assert (
        output_path_for(Path("/data/S2B_MSIL2A_X.SAFE.zip"), out, "kelp-skema", "20261014")
        == out / "S2B_MSIL2A_X_kelp_skema_20261014.tif"
    )
    assert (
        output_path_for(Path("/data/S2B_MSIL2A_X.zip"), out, "kelp-skema", "20261014")
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


def test_live_overlay_draws_parts_at_preview_scale() -> None:
    overlay = LiveOverlay(height=400, width=800, nodata=None, max_size=100)
    assert overlay.rgba.shape == (50, 100, 4)

    window = Window(col_off=400, row_off=0, width=400, height=200)
    assert overlay.to_preview(window) == (50, 0, 100, 25)
    overlay.paste(window, np.full((200, 400), 1, dtype=np.uint8))

    assert np.all(overlay.rgba[:25, 50:, 3] == 255)
    assert np.all(overlay.rgba[:25, :50, 3] == 0)
    assert np.all(overlay.rgba[25:, :, 3] == 0)


def test_live_overlay_draws_tiny_parts_as_a_pixel() -> None:
    overlay = LiveOverlay(height=4000, width=4000, nodata=None, max_size=100)
    assert overlay.to_preview(Window(col_off=3999, row_off=3999, width=1, height=1)) == (99, 99, 100, 100)


def test_live_overlay_hazes_pending_parts_until_written() -> None:
    overlay = LiveOverlay(height=100, width=100, nodata=None, max_size=100)
    tile = Window(col_off=0, row_off=0, width=50, height=50)
    finished = Window(col_off=0, row_off=0, width=25, height=25)

    overlay.mark_pending(tile)
    assert np.all(overlay.rgba[:50, :50] == PENDING_RGBA)
    assert np.all(overlay.rgba[50:, :, 3] == 0)

    # Background results are transparent, which clears the haze
    overlay.paste(finished, np.zeros((25, 25), dtype=np.uint8))
    assert np.all(overlay.rgba[:25, :25, 3] == 0)
    assert np.all(overlay.rgba[25:50, :50] == PENDING_RGBA)

    # A later tile that overlaps finished results doesn't cover them again
    overlay.mark_pending(Window(col_off=0, row_off=0, width=100, height=100))
    assert np.all(overlay.rgba[:25, :25, 3] == 0)


def test_tiles_past_the_edge_of_the_image() -> None:
    overlay = LiveOverlay(height=100, width=100, nodata=None, max_size=100)
    tile = Window(col_off=80, row_off=80, width=50, height=50)

    assert overlay.tile_to_preview(tile) == (80, 80, 130, 130)
    assert overlay.to_preview(tile) == (80, 80, 100, 100)
    overlay.mark_pending(tile)
    assert np.all(overlay.rgba[80:, 80:] == PENDING_RGBA)


def test_model_configs_name_their_classes() -> None:
    for name in model_registry.list_model_names():
        classes = model_registry[name].cfg.classes
        assert classes, f"{name} has no classes"
        assert len({c.value for c in classes}) == len(classes)


def test_class_palette_uses_model_colours_and_falls_back() -> None:
    palette = class_palette([OutputClass(value=2, name="Bull kelp", colour="#F2C230")])

    assert palette.shape == (256, 3)
    assert list(palette[2]) == [0xF2, 0xC2, 0x30]
    assert list(palette[1]) == list(DEFAULT_PALETTE[1])


def test_colourise_with_palette() -> None:
    palette = class_palette([OutputClass(value=1, name="Kelp", colour="#E8434F")])

    rgba = colourise(np.array([[0, 1]], dtype=np.uint8), nodata=None, palette=palette)

    assert list(rgba[0, 0]) == [0, 0, 0, 0]
    assert list(rgba[0, 1]) == [0xE8, 0x43, 0x4F, 255]


def test_input_description_and_default_band_order() -> None:
    assert input_description(model_registry["kelp-rgb"].cfg) == "RGB GeoTIFF"
    assert input_description(model_registry["kelp-rgbi"].cfg) == "RGB + near-infrared GeoTIFF"
    assert input_description(model_registry["kelp-skema"].cfg) == "Sentinel-2 L2A .SAFE.zip or .SAFE folder"
    assert default_band_order(model_registry["kelp-rgbi"].cfg) == "1, 2, 3, 4"


def test_accepted_inputs(tmp_path: Path) -> None:
    tif = tmp_path / "a.tif"
    tiff = tmp_path / "b.TIFF"
    other = tmp_path / "c.jpg"
    safe = tmp_path / "S2B_X.SAFE"
    safe_zip = _zip(_make_safe(tmp_path / "src"), tmp_path / "S2B_Y.SAFE.ZIP", "S2B_Y.SAFE/")
    other_zip = tmp_path / "tiffs.zip"
    damaged_zip = tmp_path / "S2B_Z.SAFE.zip"
    for path in (tif, tiff, other):
        path.touch()
    safe.mkdir()
    with zipfile.ZipFile(other_zip, "w") as archive:
        archive.writestr("survey/a.tif", "")
    damaged_zip.write_bytes(safe_zip.read_bytes()[:100])
    paths = [tif, other, safe, tiff, safe_zip, other_zip, damaged_zip]

    assert accepted_inputs(paths, safe=False) == [tif, tiff]
    assert accepted_inputs(paths, safe=True) == [safe, safe_zip]


def test_safe_folder_and_its_zip_share_an_output_stem() -> None:
    stems = {output_stem(Path(p)) for p in ("/a/S2B_X.SAFE", "/b/S2B_X.SAFE.zip", "/c/S2B_X.zip")}

    assert stems == {"S2B_X"}
    assert output_stem(Path("/a/survey.tif")) == "survey"


def test_input_preview_of_safe_zip(tmp_path: Path) -> None:
    safe = _make_safe(tmp_path)
    zipped = _zip(safe, tmp_path / "S2B_MSIL2A_X.SAFE.zip", "S2B_MSIL2A_X.SAFE/")

    preview = read_input_preview(zipped, max_size=10)

    assert preview.shape == (10, 10, 3)
    np.testing.assert_array_equal(preview, read_input_preview(safe, max_size=10))
