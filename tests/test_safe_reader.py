"""Tests for reading Sentinel-2 .SAFE products from extracted folders and from the .zip files they download as."""

from __future__ import annotations

import shutil
import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from habitat_mapper.reader import SafeProduct, SAFEReader, is_safe_zip

if TYPE_CHECKING:
    from pathlib import Path

SIZE = 20
GRANULE = "GRANULE/L2A_T09UXQ_A000001_20230801T193851/IMG_DATA"


def _write_band(path: Path, data: np.ndarray, pixel_size: int) -> None:
    # GDAL identifies files by their contents, so small GeoTIFFs stand in for the JPEG 2000 bands
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:32609",
        transform=from_origin(500000, 5500000, pixel_size, pixel_size),
    ) as dst:
        dst.write(data)


def _make_safe(parent: Path, baseline: str = "05.09") -> Path:
    """Write a small Sentinel-2 L2A .SAFE folder.

    Returns:
        The .SAFE folder.
    """
    safe = parent / "S2B_MSIL2A_X.SAFE"
    rng = np.random.default_rng(0)
    for band in ("B02", "B03", "B04", "B08"):
        data = rng.integers(1000, 5000, size=(1, SIZE, SIZE)).astype(np.uint16)
        _write_band(safe / GRANULE / "R10m" / f"T09UXQ_20230801T193851_{band}_10m.jp2", data, 10)
    b05 = rng.integers(1000, 5000, size=(1, SIZE // 2, SIZE // 2)).astype(np.uint16)
    _write_band(safe / GRANULE / "R20m" / "T09UXQ_20230801T193851_B05_20m.jp2", b05, 20)
    tci = rng.integers(0, 255, size=(3, SIZE, SIZE)).astype(np.uint8)
    _write_band(safe / GRANULE / "R10m" / "T09UXQ_20230801T193851_TCI_10m.jp2", tci, 10)
    (safe / "MTD_MSIL2A.xml").write_text(
        f"<root><General_Info><Product_Info><PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE>"
        "</Product_Info></General_Info></root>"
    )
    return safe


def _zip(folder: Path, zip_path: Path, inside: str) -> Path:
    """Zip a folder's files under a prefix, as Sentinel-2 downloads hold the .SAFE folder itself.

    Returns:
        The zip.
    """
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            archive.write(path, f"{inside}{path.relative_to(folder).as_posix()}")
    return zip_path


def _read_all(reader: SAFEReader) -> np.ndarray:
    return reader.read_window(Window(0, 0, reader.width, reader.height))


@pytest.mark.parametrize("inside", ["S2B_MSIL2A_X.SAFE/", ""], ids=["safe-folder-in-zip", "contents-at-zip-root"])
def test_zip_reads_the_same_as_folder(tmp_path: Path, inside: str) -> None:
    safe = _make_safe(tmp_path)
    zipped = _zip(safe, tmp_path / "S2B_MSIL2A_X.SAFE.zip", inside)

    from_folder = SAFEReader(safe)
    from_zip = SAFEReader(zipped)

    assert from_zip.num_bands == from_folder.num_bands == 5
    assert (from_zip.height, from_zip.width) == (SIZE, SIZE)
    assert from_zip.crs == from_folder.crs
    assert from_zip.transform == from_folder.transform
    np.testing.assert_array_equal(_read_all(from_zip), _read_all(from_folder))


def test_zip_applies_offset_from_metadata(tmp_path: Path) -> None:
    old = _zip(_make_safe(tmp_path / "old", baseline="03.00"), tmp_path / "old.zip", "X.SAFE/")
    new = _zip(_make_safe(tmp_path / "new", baseline="05.09"), tmp_path / "new.zip", "X.SAFE/")

    assert SAFEReader(old)._offset == 0
    assert SAFEReader(new)._offset == 1000


def test_zip_path_with_spaces(tmp_path: Path) -> None:
    folder = tmp_path / "my data"
    folder.mkdir()
    zipped = _zip(_make_safe(tmp_path), folder / "S2B_MSIL2A_X.SAFE.zip", "S2B_MSIL2A_X.SAFE/")

    assert SAFEReader(zipped).num_bands == 5


def test_zip_without_granule_is_rejected(tmp_path: Path) -> None:
    not_safe = tmp_path / "other"
    not_safe.mkdir()
    (not_safe / "readme.txt").write_text("hi")
    zipped = _zip(not_safe, tmp_path / "other.zip", "other/")

    with pytest.raises(ValueError, match="GRANULE"):
        SafeProduct(zipped)


def test_damaged_zip_is_reported_as_value_error(tmp_path: Path) -> None:
    zipped = _zip(_make_safe(tmp_path), tmp_path / "S2B_MSIL2A_X.SAFE.zip", "S2B_MSIL2A_X.SAFE/")
    zipped.write_bytes(zipped.read_bytes()[:100])

    with pytest.raises(ValueError, match="partly downloaded"):
        SafeProduct(zipped)


def test_macos_stub_files_are_ignored(tmp_path: Path) -> None:
    safe = _make_safe(tmp_path)
    band = next(safe.rglob("*_B02_10m.jp2"))
    # The stubs macOS writes on non-Mac drives sort before the real files and match the band patterns
    (band.parent / f"._{band.name}").write_bytes(b"\x00" * 16)
    zipped = _zip(safe, tmp_path / "S2B_MSIL2A_X.SAFE.zip", "S2B_MSIL2A_X.SAFE/")
    with zipfile.ZipFile(zipped, "a") as archive:
        archive.writestr(f"__MACOSX/S2B_MSIL2A_X.SAFE/{GRANULE}/R10m/._{band.name}", b"\x00" * 16)

    for path in (safe, zipped):
        # Folder products give native paths, so Windows separates them with backslashes
        found = SafeProduct(path).find("GRANULE/*/IMG_DATA/*_B02_10m.jp2").replace("\\", "/")
        assert found.endswith(f"/{band.name}")
        assert SAFEReader(path).num_bands == 5


def test_is_safe_zip(tmp_path: Path) -> None:
    safe = _make_safe(tmp_path)
    zipped = _zip(safe, tmp_path / "S2B_MSIL2A_X.zip", "S2B_MSIL2A_X.SAFE/")
    other = tmp_path / "tiffs.zip"
    with zipfile.ZipFile(other, "w") as archive:
        archive.writestr("survey/a.tif", "")
    damaged = tmp_path / "damaged.zip"
    damaged.write_bytes(zipped.read_bytes()[:100])

    assert is_safe_zip(zipped)
    assert not is_safe_zip(other)
    assert not is_safe_zip(damaged)
    assert not is_safe_zip(safe)


def test_folder_without_granule_is_rejected(tmp_path: Path) -> None:
    safe = _make_safe(tmp_path)
    shutil.rmtree(safe / "GRANULE")

    with pytest.raises(ValueError, match="GRANULE"):
        SAFEReader(safe)


def test_missing_band_is_reported(tmp_path: Path) -> None:
    safe = _make_safe(tmp_path)
    next(safe.rglob("*_B05_20m.jp2")).unlink()
    zipped = _zip(safe, tmp_path / "X.zip", "X.SAFE/")

    with pytest.raises(ValueError, match="Could not find required Sentinel-2 bands"):
        SAFEReader(zipped)
