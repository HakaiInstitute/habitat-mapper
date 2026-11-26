"""Builders and fakes shared across test modules.

Import from here rather than from another test module, so that tests don't depend on each other.
"""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import onnx
import rasterio
import requests
from onnx import TensorProto, helper, numpy_helper
from rasterio.transform import from_origin

from habitat_mapper.config import ModelConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rasterio.windows import Window

CRS = "EPSG:32609"
ORIGIN = (500000, 5500000)


def write_raster(path: Path, data: np.ndarray, pixel_size: float = 1) -> Path:
    """Write a (bands, height, width) array as a georeferenced GeoTIFF, creating parent folders.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=CRS,
        transform=from_origin(*ORIGIN, pixel_size, pixel_size),
    ) as dst:
        dst.write(data)
    return path


class FakeResponse:
    """Stands in for the response of requests.get(..., stream=True)."""

    def __init__(self, body: bytes, length: int | None = None, status: int = 200) -> None:
        self.body = body
        self.headers = {"content-length": str(len(body) if length is None else length)}
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} error")

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.body[i : i + chunk_size] for i in range(0, len(self.body), chunk_size)]


def make_stub_model(
    predicted_tiles: list[np.ndarray] | None = None, *, predicted_class: int = 1, default_output_value: int = 7
) -> MagicMock:
    """Model stub that predicts one class everywhere and records every tile it receives.

    Args:
        predicted_tiles: If given, every tile sent to the model is appended to it.
        predicted_class: The class predicted for every pixel.
        default_output_value: The config's value for pixels with nothing to predict.

    Returns:
        A mock model with a real ModelConfig.
    """

    def _predict(batch: np.ndarray, max_pixel_value: float) -> np.ndarray:
        if predicted_tiles is not None:
            predicted_tiles.extend(batch)
        logits = np.zeros((batch.shape[0], 2, batch.shape[2], batch.shape[3]), dtype=np.float32)
        logits[:, predicted_class] = 1.0
        return logits

    model = MagicMock()
    model.cfg = ModelConfig(
        name="stub",
        revision="20250101",
        dependencies=[],
        model_filename="model.onnx",
        default_output_value=default_output_value,
        nodata_value=255,
    )
    model.predict.side_effect = _predict
    model.postprocess.side_effect = lambda data: np.argmax(data, axis=0).astype(np.uint8)
    return model


class RecordingObserver:
    """ProcessingObserver that rebuilds the whole result from what it is told was written."""

    def __init__(self) -> None:
        self.size: tuple[int, int] | None = None
        self.batches: list[list[Window]] = []
        self.segmented: list[list[Window]] = []
        self.result: np.ndarray | None = None

    def image_started(self, height: int, width: int) -> None:
        self.size = (height, width)
        self.result = np.full((height, width), 99, dtype=np.uint8)

    def batch_started(self, windows: Sequence[Window]) -> None:
        self.batches.append(list(windows))

    def segmenting(self, windows: Sequence[Window]) -> None:
        self.segmented.append(list(windows))

    def written(self, window: Window, data: np.ndarray) -> None:
        assert self.result is not None
        rows, cols = window.toslices()
        self.result[rows, cols] = data


SAFE_SIZE = 20
GRANULE = "GRANULE/L2A_T09UXQ_A000001_20230801T193851/IMG_DATA"


def make_safe(parent: Path, baseline: str = "05.09") -> Path:
    """Write a small Sentinel-2 L2A .SAFE folder.

    GDAL identifies files by their contents, so small GeoTIFFs stand in for the JPEG 2000 bands.

    Returns:
        The .SAFE folder.
    """
    safe = parent / "S2B_MSIL2A_X.SAFE"
    rng = np.random.default_rng(0)
    for band in ("B02", "B03", "B04", "B08"):
        data = rng.integers(1000, 5000, size=(1, SAFE_SIZE, SAFE_SIZE)).astype(np.uint16)
        write_raster(safe / GRANULE / "R10m" / f"T09UXQ_20230801T193851_{band}_10m.jp2", data, 10)
    b05 = rng.integers(1000, 5000, size=(1, SAFE_SIZE // 2, SAFE_SIZE // 2)).astype(np.uint16)
    write_raster(safe / GRANULE / "R20m" / "T09UXQ_20230801T193851_B05_20m.jp2", b05, 20)
    tci = rng.integers(0, 255, size=(3, SAFE_SIZE, SAFE_SIZE)).astype(np.uint8)
    write_raster(safe / GRANULE / "R10m" / "T09UXQ_20230801T193851_TCI_10m.jp2", tci, 10)
    (safe / "MTD_MSIL2A.xml").write_text(
        f"<root><General_Info><Product_Info><PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE>"
        "</Product_Info></General_Info></root>"
    )
    return safe


def zip_folder(folder: Path, zip_path: Path, inside: str) -> Path:
    """Zip a folder's files under a prefix, as Sentinel-2 downloads hold the .SAFE folder itself.

    Returns:
        The zip.
    """
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            archive.write(path, f"{inside}{path.relative_to(folder).as_posix()}")
    return zip_path


def write_tiny_model(path: Path) -> Path:
    """Write a 1x1 convolution that scores class 1 where the first band is brighter than the second, else class 0.

    Its height and width are left unnamed, which ONNX Runtime reports as None.

    Returns:
        The model's path.
    """
    # Class 0 scores band 2 - band 1, class 1 scores band 1 - band 2
    weights = numpy_helper.from_array(np.array([[-1, 1, 0], [1, -1, 0]], dtype=np.float32).reshape(2, 3, 1, 1), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "w"], ["output"])],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, None, None])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2, None, None])],
        initializer=[weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


def tiny_model_config(path: Path, name: str = "tiny") -> ModelConfig:
    """Configure the model write_tiny_model wrote, so its classes are whichever of bands 1 and 2 is brighter.

    Returns:
        The configuration.
    """
    return ModelConfig(
        name=name,
        revision="20250101",
        dependencies=[str(path)],
        model_filename=path.name,
        # The same for every band, so normalizing keeps whichever band is brighter
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        default_output_value=0,
        nodata_value=255,
    )
