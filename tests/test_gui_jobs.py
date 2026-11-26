"""Tests for how the desktop app segments its queue: how each image ends, and the live preview, without Qt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pytest

from habitat_mapper.cache import ModelCache
from habitat_mapper.config import SegmentOptions
from habitat_mapper.gui.batch import Job, Status
from habitat_mapper.gui.jobs import RunEvents, run_jobs
from habitat_mapper.gui.preview import PENDING_RGBA, class_palette, read_output_preview
from habitat_mapper.model import ONNXModel
from tests.helpers import tiny_model_config, write_raster, write_tiny_model

if TYPE_CHECKING:
    from pathlib import Path

    from habitat_mapper.gui.preview import LiveOverlay


@dataclass
class Recorder:
    """Records what a run reports."""

    started: list[int] = field(default_factory=list)
    finished: list[tuple[int, Status, str]] = field(default_factory=list)
    overlays: dict[int, LiveOverlay] = field(default_factory=dict)
    devices: list[str] = field(default_factory=list)

    def events(self) -> RunEvents:
        return RunEvents(
            item_started=self.started.append,
            item_finished=lambda row, status, error: self.finished.append((row, status, error)),
            live_started=self.overlays.__setitem__,
            device=self.devices.append,
        )


@pytest.fixture
def model(tmp_path: Path) -> ONNXModel:
    cfg = tiny_model_config(write_tiny_model(tmp_path / "model" / "tiny.onnx"))
    return ONNXModel(cfg, device="cpu", cache=ModelCache(tmp_path / "cache"))


def _jobs(tmp_path: Path, count: int) -> list[Job]:
    rng = np.random.default_rng(0)
    jobs = []
    for row in range(count):
        image = write_raster(tmp_path / f"in{row}.tif", rng.integers(0, 255, size=(3, 40, 50), dtype=np.uint8))
        jobs.append(Job(row, image, tmp_path / f"out{row}.tif"))
    return jobs


def _options(**changes: object) -> SegmentOptions:
    return SegmentOptions(crop_size=32, **changes)  # type: ignore[arg-type]


def test_each_image_is_done(model: ONNXModel, tmp_path: Path) -> None:
    jobs = _jobs(tmp_path, 2)
    recorded = Recorder()

    run_jobs(model, jobs, _options(), recorded.events(), cancelled=lambda: False)

    assert recorded.started == [0, 1]
    assert recorded.finished == [(0, Status.DONE, ""), (1, Status.DONE, "")]
    assert all(job.output_path.exists() for job in jobs)
    assert recorded.devices
    assert recorded.devices[0].startswith("cpu")


def test_a_failed_image_doesnt_stop_the_rest(model: ONNXModel, tmp_path: Path) -> None:
    jobs = _jobs(tmp_path, 2)
    jobs[0].input_path.write_bytes(b"not an image")
    recorded = Recorder()

    run_jobs(model, jobs, _options(), recorded.events(), cancelled=lambda: False)

    [(row, status, error), done] = recorded.finished
    assert (row, status) == (0, Status.FAILED)
    assert error
    assert done == (1, Status.DONE, "")


def test_a_model_that_cant_load_fails_every_image(tmp_path: Path) -> None:
    model = ONNXModel(tiny_model_config(tmp_path / "missing.onnx"), cache=ModelCache(tmp_path / "cache"))
    recorded = Recorder()

    run_jobs(model, _jobs(tmp_path, 2), _options(), recorded.events(), cancelled=lambda: False)

    assert recorded.started == []
    assert [(row, status) for row, status, _ in recorded.finished] == [(0, Status.FAILED), (1, Status.FAILED)]
    assert all(error.startswith("Could not load the model") for _, _, error in recorded.finished)


def test_cancelling_stops_the_current_image_and_the_rest(model: ONNXModel, tmp_path: Path) -> None:
    jobs = _jobs(tmp_path, 3)
    recorded = Recorder()

    # Cancelled once the first image has started
    run_jobs(model, jobs, _options(), recorded.events(), cancelled=lambda: bool(recorded.started))

    assert recorded.started == [0]
    assert recorded.finished == [(0, Status.CANCELLED, ""), (1, Status.CANCELLED, ""), (2, Status.CANCELLED, "")]
    assert not any(job.output_path.exists() for job in jobs)


@pytest.mark.parametrize("blur", [0, 5])
def test_live_preview_ends_as_the_saved_result(model: ONNXModel, tmp_path: Path, blur: int) -> None:
    # With the blur on, the preview is painted again as the filtered result is written
    [job] = _jobs(tmp_path, 1)
    recorded = Recorder()

    run_jobs(model, [job], _options(blur_kernel_size=blur), recorded.events(), cancelled=lambda: False)

    overlay = recorded.overlays[0].rgba
    assert not np.all(overlay == PENDING_RGBA, axis=-1).any()
    np.testing.assert_array_equal(overlay, read_output_preview(job.output_path, palette=class_palette()))
