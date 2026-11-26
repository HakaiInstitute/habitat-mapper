"""Tests for the desktop app's batch queue, preview captions and legends, and advanced options, without Qt."""

from __future__ import annotations

from pathlib import Path

import pytest

from habitat_mapper.gui.batch import BatchItem, BatchQueue, Job, Status, preview_caption, preview_model
from habitat_mapper.gui.model_info import parse_band_order, parse_options
from habitat_mapper.registry import model_registry

OUT = Path("/out")


def _queue(*names: str) -> BatchQueue:
    queue = BatchQueue()
    queue.add([Path(f"/data/{name}") for name in names])
    return queue


def test_add_skips_images_already_queued_and_name_clashes() -> None:
    queue = _queue("a.tif", "S2B_X.SAFE")

    result = queue.add([Path("/data/a.tif"), Path("/data/b.tif"), Path("/zips/S2B_X.SAFE.zip"), Path("/data/b.tif")])

    assert result.added == [Path("/data/b.tif")]
    assert result.clashes == 1
    assert "1 item not added" in result.message
    assert [item.input_path.name for item in queue.items] == ["a.tif", "S2B_X.SAFE", "b.tif"]


def test_add_says_nothing_when_everything_is_added() -> None:
    assert BatchQueue().add([Path("/a.tif")]).message == ""


def test_remove_takes_rows_out() -> None:
    queue = _queue("a.tif", "b.tif", "c.tif", "d.tif")

    queue.remove({0, 2})

    assert [item.input_path.name for item in queue.items] == ["b.tif", "d.tif"]


def test_start_run_queues_every_image_and_records_the_model() -> None:
    queue = _queue("a.tif", "b.tif")

    jobs = queue.start_run(OUT, "kelp-rgb", "20240722", skip_existing=False)

    assert jobs == [
        Job(0, Path("/data/a.tif"), OUT / "a_kelp_rgb_20240722.tif"),
        Job(1, Path("/data/b.tif"), OUT / "b_kelp_rgb_20240722.tif"),
    ]
    for item in queue.items:
        assert item.status is Status.QUEUED
        assert (item.model_name, item.model_revision) == ("kelp-rgb", "20240722")


def test_images_done_with_the_same_model_and_folder_arent_run_again() -> None:
    queue = _queue("a.tif", "b.tif")
    for job in queue.start_run(OUT, "kelp-rgb", "20240722", skip_existing=False):
        queue.items[job.row].status = Status.DONE if job.row == 0 else Status.FAILED

    assert [item.input_path.name for item in queue.pending(OUT, "kelp-rgb", "20240722")] == ["b.tif"]
    assert [job.row for job in queue.start_run(OUT, "kelp-rgb", "20240722", skip_existing=False)] == [1]
    # Another model, or another folder, runs everything again
    assert len(queue.pending(OUT, "kelp-rgb", "20231214")) == 2
    assert len(queue.pending(Path("/elsewhere"), "kelp-rgb", "20240722")) == 2


def test_start_run_clears_the_last_runs_error() -> None:
    queue = _queue("a.tif")
    queue.items[0].status, queue.items[0].error = Status.FAILED, "out of memory"

    queue.start_run(OUT, "kelp-rgb", "20240722", skip_existing=False)

    assert queue.items[0].error is None


@pytest.mark.parametrize("skip_existing", [True, False])
def test_existing_results_are_skipped_or_overwritten(tmp_path: Path, skip_existing: bool) -> None:
    queue = BatchQueue()
    queue.add([tmp_path / "a.tif", tmp_path / "b.tif"])
    (tmp_path / "a_kelp_rgb_20240722.tif").touch()

    jobs = queue.start_run(tmp_path, "kelp-rgb", "20240722", skip_existing=skip_existing)

    assert [job.row for job in jobs] == ([1] if skip_existing else [0, 1])
    assert queue.items[0].status is (Status.SKIPPED if skip_existing else Status.QUEUED)


def test_summary_counts_each_state_in_order() -> None:
    queue = _queue("a.tif", "b.tif", "c.tif", "d.tif")
    for item, status in zip(queue.items, [Status.FAILED, Status.DONE, Status.DONE, Status.SKIPPED], strict=True):
        item.status = status

    assert queue.summary() == "2 done, 1 failed, 1 skipped"
    assert queue.count(Status.DONE) == 2


def test_preview_model_is_the_one_that_made_the_result() -> None:
    chosen = model_registry.config("kelp-rgb")
    made_by = BatchItem(Path("/a.tif"), model_name="mussel-rgb", model_revision="20250711")

    assert preview_model(made_by, chosen) == model_registry.config("mussel-rgb", "20250711")
    assert preview_model(BatchItem(Path("/a.tif")), chosen) == chosen
    assert preview_model(None, chosen) == chosen


@pytest.mark.parametrize(
    ("status", "has_result", "processing", "caption"),
    [
        (Status.RUNNING, False, True, "a.tif, processing"),
        (Status.FAILED, False, False, "a.tif failed: no memory"),
        (Status.DONE, True, False, "a.tif, result from kelp-rgb 20240722"),
        (Status.DONE, False, False, "a.tif, done"),
        (Status.QUEUED, False, False, "a.tif, queued"),
    ],
)
def test_preview_caption(status: Status, has_result: bool, processing: bool, caption: str) -> None:
    item = BatchItem(Path("/a.tif"), status, error="no memory", model_name="kelp-rgb", model_revision="20240722")
    assert preview_caption(item, has_result=has_result, processing=processing) == caption


@pytest.mark.parametrize(
    ("text", "bands"), [("", None), ("  ", None), ("3, 2, 1", [3, 2, 1]), ("3 2 1", [3, 2, 1]), ("4,3 ,2", [4, 3, 2])]
)
def test_parse_band_order(text: str, bands: list[int] | None) -> None:
    assert parse_band_order(text) == bands


def test_parse_band_order_explains_bad_text() -> None:
    with pytest.raises(ValueError, match="band numbers separated by commas"):
        parse_band_order("red, green")


def test_parse_options() -> None:
    options = parse_options(crop_size=0, blur=5, morph=0, band_order="3,2,1")

    assert options.crop_size is None
    assert options.blur_kernel_size == 5
    assert options.band_order == [3, 2, 1]


def test_parse_options_names_each_bad_option() -> None:
    with pytest.raises(ValueError, match="Median blur: .*\nMorphological filter: ") as error:
        parse_options(crop_size=0, blur=4, morph=2, band_order="")
    assert "Tile size" not in str(error.value)
