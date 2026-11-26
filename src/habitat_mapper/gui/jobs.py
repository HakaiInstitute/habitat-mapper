"""Segmenting the desktop app's queue, kept free of Qt so it can be tested on its own. worker.py runs it on a thread."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from habitat_mapper.gui.batch import Status
from habitat_mapper.gui.preview import LiveOverlay, class_palette
from habitat_mapper.processing import ProcessingCancelled

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import TracebackType
    from typing import Self

    import numpy as np
    from rasterio.windows import Window

    from habitat_mapper.config import SegmentOptions
    from habitat_mapper.gui.batch import Job
    from habitat_mapper.model import ONNXModel


def _ignore(*_: object) -> None:
    pass


@dataclass(frozen=True)
class RunEvents:
    """What a run reports as it goes. Each is called from the thread the run is on.

    Attributes:
        stage: What is happening, e.g. "Segmenting survey.tif".
        progress: Steps done and total for the current stage. A total of 0 means unknown.
        item_started: Queue row of the image starting.
        item_finished: Queue row, how it ended, and the error message if it failed.
        device: The hardware the model ran on, after each image.
        live_started: Queue row, and the LiveOverlay that fills in as the image is processed.
        tiles_started: Queue row, and the tiles being processed, in LiveOverlay pixels.
    """

    stage: Callable[[str], None] = _ignore
    progress: Callable[[int, int], None] = _ignore
    item_started: Callable[[int], None] = _ignore
    item_finished: Callable[[int, Status, str], None] = _ignore
    device: Callable[[str], None] = _ignore
    live_started: Callable[[int, LiveOverlay], None] = _ignore
    tiles_started: Callable[[int, list[tuple[int, int, int, int]]], None] = _ignore


class _Task:
    def __init__(self, progress: Callable[[int, int], None], total: int | None) -> None:
        self._progress = progress
        self._total = total or 0
        self._done = 0

    def update(self, advance: int = 1) -> None:
        self._done += advance
        self._progress(self._done, self._total)


class _ProgressReporter:
    """ProgressReporter that reports (done, total) to a callback."""

    def __init__(self, progress: Callable[[int, int], None], step: Callable[[str], None] | None = None) -> None:
        """Create the reporter.

        Args:
            progress: Called with (done, total) as each task advances.
            step: Called with the description of each task as it starts, e.g. "Filtering".
        """
        self._progress = progress
        self._step = step

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    def add_task(self, description: str, total: int | None = None) -> _Task:
        if self._step is not None:
            self._step(description)
        self._progress(0, total or 0)
        return _Task(self._progress, total)


class _LiveObserver:
    """ProcessingObserver that draws the result into a LiveOverlay and reports the tiles being processed."""

    def __init__(self, events: RunEvents, row: int, nodata: float, palette: np.ndarray) -> None:
        self._events = events
        self._row = row
        self._nodata = nodata
        self._palette = palette
        self._overlay: LiveOverlay | None = None

    def image_started(self, height: int, width: int) -> None:
        self._overlay = LiveOverlay(height, width, self._nodata, palette=self._palette)
        self._events.live_started(self._row, self._overlay)

    def batch_started(self, windows: Sequence[Window]) -> None:
        if self._overlay is not None:
            for window in windows:
                self._overlay.mark_pending(window)
            self._events.tiles_started(self._row, [self._overlay.tile_to_preview(w) for w in windows])

    def written(self, window: Window, data: np.ndarray) -> None:
        if self._overlay is not None:
            self._overlay.paste(window, data)


def run_jobs(
    model: ONNXModel,
    jobs: list[Job],
    options: SegmentOptions,
    events: RunEvents,
    cancelled: Callable[[], bool],
) -> None:
    """Segment each image with one model, reporting how each ends rather than raising.

    If the model can't be loaded, every image fails. Once cancelled, the current image stops after its current tiles
    and the rest are marked cancelled.

    Args:
        model: Model to run. It is downloaded first if needed.
        jobs: The images, in order.
        options: Processing options, e.g. crop_size.
        events: Told about each step.
        cancelled: Checked before each image and each batch of tiles.
    """
    events.stage(f"Loading {model.name}, which is downloaded the first time it is used…")
    try:
        # Shows download progress the first time, through the same progress bar as segmenting
        model.cache.fetch(model.cfg, lambda: _ProgressReporter(events.progress))
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not load {model.name}: {e}")
        for job in jobs:
            events.item_finished(job.row, Status.FAILED, f"Could not load the model: {e}")
        return

    palette = class_palette(model.cfg.classes)
    for job in jobs:
        if cancelled():
            events.item_finished(job.row, Status.CANCELLED, "")
            continue

        events.item_started(job.row)
        events.stage(f"Segmenting {job.input_path.name}")

        def reporter(name: str = job.input_path.name) -> _ProgressReporter:
            # Names each step in the status bar, e.g. "Filtering image.tif" once the model is done
            return _ProgressReporter(events.progress, lambda step: events.stage(f"{step} {name}"))

        try:
            model.process(
                job.input_path,
                job.output_path,
                **options.model_dump(),
                quiet=True,
                progress=reporter,
                cancel=cancelled,
                observer=_LiveObserver(events, job.row, model.cfg.nodata_value, palette),
            )
        except ProcessingCancelled:
            logger.info(f"Cancelled {job.input_path.name}")
            events.item_finished(job.row, Status.CANCELLED, "")
        except Exception as e:  # noqa: BLE001
            logger.opt(exception=e).debug("Segmentation failed")
            logger.error(f"Failed to segment {job.input_path.name}: {e}")
            events.item_finished(job.row, Status.FAILED, str(e))
        else:
            logger.success(f"Saved {job.output_path}")
            events.item_finished(job.row, Status.DONE, "")

        if model.active_device:
            events.device(model.active_device)
