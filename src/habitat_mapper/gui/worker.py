"""Background work for the desktop app: segmenting the batch queue and loading previews."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from loguru import logger
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, SignalInstance, Slot

from habitat_mapper.gui.batch import Status
from habitat_mapper.gui.preview import LiveOverlay
from habitat_mapper.processing import ImageProcessor, ProcessingCancelled

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from types import TracebackType
    from typing import Self

    import numpy as np
    from rasterio.windows import Window

    from habitat_mapper.model import ONNXModel


class _QtTask:
    def __init__(self, signal: SignalInstance, total: int | None) -> None:
        self._signal = signal
        self._total = total or 0
        self._done = 0

    def update(self, advance: int = 1) -> None:
        self._done += advance
        self._signal.emit(self._done, self._total)


class _QtProgressReporter:
    """ProgressReporter that emits (done, total) through a Qt signal."""

    def __init__(self, signal: SignalInstance) -> None:
        self._signal = signal

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    def add_task(self, description: str, total: int | None = None) -> _QtTask:
        self._signal.emit(0, total or 0)
        return _QtTask(self._signal, total)


class _LiveObserver:
    """ProcessingObserver that draws the result into a LiveOverlay and reports the tiles being processed."""

    def __init__(self, worker: SegmentWorker, row: int, nodata: float) -> None:
        self._worker = worker
        self._row = row
        self._nodata = nodata
        self._overlay: LiveOverlay | None = None

    def image_started(self, height: int, width: int) -> None:
        self._overlay = LiveOverlay(height, width, self._nodata)
        self._worker.live_started.emit(self._row, self._overlay)

    def batch_started(self, windows: Sequence[Window]) -> None:
        if self._overlay is not None:
            for window in windows:
                self._overlay.mark_pending(window)
            self._worker.tiles_started.emit(self._row, [self._overlay.tile_to_preview(w) for w in windows])

    def written(self, window: Window, data: np.ndarray) -> None:
        if self._overlay is not None:
            self._overlay.paste(window, data)


class SegmentWorker(QObject):
    """Segments a list of images with one model. Move it to a QThread and connect `run` to the thread's start."""

    item_started = Signal(int)
    item_finished = Signal(int, object, str)  # Queue row, Status, error message
    progress = Signal(int, int)  # Batches done and total, for the current image
    stage = Signal(str)
    device = Signal(str)
    live_started = Signal(int, object)  # Queue row, LiveOverlay that fills in as the image is processed
    tiles_started = Signal(int, object)  # Queue row, tiles being processed in LiveOverlay pixels, past its edges too
    finished = Signal()

    def __init__(self, model: ONNXModel, jobs: list[tuple[int, Path, Path]], options: dict[str, Any]) -> None:
        """Set up the work.

        Args:
            model: Model to run.
            jobs: Queue row, input path and output path of each image.
            options: Keyword arguments for ImageProcessor.from_model, e.g. crop_size.
        """
        super().__init__()
        self._model = model
        self._jobs = jobs
        self._options = options
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Stop after the current batch of tiles. Images that haven't started are marked cancelled."""
        self._cancelled.set()

    @Slot()
    def run(self) -> None:
        """Segment every image, reporting progress through signals."""
        try:
            self._run()
        finally:
            self.finished.emit()

    def _run(self) -> None:
        self.stage.emit(f"Loading {self._model.name}, which is downloaded the first time it is used…")
        try:
            self._model.cfg.get_local_model_path(quiet=True)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Could not load {self._model.name}: {e}")
            for row, _, _ in self._jobs:
                self.item_finished.emit(row, Status.FAILED, f"Could not load the model: {e}")
            return

        for row, input_path, output_path in self._jobs:
            if self._cancelled.is_set():
                self.item_finished.emit(row, Status.CANCELLED, "")
                continue

            self.item_started.emit(row)
            self.stage.emit(f"Segmenting {input_path.name}")
            try:
                processor = ImageProcessor.from_model(self._model, quiet=True, **self._options)
                processor.reporter_cls = lambda: _QtProgressReporter(self.progress)
                observer = _LiveObserver(self, row, self._model.cfg.nodata_value)
                processor.run(input_path, output_path, cancel=self._cancelled.is_set, observer=observer)
            except ProcessingCancelled:
                logger.info(f"Cancelled {input_path.name}")
                self.item_finished.emit(row, Status.CANCELLED, "")
            except Exception as e:  # noqa: BLE001
                logger.opt(exception=e).debug("Segmentation failed")
                logger.error(f"Failed to segment {input_path.name}: {e}")
                self.item_finished.emit(row, Status.FAILED, str(e))
            else:
                logger.success(f"Saved {output_path}")
                self.item_finished.emit(row, Status.DONE, "")

            if self._model.active_device:
                self.device.emit(self._model.active_device)


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Call(QRunnable):
    def __init__(self, fn: Callable[[], object]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as e:  # noqa: BLE001
            logger.opt(exception=e).debug("Background task failed")
            self._emit(self.signals.failed, str(e))
        else:
            self._emit(self.signals.done, result)

    @staticmethod
    def _emit(signal: SignalInstance, value: object) -> None:
        try:
            signal.emit(value)
        except RuntimeError:
            # The app closed while the task was running, so there's nothing left to tell
            pass


_running: set[_Call] = set()


def run_in_background(
    fn: Callable[[], object],
    on_done: Callable[[object], None],
    on_failed: Callable[[str], None] | None = None,
) -> None:
    """Call fn on the global thread pool, then on_done with its result on the UI thread.

    Args:
        fn: Function to call in the background.
        on_done: Called with the result of fn.
        on_failed: Called with the error message if fn raises.
    """
    call = _Call(fn)
    # Keep the task alive until it reports back, so its signals aren't garbage collected first
    _running.add(call)
    call.signals.done.connect(lambda _: _running.discard(call))
    call.signals.failed.connect(lambda _: _running.discard(call))
    call.signals.done.connect(on_done)
    if on_failed is not None:
        call.signals.failed.connect(on_failed)
    QThreadPool.globalInstance().start(call)
