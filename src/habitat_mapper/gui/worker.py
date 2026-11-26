"""Background work for the desktop app: segmenting the batch queue and loading previews."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, SignalInstance, Slot

from habitat_mapper.gui.jobs import RunEvents, run_jobs

if TYPE_CHECKING:
    from collections.abc import Callable

    from habitat_mapper.config import SegmentOptions
    from habitat_mapper.gui.batch import Job
    from habitat_mapper.model import ONNXModel


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

    def __init__(self, model: ONNXModel, jobs: list[Job], options: SegmentOptions) -> None:
        """Set up the work.

        Args:
            model: Model to run. The worker uses it from its own thread, so leave it alone until the worker is done.
            jobs: The images to segment.
            options: Processing options, e.g. crop_size.
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
        events = RunEvents(
            stage=self.stage.emit,
            progress=self.progress.emit,
            item_started=self.item_started.emit,
            item_finished=self.item_finished.emit,
            device=self.device.emit,
            live_started=self.live_started.emit,
            tiles_started=self.tiles_started.emit,
        )
        run_jobs(self._model, self._jobs, self._options, events, self._cancelled.is_set)


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
