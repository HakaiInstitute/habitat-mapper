"""Smoke tests of the desktop app's window, driving it as someone would, with a tiny model on the CPU.

The rules the window follows are tested without Qt in test_gui_queue.py and test_gui_jobs.py. These check the window
wires them up. They need the gui extra and the gui-tests group, which CI's coverage job installs; elsewhere they skip
unless HAB_REQUIRE_QT_TESTS is set.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if not os.environ.get("HAB_REQUIRE_QT_TESTS"):
    pytest.importorskip("PySide6", reason="needs the gui extra")
    pytest.importorskip("pytestqt", reason="needs the gui-tests group")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel, QMessageBox  # noqa: E402

from habitat_mapper.config import OutputClass  # noqa: E402
from habitat_mapper.gui import app as app_module, batch as batch_module  # noqa: E402
from habitat_mapper.gui.batch import Status  # noqa: E402
from habitat_mapper.registry import ModelRegistry, model_registry  # noqa: E402
from tests.helpers import tiny_model_config, write_raster, write_tiny_model  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot

RUN_TIMEOUT_MS = 30_000


class Dialogs:
    """Stands in for QMessageBox, answering questions and recording what was shown."""

    def __init__(self) -> None:
        self.answer = QMessageBox.StandardButton.Yes
        self.shown: list[str] = []

    def question(self, parent: object, title: str, text: str, *args: object) -> QMessageBox.StandardButton:
        self.shown.append(title)
        return self.answer

    def message(self, parent: object, title: str, text: str, *args: object) -> None:
        self.shown.append(title)


@pytest.fixture
def dialogs(monkeypatch: pytest.MonkeyPatch) -> Dialogs:
    dialogs = Dialogs()
    monkeypatch.setattr(QMessageBox, "question", dialogs.question)
    monkeypatch.setattr(QMessageBox, "warning", dialogs.message)
    monkeypatch.setattr(QMessageBox, "information", dialogs.message)
    return dialogs


@pytest.fixture
def window(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dialogs: Dialogs) -> app_module.MainWindow:
    """The window, with the shipped models plus a tiny one that runs offline, chosen and ready to run.

    Returns:
        The window.
    """
    registry = ModelRegistry()
    for name, revision in model_registry.list_models():
        registry.register_model(model_registry.config(name, revision))
    tiny = tiny_model_config(write_tiny_model(tmp_path / "model" / "tiny.onnx"))
    registry.register_model(tiny.with_changes(classes=[OutputClass(value=1, name="Brighter", colour="#ff0000")]))
    monkeypatch.setattr(app_module, "model_registry", registry)
    monkeypatch.setattr(batch_module, "model_registry", registry)
    monkeypatch.setattr(app_module, "newer_release", lambda version: None)

    window = app_module.MainWindow()
    qtbot.addWidget(window)
    window._model.setCurrentIndex(window._model.findData("tiny"))
    window._crop_size.setValue(32)
    (tmp_path / "out").mkdir()
    window._output_dir.setText(str(tmp_path / "out"))
    return window


def _images(tmp_path: Path, count: int) -> list[Path]:
    rng = np.random.default_rng(0)
    return [
        write_raster(tmp_path / f"img{i}.tif", rng.integers(0, 255, size=(3, 60, 70), dtype=np.uint8))
        for i in range(count)
    ]


def _run(qtbot: QtBot, window: app_module.MainWindow) -> None:
    window._run()
    qtbot.waitUntil(lambda: window._thread is None, timeout=RUN_TIMEOUT_MS)


def _statuses(window: app_module.MainWindow) -> list[Status]:
    return [item.status for item in window._batch.items]


def _legend(window: app_module.MainWindow) -> list[str]:
    layout = window._legend_layout
    widgets = (layout.itemAt(i).widget() for i in range(layout.count()))
    return [w.text() for w in widgets if isinstance(w, QLabel) and w.text()]


def test_run_segments_the_queue_and_shows_the_result(
    qtbot: QtBot, window: app_module.MainWindow, tmp_path: Path
) -> None:
    images = _images(tmp_path, 2)
    window._add_paths([*images, images[0]])
    assert window._queue.rowCount() == 2

    _run(qtbot, window)

    assert _statuses(window) == [Status.DONE, Status.DONE]
    assert window._stage.text() == "Finished: 2 done"
    assert (tmp_path / "out" / "img0_tiny_20250101.tif").exists()

    window._queue.selectRow(0)
    qtbot.waitUntil(lambda: window._preview_label.text() == "img0.tif, result from tiny 20250101")
    assert _legend(window) == ["Brighter"]
    # A result keeps the legend of the model that made it
    window._model.setCurrentIndex(window._model.findData("kelp-rgb"))
    assert _legend(window) == ["Brighter"]


def test_running_again_only_runs_what_isnt_done(
    qtbot: QtBot, window: app_module.MainWindow, tmp_path: Path, dialogs: Dialogs
) -> None:
    window._add_paths(_images(tmp_path, 1))
    _run(qtbot, window)

    window._run()

    assert dialogs.shown == ["Nothing to run"]


@pytest.mark.parametrize(
    ("answer", "statuses"),
    [
        (QMessageBox.StandardButton.No, [Status.DONE, Status.SKIPPED]),
        (QMessageBox.StandardButton.Yes, [Status.DONE, Status.DONE]),
        (QMessageBox.StandardButton.Cancel, [Status.QUEUED, Status.QUEUED]),
    ],
)
def test_existing_results_are_skipped_overwritten_or_left(
    qtbot: QtBot,
    window: app_module.MainWindow,
    tmp_path: Path,
    dialogs: Dialogs,
    answer: QMessageBox.StandardButton,
    statuses: list[Status],
) -> None:
    window._add_paths(_images(tmp_path, 2))
    (tmp_path / "out" / "img1_tiny_20250101.tif").touch()
    dialogs.answer = answer

    _run(qtbot, window)

    assert dialogs.shown == ["Results already exist"]
    assert _statuses(window) == statuses


def test_invalid_options_stop_the_run(window: app_module.MainWindow, tmp_path: Path, dialogs: Dialogs) -> None:
    window._add_paths(_images(tmp_path, 1))
    window._band_order.setText("red, green, blue")

    window._run()

    assert dialogs.shown == ["Check the advanced options"]
    assert window._thread is None
    assert _statuses(window) == [Status.QUEUED]


def test_cancel_stops_the_run(
    qtbot: QtBot, window: app_module.MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CancelsOnStart(app_module.SegmentWorker):
        # Cancels from the worker's own thread as the first image starts, so the timing is always the same
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            super().__init__(*args, **kwargs)
            self.item_started.connect(lambda row: self.cancel(), Qt.ConnectionType.DirectConnection)

    monkeypatch.setattr(app_module, "SegmentWorker", CancelsOnStart)
    window._add_paths(_images(tmp_path, 3))

    _run(qtbot, window)

    assert _statuses(window) == [Status.CANCELLED] * 3
    assert window._stage.text() == "Finished: 3 cancelled"
    assert not any((tmp_path / "out").iterdir())


def test_choosing_a_sentinel_2_model_offers_safe_inputs(window: app_module.MainWindow) -> None:
    assert window._add_images_button.isVisibleTo(window)
    assert not window._add_safe_button.isVisibleTo(window)

    window._model.setCurrentIndex(window._model.findData("kelp-skema"))

    assert not window._add_images_button.isVisibleTo(window)
    assert window._add_safe_button.isVisibleTo(window)
    assert window._add_safe_zip_button.isVisibleTo(window)
