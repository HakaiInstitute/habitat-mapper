"""Main window of the Habitat-Mapper desktop app."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger
from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from habitat_mapper import __version__
from habitat_mapper.accelerators import select_accelerators
from habitat_mapper.gui.batch import BatchItem, Status, existing_outputs, output_path_for
from habitat_mapper.gui.canvas import PreviewCanvas
from habitat_mapper.gui.diagnostics import LOG_DIR, add_log_file, diagnostics_report
from habitat_mapper.gui.models_dialog import ModelsDialog, is_downloaded
from habitat_mapper.gui.preview import read_input_preview, read_output_preview
from habitat_mapper.gui.updates import newer_release
from habitat_mapper.gui.worker import SegmentWorker, run_in_background
from habitat_mapper.registry import model_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

DOCS_URL = "https://habitat-mapper.readthedocs.io"
_IMAGE_FILTER = "GeoTIFF images (*.tif *.tiff);;All files (*)"


class _LogBridge(QObject):
    """Passes log messages from any thread to the log pane."""

    message = Signal(str)


class MainWindow(QMainWindow):
    """Queue images, choose a model and output folder, run, and preview the results."""

    def __init__(self) -> None:
        """Build the window."""
        super().__init__()
        self.setWindowTitle(f"Habitat-Mapper {__version__}")
        self.resize(1200, 800)

        self._items: list[BatchItem] = []
        self._thread: QThread | None = None
        self._worker: SegmentWorker | None = None
        self._active_device: str | None = None
        self._preview_request = 0

        self._build_menus()
        self.setCentralWidget(self._build_body())
        self._connect_log_pane()
        self._show_expected_device()
        run_in_background(lambda: newer_release(__version__), self._show_update)

    # Layout

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, slot, shortcut in [
            ("Add images…", self._add_images, QKeySequence.StandardKey.Open),
            ("Add Sentinel-2 SAFE folder…", self._add_safe_folder, None),
        ]:
            action = QAction(text, self)
            action.triggered.connect(slot)
            if shortcut is not None:
                action.setShortcut(shortcut)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        tools_menu = self.menuBar().addMenu("&Tools")
        tools: list[tuple[str, Callable[[], object]]] = [
            ("Models…", lambda: ModelsDialog(self).exec()),
            ("Copy diagnostics", self._copy_diagnostics),
            ("Open log folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_DIR)))),
        ]
        for text, callback in tools:
            action = QAction(text, self)
            action.triggered.connect(callback)
            tools_menu.addAction(action)

        help_menu = self.menuBar().addMenu("&Help")
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        help_menu.addAction(docs_action)

    def _build_body(self) -> QWidget:
        self._update_banner = QLabel()
        self._update_banner.setOpenExternalLinks(True)
        self._update_banner.setStyleSheet("background: #fff3cd; color: #664d03; padding: 6px;")
        self._update_banner.hide()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_queue())
        splitter.addWidget(self._build_preview())
        splitter.setSizes([520, 680])

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setFixedHeight(120)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(self._update_banner)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._build_run_controls())
        layout.addWidget(self._log)
        return body

    def _build_queue(self) -> QWidget:
        self._model = QComboBox()
        for name in sorted(model_registry.list_model_names()):
            cfg = model_registry[name].cfg
            self._model.addItem(f"{name}{' (beta)' if cfg.beta else ''}", name)
            self._model.setItemData(self._model.count() - 1, cfg.description, Qt.ItemDataRole.ToolTipRole)
        self._model.currentIndexChanged.connect(self._model_changed)

        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Folder to save results in")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._choose_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_dir, 1)
        output_row.addWidget(browse)

        form = QFormLayout()
        form.addRow("Model", self._model)
        form.addRow("Output folder", output_row)

        self._queue = QTableWidget(0, 2)
        self._queue.setHorizontalHeaderLabels(["Image", "Status"])
        self._queue.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._queue.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._queue.verticalHeader().hide()
        self._queue.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._queue.itemSelectionChanged.connect(self._show_selected_preview)

        queue_buttons = QHBoxLayout()
        for text, slot in [
            ("Add images…", self._add_images),
            ("Add SAFE folder…", self._add_safe_folder),
            ("Remove", self._remove_selected),
            ("Clear", self._clear_queue),
        ]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            queue_buttons.addWidget(button)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(form)
        layout.addWidget(self._queue, 1)
        layout.addLayout(queue_buttons)
        layout.addWidget(self._build_advanced())
        return widget

    def _build_advanced(self) -> QWidget:
        toggle = QToolButton()
        toggle.setText("Advanced options")
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.RightArrow)
        toggle.setStyleSheet("QToolButton { border: none; }")

        self._revision = QComboBox()
        self._crop_size = QSpinBox()
        self._crop_size.setRange(0, 8192)
        self._crop_size.setSingleStep(64)
        self._crop_size.setSpecialValueText("Model default")
        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 64)
        self._blur = QSpinBox()
        self._blur.setRange(0, 99)
        self._blur.setSingleStep(2)
        self._blur.setSpecialValueText("Off")
        self._morph = QSpinBox()
        self._morph.setRange(0, 99)
        self._morph.setSingleStep(2)
        self._morph.setSpecialValueText("Off")
        self._band_order = QLineEdit()
        self._band_order.setPlaceholderText("Model default, e.g. 1, 2, 3")

        group = QGroupBox()
        form = QFormLayout(group)
        form.addRow("Revision", self._revision)
        form.addRow("Tile size (even)", self._crop_size)
        form.addRow("Batch size", self._batch_size)
        form.addRow("Median blur (odd)", self._blur)
        form.addRow("Morphological filter (odd)", self._morph)
        form.addRow("Band order", self._band_order)
        group.hide()

        def toggled(checked: bool) -> None:
            group.setVisible(checked)
            toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

        toggle.toggled.connect(toggled)
        self._model_changed()

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toggle)
        layout.addWidget(group)
        return widget

    def _build_preview(self) -> QWidget:
        self._canvas = PreviewCanvas()
        self._preview_label = QLabel("Select an image to preview it")
        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity.setValue(60)
        self._opacity.valueChanged.connect(lambda value: self._canvas.set_overlay_opacity(value / 100))
        self._canvas.set_overlay_opacity(0.6)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self._preview_label, 1)
        opacity_row.addWidget(QLabel("Result opacity"))
        opacity_row.addWidget(self._opacity)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas, 1)
        layout.addLayout(opacity_row)
        return widget

    def _build_run_controls(self) -> QWidget:
        self._stage = QLabel("Add images to get started")
        self._progress = QProgressBar()
        self._device_label = QLabel()
        self._run_button = QPushButton("Run")
        self._run_button.setDefault(True)
        self._run_button.clicked.connect(self._run)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel)

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stage, 2)
        layout.addWidget(self._progress, 2)
        layout.addWidget(self._device_label, 1)
        layout.addWidget(self._run_button)
        layout.addWidget(self._cancel_button)
        return widget

    # Log, device and updates

    def _connect_log_pane(self) -> None:
        self._log_bridge = _LogBridge()
        self._log_bridge.message.connect(self._log.appendPlainText)
        self._log_sink = logger.add(
            lambda message: self._log_bridge.message.emit(str(message).rstrip()),
            level="INFO",
            format="{time:HH:mm:ss} | {message}",
        )

    def _show_expected_device(self) -> None:
        try:
            accelerator = select_accelerators()[0]
            self._set_device(f"{accelerator.device} ({accelerator.hardware})")
        except (RuntimeError, ValueError) as e:
            logger.warning(str(e))

    def _set_device(self, description: str) -> None:
        self._active_device = description
        if description.startswith("cpu"):
            self._device_label.setText("Running on: CPU (slower)")
            self._device_label.setToolTip(
                "No supported GPU was found, so models run on the CPU. Check that your graphics driver is up to date."
            )
        else:
            self._device_label.setText(f"Running on: {description}")
            self._device_label.setToolTip("")

    def _show_update(self, release: object) -> None:
        if isinstance(release, tuple):
            version, url = release
            self._update_banner.setText(f'Habitat-Mapper {version} is available. <a href="{url}">Download it</a>')
            self._update_banner.show()

    def _copy_diagnostics(self) -> None:
        QGuiApplication.clipboard().setText(diagnostics_report(self._active_device))
        self.statusBar().showMessage("Diagnostics copied. Paste them into an email when asking for help.", 5000)

    # Queue

    def _add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add images", "", _IMAGE_FILTER)
        self._add_paths([Path(p) for p in paths])

    def _add_safe_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add a Sentinel-2 SAFE folder")
        if not path:
            return
        if not path.endswith(".SAFE"):
            QMessageBox.warning(self, "Not a SAFE folder", "Choose a Sentinel-2 folder whose name ends in .SAFE.")
            return
        self._add_paths([Path(path)])

    def _add_paths(self, paths: list[Path]) -> None:
        queued = {item.input_path for item in self._items}
        for path in paths:
            if path in queued:
                continue
            self._items.append(BatchItem(path))
            row = self._queue.rowCount()
            self._queue.insertRow(row)
            name = QTableWidgetItem(path.name)
            name.setToolTip(str(path))
            self._queue.setItem(row, 0, name)
            self._queue.setItem(row, 1, QTableWidgetItem(Status.QUEUED.value))
        if paths and not self._output_dir.text():
            self._output_dir.setText(str(paths[0].parent))
        if self._items:
            self._stage.setText(f"{len(self._items)} image(s) queued")
        if paths and not self._queue.selectedItems():
            self._queue.selectRow(self._queue.rowCount() - 1)

    def _remove_selected(self) -> None:
        if self._thread is not None:
            return
        for row in sorted({index.row() for index in self._queue.selectedIndexes()}, reverse=True):
            self._queue.removeRow(row)
            del self._items[row]

    def _clear_queue(self) -> None:
        if self._thread is not None:
            return
        self._queue.setRowCount(0)
        self._items.clear()
        self._canvas.set_images(None)

    def _set_status(self, row: int, status: Status, error: str = "") -> None:
        item = self._items[row]
        item.status = status
        item.error = error or None
        cell = QTableWidgetItem(status.value)
        cell.setToolTip(error)
        if status is Status.FAILED:
            cell.setForeground(Qt.GlobalColor.red)
        self._queue.setItem(row, 1, cell)

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose the output folder", self._output_dir.text())
        if path:
            self._output_dir.setText(path)

    # Model options

    def _model_changed(self) -> None:
        if not hasattr(self, "_revision"):
            return
        name = self._model.currentData()
        self._revision.clear()
        revisions = (revision for model_name, revision in model_registry.list_models() if model_name == name)
        self._revision.addItems(sorted(revisions, reverse=True))

    def _options(self) -> dict[str, object]:
        """Read the advanced options as ImageProcessor.from_model keyword arguments.

        Returns:
            The keyword arguments.

        Raises:
            ValueError: If an option is invalid, with a message for the user.
        """
        crop_size = self._crop_size.value() or None
        if crop_size is not None and crop_size % 2:
            raise ValueError("Tile size must be an even number.")
        for label, box in [("Median blur", self._blur), ("Morphological filter", self._morph)]:
            if box.value() and box.value() % 2 == 0:
                raise ValueError(f"{label} must be an odd number, or off.")

        return {
            "crop_size": crop_size,
            "batch_size": self._batch_size.value(),
            "blur_kernel_size": self._blur.value(),
            "morph_kernel_size": self._morph.value(),
            "band_order": self._parse_band_order(),
        }

    def _parse_band_order(self) -> list[int] | None:
        """Read the band order option.

        Returns:
            Band numbers, starting at 1, or None to use the model default.

        Raises:
            ValueError: If the band order is invalid, with a message for the user.
        """
        text = self._band_order.text().strip()
        if not text:
            return None
        try:
            band_order = [int(band) for band in re.split(r"[\s,]+", text)]
        except ValueError:
            raise ValueError("Band order must be band numbers separated by commas, e.g. 3, 2, 1.") from None
        if any(band < 1 for band in band_order):
            raise ValueError("Band numbers start at 1.")
        return band_order

    # Running

    def _run(self) -> None:
        if not self._items:
            QMessageBox.information(self, "Nothing to run", "Add some images to the queue first.")
            return
        output_dir = Path(self._output_dir.text()).expanduser()
        if not self._output_dir.text() or not output_dir.is_dir():
            QMessageBox.warning(self, "Choose an output folder", "Choose an existing folder to save the results in.")
            return
        try:
            options = self._options()
        except ValueError as e:
            QMessageBox.warning(self, "Check the advanced options", str(e))
            return

        name = self._model.currentData()
        model = model_registry[name, self._revision.currentText()]

        # Images already done with this model and output folder aren't run again
        def is_done(item: BatchItem) -> bool:
            return item.status is Status.DONE and item.output_path == output_path_for(item.input_path, output_dir, name)

        pending = [item for item in self._items if not is_done(item)]
        if not pending:
            QMessageBox.information(self, "Nothing to run", "Every image has already been run with this model.")
            return

        skip: set[Path] = set()
        if existing := existing_outputs(pending, output_dir, name):
            answer = QMessageBox.question(
                self,
                "Results already exist",
                f"{len(existing)} result(s) already exist in {output_dir}. Overwrite them?\n\n"
                "Choose No to skip those images.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return
            if answer == QMessageBox.StandardButton.No:
                skip = set(existing)

        jobs = []
        for row, item in enumerate(self._items):
            if is_done(item):
                continue
            item.output_path = output_path_for(item.input_path, output_dir, name)
            if item.output_path in skip:
                self._set_status(row, Status.SKIPPED)
            else:
                self._set_status(row, Status.QUEUED)
                jobs.append((row, item.input_path, item.output_path))
        if not jobs:
            return

        if not is_downloaded(model.cfg):
            logger.info(f"Downloading {name}. This only happens the first time it's used.")

        self._worker = SegmentWorker(model, jobs, options)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.item_started.connect(lambda row: self._set_status(row, Status.RUNNING))
        self._worker.item_finished.connect(self._item_finished)
        self._worker.progress.connect(self._show_progress)
        self._worker.stage.connect(self._stage.setText)
        self._worker.device.connect(self._set_device)
        self._worker.finished.connect(self._run_finished)
        self._set_running(True)
        self._progress.setRange(0, 0)
        self._thread.start()

    def _show_progress(self, done: int, total: int) -> None:
        self._progress.setRange(0, max(total, 1))
        self._progress.setValue(done)

    def _item_finished(self, row: int, status: Status, error: str) -> None:
        self._set_status(row, status, error)
        if status is Status.DONE and self._queue.currentRow() == row:
            self._show_selected_preview()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_button.setEnabled(False)
            self._stage.setText("Cancelling after the current tiles…")

    def _run_finished(self) -> None:
        if self._thread is None:
            return
        self._thread.quit()
        self._thread.wait()
        self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._set_running(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)

        counts = {status: sum(item.status is status for item in self._items) for status in Status}
        summary = ", ".join(f"{count} {status.value.lower()}" for status, count in counts.items() if count)
        self._stage.setText(f"Finished: {summary}")
        if counts[Status.FAILED]:
            QMessageBox.warning(
                self,
                "Some images failed",
                f"{counts[Status.FAILED]} image(s) failed. Hover over their status to see why, or check the log.",
            )

    def _set_running(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._model.setEnabled(not running)

    # Preview

    def _show_selected_preview(self) -> None:
        row = self._queue.currentRow()
        if row < 0 or row >= len(self._items):
            return
        item = self._items[row]
        output = item.output_path if item.status is Status.DONE else None
        try:
            band_order = self._parse_band_order()
        except ValueError:
            band_order = None

        self._preview_request += 1
        request = self._preview_request
        self._preview_label.setText(f"Loading {item.input_path.name}…")

        def load() -> tuple[np.ndarray, np.ndarray | None]:
            image = read_input_preview(item.input_path, band_order=band_order)
            overlay = read_output_preview(output) if output is not None and output.exists() else None
            return image, overlay

        def show(result: object) -> None:
            if request != self._preview_request:
                return
            image, overlay = cast("tuple[np.ndarray, np.ndarray | None]", result)
            self._canvas.set_images(image, overlay)
            self._preview_label.setText(item.input_path.name + ("" if overlay is not None else " (not run yet)"))

        def failed(error: str) -> None:
            if request == self._preview_request:
                self._canvas.set_images(None)
                self._preview_label.setText(f"Could not preview {item.input_path.name}: {error}")

        run_in_background(load, show, failed)

    def closeEvent(self, event: object) -> None:  # noqa: N802
        """Stop processing before closing, so the output isn't left half written."""
        if self._worker is not None and self._thread is not None:
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait()
        logger.remove(self._log_sink)
        super().closeEvent(event)  # type: ignore[arg-type]


def _ensure_std_streams() -> None:
    """Give windowed builds somewhere to write output.

    Windowed Windows builds start without stdout and stderr, but loguru, Rich, and the code that captures native
    library output from file descriptor 2 all expect them.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    devnull = open(os.devnull, "w")  # noqa: SIM115
    for fd in (1, 2):
        try:
            os.fstat(fd)
        except OSError:
            os.dup2(devnull.fileno(), fd)
    sys.stdout = sys.stdout or devnull
    sys.stderr = sys.stderr or devnull


def main() -> None:
    """Start the desktop app."""
    _ensure_std_streams()
    add_log_file()
    logger.info(f"Starting Habitat-Mapper {__version__}")

    app = QApplication(sys.argv)
    app.setApplicationName("Habitat-Mapper")
    app.setOrganizationName("Hakai Institute")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
