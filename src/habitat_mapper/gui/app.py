"""Main window of the Habitat-Mapper desktop app."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger
from pydantic import ValidationError
from PySide6.QtCore import QObject, QPoint, QProcess, QRectF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from habitat_mapper import __version__
from habitat_mapper.accelerators import select_accelerators
from habitat_mapper.cache import model_cache
from habitat_mapper.config import SegmentOptions
from habitat_mapper.gui import theme
from habitat_mapper.gui.batch import BatchItem, Status, existing_outputs, output_path_for, output_stem
from habitat_mapper.gui.canvas import PreviewCanvas
from habitat_mapper.gui.diagnostics import LOG_DIR, add_log_file, diagnostics_report
from habitat_mapper.gui.model_info import (
    accepted_inputs,
    default_band_order,
    describe_option_errors,
    input_description,
    reads_safe,
)
from habitat_mapper.gui.models_dialog import ModelsDialog
from habitat_mapper.gui.preview import (
    DEFAULT_PALETTE,
    LiveOverlay,
    class_palette,
    read_input_preview,
    read_output_preview,
)
from habitat_mapper.gui.tray import TrayIcon
from habitat_mapper.gui.updates import newer_release
from habitat_mapper.gui.worker import SegmentWorker, run_in_background
from habitat_mapper.registry import model_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from PySide6.QtCore import QModelIndex, QPersistentModelIndex
    from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDragLeaveEvent, QDropEvent

    from habitat_mapper.config import ModelConfig, OutputClass
    from habitat_mapper.model import ONNXModel

# How often the preview of the image being processed is redrawn
_LIVE_REFRESH_MS = 200
DOCS_URL = "https://habitat-mapper.readthedocs.io"
_IMAGE_FILTER = "GeoTIFF images (*.tif *.tiff);;All files (*)"
_SAFE_ZIP_FILTER = "Sentinel-2 SAFE zips (*.zip);;All files (*)"
# Fraction of an image done, stored on its status cell while it is processed
_PROGRESS_ROLE = Qt.ItemDataRole.UserRole + 1
_LIVE_HELP = (
    "The outlined box is the tile being processed, dashed where it extends past the edge of the image. The haze "
    "covers parts the model has seen but that aren't finished: overlapping predictions are blended for smooth edges, "
    "so each part waits for the tiles that overlap it to the right and below. The final result, with any filters "
    "applied, replaces this preview when the image is done."
)


class _LogBridge(QObject):
    """Passes log messages from any thread to the log pane."""

    message = Signal(str)


class _OddSpinBox(QSpinBox):
    """Kernel size for a filter, which must be odd, or 0 to turn the filter off."""

    def __init__(self) -> None:
        """Create the spin box, starting at off."""
        super().__init__()
        self.setRange(0, 99)
        self.setSpecialValueText("Off")

    def stepBy(self, steps: int) -> None:  # noqa: N802
        """Step through 0 (off), 3, 5, 7 and so on. A kernel size of 1 has no effect, so it is skipped."""
        value = self.value()
        for _ in range(abs(steps)):
            # Values typed in can be even, so step to the next odd number either way
            if steps > 0:
                value = 3 if value < 3 else value + (2 if value % 2 else 1)
            else:
                value = 0 if value <= 3 else value - (2 if value % 2 else 1)
        self.setValue(min(value, self.maximum()))


class _StatusDelegate(QStyledItemDelegate):
    """Draws a thin progress bar along the bottom of the status of the image being processed."""

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> None:
        """Draw the status, then the progress bar if the image is being processed."""
        super().paint(painter, option, index)
        fraction = index.data(_PROGRESS_ROLE)
        if fraction is None:
            return
        rect = option.rect  # type: ignore[attr-defined]
        bar = QRectF(rect.left() + 6, rect.bottom() - 4, (rect.width() - 12) * float(fraction), 3)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.OCHRE))
        painter.drawRoundedRect(bar, 1.5, 1.5)
        painter.restore()


def _dot(colour: QColor | str, size: int = 10) -> QIcon:
    """A filled circle, for marking the status of an image.

    Returns:
        The icon.
    """
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(colour))
    painter.drawEllipse(QRectF(1, 1, size - 2, size - 2))
    painter.end()
    return QIcon(pixmap)


def _swatch(colour: str, border: str | None = None, haze: bool = False) -> QLabel:
    """A small square of colour for a legend.

    Returns:
        The swatch.
    """
    swatch = QLabel()
    swatch.setFixedSize(12, 12)
    # The haze as it looks over the dark preview background
    background = "#7A9596" if haze else colour
    outline = f"2px solid {border}" if border else "1px solid rgba(0, 0, 0, 60)"
    swatch.setStyleSheet(f"background: {background}; border: {outline}; border-radius: 2px;")
    return swatch


def _muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    return label


def _reveal(path: Path) -> None:
    """Show a file in Finder or File Explorer, selected, or open its folder elsewhere."""
    if sys.platform == "darwin":
        QProcess.startDetached("open", ["-R", str(path)])
    elif sys.platform == "win32":
        QProcess.startDetached("explorer", [f"/select,{path}"])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))


class MainWindow(QMainWindow):
    """Choose a model, queue images and an output folder, run, and preview the results."""

    def __init__(self) -> None:
        """Build the window."""
        super().__init__()
        self.setWindowTitle(f"Habitat-Mapper {__version__}")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)
        self.setAcceptDrops(True)

        self._items: list[BatchItem] = []
        self._thread: QThread | None = None
        self._worker: SegmentWorker | None = None
        # The model of the last run, kept so its session stays loaded for the next run with the same model
        self._loaded_model: ONNXModel | None = None
        self._active_device: str | None = None
        self._preview_request = 0
        # Queue row the preview shows, and the row being processed with its result so far and current tiles
        self._preview_row: int | None = None
        self._live_row: int | None = None
        self._live_overlay: LiveOverlay | None = None
        self._live_tiles: list[tuple[int, int, int, int]] = []
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(_LIVE_REFRESH_MS)
        self._live_timer.timeout.connect(self._refresh_live_preview)
        # Images in the current run, how many have started, and where the run writes
        self._run_total = 0
        self._run_started = 0
        self._run_output_dir: Path | None = None

        self._build_menus()
        self.setCentralWidget(self._build_body())
        self._tray = TrayIcon(self, self._cancel)
        self._connect_log_pane()
        self._show_expected_device()
        self._model.currentIndexChanged.connect(self._model_changed)
        self._revision.currentIndexChanged.connect(self._revision_changed)
        self._model_changed()
        run_in_background(lambda: newer_release(__version__), self._show_update)

    # Layout

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._add_images_action = QAction("Add GeoTIFFs…", self)
        self._add_images_action.setShortcut(QKeySequence.StandardKey.Open)
        self._add_images_action.triggered.connect(self._add_images)
        self._add_safe_zip_action = QAction("Add Sentinel-2 zip downloads…", self)
        self._add_safe_zip_action.setShortcut(QKeySequence.StandardKey.Open)
        self._add_safe_zip_action.triggered.connect(self._add_safe_zips)
        self._add_safe_action = QAction("Add extracted Sentinel-2 .SAFE folder…", self)
        self._add_safe_action.triggered.connect(self._add_safe_folder)
        file_menu.addAction(self._add_images_action)
        file_menu.addAction(self._add_safe_zip_action)
        file_menu.addAction(self._add_safe_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        tools_menu = self.menuBar().addMenu("&Tools")
        tools: list[tuple[str, Callable[[], object]]] = [
            ("Models…", self._open_models_dialog),
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
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_steps())
        splitter.addWidget(self._build_preview())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 880])

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setFixedHeight(140)
        self._log.hide()

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        layout.addWidget(self._build_update_bar())
        layout.addWidget(splitter, 1)
        layout.addWidget(self._build_run_bar())
        layout.addWidget(self._log)
        return body

    def _build_update_bar(self) -> QWidget:
        self._update_bar = QFrame()
        self._update_bar.setObjectName("infoBar")
        self._update_label = QLabel()
        self._update_url = ""
        download = QPushButton("Download")
        download.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self._update_url)))
        dismiss = QToolButton()
        dismiss.setText("✕")
        dismiss.setToolTip("Hide until next time")
        dismiss.setAccessibleName("Hide update message")
        dismiss.clicked.connect(self._update_bar.hide)
        layout = QHBoxLayout(self._update_bar)
        layout.setContentsMargins(12, 6, 6, 6)
        layout.addWidget(self._update_label, 1)
        layout.addWidget(download)
        layout.addWidget(dismiss)
        self._update_bar.hide()
        return self._update_bar

    @staticmethod
    def _step(number: int, title: str, trailing: QWidget | None = None) -> QHBoxLayout:
        badge = QLabel(str(number))
        badge.setObjectName("stepNumber")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(title)
        label.setObjectName("stepTitle")
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(badge)
        row.addWidget(label)
        row.addStretch(1)
        if trailing is not None:
            row.addWidget(trailing)
        return row

    def _build_steps(self) -> QWidget:
        widget = QWidget()
        widget.setMinimumWidth(340)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(6)

        layout.addLayout(self._step(1, "Model"))
        layout.addLayout(self._build_model_picker())
        layout.addSpacing(14)

        self._queue_count = _muted()
        layout.addLayout(self._step(2, "Images", self._queue_count))
        layout.addWidget(self._build_queue(), 1)
        layout.addSpacing(14)

        layout.addLayout(self._step(3, "Output folder"))
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Folder to save results in")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._choose_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_dir, 1)
        output_row.addWidget(browse)
        layout.addLayout(output_row)
        self._output_hint = _muted("Results are named after the image, model and revision.")
        self._output_hint.setWordWrap(True)
        layout.addWidget(self._output_hint)
        layout.addSpacing(8)

        layout.addWidget(self._build_advanced())
        return widget

    def _build_model_picker(self) -> QVBoxLayout:
        self._model = QComboBox()
        for name in model_registry.list_model_names():
            self._model.addItem(name, name)
        self._model.setAccessibleName("Model")
        self._revision = QComboBox()
        self._revision.setToolTip("Model version. The newest stable revision is chosen by default.")
        self._revision.setAccessibleName("Model revision")

        self._revision.setMinimumContentsLength(8)
        model_row = QHBoxLayout()
        model_row.addWidget(self._model, 1)
        model_row.addWidget(_muted("Revision"))
        model_row.addWidget(self._revision)

        self._model_description = QLabel()
        self._model_description.setWordWrap(True)
        self._model_input = _muted()
        self._model_download = _muted()
        self._beta_tag = QLabel("Beta")
        self._beta_tag.setObjectName("betaTag")
        self._beta_tag.setToolTip("This model is still being tested. Check its results carefully.")
        meta_row = QHBoxLayout()
        meta_row.setSpacing(10)
        meta_row.addWidget(self._model_input)
        meta_row.addWidget(self._model_download)
        meta_row.addStretch(1)
        meta_row.addWidget(self._beta_tag)

        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.addLayout(model_row)
        layout.addWidget(self._model_description)
        layout.addLayout(meta_row)
        return layout

    def _build_queue(self) -> QWidget:
        self._queue = QTableWidget(0, 2)
        self._queue.setHorizontalHeaderLabels(["Image", "Status"])
        self._queue.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._queue.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._queue.setShowGrid(False)
        self._queue.setWordWrap(False)
        self._queue.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._queue.verticalHeader().hide()
        self._queue.verticalHeader().setDefaultSectionSize(28)
        header = self._queue.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 120)
        self._queue.setItemDelegateForColumn(1, _StatusDelegate(self._queue))
        self._queue.itemSelectionChanged.connect(self._show_selected_preview)
        self._queue.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._queue.customContextMenuRequested.connect(self._queue_menu)
        self._queue.cellDoubleClicked.connect(self._open_row)

        # Only the add buttons for what the model reads are shown. Sentinel-2 products come as zips or extracted
        # folders, which need different file dialogs, so they get a button each. The model already says it reads
        # Sentinel-2, so the labels only name the format, keeping the row from crowding Remove and Clear
        self._add_images_button = QPushButton("Add GeoTIFFs…")
        self._add_images_button.clicked.connect(self._add_images)
        self._add_safe_zip_button = QPushButton("Add .SAFE.zip…")
        self._add_safe_zip_button.setToolTip("Add Sentinel-2 products as downloaded, without extracting them")
        self._add_safe_zip_button.clicked.connect(self._add_safe_zips)
        self._add_safe_button = QPushButton("Add .SAFE…")
        self._add_safe_button.setToolTip("Add an extracted Sentinel-2 .SAFE folder")
        self._add_safe_button.clicked.connect(self._add_safe_folder)
        self._remove_button = QPushButton("Remove")
        self._remove_button.clicked.connect(self._remove_selected)
        self._clear_button = QPushButton("Clear")
        self._clear_button.clicked.connect(self._clear_queue)
        buttons = QHBoxLayout()
        buttons.addWidget(self._add_images_button)
        buttons.addWidget(self._add_safe_zip_button)
        buttons.addWidget(self._add_safe_button)
        buttons.addStretch(1)
        buttons.addWidget(self._remove_button)
        buttons.addWidget(self._clear_button)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._queue, 1)
        layout.addLayout(buttons)
        return widget

    def _build_advanced(self) -> QWidget:
        toggle = QToolButton()
        toggle.setText("Advanced options")
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.RightArrow)
        toggle.setStyleSheet("QToolButton { border: none; }")

        self._crop_size = QSpinBox()
        self._crop_size.setRange(0, 8192)
        self._crop_size.setSingleStep(64)
        self._crop_size.setSpecialValueText("Model default")
        self._crop_size.setToolTip("Most models only run at their own tile size, so leave this at the default.")
        self._blur = _OddSpinBox()
        self._morph = _OddSpinBox()
        self._band_order = QLineEdit()

        for box in (self._crop_size, self._blur, self._morph):
            box.setMinimumWidth(140)
        self._band_order.setMinimumWidth(140)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(12, 12, 12, 12)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Tile size (even)", self._crop_size)
        form.addRow("Median blur (odd)", self._blur)
        form.addRow("Morphological filter (odd)", self._morph)
        form.addRow("Band order", self._band_order)
        group.hide()

        def toggled(checked: bool) -> None:
            group.setVisible(checked)
            toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

        toggle.toggled.connect(toggled)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toggle)
        layout.addWidget(group)
        return widget

    def _build_preview(self) -> QWidget:
        self._canvas = PreviewCanvas()
        self._canvas.setToolTip("Scroll to zoom, drag to pan, double click to fit. Hold Space to hide the result.")
        self._preview_label = QLabel()
        self._preview_label.setWordWrap(True)
        self._preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity.setValue(60)
        self._opacity.setFixedWidth(140)
        self._opacity.setAccessibleName("Result opacity")
        self._opacity.setToolTip("Hold Space over the preview to hide the result")
        self._opacity.valueChanged.connect(lambda value: self._canvas.set_overlay_opacity(value / 100))
        self._canvas.set_overlay_opacity(0.6)

        caption_row = QHBoxLayout()
        caption_row.addWidget(self._preview_label, 1)
        caption_row.addWidget(_muted("Result opacity"))
        caption_row.addWidget(self._opacity)

        # Key to the colours of the result, and to the tile outline and haze while an image is processed
        self._legend = QWidget()
        self._legend_layout = QHBoxLayout(self._legend)
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_layout.setSpacing(6)
        self._live_key = QWidget()
        self._live_key.setToolTip(_LIVE_HELP)
        live_layout = QHBoxLayout(self._live_key)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(6)
        live_layout.addWidget(_swatch("transparent", border=theme.OCHRE))
        live_layout.addWidget(_muted("Tile being processed"))
        live_layout.addSpacing(10)
        live_layout.addWidget(_swatch("", haze=True))
        live_layout.addWidget(_muted("Waiting for overlapping tiles"))
        self._live_key.hide()
        key_row = QHBoxLayout()
        key_row.addWidget(self._legend)
        key_row.addStretch(1)
        key_row.addWidget(self._live_key)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._canvas, 1)
        layout.addLayout(caption_row)
        layout.addLayout(key_row)
        return widget

    def _build_run_bar(self) -> QWidget:
        self._stage = QLabel()
        self._stage.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._progress = QProgressBar()
        self._progress.setObjectName("runProgress")
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setAccessibleName("Progress")
        status = QVBoxLayout()
        status.setSpacing(5)
        status.addWidget(self._stage)
        status.addWidget(self._progress)

        self._device_label = _muted()
        self._open_output_button = QPushButton("Open output folder")
        self._open_output_button.clicked.connect(self._open_output_dir)
        self._open_output_button.hide()
        self._log_toggle = QToolButton()
        self._log_toggle.setText("Show log")
        self._log_toggle.setCheckable(True)
        self._log_toggle.setObjectName("quietButton")
        self._log_toggle.toggled.connect(self._show_log)
        self._run_button = QPushButton("Run")
        self._run_button.setObjectName("runButton")
        self._run_button.setDefault(True)
        self._run_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._run_button.clicked.connect(self._run)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel)

        widget = QFrame()
        widget.setObjectName("runBar")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(status, 1)
        layout.addWidget(self._device_label)
        layout.addWidget(self._open_output_button)
        layout.addWidget(self._log_toggle)
        layout.addWidget(self._cancel_button)
        layout.addWidget(self._run_button)
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

    def _show_log(self, show: bool) -> None:
        self._log.setVisible(show)
        self._log_toggle.setText("Hide log" if show else "Show log")
        if show:
            self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

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
            self._update_url = url
            self._update_label.setText(f"Habitat-Mapper {version} is available. You have {__version__}.")
            self._update_bar.show()

    def _copy_diagnostics(self) -> None:
        QGuiApplication.clipboard().setText(diagnostics_report(self._active_device))
        self.statusBar().showMessage("Diagnostics copied. Paste them into an email when asking for help.", 5000)

    def _open_models_dialog(self) -> None:
        ModelsDialog(self).exec()
        # Models may have been downloaded
        self._show_model_info()

    # Queue

    def _add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add GeoTIFFs", "", _IMAGE_FILTER)
        self._add_paths([Path(p) for p in paths])

    def _add_safe_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add a Sentinel-2 SAFE folder")
        if not path:
            return
        if not path.endswith(".SAFE"):
            QMessageBox.warning(
                self,
                "Not a SAFE folder",
                "Choose a Sentinel-2 folder whose name ends in .SAFE. For a .zip download, use Add .SAFE.zip… instead.",
            )
            return
        self._add_paths([Path(path)])

    def _add_safe_zips(self) -> None:
        chosen = [
            Path(p) for p in QFileDialog.getOpenFileNames(self, "Add Sentinel-2 SAFE zips", "", _SAFE_ZIP_FILTER)[0]
        ]
        paths = accepted_inputs(chosen, safe=True)
        rejected = len(chosen) - len(paths)
        note = (
            f"{rejected} file{'s' if rejected != 1 else ''} not added. Choose Sentinel-2 .SAFE.zip downloads. "
            "A zip that won't add may be damaged or only partly downloaded."
            if rejected
            else ""
        )
        self._add_paths(paths, note)

    def _add_paths(self, paths: list[Path], note: str = "") -> None:
        """Queue images, skipping ones already queued.

        Args:
            paths: Images to queue.
            note: Status message about paths the caller didn't pass on, shown alongside any about these.
        """
        queued = {item.input_path for item in self._items}
        # A .SAFE folder and its .zip give results with the same name, so only one of them is queued
        stems = {output_stem(item.input_path) for item in self._items}
        clashes = 0
        added: list[Path] = []
        for path in paths:
            if path in queued:
                continue
            if output_stem(path) in stems:
                clashes += 1
                continue
            queued.add(path)
            stems.add(output_stem(path))
            added.append(path)
            self._items.append(BatchItem(path))
            row = self._queue.rowCount()
            self._queue.insertRow(row)
            name = QTableWidgetItem(path.name)
            name.setToolTip(str(path))
            self._queue.setItem(row, 0, name)
            self._set_status(row, Status.QUEUED)
        # Skipped paths are left out, so the results go beside an image that was queued
        if added and not self._output_dir.text():
            self._output_dir.setText(str(added[0].parent))
        if added and not self._queue.selectedItems():
            self._queue.selectRow(self._queue.rowCount() - 1)
        self._update_queue_state()
        messages = [note] if note else []
        if clashes:
            messages.append(
                f"{clashes} item{'s' if clashes != 1 else ''} not added. An image with the same name is already "
                "queued, and their results would overwrite each other."
            )
        if messages:
            self.statusBar().showMessage(" ".join(messages), 8000)

    def _remove_rows(self, rows: set[int]) -> None:
        if self._thread is not None:
            return
        for row in sorted(rows, reverse=True):
            self._queue.removeRow(row)
            del self._items[row]
        self._preview_row = None
        if not self._items:
            self._canvas.set_images(None)
        self._update_queue_state()

    def _remove_selected(self) -> None:
        self._remove_rows({index.row() for index in self._queue.selectedIndexes()})

    def _clear_queue(self) -> None:
        self._remove_rows(set(range(len(self._items))))

    def _update_queue_state(self) -> None:
        """Update the parts of the window that depend on how many images are queued."""
        count = len(self._items)
        self._queue_count.setText(f"{count} image{'s' if count != 1 else ''}" if count else "")
        running = self._thread is not None
        self._remove_button.setEnabled(bool(count) and not running)
        self._clear_button.setEnabled(bool(count) and not running)
        safe = reads_safe(self._current_cfg())
        if not count:
            self._preview_label.setText("")
            if safe:
                self._canvas.set_placeholder(
                    "Drop .SAFE.zip files or .SAFE folders here",
                    "Or use Add .SAFE.zip… or Add .SAFE… to choose them",
                )
            else:
                self._canvas.set_placeholder("Drop GeoTIFFs here", "Or use Add GeoTIFFs… to choose them")
        elif not self._canvas.has_image():
            self._canvas.set_placeholder("Select an image to preview it")
        if not running:
            self._stage.setText(
                f"{count} image{'s' if count != 1 else ''} ready to run" if count else "Add images to get started"
            )

    def _set_status(self, row: int, status: Status, error: str = "") -> None:
        item = self._items[row]
        item.status = status
        item.error = error or None
        colours = {
            Status.RUNNING: theme.running_colour(),
            Status.DONE: theme.done_colour(),
            Status.FAILED: theme.failed_colour(),
        }
        cell = QTableWidgetItem(status.value)
        cell.setToolTip(error)
        if status in colours:
            cell.setForeground(colours[status])
            cell.setIcon(_dot(colours[status]))
        else:
            cell.setIcon(_dot(self.palette().placeholderText().color()))
        if status is Status.RUNNING:
            cell.setData(_PROGRESS_ROLE, 0.0)
        self._queue.setItem(row, 1, cell)

    def _queue_menu(self, position: QPoint) -> None:
        row = self._queue.rowAt(position.y())
        if row < 0:
            return
        item = self._items[row]
        menu = QMenu(self)
        has_result = item.status is Status.DONE and item.output_path is not None and item.output_path.exists()
        open_result = menu.addAction("Open result")
        open_result.setEnabled(has_result)
        show_result = menu.addAction("Show result in folder")
        show_result.setEnabled(has_result)
        show_input = menu.addAction("Show image in folder")
        menu.addSeparator()
        remove = menu.addAction("Remove")
        remove.setEnabled(self._thread is None)
        if item.error:
            menu.addSeparator()
            copy_error = menu.addAction("Copy error")
        else:
            copy_error = None

        chosen = menu.exec(self._queue.viewport().mapToGlobal(position))
        if chosen is None:
            return
        if chosen is open_result and item.output_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(item.output_path)))
        elif chosen is show_result and item.output_path is not None:
            _reveal(item.output_path)
        elif chosen is show_input:
            _reveal(item.input_path)
        elif chosen is remove:
            self._remove_rows({row})
        elif chosen is copy_error and item.error:
            QGuiApplication.clipboard().setText(item.error)

    def _open_row(self, row: int, _column: int) -> None:
        """Show the result of a finished image in its folder."""
        item = self._items[row]
        if item.status is Status.DONE and item.output_path is not None and item.output_path.exists():
            _reveal(item.output_path)

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose the output folder", self._output_dir.text())
        if path:
            self._output_dir.setText(path)

    def _open_output_dir(self) -> None:
        if self._run_output_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._run_output_dir)))

    # Drag and drop

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        """Accept files and folders dragged onto the window."""
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
            self._canvas.set_drop_highlight(True)

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # noqa: N802
        """Stop showing that files can be dropped."""
        self._canvas.set_drop_highlight(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        """Queue the dropped images the model can read."""
        self._canvas.set_drop_highlight(False)
        dropped = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        safe = reads_safe(self._current_cfg())
        paths = accepted_inputs(dropped, safe=safe)
        event.acceptProposedAction()
        skipped = len(dropped) - len(paths)
        note = ""
        if skipped:
            needs = "Sentinel-2 .SAFE.zip files or .SAFE folders" if safe else "GeoTIFF images (.tif or .tiff)"
            note = f"{skipped} item{'s' if skipped != 1 else ''} not added. {self._model.currentData()} reads {needs}."
        self._add_paths(paths, note)

    # Model options

    def _current_cfg(self) -> ModelConfig:
        name = self._model.currentData()
        revision = self._revision.currentText()
        return model_registry.config(name, revision or "latest")

    def _model_changed(self) -> None:
        name = self._model.currentData()
        self._revision.blockSignals(True)
        self._revision.clear()
        self._revision.addItems(model_registry.revisions(name))
        self._revision.setCurrentText(model_registry.get_latest_revision(name))
        self._revision.blockSignals(False)

        safe = reads_safe(model_registry.config(name))
        self._add_images_button.setVisible(not safe)
        self._add_images_action.setVisible(not safe)
        self._add_safe_button.setVisible(safe)
        self._add_safe_zip_button.setVisible(safe)
        self._add_safe_action.setVisible(safe)
        self._add_safe_zip_action.setVisible(safe)
        self._revision_changed()

    def _revision_changed(self) -> None:
        cfg = self._current_cfg()
        self._band_order.setPlaceholderText(f"Model default: {default_band_order(cfg)}")
        self._show_model_info()
        self._update_queue_state()
        if not self._canvas.has_image() or self._preview_row is None:
            self._show_legend(cfg.classes)

    def _show_model_info(self) -> None:
        cfg = self._current_cfg()
        self._model_description.setText(cfg.description or "")
        self._model_input.setText(f"Reads: {input_description(cfg)}")
        if model_cache.is_cached(cfg):
            self._model_download.setText("Downloaded")
            self._model_download.setToolTip("This model can run without an internet connection.")
        else:
            self._model_download.setText("Downloads on first run")
            self._model_download.setToolTip("Download it ahead of time from Tools → Models… to work offline.")
        self._beta_tag.setVisible(cfg.beta)

    def _show_legend(self, classes: list[OutputClass]) -> None:
        while (child := self._legend_layout.takeAt(0)) is not None:
            if (widget := child.widget()) is not None:
                # Hide it now, since deleteLater leaves it drawn until the next pass of the event loop
                widget.hide()
                widget.deleteLater()
        for output_class in classes:
            self._legend_layout.addWidget(_swatch(output_class.colour))
            self._legend_layout.addWidget(QLabel(output_class.name))
            self._legend_layout.addSpacing(8)

    def _options(self) -> SegmentOptions:
        """Read the advanced options.

        Returns:
            The options.

        Raises:
            ValueError: If an option is invalid, with a message for the user.
        """
        try:
            return SegmentOptions(
                crop_size=self._crop_size.value() or None,
                # Batches larger than 1 were never faster in testing
                batch_size=1,
                blur_kernel_size=self._blur.value(),
                morph_kernel_size=self._morph.value(),
                band_order=self._parse_band_order(),
            )
        except ValidationError as e:
            raise ValueError(describe_option_errors(e)) from None

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
            return [int(band) for band in re.split(r"[\s,]+", text)]
        except ValueError:
            raise ValueError("Band order must be band numbers separated by commas, e.g. 3, 2, 1.") from None

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
        revision = self._revision.currentText()
        model = self._loaded_model
        if model is None or (model.name, model.revision) != (name, revision):
            # Only one model is kept loaded, since each can take a lot of memory
            model = self._loaded_model = model_registry[name, revision]

        # Images already done with this model and output folder aren't run again
        def is_done(item: BatchItem) -> bool:
            output_path = output_path_for(item.input_path, output_dir, name, revision)
            return item.status is Status.DONE and item.output_path == output_path

        pending = [item for item in self._items if not is_done(item)]
        if not pending:
            QMessageBox.information(self, "Nothing to run", "Every image has already been run with this model.")
            return

        skip: set[Path] = set()
        if existing := existing_outputs(pending, output_dir, name, revision):
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
            item.output_path = output_path_for(item.input_path, output_dir, name, revision)
            item.model_name = name
            if item.output_path in skip:
                self._set_status(row, Status.SKIPPED)
            else:
                self._set_status(row, Status.QUEUED)
                jobs.append((row, item.input_path, item.output_path))
        if not jobs:
            return

        if not model_cache.is_cached(model.cfg):
            logger.info(f"Downloading {name}. This only happens the first time it's used.")

        self._run_total, self._run_started, self._run_output_dir = len(jobs), 0, output_dir
        self._open_output_button.hide()
        self._worker = SegmentWorker(model, jobs, options)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.item_started.connect(self._item_started)
        self._worker.item_finished.connect(self._item_finished)
        self._worker.progress.connect(self._show_progress)
        self._worker.stage.connect(self._show_stage)
        self._worker.device.connect(self._set_device)
        self._worker.live_started.connect(self._live_started)
        self._worker.tiles_started.connect(self._tiles_started)
        self._worker.finished.connect(self._run_finished)
        self._set_running(True)
        self._progress.setRange(0, 0)
        self._tray.set_status("Loading the model", running=True)
        self._thread.start()

    def _show_stage(self, text: str) -> None:
        if self._run_started:
            text = f"Image {self._run_started} of {self._run_total}: {text.removeprefix('Segmenting ')}"
        self._stage.setText(text)

    def _item_started(self, row: int) -> None:
        self._run_started += 1
        self._set_status(row, Status.RUNNING)
        self._tray.set_status(f"Image {self._run_started} of {self._run_total}", running=True)

    def _show_progress(self, done: int, total: int) -> None:
        # A total of 0 is unknown, e.g. a download without a size, so show a busy bar
        self._progress.setRange(0, total)
        self._progress.setValue(done)
        fraction = done / total if total else 0.0
        if self._live_row is not None and (cell := self._queue.item(self._live_row, 1)) is not None:
            cell.setData(_PROGRESS_ROLE, fraction)
        if self._run_started:
            self._tray.set_status(
                f"Image {self._run_started} of {self._run_total}, {fraction:.0%}",
                running=True,
            )

    def _item_finished(self, row: int, status: Status, error: str) -> None:
        self._set_status(row, status, error)
        if row == self._live_row:
            self._live_timer.stop()
            self._live_row, self._live_overlay, self._live_tiles = None, None, []
            self._live_key.hide()
            if self._preview_row == row:
                self._canvas.set_tiles([])
        # Show the finished result, which has the final filters applied, or clear a cancelled one
        if self._queue.currentRow() == row:
            self._show_selected_preview()

    def _live_started(self, row: int, overlay: LiveOverlay) -> None:
        self._live_row, self._live_overlay, self._live_tiles = row, overlay, []
        self._live_timer.start()
        # Follow the batch as it moves to the next image
        if self._queue.currentRow() != row:
            self._queue.selectRow(row)
        elif self._preview_row == row:
            self._preview_label.setText(f"{self._items[row].input_path.name}, processing")

    def _tiles_started(self, row: int, tiles: list[tuple[int, int, int, int]]) -> None:
        if row == self._live_row:
            self._live_tiles = tiles

    def _refresh_live_preview(self) -> None:
        showing_live = self._live_overlay is not None and self._preview_row == self._live_row
        self._live_key.setVisible(showing_live)
        if self._live_overlay is None or not showing_live:
            return
        self._canvas.set_overlay(self._live_overlay.rgba)
        self._canvas.set_tiles(self._live_tiles)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_button.setEnabled(False)
            self._stage.setText("Cancelling after the current tiles…")
            self._tray.set_status("Cancelling", running=False)

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
        self._show_model_info()

        counts = {status: sum(item.status is status for item in self._items) for status in Status}
        summary = ", ".join(f"{count} {status.value.lower()}" for status, count in counts.items() if count)
        self._stage.setText(f"Finished: {summary}")
        self._tray.set_status(f"Finished: {summary}", running=False)
        self._open_output_button.setVisible(bool(counts[Status.DONE]))
        QApplication.alert(self)

        failed = counts[Status.FAILED]
        if failed:
            self._tray.notify("Some images failed", f"{failed} image(s) failed. {summary}.", failed=True)
            self._log_toggle.setChecked(True)
            QMessageBox.warning(
                self,
                "Some images failed",
                f"{failed} image(s) failed. Select one to see why, or check the log.",
            )
        elif not self.isActiveWindow():
            self._tray.notify("Habitat-Mapper finished", summary.capitalize() + ".")

    def _set_running(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._model.setEnabled(not running)
        self._revision.setEnabled(not running)
        self._update_queue_state()

    # Preview

    def _show_selected_preview(self) -> None:
        row = self._queue.currentRow()
        if row < 0 or row >= len(self._items):
            return
        item = self._items[row]
        output = item.output_path if item.status is Status.DONE else None
        model_cfg = model_registry.config(item.model_name) if item.model_name else self._current_cfg()
        palette = class_palette(model_cfg.classes) if model_cfg.classes else DEFAULT_PALETTE
        try:
            band_order = self._parse_band_order()
        except ValueError:
            band_order = None

        self._preview_request += 1
        request = self._preview_request
        self._preview_label.setText(f"Loading {item.input_path.name}…")

        def load() -> tuple[np.ndarray, np.ndarray | None]:
            image = read_input_preview(item.input_path, band_order=band_order)
            overlay = read_output_preview(output, palette=palette) if output is not None and output.exists() else None
            return image, overlay

        def show(result: object) -> None:
            if request != self._preview_request:
                return
            image, overlay = cast("tuple[np.ndarray, np.ndarray | None]", result)
            self._canvas.set_images(image, overlay)
            self._preview_row = row
            self._show_legend(model_cfg.classes)
            name = item.input_path.name
            if row == self._live_row:
                self._preview_label.setText(f"{name}, processing")
                self._refresh_live_preview()
                return
            self._live_key.hide()
            if item.status is Status.FAILED:
                self._preview_label.setText(f"{name} failed: {item.error}")
            elif overlay is not None:
                self._preview_label.setText(f"{name}, result from {item.model_name}")
            else:
                self._preview_label.setText(f"{name}, {item.status.value.lower()}")

        def failed(error: str) -> None:
            if request == self._preview_request:
                self._canvas.set_images(None)
                self._canvas.set_placeholder("Can't preview this image", error)
                self._preview_label.setText(f"Could not preview {item.input_path.name}: {error}")

        run_in_background(load, show, failed)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Stop processing before closing, so the output isn't left half written."""
        if self._worker is not None and self._thread is not None:
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait()
        self._tray.hide()
        logger.remove(self._log_sink)
        super().closeEvent(event)


def _ensure_std_streams() -> None:
    """Give windowed builds somewhere to write output.

    Windowed Windows builds start without stdout and stderr, but loguru, Rich, and the code that captures native
    library output from file descriptor 2 all expect them.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    devnull = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
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
    # Replace loguru's default handler, which shows debug messages, with one for people running the app from a terminal
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if os.environ.get("DEBUG") else "INFO")
    add_log_file()
    logger.info(f"Starting Habitat-Mapper {__version__}")

    app = QApplication(sys.argv)
    app.setApplicationName("Habitat-Mapper")
    app.setApplicationDisplayName("Habitat-Mapper")
    app.setOrganizationName("Hakai Institute")
    # Frozen builds also embed the icon in the executable or app bundle; this covers running from source
    app.setWindowIcon(QIcon(str(Path(__file__).with_name("icon.png"))))
    app.setStyleSheet(theme.STYLE_SHEET)
    window = MainWindow()
    window.show()
    # Used by CI to check that a frozen build can start
    if os.environ.get("HAB_GUI_SMOKE_TEST"):
        QTimer.singleShot(0, app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
