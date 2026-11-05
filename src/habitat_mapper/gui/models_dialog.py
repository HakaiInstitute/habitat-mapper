"""Dialog for listing models and downloading them ahead of time, e.g. before going somewhere without internet."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from habitat_mapper.gui.worker import run_in_background
from habitat_mapper.registry import model_registry
from habitat_mapper.utils import get_local_model_dir, is_url

if TYPE_CHECKING:
    from habitat_mapper.config import ModelConfig


def is_downloaded(cfg: ModelConfig) -> bool:
    """Check whether a model's files are in the local cache.

    Args:
        cfg: The model's configuration.

    Returns:
        True if the model can run without downloading anything.
    """
    if not any(is_url(dep) for dep in cfg.dependencies):
        return True
    return (get_local_model_dir(cfg.name, cfg.revision) / cfg.model_filename).exists()


class ModelsDialog(QDialog):
    """Lists the latest revision of each model and downloads the selected ones."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the dialog."""
        super().__init__(parent)
        self.setWindowTitle("Models")
        self.resize(760, 360)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Model", "Description", "Status"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().hide()
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        self._download = QPushButton("Download selected")
        self._download.clicked.connect(self._download_selected)
        open_folder = QPushButton("Open model folder")
        open_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(get_local_model_dir()))))
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.hide()

        buttons = QHBoxLayout()
        buttons.addWidget(self._download)
        buttons.addWidget(open_folder)
        buttons.addWidget(self._progress, 1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Models download the first time they're used. Download them here to work offline."))
        layout.addWidget(self._table)
        layout.addLayout(buttons)
        layout.addWidget(close)
        self._refresh()

    def _refresh(self) -> None:
        names = sorted(model_registry.list_model_names())
        self._table.setRowCount(len(names))
        for row, name in enumerate(names):
            cfg = model_registry[name].cfg
            label = f"{name} {cfg.revision}{' (beta)' if cfg.beta else ''}"
            for column, text in enumerate([
                label,
                cfg.description or "",
                "Downloaded" if is_downloaded(cfg) else "Not downloaded",
            ]):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, name)
                self._table.setItem(row, column, item)
        self._table.resizeColumnToContents(0)

    def _download_selected(self) -> None:
        rows = {index.row() for index in self._table.selectionModel().selectedRows()}
        items = [self._table.item(row, 0) for row in sorted(rows)]
        names = [item.data(Qt.ItemDataRole.UserRole) for item in items if item is not None]
        if not names:
            return

        def download() -> None:
            for name in names:
                model_registry[name].cfg.get_local_model_path(quiet=True)

        self._download.setEnabled(False)
        self._progress.show()
        run_in_background(download, lambda _: self._finish_download(), self._finish_download)

    def _finish_download(self, error: str | None = None) -> None:
        self._download.setEnabled(True)
        self._progress.hide()
        self._refresh()
        if error:
            QMessageBox.warning(self, "Download failed", f"Could not download the model.\n\n{error}")
