"""Icon in the menu bar (macOS) or notification area (Windows) that shows how a run is going."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QWidget


class TrayIcon(QSystemTrayIcon):
    """Shows progress while the window is in the background, and says when a run finishes."""

    def __init__(self, window: QWidget, cancel: Callable[[], None]) -> None:
        """Create the icon and its menu.

        Args:
            window: Main window, brought to the front from the menu or by clicking the icon.
            cancel: Cancels the current run.
        """
        icon = QApplication.windowIcon()
        if icon.isNull():
            icon = QIcon(str(Path(__file__).with_name("icon.png")))
        super().__init__(icon, window)
        self._window = window
        self.setToolTip("Habitat-Mapper")

        # Parentless, since QSystemTrayIcon isn't a widget; kept alive by self
        self._menu = QMenu()
        self._status = QAction("Ready", self._menu)
        self._status.setEnabled(False)
        show = QAction("Show Habitat-Mapper", self._menu)
        show.triggered.connect(self.show_window)
        self._cancel = QAction("Cancel run", self._menu)
        self._cancel.setEnabled(False)
        self._cancel.triggered.connect(cancel)
        quit_action = QAction("Quit Habitat-Mapper", self._menu)
        quit_action.triggered.connect(window.close)
        self._menu.addAction(self._status)
        self._menu.addSeparator()
        self._menu.addAction(show)
        self._menu.addAction(self._cancel)
        self._menu.addSeparator()
        self._menu.addAction(quit_action)
        self.setContextMenu(self._menu)

        # On macOS a click opens the menu; on Windows a left click shows the window and a right click opens the menu
        if sys.platform != "darwin":
            self.activated.connect(self._activated)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.show()

    def _activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_window()

    def show_window(self) -> None:
        """Bring the main window to the front."""
        if self._window.isMinimized():
            self._window.showNormal()
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()

    def set_status(self, text: str, running: bool) -> None:
        """Show what's happening in the menu and the tooltip.

        Args:
            text: Short status, e.g. "Image 2 of 5, 40%".
            running: Whether a run is in progress, so it can be cancelled.
        """
        self._status.setText(text)
        self._cancel.setEnabled(running)
        self.setToolTip(f"Habitat-Mapper: {text}")

    def notify(self, title: str, message: str, failed: bool = False) -> None:
        """Show a system notification, e.g. when a run finishes."""
        if self.isVisible() and QSystemTrayIcon.supportsMessages():
            icon = QSystemTrayIcon.MessageIcon.Warning if failed else QSystemTrayIcon.MessageIcon.Information
            self.showMessage(title, message, icon, 8000)
