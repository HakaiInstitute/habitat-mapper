"""Colours of the desktop app, taken from its icon: sea teal and kelp ochre.

The app keeps the native look of each platform and uses these colours only to accent the parts that matter: the run
button, progress, the preview and the status of each image.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication

TEAL = "#0E4A4C"
# Behind the preview, darker than the teal so the image stands out
TEAL_DEEP = "#0A3335"
OCHRE = "#D6B04A"
# Ochre dark enough to read as text on a light background
OCHRE_TEXT = "#8A6D1E"
DONE = "#2E7D5B"
DONE_ON_DARK = "#5CC08F"
FAILED = "#B3261E"
FAILED_ON_DARK = "#FF8A80"
WHITE = "#FFFFFF"


def is_dark() -> bool:
    """Check whether the system is in dark mode.

    Returns:
        True in dark mode.
    """
    return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark


def running_colour() -> QColor:
    """Text colour for an image that is being processed.

    Returns:
        The colour for the current light or dark mode.
    """
    return QColor(OCHRE if is_dark() else OCHRE_TEXT)


def done_colour() -> QColor:
    """Text colour for an image that has been processed.

    Returns:
        The colour for the current light or dark mode.
    """
    return QColor(DONE_ON_DARK if is_dark() else DONE)


def failed_colour() -> QColor:
    """Text colour for an image that failed.

    Returns:
        The colour for the current light or dark mode.
    """
    return QColor(FAILED_ON_DARK if is_dark() else FAILED)


# Styles for named widgets only, so everything else keeps the platform's look
STYLE_SHEET = f"""
QPushButton#runButton {{
    background: {OCHRE};
    color: {TEAL};
    border: none;
    border-radius: 6px;
    padding: 7px 22px;
    font-weight: 600;
}}
QPushButton#runButton:hover {{ background: #E2BF5E; }}
QPushButton#runButton:pressed {{ background: #C29C38; }}
QPushButton#runButton:disabled {{ background: palette(mid); color: palette(window); }}
QPushButton#runButton:focus {{ outline: none; border: 2px solid {TEAL}; }}

QProgressBar#runProgress {{
    border: none;
    border-radius: 3px;
    background: palette(midlight);
    max-height: 6px;
    min-height: 6px;
}}
QProgressBar#runProgress::chunk {{ background: {OCHRE}; border-radius: 3px; }}

QLabel#stepNumber {{
    background: {TEAL};
    color: {WHITE};
    border-radius: 11px;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    font-weight: 600;
}}
QLabel#stepTitle {{ font-weight: 600; }}
QLabel#muted {{ color: palette(placeholder-text); }}
QLabel#betaTag {{
    color: {OCHRE_TEXT};
    border: 1px solid {OCHRE};
    border-radius: 4px;
    padding: 0 5px;
}}

QFrame#infoBar {{ background: {TEAL}; border-radius: 6px; }}
QFrame#infoBar QLabel {{ color: {WHITE}; }}
QFrame#infoBar QPushButton {{
    background: {OCHRE};
    color: {TEAL};
    border: none;
    border-radius: 4px;
    padding: 4px 12px;
    font-weight: 600;
}}
QFrame#infoBar QToolButton {{ color: {WHITE}; border: none; padding: 2px 6px; }}

QToolButton#quietButton {{ border: none; color: palette(placeholder-text); padding: 4px 6px; }}
QToolButton#quietButton:hover, QToolButton#quietButton:checked {{ color: palette(text); }}

QToolButton#canvasButton {{
    background: rgba(10, 51, 53, 210);
    color: {WHITE};
    border: 1px solid rgba(255, 255, 255, 40);
    border-radius: 5px;
    min-width: 26px;
    min-height: 26px;
    font-weight: 600;
}}
QToolButton#canvasButton:hover {{ background: {TEAL}; }}
QToolButton#canvasButton:focus {{ border: 1px solid {OCHRE}; }}
QLabel#canvasZoom {{
    background: rgba(10, 51, 53, 210);
    color: {WHITE};
    border-radius: 5px;
    padding: 4px 8px;
}}
"""
