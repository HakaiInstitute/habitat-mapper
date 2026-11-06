"""Log file and diagnostics report for support requests from desktop app users."""

from __future__ import annotations

import platform
import sys
from collections import deque
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import platformdirs
from loguru import logger

from habitat_mapper.accelerators import find_accelerators

LOG_DIR = Path(platformdirs.user_log_dir(appname="habitat_mapper", appauthor="hakai"))
LOG_FILE = LOG_DIR / "habitat-mapper.log"
_PACKAGES = ("habitat-mapper", "onnxruntime", "onnxruntime-gpu", "onnxruntime-ep-webgpu")


def add_log_file() -> None:
    """Log everything, including debug messages, to a rotating file in the user log folder."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOG_FILE, level="DEBUG", rotation="5 MB", retention=3, encoding="utf-8")


def _tail(path: Path, lines: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=lines))
    except OSError:
        return "(no log file)\n"


def diagnostics_report(active_device: str | None, log_lines: int = 200) -> str:
    """Describe the app, system and recent log messages, for pasting into a support email.

    Args:
        active_device: Device and hardware the last model ran on, if one has run.
        log_lines: Number of lines from the end of the log file to include.

    Returns:
        The report as plain text.
    """
    packages = []
    for package in _PACKAGES:
        try:
            packages.append(f"{package} {version(package)}")
        except PackageNotFoundError:
            pass
    try:
        accelerators = ", ".join(f"{a.device} ({a.hardware})" for a in find_accelerators())
    except Exception as e:  # noqa: BLE001
        accelerators = f"failed to list: {e}"

    return (
        f"Packages: {', '.join(packages)}\n"
        f"System: {platform.platform()}, Python {platform.python_version()}"
        f"{', frozen' if getattr(sys, 'frozen', False) else ''}\n"
        f"Devices: {accelerators}\n"
        f"Last run on: {active_device or 'no model has run yet'}\n"
        f"\nLast {log_lines} lines of {LOG_FILE}:\n"
        f"{_tail(LOG_FILE, log_lines)}"
    )
