"""Tests for the CLI log level, which hides debug output unless DEBUG is set."""

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys

_SCRIPT = """
import habitat_mapper.main
from loguru import logger
logger.debug("debug-marker")
logger.info("info-marker")
"""


def _run_cli_logging(debug: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "DEBUG"}
    if debug is not None:
        env["DEBUG"] = debug
    result = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, check=True)  # noqa: S603
    return result.stdout + result.stderr


def test_debug_hidden_by_default() -> None:
    output = _run_cli_logging(None)
    assert "info-marker" in output
    assert "debug-marker" not in output


def test_debug_shown_when_debug_env_set() -> None:
    output = _run_cli_logging("1")
    assert "info-marker" in output
    assert "debug-marker" in output
