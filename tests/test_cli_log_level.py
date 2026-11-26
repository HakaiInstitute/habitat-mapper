"""Tests for logging: the CLI hides debug output unless DEBUG is set, and importing the package changes nothing."""

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys

_CLI_SCRIPT = """
import sys
from loguru import logger
import habitat_mapper.main as main

@main.app.command
def log():
    logger.debug("debug-marker")
    logger.info("info-marker")

sys.argv = ["hab", "log"]
main.main()
"""

_IMPORT_SCRIPT = """
import sys
from loguru import logger
logger.remove()
sink = logger.add(sys.stdout, level="DEBUG", format="{message}")
import habitat_mapper
logger.debug("library-debug-marker")
print("cli-imported" if "habitat_mapper.main" in sys.modules else "cli-not-imported")
"""


def _run(script: str, debug: str | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "DEBUG"}
    if debug is not None:
        env["DEBUG"] = debug
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)  # noqa: S603
    return result.stdout + result.stderr


def test_debug_hidden_by_default() -> None:
    output = _run(_CLI_SCRIPT)
    assert "info-marker" in output
    assert "debug-marker" not in output


def test_debug_shown_when_debug_env_set() -> None:
    output = _run(_CLI_SCRIPT, debug="1")
    assert "info-marker" in output
    assert "debug-marker" in output


def test_importing_the_package_leaves_logging_alone() -> None:
    output = _run(_IMPORT_SCRIPT)
    assert "library-debug-marker" in output
    assert "cli-not-imported" in output
