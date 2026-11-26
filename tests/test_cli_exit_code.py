"""Tests for the exit code when a command fails, which scripts rely on to detect failures."""

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_SCRIPT = """
import habitat_mapper.main as main

@main.logger_catch
def fail():
    raise RuntimeError("failure-marker")

main._exit_on_error = {from_cli}
fail()
print("returned-marker")
"""


def _run(*, from_cli: bool) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "DEBUG"}
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SCRIPT.format(from_cli=from_cli)], env=env, capture_output=True, text=True, check=False
    )


def test_failed_command_exits_with_an_error_code() -> None:
    result = _run(from_cli=True)
    assert result.returncode == 1
    assert "failure-marker" in result.stdout + result.stderr
    assert "returned-marker" not in result.stdout


def test_failed_command_called_from_python_returns() -> None:
    result = _run(from_cli=False)
    assert result.returncode == 0
    assert "returned-marker" in result.stdout


def test_unknown_revision_exits_with_an_error_code(tmp_path: Path) -> None:
    image = tmp_path / "input.tif"
    image.write_bytes(b"")
    env = {k: v for k, v in os.environ.items() if k != "DEBUG"} | {"COLUMNS": "500"}
    args = ["segment", "-m", "kelp-rgb", "-i", str(image), "-o", str(tmp_path / "out.tif"), "--rev", "19990101"]
    script = f"import sys; sys.argv = ['hab', *{args!r}]; from habitat_mapper.main import main; main()"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Revision 19990101 was not found" in result.stdout + result.stderr
    assert not (tmp_path / "out.tif").exists()
