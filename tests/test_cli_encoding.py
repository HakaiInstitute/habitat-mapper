"""Tests for CLI output on streams that can't encode every character, like piped output on Windows."""

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys

_SCRIPT = """
import sys
import habitat_mapper.main as main

@main.app.command
def tick():
    main.console.print("✓ tick-marker")

sys.argv = ["hab", "tick"]
main.main()
"""


def test_unencodable_output_is_replaced() -> None:
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SCRIPT], env=env, capture_output=True, encoding="cp1252", check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "? tick-marker" in result.stdout
