"""Check that the generated CLI reference page matches the current CLI."""

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gen_cli_docs.py"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_cli_docs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_reference_is_up_to_date() -> None:
    gen = _load_generator()
    committed = gen.OUTPUT_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == gen.render(), (
        "docs/reference/cli.md is out of date. Run `uv run python scripts/gen_cli_docs.py` and commit the result."
    )
