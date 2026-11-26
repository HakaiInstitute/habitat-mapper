"""Tests for the Models dialog, which downloads models ahead of time. Needs Qt, like test_gui_window.py."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if not os.environ.get("HAB_REQUIRE_QT_TESTS"):
    pytest.importorskip("PySide6", reason="needs the gui extra")
    pytest.importorskip("pytestqt", reason="needs the gui-tests group")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from habitat_mapper import cache as cache_module  # noqa: E402
from habitat_mapper.cache import ModelCache  # noqa: E402
from habitat_mapper.config import ModelConfig  # noqa: E402
from habitat_mapper.gui import models_dialog  # noqa: E402
from habitat_mapper.registry import ModelRegistry  # noqa: E402
from tests.helpers import FakeResponse  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot


def _config(name: str, revision: str, **changes: object) -> ModelConfig:
    return ModelConfig(
        name=name,
        revision=revision,
        dependencies=[f"https://example.com/{name}/{revision}/model.onnx"],
        model_filename="model.onnx",
        description=f"About {name}",
        **changes,  # type: ignore[arg-type]
    )


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModelCache:
    cache = ModelCache(tmp_path / "cache")
    monkeypatch.setattr(models_dialog, "model_cache", cache)
    registry = ModelRegistry()
    for cfg in [_config("kelp", "20240101"), _config("kelp", "20250101"), _config("mussel", "20250101", beta=True)]:
        registry.register_model(cfg)
    monkeypatch.setattr(models_dialog, "model_registry", registry)
    # kelp's latest revision is already downloaded
    path = cache.paths(registry.config("kelp"))["model.onnx"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"model")
    return cache


@pytest.fixture
def dialog(qtbot: QtBot, cache: ModelCache) -> models_dialog.ModelsDialog:
    dialog = models_dialog.ModelsDialog()
    qtbot.addWidget(dialog)
    return dialog


def _table(dialog: models_dialog.ModelsDialog) -> list[list[str]]:
    table = dialog._table
    return [[table.item(row, column).text() for column in range(3)] for row in range(table.rowCount())]


def test_lists_each_models_latest_revision_and_whether_it_is_downloaded(dialog: models_dialog.ModelsDialog) -> None:
    assert _table(dialog) == [
        ["kelp 20250101", "About kelp", "Downloaded"],
        ["mussel 20250101 (beta)", "About mussel", "Not downloaded"],
    ]


def test_downloads_the_selected_models(
    qtbot: QtBot, dialog: models_dialog.ModelsDialog, cache: ModelCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def get(url: str, **kwargs: object) -> FakeResponse:
        urls.append(url)
        return FakeResponse(b"onnx bytes")

    monkeypatch.setattr(cache_module.requests, "get", get)
    dialog._table.selectRow(1)

    dialog._download_selected()
    qtbot.waitUntil(lambda: dialog._download.isEnabled())

    assert urls == ["https://example.com/mussel/20250101/model.onnx"]
    assert _table(dialog)[1][2] == "Downloaded"
    assert (cache.root / "mussel" / "20250101" / "model.onnx").read_bytes() == b"onnx bytes"


def test_a_failed_download_is_reported(
    qtbot: QtBot, dialog: models_dialog.ModelsDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: warnings.append(text))
    monkeypatch.setattr(cache_module.requests, "get", lambda url, **kwargs: FakeResponse(b"", status=503))
    dialog._table.selectRow(1)

    dialog._download_selected()
    qtbot.waitUntil(lambda: dialog._download.isEnabled())

    [warning] = warnings
    assert "503" in warning
    assert _table(dialog)[1][2] == "Not downloaded"


def test_downloading_nothing_selected_does_nothing(
    dialog: models_dialog.ModelsDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache_module.requests, "get", lambda url, **kwargs: pytest.fail("downloaded"))
    dialog._table.clearSelection()

    dialog._download_selected()

    assert dialog._download.isEnabled()
