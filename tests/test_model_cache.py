"""Tests for the model cache: where model files go, and downloading them safely."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from habitat_mapper import cache as cache_module
from habitat_mapper.cache import ModelCache
from habitat_mapper.config import ModelConfig
from tests.helpers import FakeResponse

if TYPE_CHECKING:
    from pathlib import Path

URL = "https://example.com/models/model.onnx?download=1"
BODY = b"onnx bytes" * 1000


class _Reporter:
    def __init__(self) -> None:
        self.tasks: list[tuple[str, int | None]] = []
        self.advanced = 0

    def __enter__(self) -> _Reporter:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def add_task(self, description: str, total: int | None = None) -> _Reporter:
        self.tasks.append((description, total))
        return self

    def update(self, advance: int = 1) -> None:
        self.advanced += advance


def _config(*dependencies: str) -> ModelConfig:
    return ModelConfig(
        name="stub", revision="20250101", dependencies=list(dependencies or [URL]), model_filename="model.onnx"
    )


@pytest.fixture
def responses(monkeypatch: pytest.MonkeyPatch) -> list[FakeResponse]:
    # Responses for each request, in order
    queue: list[FakeResponse] = []
    monkeypatch.setattr(cache_module.requests, "get", lambda url, **kwargs: queue.pop(0))
    return queue


def test_url_files_go_in_a_folder_per_revision(tmp_path: Path) -> None:
    cache = ModelCache(tmp_path)
    assert cache.paths(_config()) == {"model.onnx": tmp_path / "stub" / "20250101" / "model.onnx"}


def test_checking_the_cache_creates_nothing(tmp_path: Path) -> None:
    cache = ModelCache(tmp_path / "cache")
    assert not cache.is_cached(_config())
    assert not (tmp_path / "cache").exists()


def test_fetch_downloads_and_reports_progress(tmp_path: Path, responses: list[FakeResponse]) -> None:
    cache = ModelCache(tmp_path)
    responses.append(FakeResponse(BODY))
    reporter = _Reporter()

    path = cache.fetch(_config(), lambda: reporter)

    assert path.read_bytes() == BODY
    assert cache.is_cached(_config())
    assert reporter.tasks == [("Downloading model.onnx", len(BODY))]
    assert reporter.advanced == len(BODY)


def test_fetch_does_not_download_cached_files(tmp_path: Path, responses: list[FakeResponse]) -> None:
    cache = ModelCache(tmp_path)
    responses.append(FakeResponse(BODY))
    cache.fetch(_config())
    cache.fetch(_config())  # Would fail with an empty response queue if it downloaded again


def test_truncated_download_leaves_nothing_in_the_cache(tmp_path: Path, responses: list[FakeResponse]) -> None:
    cache = ModelCache(tmp_path)
    responses.append(FakeResponse(BODY[:100], length=len(BODY)))

    with pytest.raises(RuntimeError, match="stopped early, after 100 of"):
        cache.fetch(_config())

    folder = cache.dir_for(_config())
    assert list(folder.iterdir()) == []


def test_http_errors_raise(tmp_path: Path, responses: list[FakeResponse]) -> None:
    responses.append(FakeResponse(b"", status=404))
    with pytest.raises(RuntimeError, match="Could not download model.onnx for stub: 404 error"):
        ModelCache(tmp_path).fetch(_config())


def test_local_files_are_used_where_they_are(tmp_path: Path) -> None:
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(BODY)
    assert ModelCache(tmp_path / "cache").fetch(_config(str(model_file))) == model_file


def test_missing_local_files_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ModelCache(tmp_path).fetch(_config(str(tmp_path / "model.onnx")))


def test_clear_deletes_every_cached_file(tmp_path: Path, responses: list[FakeResponse]) -> None:
    cache = ModelCache(tmp_path / "cache")
    responses.append(FakeResponse(BODY))
    cache.fetch(_config())
    assert tmp_path / "cache" / "stub" / "20250101" / "model.onnx" in cache.files()

    assert cache.clear() == []
    assert not cache.is_cached(_config())
