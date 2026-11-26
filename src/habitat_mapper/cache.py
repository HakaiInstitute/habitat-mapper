"""The model cache: where each model's files are kept on this computer, and downloading them."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import platformdirs
import requests
from loguru import logger

from habitat_mapper.progress import NullProgressReporter, ProgressTask
from habitat_mapper.utils import is_url

if TYPE_CHECKING:
    from collections.abc import Callable

    from habitat_mapper.config import ModelConfig
    from habitat_mapper.progress import ProgressReporter

_TIMEOUT = (15, 120)  # Seconds to connect, and to wait for each part of the file
_CHUNK_SIZE = 1 << 16


class ModelCache:
    """The folder that downloaded model files are kept in, one subfolder per model revision.

    Models list their files as `dependencies` in their `ModelConfig`. URLs are downloaded into the cache the first time
    they're needed. Local paths are used where they are.
    """

    def __init__(self, root: Path | None = None) -> None:
        """Use a cache folder.

        Args:
            root: The folder. Defaults to the user's cache folder, e.g. ~/.cache/habitat_mapper/models on Linux.
        """
        self.root = root or Path(platformdirs.user_cache_dir(appname="habitat_mapper", appauthor="hakai")) / "models"

    def dir_for(self, cfg: ModelConfig) -> Path:
        """The folder a model revision's downloads go in. It is created when something is downloaded.

        Returns:
            The folder.
        """
        return self.root / cfg.name / cfg.revision

    def paths(self, cfg: ModelConfig) -> dict[str, Path]:
        """Where each of a model's files is, or will be once it is downloaded.

        Returns:
            Local paths, by file name.
        """
        paths = {}
        for dependency in cfg.dependencies:
            if is_url(dependency):
                path = self.dir_for(cfg) / dependency.split("?")[0].rsplit("/", 1)[-1]
            else:
                path = Path(dependency).expanduser().resolve()
            paths[path.name] = path
        return paths

    def is_cached(self, cfg: ModelConfig) -> bool:
        """Check whether a model can run without downloading anything.

        Returns:
            True if every file the model downloads is in the cache.
        """
        return all(
            path.exists() for dependency, path in zip(cfg.dependencies, self.paths(cfg).values()) if is_url(dependency)
        )

    def fetch(self, cfg: ModelConfig, progress: Callable[[], ProgressReporter] = NullProgressReporter) -> Path:
        """Download any of a model's files that aren't in the cache yet.

        Each file is downloaded beside its final path and moved into place once it is complete, so an interrupted
        download never leaves a partial file in the cache.

        Args:
            cfg: The model.
            progress: Makes the reporter that shows download progress, with one task per file.

        Returns:
            The local path of the model's ONNX file.

        Raises:
            FileNotFoundError: If a local file the model lists doesn't exist.
            ValueError: If the model's ONNX file isn't one of its dependencies.
            RuntimeError: If a download fails.
        """
        paths = self.paths(cfg)
        if cfg.model_filename not in paths:
            raise ValueError(f"Model file '{cfg.model_filename}' not found in dependencies: {list(paths)}")

        missing = []
        for dependency, path in zip(cfg.dependencies, paths.values()):
            if is_url(dependency):
                if not path.exists():
                    missing.append((dependency, path))
            elif not path.is_file():
                raise FileNotFoundError(f"Local dependency file not found: {path}")

        if missing:
            with progress() as reporter:
                for url, path in missing:
                    try:
                        _download(
                            url, path, lambda total, name=path.name: reporter.add_task(f"Downloading {name}", total)
                        )
                    except (requests.RequestException, OSError) as e:
                        raise RuntimeError(f"Could not download {path.name} for {cfg.name}: {e}") from e
                    logger.success(f"Downloaded {path.name}")

        model_path = paths[cfg.model_filename]
        if model_path.suffix.lower() != ".onnx":
            logger.warning(f"Model file does not have .onnx extension: {model_path}")
        return model_path

    def files(self) -> list[Path]:
        """List every cached file, including models left by Kelp-O-Matic, largest first.

        Returns:
            The files.
        """
        legacy = Path(platformdirs.user_cache_dir(appname="kelp_o_matic", appauthor="hakai")) / "models"
        files = [path for path in itertools.chain(self.root.rglob("*"), legacy.glob("*")) if path.is_file()]
        return sorted(files, key=lambda path: path.stat().st_size, reverse=True)

    def clear(self) -> list[tuple[Path, OSError]]:
        """Delete every cached file. Models download again the next time they're used.

        Returns:
            The files that couldn't be deleted, with why.
        """
        failed = []
        for path in self.files():
            try:
                path.unlink()
            except OSError as e:
                failed.append((path, e))
        return failed


def _download(url: str, path: Path, add_task: Callable[[int | None], ProgressTask]) -> None:
    """Download a file to path, through a temporary file beside it.

    Raises:
        OSError: If the server sent less than it said it would.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    try:
        with requests.get(url, stream=True, timeout=_TIMEOUT) as response:
            response.raise_for_status()
            # The length is of the encoded body, so it can only be checked when the file isn't compressed in transit
            expected = int(response.headers.get("content-length", 0)) or None
            if response.headers.get("content-encoding"):
                expected = None
            task = add_task(expected)
            received = 0
            with partial.open("wb") as file:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    file.write(chunk)
                    received += len(chunk)
                    task.update(advance=len(chunk))
        if expected is not None and received != expected:
            raise OSError(f"the download stopped early, after {received} of {expected} bytes. Please try again")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


model_cache = ModelCache()
