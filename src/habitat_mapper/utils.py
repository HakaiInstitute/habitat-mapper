"""Utilities and helper functions for the Habitat-Mapper package."""

from __future__ import annotations

import itertools
import os
import shutil
import site
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
import platformdirs
import requests
from loguru import logger
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TaskID, TextColumn, TransferSpeedColumn
from rich.prompt import Confirm

if TYPE_CHECKING:
    from collections.abc import Iterable

    from habitat_mapper.config import ModelConfig


def download_file_with_progress(url: str, out_path: Path, timeout: tuple[int, int] = (15, 120)) -> None:
    """Download the file at url to out_path."""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        # Get total file size
        total_size = int(response.headers.get("content-length", 0))

        # Download with progress to temp location
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                open(tmp_out := Path(tmp_dir) / out_path.name, "wb") as f,
                Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    console=None,  # Use default console
                ) as progress,
            ):
                task = progress.add_task(f"Downloading {out_path.name}", total=total_size)
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

            # Move the completed file to the proper location
            shutil.move(tmp_out, out_path)
        # /tempdir
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        logger.error(f"Download timed out: {e}")

        if Confirm.ask("Do you want to retry the download?"):
            logger.trace("Retrying download...")
            # Retry with longer timeout
            download_file_with_progress(url, out_path, timeout=(timeout[0] * 2, timeout[1] * 2))
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")


def _get_dependency_local_path(dependency_url: str, model_config: ModelConfig) -> Path:
    """Get the local path for a dependency file.

    Args:
        dependency_url: URL or local path of the dependency
        model_config: The model configuration (used to determine cache directory)

    Returns:
        Path to the local dependency file
    """
    # Check if it's a local file path
    if dependency_url.startswith(("/", "./", "../", "~")) or (len(dependency_url) > 1 and dependency_url[1] == ":"):
        # It's a local path (Unix absolute/relative or Windows drive path)
        return Path(dependency_url).expanduser().resolve()

    # It's a URL - use model-specific cache directory
    if is_url(dependency_url):
        model_dir = get_local_model_dir(model_config.name, model_config.revision)
        # Extract filename from URL
        filename = dependency_url.split("/")[-1].split("?")[0]  # Handle query parameters
        return model_dir / filename

    # If it doesn't match URL patterns, assume it's a local path
    return Path(dependency_url).expanduser().resolve()


def _download_single_dependency(
    dependency_url: str,
    local_path: Path,
    progress: Progress,
    task_id: TaskID,
    timeout: tuple[int, int] = (15, 120),
) -> tuple[str, Path, Exception | None]:
    """Download a single dependency file with progress tracking.

    Args:
        dependency_url: URL to download from
        local_path: Local path to save to
        progress: Rich Progress instance for tracking
        task_id: Task ID for this download in the progress bar
        timeout: Request timeout tuple (connect, read)

    Returns:
        Tuple of (url, local_path, error). error is None if successful.
    """
    try:
        if local_path.exists():
            # File already cached
            progress.update(task_id, completed=100, total=100)
            return (dependency_url, local_path, None)

        # Download the file
        response = requests.get(dependency_url, stream=True, timeout=timeout)
        response.raise_for_status()

        # Get total file size
        total_size = int(response.headers.get("content-length", 0))
        progress.update(task_id, total=total_size)

        # Download with progress to temp location
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_out = Path(tmp_dir) / local_path.name
            with open(tmp_out, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress.update(task_id, completed=downloaded)

            # Move the completed file to the proper location
            shutil.move(tmp_out, local_path)

        return (dependency_url, local_path, None)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        return (dependency_url, local_path, e)
    except Exception as e:
        return (dependency_url, local_path, e)


def download_dependencies(model_config: ModelConfig, quiet: bool = False) -> list[Path]:
    """Download all dependencies for a model in parallel.

    Args:
        model_config: The model configuration containing dependency URLs
        quiet: If True, suppress download progress bars.

    Returns:
        List of paths to downloaded dependency files

    Raises:
        RuntimeError: If any download fails
        FileNotFoundError: If any local dependency file doesn't exist
        ValueError: If dependency path is not a file
    """
    if not model_config.dependencies:
        return []

    local_paths: list[Path] = []
    urls_to_download: list[tuple[str, Path]] = []

    # First pass: collect paths and determine what needs downloading
    for dep_url in model_config.dependencies:
        local_path = _get_dependency_local_path(dep_url, model_config)
        local_paths.append(local_path)

        # Only download if it's a URL
        if is_url(dep_url):
            urls_to_download.append((dep_url, local_path))
        else:
            # It's a local path - validate it exists
            if not local_path.exists():
                raise FileNotFoundError(f"Local dependency file not found: {local_path}")
            if not local_path.is_file():
                raise ValueError(f"Dependency path is not a file: {local_path}")

    # Download URLs in parallel with multi-task progress bar
    if urls_to_download:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            disable=quiet,
        ) as progress:
            # Create a task for each download
            tasks = {}
            for url, path in urls_to_download:
                filename = path.name
                status = "Cached" if path.exists() else "Downloading"
                task_id = progress.add_task(f"[cyan]{status}:[/cyan] {filename}", total=100)
                tasks[(url, path)] = task_id

            # Submit all downloads to the thread pool
            with ThreadPoolExecutor(max_workers=min(len(urls_to_download), 4)) as executor:
                futures = {
                    executor.submit(_download_single_dependency, url, path, progress, tasks[(url, path)]): (url, path)
                    for url, path in urls_to_download
                }

                errors = []
                for future in as_completed(futures):
                    url, local_path, error = future.result()
                    if error:
                        errors.append(f"Failed to download {url}: {error}")

                if errors:
                    raise RuntimeError("\n".join(errors))

    # Log success for already cached files
    if not quiet:
        for url, path in urls_to_download:
            if path.exists():
                logger.success(f"✓ Loaded: {path.name}")

    return local_paths


def get_ort_providers() -> list[str]:
    """Get the ORT provider list for the current system and installation.

    Returns:
       list[str]: List of available ONNX Runtime providers in order of preference.
    """
    available_providers = ort.get_available_providers()
    providers = []

    if "NvTensorRtRtxExecutionProvider" in available_providers:
        providers.append("NvTensorRtRtxExecutionProvider")

    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


def get_local_model_dir(model_name: str | None = None, revision: str | None = None) -> Path:
    """Get the path to the directory containing cached models.

    Args:
        model_name: Optional model name for creating a model-specific subdirectory
        revision: Optional revision for creating a revision-specific subdirectory

    Returns:
        Path: The path to the local model directory, creating it if it doesn't exist.
    """
    cache_dir = platformdirs.user_cache_dir(appname="habitat_mapper", appauthor="hakai")
    model_dir = Path(cache_dir) / "models"

    # Create model-specific subdirectory if model_name is provided
    if model_name:
        model_dir = model_dir / model_name
        if revision:
            model_dir = model_dir / revision

    model_dir.mkdir(exist_ok=True, parents=True)
    return model_dir


def is_url(uri: str) -> bool:
    """Check if a URI is a URL or a local file path.

    Args:
        uri: The URI to check

    Returns:
        True if it's a URL, False if it's a local path

    """
    return uri.startswith(("http://", "https://", "ftp://", "ftps://"))


def batched(iterable: Iterable[Any], n: int) -> Iterable[tuple[Any, ...]]:
    """Batch data from the iterable into tuples of length n. The last batch may be shorter.

    This function emulates itertools.batched() which was introduced in Python 3.12.

    Args:
        iterable: Any iterable (list, string, generator, etc.)
        n: Batch size (positive integer)

    Yields:
        Tuples containing up to n elements from the iterable

    Raises:
        ValueError: If n is less than 1

    Examples:
        >>> list(batched('ABCDEFG', 3))
        [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

        >>> list(batched([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
        [(1, 2, 3, 4), (5, 6, 7, 8), (9,)]

        >>> list(batched(range(10), 2))
        [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    """
    if n < 1:
        raise ValueError("n must be at least one")

    iterator = iter(iterable)
    while True:
        batch = tuple(itertools.islice(iterator, n))
        if not batch:
            break
        yield batch


def setup_cuda_paths() -> None:
    """Add CUDA DLL directories to PATH on Windows."""
    if sys.platform != "win32":
        return

    # Find all NVIDIA bin directories in site-packages
    dll_paths = []
    for site_dir in site.getsitepackages():
        nvidia_path = Path(site_dir) / "nvidia"
        if nvidia_path.exists():
            # Find all bin directories under nvidia/*
            dll_paths.extend(str(p) for p in nvidia_path.glob("*/bin") if p.is_dir())

    if not dll_paths:
        return

    # Add to PATH environment variable
    os.environ["PATH"] = ";".join(dll_paths) + ";" + os.environ.get("PATH", "")

    # Add as DLL directories for Python 3.8+
    if hasattr(os, "add_dll_directory"):
        for path in dll_paths:
            try:
                os.add_dll_directory(path)
            except OSError:
                pass


def _is_odd_or_zero(value: int) -> int:
    if value == 0:
        return value
    if value % 2 == 0:
        raise ValueError(f"{value} is not an odd number")
    return value


def _all_positive(value: list[int]) -> list[int]:
    if any(v <= 0 for v in value):
        raise ValueError("All values must be positive integers")
    return value


def softmax(x: np.ndarray, axis: int = 0, keepdims: bool = False) -> np.ndarray:
    """Apply softmax function to ndarray x.

    Args:
        x: A numpy array
        axis: Axis along which to apply the softmax function
        keepdims: Whether to keep dimensions or not

    Returns:
        A numpy array representing the softmax output
    """
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=keepdims)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the sigmoid activation function to array x.

    Args:
        x: A numpy array

    Returns:
        A numpy array of the sigmoid output
    """
    ex = np.exp(x)
    return ex / (1 + ex)
