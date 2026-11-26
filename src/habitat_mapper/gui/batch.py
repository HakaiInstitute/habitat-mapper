"""Batch queue state for the desktop app, kept free of Qt so it can be tested on its own."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class Status(Enum):
    """Processing state of one queued image."""

    QUEUED = "Queued"
    RUNNING = "Running"
    DONE = "Done"
    FAILED = "Failed"
    SKIPPED = "Skipped"
    CANCELLED = "Cancelled"


@dataclass
class BatchItem:
    """An image in the batch queue.

    Attributes:
        input_path: GeoTIFF, or Sentinel-2 .SAFE directory or .zip, to segment.
        status: Processing state.
        output_path: Where the segmentation was written, once it is done.
        error: Why processing failed, if it did.
        model_name: Model the output was made with, to colour and label it.
        model_revision: Revision of that model.
    """

    input_path: Path
    status: Status = Status.QUEUED
    output_path: Path | None = None
    error: str | None = None
    model_name: str | None = None
    model_revision: str | None = None


def output_stem(input_path: Path) -> str:
    """Name an input's results start with, which is the same for a .SAFE folder and its .zip.

    Returns:
        e.g. `S2B_X` for `S2B_X.SAFE`, `S2B_X.SAFE.zip` or `S2B_X.zip`.
    """
    stem = input_path.stem
    # S2B_X.SAFE.zip is named like the S2B_X.SAFE folder inside it
    if input_path.suffix.lower() == ".zip" and stem.upper().endswith(".SAFE"):
        stem = stem[: -len(".SAFE")]
    return stem


def output_path_for(input_path: Path, output_dir: Path, model_name: str, revision: str) -> Path:
    """Name the output for an input image, e.g. `survey_kelp_rgb_20240722.tif`.

    Args:
        input_path: GeoTIFF, or Sentinel-2 .SAFE directory or .zip.
        output_dir: Folder the batch writes to.
        model_name: Name of the model that produces the output.
        revision: Revision of the model.

    Returns:
        The output path.
    """
    return output_dir / f"{output_stem(input_path)}_{model_name.replace('-', '_')}_{revision}.tif"


def existing_outputs(items: list[BatchItem], output_dir: Path, model_name: str, revision: str) -> list[Path]:
    """Find the queued items whose output would overwrite an existing file.

    Args:
        items: The batch queue.
        output_dir: Folder the batch writes to.
        model_name: Name of the model that produces the outputs.
        revision: Revision of the model.

    Returns:
        Output paths that already exist.
    """
    paths = (output_path_for(item.input_path, output_dir, model_name, revision) for item in items)
    return [path for path in paths if path.exists()]
