"""Batch queue state for the desktop app, kept free of Qt so it can be tested on its own."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from habitat_mapper.registry import model_registry

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from habitat_mapper.config import ModelConfig


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


def preview_model(item: BatchItem | None, chosen: ModelConfig) -> ModelConfig:
    """The model whose classes colour and label an image's preview.

    A result keeps the classes of the model that made it, even after another model is chosen.

    Args:
        item: The image being previewed, or None if there isn't one.
        chosen: The model chosen in the window.

    Returns:
        The model that ran on the image, or the chosen one if none has.
    """
    if item is None or item.model_name is None:
        return chosen
    return model_registry.config(item.model_name, item.model_revision or "latest")


def preview_caption(item: BatchItem, *, has_result: bool, processing: bool) -> str:
    """Caption an image's preview, e.g. "survey.tif, result from kelp-rgb 20240722".

    Args:
        item: The image being previewed.
        has_result: Whether its result is drawn over it.
        processing: Whether it is being segmented now.

    Returns:
        The caption.
    """
    name = item.input_path.name
    if processing:
        return f"{name}, processing"
    if item.status is Status.FAILED:
        return f"{name} failed: {item.error}"
    if has_result:
        return f"{name}, result from {item.model_name} {item.model_revision}"
    return f"{name}, {item.status.value.lower()}"


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


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'s' if count != 1 else ''}"


@dataclass(frozen=True)
class AddResult:
    """What happened to images offered to the queue.

    Attributes:
        added: Images queued, in the order given.
        clashes: Images not queued because one with the same output name already is.
    """

    added: list[Path]
    clashes: int

    @property
    def message(self) -> str:
        """Why some images weren't added, for the user, or "" if there's nothing to say."""
        if not self.clashes:
            return ""
        return (
            f"{_plural(self.clashes, 'item')} not added. An image with the same name is already queued, and their "
            "results would overwrite each other."
        )


@dataclass(frozen=True)
class Job:
    """An image to segment: its row in the queue, and where it is read from and written to."""

    row: int
    input_path: Path
    output_path: Path


@dataclass
class BatchQueue:
    """The images queued to segment, and what happened to each.

    Attributes:
        items: The images, in queue order. A row is an index into it, as the window lists them.
    """

    items: list[BatchItem] = field(default_factory=list)

    def add(self, paths: Iterable[Path]) -> AddResult:
        """Queue images, skipping ones already queued.

        A .SAFE folder and its .zip give results with the same name, so only one of them is queued.

        Returns:
            Which images were queued.
        """
        queued = {item.input_path for item in self.items}
        stems = {output_stem(item.input_path) for item in self.items}
        added: list[Path] = []
        clashes = 0
        for path in paths:
            if path in queued:
                continue
            if output_stem(path) in stems:
                clashes += 1
                continue
            queued.add(path)
            stems.add(output_stem(path))
            added.append(path)
            self.items.append(BatchItem(path))
        return AddResult(added, clashes)

    def remove(self, rows: Iterable[int]) -> None:
        """Take images out of the queue."""
        for row in sorted(set(rows), reverse=True):
            del self.items[row]

    def pending(self, output_dir: Path, model_name: str, revision: str) -> list[BatchItem]:
        """The images a run would segment, leaving out ones already done with this model and output folder.

        Returns:
            The images, in queue order.
        """
        return [item for item in self.items if not self._is_done(item, output_dir, model_name, revision)]

    def start_run(self, output_dir: Path, model_name: str, revision: str, *, skip_existing: bool) -> list[Job]:
        """Mark each pending image queued or skipped, and record the model and output it will have.

        Args:
            output_dir: Folder to write the results to.
            model_name: Model to run.
            revision: Revision of the model.
            skip_existing: Skip images whose result already exists, rather than overwrite it.

        Returns:
            The images to segment.
        """
        jobs = []
        for row, item in enumerate(self.items):
            if self._is_done(item, output_dir, model_name, revision):
                continue
            item.output_path = output_path_for(item.input_path, output_dir, model_name, revision)
            item.model_name, item.model_revision = model_name, revision
            item.error = None
            if skip_existing and item.output_path.exists():
                item.status = Status.SKIPPED
            else:
                item.status = Status.QUEUED
                jobs.append(Job(row, item.input_path, item.output_path))
        return jobs

    def count(self, status: Status) -> int:
        """Count the images in a state.

        Returns:
            The count.
        """
        return sum(item.status is status for item in self.items)

    def summary(self) -> str:
        """Say how many images are in each state, e.g. "2 done, 1 failed".

        Returns:
            The summary, leaving out states no image is in.
        """
        return ", ".join(f"{count} {status.value.lower()}" for status in Status if (count := self.count(status)))

    @staticmethod
    def _is_done(item: BatchItem, output_dir: Path, model_name: str, revision: str) -> bool:
        output_path = output_path_for(item.input_path, output_dir, model_name, revision)
        return item.status is Status.DONE and item.output_path == output_path
