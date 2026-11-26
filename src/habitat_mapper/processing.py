"""High-level image processing with tiled segmentation and post-processing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np
import rasterio
from loguru import logger
from rasterio.windows import Window

from habitat_mapper.config import ProcessingConfig, SegmentOptions
from habitat_mapper.hann import BartlettHannKernel, NumpyMemoryRegister
from habitat_mapper.progress import NullProgressReporter, ProgressReporter
from habitat_mapper.utils import batched

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence

    from habitat_mapper.config import ModelConfig
    from habitat_mapper.reader import ImageReader


def _tile_count(size: int, tile_size: int, stride: int) -> int:
    """Count the tiles needed along one side of an image for the last tile to reach its end.

    Returns:
        The number of tiles.
    """
    return 1 if size <= tile_size else math.ceil((size - tile_size) / stride) + 1


class ProcessingCancelled(Exception):  # noqa: N818
    """Raised when processing is cancelled before it finishes."""


class ProcessingObserver(Protocol):
    """Follows an image as it is processed, e.g. to preview the result while it is made."""

    def image_started(self, height: int, width: int) -> None:
        """Called once the image is opened, with its size in pixels."""
        ...

    def batch_started(self, windows: Sequence[Window]) -> None:
        """Called before each batch with its tiles, which can extend past the bottom and right of the image.

        If the blur or morphological filters are on, also called before each tile they filter, once the model is done.
        """
        ...

    def written(self, window: Window, data: np.ndarray) -> None:
        """Called with each part of the result as it is written, then again as the filters rewrite it, if on."""
        ...


class Segmenter(Protocol):
    """What ImageProcessor needs from a model. ONNXModel is one."""

    @property
    def cfg(self) -> ModelConfig:
        """The model's configuration."""
        ...

    @property
    def input_size(self) -> int | None:
        """The square tile size the model takes, or None if it takes any size."""
        ...

    def predict(self, batch: np.ndarray, max_pixel_value: float) -> np.ndarray:
        """Score each class for a batch of tiles with shape [batch_size, channels, height, width]."""
        ...

    def postprocess(self, batch: np.ndarray) -> np.ndarray:
        """Turn the scores for one tile into class labels."""
        ...


def band_order_for(cfg: ModelConfig, band_order: list[int] | None) -> list[int]:
    """Check a band order against the number of bands a model takes.

    Args:
        cfg: The model's configuration.
        band_order: Band numbers, starting at 1, or None for the image's first bands in order.

    Returns:
        The band order to use.

    Raises:
        ValueError: If band_order doesn't have one band for each of the model's inputs.
    """
    channels = cfg.input_channels
    band_order = band_order or list(range(1, channels + 1))
    if len(band_order) != channels:
        raise ValueError(f"{cfg.name} takes {channels} bands, but band_order has {len(band_order)}.")
    return band_order


@dataclass(frozen=True)
class ImageProcessor:
    """Segments a large image tile by tile, blending overlapping tiles, then smooths the result."""

    model: Segmenter
    config: ProcessingConfig
    progress: Callable[[], ProgressReporter] = field(default=NullProgressReporter)

    @classmethod
    def from_model(
        cls,
        model: Segmenter,
        options: SegmentOptions | None = None,
        *,
        progress: Callable[[], ProgressReporter] = NullProgressReporter,
    ) -> ImageProcessor:
        """Resolve the options against the model, e.g. using the model's tile size.

        The band order is checked with `band_order_for`, which raises a ValueError if it doesn't suit the model.

        Args:
            model: The model to run.
            options: Processing options. Defaults to SegmentOptions().
            progress: Makes the progress reporter for each stage.

        Returns:
            Configured ImageProcessor instance
        """
        options = options or SegmentOptions()
        band_order = band_order_for(model.cfg, options.band_order)

        crop_size = options.crop_size
        if crop_size is None:
            crop_size = model.input_size or 1024
        elif model.input_size is not None and crop_size != model.input_size:
            logger.warning(
                f"Specified tile size {crop_size} does not match model preferred size "
                f"{model.input_size}. Using model preferred size of {model.input_size}.",
            )
            crop_size = model.input_size

        config = ProcessingConfig(
            crop_size=crop_size,
            batch_size=options.batch_size,
            blur_kernel_size=options.blur_kernel_size,
            morph_kernel_size=options.morph_kernel_size,
            band_order=band_order,
        )
        return cls(model=model, config=config, progress=progress)

    def run(
        self,
        img_path: str | Path,
        output_path: str | Path,
        cancel: Callable[[], bool] | None = None,
        observer: ProcessingObserver | None = None,
    ) -> None:
        """Process a raster file with tiled segmentation.

        Args:
            img_path: Path to input raster or SAFE directory
            output_path: Path to output segmentation raster
            cancel: Called before each batch. If it returns True, processing stops and the partial output is deleted.
            observer: Told about each batch and each part of the result as it is written.

        Raises:
            ProcessingCancelled: If cancel returned True.
        """
        try:
            self._run(img_path, output_path, cancel, observer)
        except ProcessingCancelled:
            Path(output_path).unlink(missing_ok=True)
            raise

    def _run(
        self,
        img_path: str | Path,
        output_path: str | Path,
        cancel: Callable[[], bool] | None,
        observer: ProcessingObserver | None,
    ) -> None:
        """Process a raster file with tiled segmentation. See run.

        Raises:
            ProcessingCancelled: If cancel returned True.
        """
        register = None

        # Instantiate the image reader using model configuration
        with self.model.cfg.get_reader(img_path) as reader:
            # Get raster properties
            height, width = reader.height, reader.width
            dtype = reader.dtype
            if observer is not None:
                observer.image_started(height, width)

            max_pixel_value = self.model.cfg.max_pixel_value_for(dtype)

            # The output has the input's size and location, with one band of class values
            profile = {
                **reader.profile,
                "dtype": "uint8",
                "count": 1,
                "compress": "lzw",
                "nodata": self.model.cfg.nodata_value,
            }

            # Full-size tiles, which can extend past the bottom and right of the image. They are read boundless
            windows = list(
                self._generate_windows(height, width, tile_size=self.config.crop_size, stride=self.config.stride)
            )

            window_batches = list(batched(windows, n=self.config.batch_size))

            with rasterio.open(output_path, "w", **profile) as dst:
                with self.progress() as reporter:
                    task = reporter.add_task("Segmenting", total=len(window_batches))
                    for window_batch in window_batches:
                        if cancel is not None and cancel():
                            raise ProcessingCancelled
                        if observer is not None:
                            observer.batch_started(window_batch)

                        # Read tile data from source raster
                        input_batch = self._load_batch(reader, window_batch)

                        # Shortcut to write the default value if all pixels have the same value in each img
                        blank = np.array([np.all(img == img.flat[0]) for img in input_batch], dtype=bool)
                        for window in (w for w, b in zip(window_batch, blank) if b):
                            clipped_window = self._clip_window_to_image_bounds(window, height, width)

                            # Skip windows that are completely outside the image bounds
                            if clipped_window is not None:
                                data = np.full(
                                    (clipped_window.height, clipped_window.width),
                                    self.model.cfg.default_output_value,
                                    dtype=np.uint8,
                                )
                                dst.write(data, 1, window=clipped_window)
                                if observer is not None:
                                    observer.written(clipped_window, data)

                        # Remove blank tiles from batch
                        window_batch = tuple(w for w, b in zip(window_batch, blank) if not b)
                        input_batch = input_batch[~blank]

                        # If all windows were removed, skip processing and update progress
                        if len(window_batch) == 0:
                            task.update(advance=1)
                            continue

                        # Process batch through model
                        result_batch = self.model.predict(input_batch, max_pixel_value)

                        if register is None:
                            num_classes = result_batch.shape[1]
                            register = NumpyMemoryRegister(
                                image_width=width,
                                register_depth=num_classes,
                                window_size=self.config.crop_size,
                                kernel=BartlettHannKernel,
                            )

                        # Apply post-processing to each result
                        for read_window, model_output in zip(
                            window_batch,
                            result_batch,
                            strict=False,
                        ):
                            clipped_window = self._clip_window_to_image_bounds(read_window, height, width)

                            # Skip windows that are completely outside the image bounds
                            if clipped_window is None:
                                continue

                            # Edge detection based on clipped window
                            is_top = clipped_window.row_off == 0
                            is_bottom = clipped_window.row_off + clipped_window.height >= height
                            is_left = clipped_window.col_off == 0
                            is_right = clipped_window.col_off + clipped_window.width >= width

                            data, write_window = register._step(
                                model_output,
                                clipped_window,
                                top=is_top,
                                bottom=is_bottom,
                                left=is_left,
                                right=is_right,
                            )
                            data = self.model.postprocess(data)
                            dst.write(data, 1, window=write_window)
                            if observer is not None:
                                observer.written(write_window, data)

                        task.update(advance=1)

            # Apply final post-processing
            self._apply_final_postprocessing(output_path, observer)

    def _load_batch(
        self,
        reader: ImageReader,
        windows: Iterable[Window],
    ) -> np.ndarray:
        """Load a batch of tiles from the source image using boundless reading.

        Args:
            reader: ImageReader instance for reading tiles.
            windows: Iterable of windows to load.

        Returns:
            Numpy array of shape [batch_size, channels, height, width] containing the tile data.
        """
        tile_data = []
        for window in windows:
            # Use boundless reading to handle out-of-bounds areas with fill_value=0
            tile_img = reader.read_window(
                window=window,
                band_order=self.config.band_order,
                boundless=True,
                fill_value=0,
            )
            _, th, tw = tile_img.shape

            # Pad out to tile size if needed (shouldn't be necessary with boundless reading)
            if th < self.config.crop_size or tw < self.config.crop_size:
                tile_img = np.pad(
                    tile_img,
                    (
                        (0, 0),
                        (0, self.config.crop_size - th),
                        (0, self.config.crop_size - tw),
                    ),
                    mode="constant",
                    constant_values=0,
                )

            tile_data.append(tile_img)

        # Stack into batch array [b, c, h, w]
        return np.stack(tile_data, axis=0)

    def _apply_final_postprocessing(
        self,
        dst_path: str | Path,
        observer: ProcessingObserver | None = None,
    ) -> None:
        """Apply final post-processing using tiled approach for large rasters.

        Args:
            dst_path: Destination rasterio dataset writer.
            observer: Told about each tile being filtered and each filtered part as it is written.
        """
        if not (self.config.apply_median_blur or self.config.apply_morphological_ops):
            return  # No processing needed

        # Calculate required overlap for operations
        blur_overlap = (self.config.blur_kernel_size - 1) // 2 if self.config.apply_median_blur else 0
        morph_overlap = (self.config.morph_kernel_size - 1) // 2 if self.config.apply_morphological_ops else 0
        overlap = max(blur_overlap, morph_overlap)

        if overlap == 0:
            return

        # Process in tiles with overlap
        with rasterio.open(str(dst_path), "r+") as dst:
            height, width = dst.height, dst.width

            # Use smaller tiles for post-processing to manage memory
            tile_size = min(512, self.config.crop_size)
            # Calculate stride to create overlapping tiles
            stride = tile_size - 2 * overlap

            # Generate overlapping windows for post-processing
            windows = self._generate_windows(
                height,
                width,
                tile_size=tile_size,
                stride=stride,
                boundless=False,
            )

            windows_list = list(windows)
            if windows_list:  # Only show progress if there are windows to process
                with self.progress() as reporter:
                    task = reporter.add_task("Filtering", total=len(windows_list))
                    for window in windows_list:
                        if observer is not None:
                            observer.batch_started([window])
                        # Read tile with overlap
                        tile_data = dst.read(1, window=window)

                        # Apply median blur (key for quality improvement)
                        if self.config.apply_median_blur:
                            tile_data = cv2.medianBlur(tile_data, self.config.blur_kernel_size)

                        # Apply morphological operations if enabled
                        if self.config.apply_morphological_ops:
                            kernel_size = self.config.morph_kernel_size
                            kernel = np.ones((kernel_size, kernel_size), np.uint8)

                            # Opening (remove noise)
                            tile_data = cv2.morphologyEx(
                                tile_data,
                                cv2.MORPH_OPEN,
                                kernel,
                            )
                            # Closing (fill gaps)
                            tile_data = cv2.morphologyEx(
                                tile_data,
                                cv2.MORPH_CLOSE,
                                kernel,
                            )

                        # Place result, removing overlap except at boundaries
                        write_window, written = self._place_window_result(dst, window, tile_data, overlap)
                        if observer is not None:
                            observer.written(write_window, written)
                        task.update(advance=1)

    @staticmethod
    def _clip_window_to_image_bounds(
        window: Window,
        height: int,
        width: int,
    ) -> Window | None:
        """Clip a window to fit within image bounds.

        Args:
            window: The original window.
            height: Height of the image.
            width: Width of the image.

        Returns:
            A new Window object clipped to the image bounds or None if the window is completely outside the image.
        """
        clipped_height = max(0, min(window.height, height - window.row_off))
        clipped_width = max(0, min(window.width, width - window.col_off))

        # Test if windows is completely outside the image bounds
        if clipped_height <= 0 or clipped_width <= 0 or window.row_off >= height or window.col_off >= width:
            return None

        return Window(
            col_off=window.col_off,
            row_off=window.row_off,
            height=clipped_height,
            width=clipped_width,
        )

    @staticmethod
    def _generate_windows(
        height: int,
        width: int,
        tile_size: int,
        stride: int,
        boundless: bool = True,
    ) -> Generator[Window]:
        """Generate tile windows for processing.

        Args:
            height: Original height of the image.
            width: Original width of the image.
            tile_size: Size of each tile (defaults to config.crop_size if config provided).
            stride: Size of each tile stride (defaults to config.stride if config provided).
            boundless: If True (default), windows can extend beyond image bounds.
                      If False, clips windows to image bounds.

        Yields:
            Window objects for each tile.
        """
        for y in range(_tile_count(height, tile_size, stride)):
            for x in range(_tile_count(width, tile_size, stride)):
                row_start = y * stride
                col_start = x * stride

                if boundless:
                    row_end = row_start + tile_size
                    col_end = col_start + tile_size
                else:
                    # For post-processing, clip to actual image bounds
                    row_end = min(row_start + tile_size, height)
                    col_end = min(col_start + tile_size, width)

                # Skip windows that would be entirely outside the image bounds
                if row_start >= height or col_start >= width:
                    continue

                # Ensure we have a valid window size (only needed for bounded mode)
                if not boundless and (row_end <= row_start or col_end <= col_start):
                    continue

                yield Window.from_slices(
                    rows=(row_start, row_end),
                    cols=(col_start, col_end),
                )

    @staticmethod
    def _place_window_result(
        dst: rasterio.io.BufferedDatasetWriter,
        window: Window,
        tile_result: np.ndarray,
        overlap: int,
    ) -> tuple[Window, np.ndarray]:
        """Place tile result into the full result array using overlap strategy.

        Args:
            dst: Destination rasterio dataset writer.
            window: Window object defining the region to write.
            tile_result: Result array from the model for the tile.
            overlap: Overlap size to handle edge cases.

        Returns:
            The window that was written and the data written to it.
        """
        # Calculate the region to copy (excluding overlap except at image boundaries)
        copy_row_start = window.row_off
        copy_col_start = window.col_off
        copy_row_end = window.row_off + window.height
        copy_col_end = window.col_off + window.width

        # Exclude overlap regions except at image boundaries
        if window.row_off > 0:  # Not top edge
            copy_row_start += overlap
        if window.col_off > 0:  # Not left edge
            copy_col_start += overlap
        if window.row_off + window.height < dst.height:  # Not bottom edge
            copy_row_end -= overlap
        if window.col_off + window.width < dst.width:  # Not right edge
            copy_col_end -= overlap

        # Calculate corresponding region in tile result
        tile_row_start = copy_row_start - window.row_off
        tile_col_start = copy_col_start - window.col_off
        tile_row_end = tile_row_start + (copy_row_end - copy_row_start)
        tile_col_end = tile_col_start + (copy_col_end - copy_col_start)

        result = tile_result[tile_row_start:tile_row_end, tile_col_start:tile_col_end]
        write_window = Window(
            col_off=copy_col_start,
            row_off=copy_row_start,
            width=copy_col_end - copy_col_start,
            height=copy_row_end - copy_row_start,
        )

        dst.write(result, 1, window=write_window)
        return write_window, result
