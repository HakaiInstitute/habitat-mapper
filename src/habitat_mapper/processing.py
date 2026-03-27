"""High-level image processing with tiled segmentation and post-processing."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import rasterio
from loguru import logger
from rasterio.windows import Window
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from habitat_mapper.config import ProcessingConfig
from habitat_mapper.hann import BartlettHannKernel, NumpyMemoryRegister
from habitat_mapper.utils import batched

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from pathlib import Path

    from habitat_mapper.model import ONNXModel
    from habitat_mapper.reader import ImageReader


@dataclass
class ImageProcessor:
    """High-level image processor that orchestrates model inference and tiled processing."""

    model: ONNXModel
    config: ProcessingConfig
    quiet: bool = False

    @classmethod
    def from_model(
        cls,
        model: ONNXModel,
        *,
        crop_size: int | None = None,
        batch_size: int = 1,
        blur_kernel_size: int = 5,
        morph_kernel_size: int = 0,
        band_order: list[int] | None = None,
        quiet: bool = False,
    ) -> ImageProcessor:
        """Create an ImageProcessor from a model with processing parameters.

        Args:
            model: The ONNX model to use
            crop_size: Tile size for processing (uses model's preferred size if None)
            batch_size: Batch size for processing
            blur_kernel_size: Size of median blur kernel (must be odd)
            morph_kernel_size: Size of morphological kernel (0 to disable)
            band_order: List of band indices (1-based like GDAL)
            quiet: If True, suppress all progress bars.

        Returns:
            Configured ImageProcessor instance

        Raises:
            ValueError: If band_order contains invalid indices
        """
        if band_order is None:
            band_order = [i + 1 for i in range(model.cfg.input_channels)]
        else:
            if max(band_order) > model.cfg.input_channels:
                raise ValueError(
                    f"band_order contains index {max(band_order)} which exceeds model input "
                    f"channels ({model.cfg.input_channels})",
                )

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
            batch_size=batch_size,
            blur_kernel_size=blur_kernel_size,
            morph_kernel_size=morph_kernel_size,
            band_order=band_order,
        )
        return cls(model=model, config=config, quiet=quiet)

    def run(
        self,
        img_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """Process a raster file with tiled segmentation.

        Args:
            img_path: Path to input raster or SAFE directory
            output_path: Path to output segmentation raster
        """
        register = None

        # Instantiate the image reader using model configuration
        with self.model.cfg.get_reader(img_path) as reader:
            # Get raster properties
            height, width = reader.height, reader.width
            dtype = reader.dtype

            # For TIFF, we can get the profile; for other formats, create a minimal one
            try:
                from habitat_mapper.reader import TIFFReader

                if isinstance(reader, TIFFReader):
                    profile = reader.profile.copy()
                else:
                    profile = self._create_profile_from_reader(reader)
            except ImportError:
                profile = self._create_profile_from_reader(reader)

            if self.model.cfg.max_pixel_value == "auto":
                # Automatically determine max pixel value based on dtype
                if dtype == "uint8":
                    self.model.cfg.max_pixel_value = 255
                elif dtype == "uint16":
                    self.model.cfg.max_pixel_value = 65535
                elif dtype in ["float16", "float32", "float64"]:
                    # Warn that float types should be normalized
                    logger.warning(
                        "Warning: Input image has float data type. "
                        "Ensure pixel values are in range [0,1] to avoid unexpected results."
                    )
                    self.model.cfg.max_pixel_value = 1.0
                else:
                    logger.error(
                        f"Unsupported image data type {dtype} for model. "
                        f"Convert your image to uint8 or uint16 before processing.",
                    )
                    sys.exit(0)

            # Update profile for output
            profile.update({"dtype": "uint8", "count": 1, "compress": "lzw", "nodata": self.model.cfg.nodata_value})

            # Calculate extended dimensions to accommodate full tiles
            extended_height, extended_width = self._calculate_extended_dimensions(
                height,
                width,
                self.config,
            )

            # Generate tiles and process in batches
            windows = list(
                self._generate_windows(
                    extended_height, extended_width, tile_size=self.config.crop_size, stride=self.config.stride
                ),
            )

            window_batches = list(batched(windows, n=self.config.batch_size))

            with rasterio.open(output_path, "w", **profile) as dst:
                with Progress(
                    TextColumn("[bold blue]Segmenting"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=Console(quiet=True) if self.quiet else None,
                ) as progress:
                    task = progress.add_task("Processing", total=len(window_batches))
                    for window_batch in window_batches:
                        # Read tile data from source raster
                        input_batch = self._load_batch(reader, window_batch)

                        # Shortcut to write zeros if all pixels have the same value in each img
                        for i, (window, img) in enumerate(zip(window_batch, input_batch)):
                            if np.all(img == img.flatten()[0]):
                                clipped_window = self._clip_window_to_image_bounds(window, height, width)

                                # Skip windows that are completely outside the image bounds
                                if clipped_window is not None:
                                    data = (
                                        np.ones((clipped_window.height, clipped_window.width), dtype=np.uint8)
                                        * self.model.cfg.default_output_value
                                    )
                                    dst.write(data, 1, window=clipped_window)

                                # Remove from batch
                                window_batch_list = list(window_batch)
                                window_batch_list.pop(i)
                                window_batch = tuple(window_batch_list)
                                input_batch = np.delete(input_batch, i, axis=0)

                        # If all windows were removed, skip processing and update progress
                        if len(window_batch) == 0:
                            progress.update(task, advance=1, refresh=True)
                            continue

                        # Process batch through model
                        result_batch = self.model._predict(input_batch)

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
                            data = self.model._postprocess(data)
                            dst.write(data, 1, window=write_window)

                        progress.update(task, advance=1, refresh=True)

            # Apply final post-processing
            self._apply_final_postprocessing(output_path)

    def _create_profile_from_reader(self, reader: ImageReader) -> dict[str, Any]:
        """Create a rasterio profile from an ImageReader.

        Args:
            reader: ImageReader instance to extract properties from

        Returns:
            Dictionary suitable for rasterio profile
        """
        profile = {
            "driver": "GTiff",
            "height": reader.height,
            "width": reader.width,
            "count": 1,
            "dtype": "uint8",
        }

        # Add CRS and transform if available
        if reader.crs is not None:
            profile["crs"] = reader.crs
        if reader.transform is not None:
            profile["transform"] = reader.transform

        return profile

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
    ) -> None:
        """Apply final post-processing using tiled approach for large rasters.

        Args:
            dst_path: Destination rasterio dataset writer.
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
                with Progress(
                    TextColumn("[bold green]Post-processing"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=Console(quiet=True) if self.quiet else None,
                ) as progress:
                    task = progress.add_task("Post-processing", total=len(windows_list))
                    for window in windows_list:
                        # Read tile with overlap
                        tile_data = dst.read(1, window=window)

                        # Apply median blur (key for quality improvement)
                        if self.config.apply_median_blur:
                            blur_size = self.config.blur_kernel_size
                            if blur_size % 2 == 0:
                                blur_size += 1  # Ensure odd size
                            tile_data = cv2.medianBlur(tile_data, blur_size)

                        # Apply morphological operations if enabled
                        if self.config.morph_kernel_size > 0:
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
                        self._place_window_result(dst, window, tile_data, overlap)
                        progress.update(task, advance=1, refresh=True)

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
    def _calculate_extended_dimensions(
        height: int,
        width: int,
        config: ProcessingConfig,
    ) -> tuple[int, int]:
        """Calculate extended dimensions to fit complete tiles.

        Args:
            height: Original height of the image.
            width: Original width of the image.
            config: Processing configuration.

        Returns:
            Tuple of (extended_height, extended_width) that ensures full tile coverage.
        """
        tile_size = config.crop_size
        stride = config.stride

        # Calculate number of tiles needed using same logic as _generate_windows
        if height <= tile_size:
            tiles_y = 1
        else:
            tiles_y = math.ceil((height - tile_size) / stride) + 1

        if width <= tile_size:
            tiles_x = 1
        else:
            tiles_x = math.ceil((width - tile_size) / stride) + 1

        # Calculate extended dimensions to ensure full coverage
        extended_height = (tiles_y - 1) * stride + tile_size
        extended_width = (tiles_x - 1) * stride + tile_size

        return max(extended_height, height), max(extended_width, width)

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
        # Calculate number of tiles needed to ensure full coverage
        # Use ceiling division to handle cases where dimensions don't align perfectly with stride
        if height <= tile_size:
            tiles_y = 1
        else:
            tiles_y = math.ceil((height - tile_size) / stride) + 1

        if width <= tile_size:
            tiles_x = 1
        else:
            tiles_x = math.ceil((width - tile_size) / stride) + 1

        for y in range(tiles_y):
            for x in range(tiles_x):
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
    ) -> None:
        """Place tile result into the full result array using overlap strategy.

        Args:
            dst: Destination rasterio dataset writer.
            window: Window object defining the region to write.
            tile_result: Result array from the model for the tile.
            overlap: Overlap size to handle edge cases.
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
