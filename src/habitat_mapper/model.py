"""ONNX model handling and inference."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf

from habitat_mapper.accelerators import create_session, describe_provider
from habitat_mapper.cache import ModelCache, model_cache
from habitat_mapper.config import ModelConfig, SegmentOptions
from habitat_mapper.processing import ImageProcessor, band_order_for
from habitat_mapper.progress import NullProgressReporter, RichProgressReporter
from habitat_mapper.utils import sigmoid, softmax

if TYPE_CHECKING:
    from collections.abc import Callable

    import onnxruntime as ort

    from habitat_mapper.accelerators import Device
    from habitat_mapper.processing import ProcessingObserver
    from habitat_mapper.progress import ProgressReporter


class FallbackSession:
    """An ONNX Runtime session that moves to the next device if the model fails on its first batch.

    Some failures only show up when the model first runs, e.g. a cuDNN library that can't be loaded. Failures after the
    first batch, e.g. running out of memory, don't mean the device can't run the model, so they are raised.
    """

    def __init__(self, model_path: Path, device: Device | None = None) -> None:
        """Load the model on the best available device.

        Args:
            model_path: Local path to the ONNX model.
            device: Device to run on, as for `hab segment --device`.
        """
        self._model_path = model_path
        self._device = device
        self._failed: set[str] = set()
        self._has_run = False
        self._session = self._create()

    def _create(self) -> ort.InferenceSession:
        """Create a session that skips the devices that have failed.

        Returns:
            The session.

        Raises:
            RuntimeError: If the model file is corrupt. It is deleted so that it downloads again next time.
        """
        try:
            return create_session(self._model_path, device=self._device, skip=self._failed)
        except InvalidProtobuf:
            self._model_path.unlink()
            raise RuntimeError(
                "Failed to load ONNX model. Removed the corrupted model file. Please try again."
            ) from None

    @property
    def provider(self) -> str:
        """The execution provider the session runs on, e.g. "CUDAExecutionProvider"."""
        return self._session.get_providers()[0]

    def get_inputs(self) -> list[ort.NodeArg]:
        """The model's inputs.

        Returns:
            The inputs, as ONNX Runtime describes them.
        """
        return self._session.get_inputs()

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Run the model, retrying the first batch on the next device if it fails.

        Args:
            inputs: Model inputs by name.

        Returns:
            The model's first output.
        """
        while True:
            try:
                outputs = self._session.run(None, inputs)
            except Exception as e:
                provider = self.provider
                if self._has_run or provider == "CPUExecutionProvider":
                    raise
                logger.warning(
                    f"The model failed to run on {describe_provider(provider)}. Trying the next device. "
                    "Set DEBUG=1 to see why."
                )
                logger.debug(f"{provider} failed: {e}")
                self._failed.add(provider)
                self._session = self._create()
                continue
            self._has_run = True
            return outputs[0]


class ONNXModel:
    """An ONNX model for image segmentation.

    Each model loads its own ONNX Runtime session the first time it runs, and keeps it for later images. Get a model
    from `model_registry`, which makes a new one for every lookup.
    """

    def __init__(self, config: ModelConfig, device: Device | None = None, cache: ModelCache | None = None) -> None:
        """Create a new ONNXModel instance.

        Args:
            config: Configuration for the model.
            device: Hardware to run on: "auto", "cpu", "cuda", "coreml", or "webgpu". Defaults to the HAB_DEVICE
                environment variable, or "auto".
            cache: Where the model's files are downloaded to. Defaults to `habitat_mapper.model_cache`.
        """
        self.cfg = config
        self.device = device
        self.cache = cache or model_cache
        self._session: FallbackSession | None = None
        self._session_device: Device | None = None

    @classmethod
    def from_json_config(cls, config_path: str | Path) -> ONNXModel:
        """Create an ONNXModel instance from a JSON configuration file.

        Returns:
            An instance of the ONNXModel class initialized with the configuration.
        """
        return cls(config=ModelConfig.model_validate_json(Path(config_path).read_text()))

    @property
    def name(self) -> str:
        """The model name."""
        return self.cfg.name

    @property
    def description(self) -> str:
        """A description of the model."""
        return self.cfg.description or ""

    @property
    def revision(self) -> str:
        """The model revision string."""
        return self.cfg.revision

    @property
    def active_device(self) -> str | None:
        """The device and hardware the model last ran on, e.g. "webgpu (Apple GPU)", or None before it has run."""
        if self._session is None:
            return None
        return describe_provider(self._session.provider)

    @property
    def session(self) -> FallbackSession:
        """The ONNX Runtime session for the model's device, loaded (and downloaded) the first time it is needed."""
        if self._session is None or self._session_device != self.device:
            self._session = FallbackSession(self.cache.fetch(self.cfg), self.device)
            self._session_device = self.device
        return self._session

    @cached_property
    def _input_name(self) -> str:
        """The name of the model's input.

        Raises:
            ValueError: If the model has no inputs defined.
        """
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("No inputs found in the ONNX model.")
        return inputs[0].name

    @cached_property
    def input_size(self) -> int | None:
        """The square tile size the model takes, or None if it takes any size.

        Raises:
            ValueError: If the model input shape is not valid.
        """
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("No inputs found in the ONNX model.")
        input_shape = inputs[0].shape
        if len(input_shape) < 3:
            raise ValueError(
                "ONNX model input shape must have at least 3 dimensions (batch, channels, height, width).",
            )

        h, w = input_shape[2], input_shape[3]

        if isinstance(h, str) and isinstance(w, str):
            return None  # Dynamic shape, no fixed tile size

        if isinstance(h, int) and h > 0:
            if h == w or isinstance(w, str):
                return h

        elif isinstance(w, int) and w > 0:
            if w == h or isinstance(h, str):
                return w

        msg = (
            f""
            f"Model input shape must be square or dynamic (e.g., [1, 8, 224, 224]). "
            f"Bad model input shape: {input_shape}"
        )
        raise ValueError(msg)

    def preprocess(self, batch: np.ndarray, max_pixel_value: float) -> np.ndarray:
        """Scale and normalize a batch of tiles as the model expects.

        Args:
            batch: Tiles with shape [batch_size, channels, height, width].
            max_pixel_value: Value to divide the pixels by, from `ModelConfig.max_pixel_value_for`.

        Returns:
            Batch of preprocessed float32 images with shape [batch_size, channels, height, width]

        Raises:
            NotImplementedError: If the configured normalization isn't supported.
        """
        batch = batch.astype(np.float32)

        if self.cfg.normalization == "standard":
            # Fold pixel scaling into the normalization, (x / max - mean) / std == x * scale + offset,
            # and apply it in place in float32 (float64 mean/std would otherwise promote the whole batch)
            mean = np.asarray(self.cfg.mean, dtype=np.float64)
            std = np.asarray(self.cfg.std, dtype=np.float64)
            scale = (1.0 / (max_pixel_value * std)).astype(np.float32)[None, :, None, None]
            offset = (-mean / std).astype(np.float32)[None, :, None, None]
            batch *= scale
            batch += offset
            return batch

        batch /= np.float32(max_pixel_value)

        if self.cfg.normalization is None:
            return batch

        if self.cfg.normalization == "min_max":
            bmin = batch.min(axis=(1, 2, 3), keepdims=True)
            bmax = batch.max(axis=(1, 2, 3), keepdims=True)
            return (batch - bmin) / (bmax - bmin + 1e-8)

        if self.cfg.normalization == "min_max_per_channel":
            bcmin = batch.min(axis=(2, 3), keepdims=True)
            bcmax = batch.max(axis=(2, 3), keepdims=True)
            return (batch - bcmin) / (bcmax - bcmin + 1e-8)

        raise NotImplementedError("Normalization method not implemented.")

    def postprocess(self, batch: np.ndarray) -> np.ndarray:
        """Turn model outputs into class labels. Subclasses override this for models with unusual outputs.

        Supports batch inputs and single samples.

        Returns:
            Batch of class labels with shape [batch_size, height, width] or [height, width]
        """
        undo_batch = False
        if len(batch.shape) == 3:
            undo_batch = True
            # Expand single class dim
            batch = np.expand_dims(batch, axis=0)

        if self.cfg.activation == "softmax":
            batch = softmax(batch, axis=1, keepdims=True)
        elif self.cfg.activation == "sigmoid":
            batch = sigmoid(batch)

        if batch.shape[1] == 1:
            # If single dim class, make into binary tensor
            batch = np.concat([1 - batch, batch], axis=1)

        batch = np.argmax(batch, axis=1, keepdims=False).astype(np.uint8)

        if undo_batch:
            batch = batch.squeeze(0)

        return batch

    def predict(self, batch: np.ndarray, max_pixel_value: float) -> np.ndarray:
        """Run the model on a batch of tiles.

        Args:
            batch: Tiles with shape [batch_size, channels, height, width].
            max_pixel_value: Value to divide the pixels by, from `ModelConfig.max_pixel_value_for`.

        Returns:
            The model's output scores, with shape [batch_size, classes, height, width]. See `postprocess`.
        """
        return self.session.run({self._input_name: self.preprocess(batch, max_pixel_value)})

    def process(
        self,
        img_path: str | Path,
        output_path: str | Path,
        *,
        batch_size: int = 1,
        crop_size: int | None = None,
        blur_kernel_size: int = 0,
        morph_kernel_size: int = 0,
        band_order: list[int] | None = None,
        quiet: bool = False,
        device: Device | None = None,
        progress: Callable[[], ProgressReporter] | None = None,
        cancel: Callable[[], bool] | None = None,
        observer: ProcessingObserver | None = None,
    ) -> None:
        """Segment an image and write the result to a GeoTIFF.

        The options are checked with `SegmentOptions`, so an invalid one raises a `ValueError`.

        Args:
            img_path: Path to input raster, or Sentinel-2 .SAFE folder or .zip for models that read them.
            output_path: Path to output segmentation raster
            batch_size: Batch size for processing
            crop_size: Tile size for processing (uses model's preferred size if None)
            blur_kernel_size: Size of median blur kernel (must be odd, 0 to disable)
            morph_kernel_size: Size of morphological kernel (must be odd, 0 to disable)
            band_order: A list of integers used to rearrange the input image channels. Indexed from 1 (like GDAL).
            quiet: If True, suppress progress bars during downloads and processing; log messages are not affected.
            device: Hardware to run the model on: "auto", "cpu", "cuda", "coreml", or "webgpu". Defaults to the
                device the model was made with, then the HAB_DEVICE environment variable, then "auto".
            progress: Makes the progress reporter for each stage, including downloading the model the first time, e.g.
                to show progress in your own app. Defaults to progress bars in the terminal, or none if quiet.
            cancel: Called before each batch of tiles. If it returns True, processing stops, the partial output is
                deleted, and ProcessingCancelled is raised.
            observer: Told about each batch and each part of the result as it is written, e.g. to preview it.
        """
        options = SegmentOptions(
            batch_size=batch_size,
            crop_size=crop_size,
            blur_kernel_size=blur_kernel_size,
            morph_kernel_size=morph_kernel_size,
            band_order=band_order,
            device=device,
        )
        # Checked before downloading, so a wrong band order doesn't wait for a large download
        band_order_for(self.cfg, options.band_order)
        if self._session is None:
            # Download the model first, so that its progress is shown like the rest
            self.cache.fetch(self.cfg, progress or (NullProgressReporter if quiet else _download_bars))
        if progress is None:
            progress = NullProgressReporter if quiet else RichProgressReporter

        # A device given here is for this image only
        default_device = self.device
        if options.device is not None:
            self.device = options.device
        try:
            processor = ImageProcessor.from_model(self, options, progress=progress)
            processor.run(img_path, output_path, cancel=cancel, observer=observer)
        finally:
            self.device = default_device


def _download_bars() -> RichProgressReporter:
    return RichProgressReporter(transfer=True)


class LegacyKelpRGBModel(ONNXModel):
    """A legacy model class for Kelp Species Segmentation in 3-band imagery.

    This class is used to handle the multi-model kelp detection models.

    Previous kelp models used one model for kelp presence detection and one model for species detection.

    This class should be considered deprecated and should not be used for new model development.
    """

    def postprocess(self, batch: np.ndarray) -> np.ndarray:
        """Turn the presence and species outputs into class labels.

        Returns:
            Class labels with shape [height, width]: 0 background, 1 macrocystis, 2 nereocystis.
        """
        presence_logits = batch[:1, :, :]
        species_logits = batch[1:, :, :]

        pa_probs = sigmoid(presence_logits)
        pa_label = (pa_probs > 0.5).astype(np.uint8).squeeze(0)  # 0 = bg, 1 = kelp

        sp_probs = softmax(species_logits, axis=0, keepdims=True)
        sp_label = np.argmax(sp_probs, axis=0).astype(np.uint8) + 1  # 1 = macro, 2 = nereo
        return pa_label * sp_label


class LegacyKelpRGBIModel(ONNXModel):
    """A legacy model class for Kelp Species Segmentation in 4-band imagery.

    This class is used to handle the multi-model kelp detection models.

    Previous kelp models used one model for kelp presence detection and one model for species detection.

    This class should be considered deprecated and should not be used for new model development.
    """

    def postprocess(self, batch: np.ndarray) -> np.ndarray:
        """Turn the presence and species outputs into class labels.

        Returns:
            Class labels with shape [height, width]: 0 background, 1 macrocystis, 2 nereocystis.
        """
        presence_logits = batch[:2, :, :]
        species_logits = batch[2:, :, :]

        pa_probs = softmax(presence_logits, axis=0, keepdims=True)
        pa_label = np.argmax(pa_probs, axis=0).astype(np.uint8)  # 0 = bg, 1 = kelp

        sp_probs = softmax(species_logits, axis=0, keepdims=True)
        sp_label = np.argmax(sp_probs, axis=0).astype(np.uint8) + 1  # 1 = macro, 2 = nereo
        return pa_label * sp_label
