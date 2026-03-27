"""ONNX model handling and inference."""

from __future__ import annotations

import sys
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from loguru import logger
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf

from habitat_mapper.config import ModelConfig
from habitat_mapper.processing import ImageProcessor
from habitat_mapper.utils import get_ort_providers, setup_cuda_paths, sigmoid, softmax

if TYPE_CHECKING:
    from pathlib import Path

# Load CUDA and CUDNN DLLs
setup_cuda_paths()  # Workaround for Windows to ensure CUDA paths are set correctly
ort.preload_dlls(directory="")

ort.set_default_logger_severity(3)  # Set to 3=ERROR


class ONNXModel:
    """A class representing an ONNX model for image segmentation."""

    def __init__(self, config: ModelConfig) -> None:
        """Create a new ONNXModel instance.

        Args:
            config: Configuration for the model.
        """
        self.cfg = config
        self.__ort_sess: ort.InferenceSession | None = None

    @classmethod
    def from_json_config(cls, config_path: str) -> ONNXModel:
        """Create an ONNXModel instance from a JSON configuration file.

        Returns:
            An instance of the ONNXModel class initialized with the configuration.
        """
        config = ModelConfig.model_validate_json(open(config_path).read())
        return cls(config=config)

    @property
    def name(self) -> str:
        """Get the model name."""
        return self.cfg.name

    @property
    def description(self) -> str:
        """Get a description of the model."""
        return self.cfg.description or ""

    @property
    def revision(self) -> str:
        """Get the model revision string."""
        return self.cfg.revision

    @property
    def _ort_sess(self) -> ort.InferenceSession:
        """Load and cache the ONNX Runtime session."""
        if self.__ort_sess is not None:
            return self.__ort_sess

        # Get local model path (handles download if needed)
        local_model_path = self.cfg.local_model_path

        # Load appropriate ONNX providers for OS capabilities
        providers = get_ort_providers()

        # Load the model
        try:
            self.__ort_sess = ort.InferenceSession(
                str(local_model_path),
                providers=providers,
            )
            self.__ort_sess.enable_fallback()
        except InvalidProtobuf:
            logger.error("Failed to load ONNX model. Removing corrupted model file. Please try again.")
            local_model_path.unlink()
            sys.exit(1)

        return self.__ort_sess

    @cached_property
    def _input_name(self) -> str:
        """Get the input name for the model.

        Returns:
            Input name as a string.

        Raises:
            ValueError: If the model has no inputs defined.
        """
        inputs = self._ort_sess.get_inputs()
        if not inputs:
            raise ValueError("No inputs found in the ONNX model.")
        return inputs[0].name

    @cached_property
    def input_size(self) -> int | None:
        """Get the input tile size from the model configuration.

        Returns:
            Tuple of (height, width) or None if not specified.

        Raises:
            ValueError: If the model input shape is not valid.
        """
        inputs = self._ort_sess.get_inputs()
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

    def _preprocess(self, batch: np.ndarray) -> np.ndarray:
        """Preprocess the input image according to the model's configuration.

        Returns:
            Batch of preprocessed images with shape [batch_size, channels, height, width]
        """
        if self.cfg.max_pixel_value == "auto":
            self.cfg.max_pixel_value = np.iinfo(batch.dtype).max

        batch = batch.astype(np.float32) / self.cfg.max_pixel_value

        if self.cfg.normalization is None:
            return batch

        if self.cfg.normalization == "standard":
            # Expand dims for broadcasting
            mean = np.array(self.cfg.mean)[None, :, None, None]
            std = np.array(self.cfg.std)[None, :, None, None]
            return (batch - mean) / std

        if self.cfg.normalization == "min_max":
            bmin = batch.min(axis=(1, 2, 3), keepdims=True)
            bmax = batch.max(axis=(1, 2, 3), keepdims=True)
            return (batch - bmin) / (bmax - bmin + 1e-8)

        if self.cfg.normalization == "min_max_per_channel":
            bcmin = batch.min(axis=(2, 3), keepdims=True)
            bcmax = batch.max(axis=(2, 3), keepdims=True)
            return (batch - bcmin) / (bcmax - bcmin + 1e-8)

        raise NotImplementedError("Normalization method not implemented.")

    def _postprocess(self, batch: np.ndarray) -> np.ndarray:
        """Postprocess the output to get class label from logits or probs.

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

    def _predict(self, batch: np.ndarray) -> np.ndarray:
        """Run inference on a batch of image tiles.

        Args:
            batch: Input batch with shape [batch_size, channels, height, width]

        Returns:
            Predictions with shape [batch_size, height, width]

        """
        batch = self._preprocess(batch)

        batch = self._ort_sess.run(
            None,
            {f"{self._input_name}": batch.astype(np.float32)},
        )
        return batch[0]

    def process(
        self,
        img_path: str | Path,
        output_path: str | Path,
        *,
        batch_size: int = 1,
        crop_size: int | None = None,
        blur_kernel_size: int = 5,
        morph_kernel_size: int = 0,
        band_order: list[int] | None = None,
        quiet: bool = False,
    ) -> None:
        """Process an image using the default tiled processing strategy.

        This is a convenience method that creates an ImageProcessor with
        TiledProcessingStrategy and processes the image.

        Args:
            img_path: Path to input raster
            output_path: Path to output segmentation raster
            batch_size: Batch size for processing
            crop_size: Tile size for processing (uses model's preferred size if None)
            blur_kernel_size: Size of median blur kernel (must be odd)
            morph_kernel_size: Size of morphological kernel (0 to disable)
            band_order: A list of integers used to rearrange the input image channels. Indexed from 1 (like GDAL).
            quiet: If True, suppress all progress bars and log messages.

        """
        if quiet:
            logger.disable("habitat_mapper")
        try:
            processor = ImageProcessor.from_model(
                model=self,
                batch_size=batch_size,
                crop_size=crop_size,
                blur_kernel_size=blur_kernel_size,
                morph_kernel_size=morph_kernel_size,
                band_order=band_order,
                quiet=quiet,
            )
            processor.run(img_path=img_path, output_path=output_path)
        finally:
            if quiet:
                logger.enable("habitat_mapper")


class LegacyKelpRGBModel(ONNXModel):
    """A legacy model class for Kelp Species Segmentation in 3-band imagery.

    This class is used to handle the multi-model kelp detection models.

    Previous kelp models used one model for kelp presence detection and one model for species detection.

    This class should be considered deprecated and should not be used for new model development.
    """

    def _postprocess(self, batch: np.ndarray) -> np.ndarray:
        """Postprocess the output to get class label from logits or probs.

        Supports batch inputs and single samples.

        Returns:
            Batch of class labels with shape [batch_size, height, width] or [height, width]
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

    def _postprocess(self, batch: np.ndarray) -> np.ndarray:
        """Postprocess the output to get class label from logits or probs.

        Supports batch inputs and single samples.

        Returns:
            Batch of class labels with shape [batch_size, height, width] or [height, width]
        """
        presence_logits = batch[:2, :, :]
        species_logits = batch[2:, :, :]

        pa_probs = softmax(presence_logits, axis=0, keepdims=True)
        pa_label = np.argmax(pa_probs, axis=0).astype(np.uint8)  # 0 = bg, 1 = kelp

        sp_probs = softmax(species_logits, axis=0, keepdims=True)
        sp_label = np.argmax(sp_probs, axis=0).astype(np.uint8) + 1  # 1 = macro, 2 = nereo
        return pa_label * sp_label
