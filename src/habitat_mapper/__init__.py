"""Habitat-Mapper: segment habitats, like kelp and eelgrass, in drone and satellite imagery with AI."""

from importlib.metadata import version

from habitat_mapper.cache import ModelCache, model_cache
from habitat_mapper.config import ModelConfig, SegmentOptions
from habitat_mapper.model import ONNXModel
from habitat_mapper.processing import ProcessingCancelled, ProcessingObserver
from habitat_mapper.progress import NullProgressReporter, ProgressReporter, RichProgressReporter
from habitat_mapper.registry import model_registry

__all__ = [
    "ModelCache",
    "ModelConfig",
    "NullProgressReporter",
    "ONNXModel",
    "ProcessingCancelled",
    "ProcessingObserver",
    "ProgressReporter",
    "RichProgressReporter",
    "SegmentOptions",
    "model_cache",
    "model_registry",
]

__version__ = version("habitat_mapper")
