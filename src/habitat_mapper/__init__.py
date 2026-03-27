"""Habitat-Mapper: A tool for segmenting audio files using machine learning models."""

from importlib.metadata import version

from habitat_mapper.config import ModelConfig
from habitat_mapper.main import clean, models, revisions, segment
from habitat_mapper.model import ONNXModel
from habitat_mapper.progress import NullProgressReporter, ProgressReporter, RichProgressReporter
from habitat_mapper.registry import model_registry

__all__ = [
    "ModelConfig",
    "NullProgressReporter",
    "ONNXModel",
    "ProgressReporter",
    "RichProgressReporter",
    "clean",
    "model_registry",
    "models",
    "revisions",
    "segment",
]

__version__ = version("habitat_mapper")
