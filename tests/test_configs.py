"""Tests that all config files in the configs directory load correctly."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from habitat_mapper.config import ModelConfig
from habitat_mapper.model import ONNXModel
from habitat_mapper.reader import TIFFReader
from habitat_mapper.registry import ModelRegistry

CONFIGS_DIR = Path(__file__).parent.parent / "src" / "habitat_mapper" / "configs"
CONFIG_FILES = sorted(CONFIGS_DIR.glob("*.json"), key=lambda p: p.name)


@pytest.mark.parametrize("config_file", CONFIG_FILES, ids=[f.name for f in CONFIG_FILES])
def test_config_loads(config_file: Path) -> None:
    config = ModelConfig.model_validate_json(config_file.read_text())
    assert config.name
    assert config.revision
    assert config.dependencies
    assert config.model_filename in [Path(dep).name for dep in config.dependencies]


def test_registry_loads_all_configs() -> None:
    registry = ModelRegistry.from_config_dir(CONFIGS_DIR)
    assert len(registry) == len(CONFIG_FILES)


def _config(**kwargs: object) -> ModelConfig:
    return ModelConfig(name="stub", revision="20250101", dependencies=[], model_filename="model.onnx", **kwargs)


def test_config_is_frozen() -> None:
    with pytest.raises(ValidationError):
        _config().max_pixel_value = 255  # type: ignore[misc]


def test_import_string_defaults_are_resolved() -> None:
    config = _config()
    assert config.model_cls is ONNXModel
    assert config.reader_cls is TIFFReader


def test_registry_makes_a_new_model_for_each_lookup() -> None:
    registry = ModelRegistry.from_config_dir(CONFIGS_DIR)
    first, second = registry["kelp-rgb"], registry["kelp-rgb"]
    assert first is not second
    assert first.cfg is second.cfg


@pytest.mark.parametrize(("dtype", "expected"), [("uint8", 255), ("uint16", 65535), ("float32", 1.0)])
def test_auto_max_pixel_value_follows_the_data_type(dtype: str, expected: float) -> None:
    assert _config().max_pixel_value_for(dtype) == expected


def test_configured_max_pixel_value_is_kept() -> None:
    assert _config(max_pixel_value=10000.0).max_pixel_value_for("uint16") == 10000.0


def test_unsupported_data_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported image data type int32"):
        _config().max_pixel_value_for("int32")
