"""Tests that all config files in the configs directory load correctly."""

from pathlib import Path

import pytest

from habitat_mapper.config import ModelConfig
from habitat_mapper.registry import ModelRegistry

CONFIGS_DIR = Path(__file__).parent.parent / "src" / "habitat_mapper" / "configs"
CONFIG_FILES = list(CONFIGS_DIR.glob("*.json"))


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
