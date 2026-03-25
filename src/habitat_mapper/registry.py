"""A module containing the ModelRegistry class that indexes the available models for inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from habitat_mapper.config import ModelConfig

if TYPE_CHECKING:
    from habitat_mapper.model import ONNXModel


class ModelRegistry:
    """A model_registry for models that allows for dynamic registration and retrieval.

    Supports multiple revisions of the same model with calendar versioning.
    """

    def __init__(self) -> None:
        """Create a new ModelRegistry instance."""
        # Changed to nested dict: {name: {revision: model}}
        self._models: dict[str, dict[str, ONNXModel]] = {}

    def list_models(self) -> list[tuple[str, str]]:
        """List all registered model names and revisions.

        Returns:
             A list of (name, revision) tuples.
        """
        models = []
        for name, revisions in self._models.items():
            for revision in revisions.keys():
                models.append((name, revision))
        return models

    def list_model_names(self) -> list[str]:
        """List unique model names (without revisions).

        Returns:
            List of unique model names registered in the registry.
        """
        return list(self._models.keys())

    def register_model(self, model_config: ModelConfig) -> None:
        """Register a model configuration.

        Raises:
            TypeError: If model_config is not an instance of ModelConfig.
        """
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be an instance of ModelConfig")

        name = model_config.name
        revision = model_config.revision

        if name not in self._models:
            self._models[name] = {}

        # model_cls is resolved by Pydantic's ImportString
        model_class = model_config.model_cls
        self._models[name][revision] = model_class(model_config)

    @classmethod
    def from_config_dir(cls, config_dir: str | Path) -> ModelRegistry:
        """Create a ModelRegistry instance from a directory containing model configuration files.

        Config files should be named with pattern: {name}_{revision}.json

        Returns:
            ModelRegistry instance with models loaded from the specified directory.
        """
        reg = cls()
        config_path = Path(config_dir)

        for config_file in config_path.glob("*.json"):
            # Validate and parse the JSON configuration
            config_data = ModelConfig.model_validate_json(open(config_file).read())
            reg.register_model(config_data)

        return reg

    def get_latest_revision(self, name: str) -> str:
        """Get the latest revision of a model by calendar versioning.

        Returns:
            Latest revision string

        Raises:
            KeyError: If the model is not registered.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered.")

        revisions = list(self._models[name].keys())
        # Sort by revision string (calendar versioning works with string sort)
        return max(revisions)

    def __getitem__(self, key: str | tuple[str, str]) -> ONNXModel:
        """Retrieve a model by its name (latest revision) or by (name, revision) tuple.

        Args:
            key: Either model name (str) for latest revision, or (name, revision) tuple

        Returns:
            ONNXModel instance

        Raises:
            KeyError: If the model or revision is not registered.
        """
        if isinstance(key, tuple):
            name, revision = key
            if name not in self._models:
                raise KeyError(f"Model '{name}' is not registered.")
            if revision not in self._models[name]:
                available_revisions = list(self._models[name].keys())
                raise KeyError(
                    f"Revision '{revision}' of model '{name}' is not registered. "
                    f"Available revisions: {available_revisions}",
                )
            return self._models[name][revision]
        # Single string key - get latest revision
        name = key
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered.")
        latest_revision = self.get_latest_revision(name)
        return self._models[name][latest_revision]

    def __contains__(self, key: str | tuple[str, str]) -> bool:
        """Check if a model is registered by its name or (name, revision).

        Returns:
            True if the model or revision exists, False otherwise.
        """
        if isinstance(key, tuple):
            name, revision = key
            return name in self._models and revision in self._models[name]
        return key in self._models

    def __repr__(self) -> str:
        """Get a representative string describing this instance.

        Returns:
            Representative string of the ModelRegistry instance.
        """
        model_info = []
        for name, revisions in self._models.items():
            revision_list = list(revisions.keys())
            model_info.append(f"{name}: {revision_list}")
        return f"ModelRegistry({model_info})"

    def __len__(self) -> int:
        """Get the total number of registered model revisions.

        Returns:
            Total number of model revisions registered.
        """
        return sum(len(revisions) for revisions in self._models.values())


# Initialize the model registry from the default configuration directory
model_registry = ModelRegistry.from_config_dir(Path(__file__).parent / "configs")
