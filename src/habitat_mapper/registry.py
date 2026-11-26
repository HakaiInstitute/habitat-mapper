"""The ModelRegistry, which lists the available models and their revisions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from habitat_mapper.config import ModelConfig

if TYPE_CHECKING:
    from habitat_mapper.model import ONNXModel


class ModelNotFoundError(KeyError):
    """Raised for a model name or revision that isn't in the registry."""

    def __str__(self) -> str:
        """The message, without the quotes KeyError adds.

        Returns:
            The message.
        """
        return str(self.args[0])


def _newest_first(revision: str) -> tuple[int, str]:
    """Sort key for revisions: newest date (YYYYMMDD) first, then the plain revision before its variants.

    Returns:
        The key.
    """
    return -int(revision[:8]), revision[8:]


class ModelRegistry:
    """The configurations of the available models, by name and revision.

    Revisions are dates, YYYYMMDD, optionally followed by a variant, e.g. 20250101-small. The latest revision is the
    newest date, preferring the plain revision over its variants.

    Indexing the registry makes a new model from the configuration each time, so models are never shared between
    callers (or threads).
    """

    def __init__(self) -> None:
        """Create a new ModelRegistry instance."""
        self._configs: dict[str, dict[str, ModelConfig]] = {}

    def register_model(self, model_config: ModelConfig) -> None:
        """Register a model configuration.

        Raises:
            TypeError: If model_config is not an instance of ModelConfig.
        """
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be an instance of ModelConfig")
        self._configs.setdefault(model_config.name, {})[model_config.revision] = model_config

    @classmethod
    def from_config_dir(cls, config_dir: str | Path) -> ModelRegistry:
        """Create a ModelRegistry instance from a directory containing model configuration files.

        Config files should be named with pattern: {name}_{revision}.json

        Returns:
            ModelRegistry instance with models loaded from the specified directory.
        """
        reg = cls()
        for config_file in Path(config_dir).glob("*.json"):
            reg.register_model(ModelConfig.model_validate_json(config_file.read_text()))
        return reg

    def list_model_names(self) -> list[str]:
        """List the model names, sorted.

        Returns:
            The names.
        """
        return sorted(self._configs)

    def list_models(self) -> list[tuple[str, str]]:
        """List every model name and revision.

        Returns:
            (name, revision) pairs, by name and then newest revision first.
        """
        return [(name, revision) for name in self.list_model_names() for revision in self.revisions(name)]

    def revisions(self, name: str) -> list[str]:
        """List a model's revisions, latest first.

        Returns:
            The revisions.

        Raises:
            ModelNotFoundError: If there is no such model.
        """
        if name not in self._configs:
            raise ModelNotFoundError(f"Model {name} was not found. Run `hab models` to see the options.")
        return sorted(self._configs[name], key=_newest_first)

    def get_latest_revision(self, name: str) -> str:
        """Get the latest revision of a model.

        Returns:
            Latest revision string. Raises ModelNotFoundError if there is no such model.
        """
        return self.revisions(name)[0]

    def config(self, name: str, revision: str = "latest") -> ModelConfig:
        """Get the configuration of a model revision.

        Args:
            name: Model name, e.g. "kelp-rgb".
            revision: Revision, or "latest".

        Returns:
            The configuration.

        Raises:
            ModelNotFoundError: If there is no such model or revision.
        """
        revisions = self.revisions(name)
        if revision == "latest":
            revision = revisions[0]
        if revision not in revisions:
            raise ModelNotFoundError(
                f"Revision {revision} was not found for model {name}. Available revisions: {', '.join(revisions)}. "
                f"Run `hab revisions --model {name}` for details."
            )
        return self._configs[name][revision]

    def __getitem__(self, key: str | tuple[str, str]) -> ONNXModel:
        """Make a model by its name (latest revision) or by (name, revision) tuple.

        Args:
            key: Either model name (str) for latest revision, or (name, revision) tuple

        Returns:
            A new instance of the configured model class, usually ONNXModel. Keep it to reuse its loaded session.
        """
        cfg = self.config(*key) if isinstance(key, tuple) else self.config(key)
        return cfg.model_cls(cfg)

    def __contains__(self, key: str | tuple[str, str]) -> bool:
        """Check if a model is registered by its name or (name, revision).

        Returns:
            True if the model or revision exists, False otherwise.
        """
        if isinstance(key, tuple):
            name, revision = key
            return revision in self._configs.get(name, {})
        return key in self._configs

    def __repr__(self) -> str:
        """Get a representative string describing this instance.

        Returns:
            Representative string of the ModelRegistry instance.
        """
        return f"ModelRegistry({[f'{name}: {self.revisions(name)}' for name in self.list_model_names()]})"

    def __len__(self) -> int:
        """Get the total number of registered model revisions.

        Returns:
            Total number of model revisions registered.
        """
        return sum(len(revisions) for revisions in self._configs.values())


# Initialize the model registry from the default configuration directory
model_registry = ModelRegistry.from_config_dir(Path(__file__).parent / "configs")
