"""Tests for looking up models and revisions in the ModelRegistry."""

from __future__ import annotations

import pytest

from habitat_mapper.config import ModelConfig
from habitat_mapper.registry import ModelNotFoundError, ModelRegistry


def _registry(*revisions: str, beta: tuple[str, ...] = ()) -> ModelRegistry:
    registry = ModelRegistry()
    for revision in revisions:
        registry.register_model(
            ModelConfig(
                name="stub",
                revision=revision,
                beta=revision in beta,
                dependencies=[],
                model_filename="model.onnx",
            )
        )
    return registry


def test_revisions_are_newest_first_with_the_plain_revision_before_variants() -> None:
    registry = _registry("20240101", "20250101-small", "20250101", "20230101")
    assert registry.revisions("stub") == ["20250101", "20250101-small", "20240101", "20230101"]
    assert registry.get_latest_revision("stub") == "20250101"


def test_latest_is_the_first_revision() -> None:
    registry = _registry("20240101", "20250101")
    assert registry.config("stub").revision == "20250101"
    assert registry["stub"].revision == "20250101"
    assert registry["stub", "20240101"].revision == "20240101"


def test_latest_skips_betas_and_variants() -> None:
    registry = _registry("20240101", "20250101-small", "20260101", beta=("20260101",))
    assert registry.revisions("stub") == ["20260101", "20250101-small", "20240101"]
    assert registry.get_latest_revision("stub") == "20240101"
    assert registry.config("stub").revision == "20240101"
    assert registry["stub"].revision == "20240101"
    assert registry["stub", "20260101"].revision == "20260101"


def test_latest_falls_back_to_the_newest_when_every_revision_is_beta_or_a_variant() -> None:
    registry = _registry("20240101-small", "20250101", beta=("20250101",))
    assert registry.get_latest_revision("stub") == "20250101"


def test_unknown_revision_lists_the_options() -> None:
    with pytest.raises(
        ModelNotFoundError,
        match="Revision 19990101 was not found for model stub. Available revisions: 20250101, 20240101",
    ):
        _registry("20240101", "20250101").config("stub", "19990101")


def test_unknown_model_raises() -> None:
    with pytest.raises(ModelNotFoundError, match="Model nope was not found"):
        _registry("20240101")["nope"]


def test_lookup_errors_are_still_key_errors() -> None:
    # Callers that caught KeyError before ModelNotFoundError existed keep working
    with pytest.raises(KeyError):
        _registry("20240101")["stub", "19990101"]


def test_list_models_is_by_name_then_newest_revision() -> None:
    assert _registry("20240101", "20250101").list_models() == [("stub", "20250101"), ("stub", "20240101")]
