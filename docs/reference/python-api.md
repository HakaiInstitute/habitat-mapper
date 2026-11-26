# Python API Reference

Generated from the docstrings in the `habitat_mapper` package. For a guided introduction, see
[Use from Python](../guides/python.md).

## Model registry

`habitat_mapper.model_registry` is the shared `ModelRegistry` instance with every built-in model registered. Index it
with a model name, or with a `(name, revision)` tuple: `model_registry["kelp-rgb", "20240722"]`.

::: habitat_mapper.registry.ModelRegistry

::: habitat_mapper.registry.ModelNotFoundError

## Models

::: habitat_mapper.ONNXModel
    options:
      members:
        - name
        - revision
        - description
        - device
        - input_size
        - active_device
        - process
        - predict
        - postprocess

::: habitat_mapper.ModelConfig
    options:
      show_if_no_docstring: true

::: habitat_mapper.SegmentOptions
    options:
      show_if_no_docstring: true

## Processing

::: habitat_mapper.ProcessingObserver

::: habitat_mapper.ProcessingCancelled

## Model cache

`habitat_mapper.model_cache` is the shared `ModelCache` that models download their files to.

::: habitat_mapper.cache.ModelCache

## Progress reporting

::: habitat_mapper.ProgressReporter

::: habitat_mapper.RichProgressReporter

::: habitat_mapper.NullProgressReporter
