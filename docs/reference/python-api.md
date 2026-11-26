# Python API Reference

Generated from the docstrings in the `habitat_mapper` package. For a guided introduction, see
[Use from Python](../guides/python.md).

## Model registry

`habitat_mapper.model_registry` is the shared `ModelRegistry` instance with every built-in model registered. Index it
with a model name, or with a `(name, revision)` tuple: `model_registry["kelp-rgb", "20240722"]`.

::: habitat_mapper.registry.ModelRegistry

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

## Progress reporting

::: habitat_mapper.ProgressReporter

::: habitat_mapper.RichProgressReporter

::: habitat_mapper.NullProgressReporter
