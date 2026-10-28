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
        - input_size
        - process

::: habitat_mapper.ModelConfig
    options:
      show_if_no_docstring: true

## Progress reporting

::: habitat_mapper.ProgressReporter

::: habitat_mapper.RichProgressReporter

::: habitat_mapper.NullProgressReporter
