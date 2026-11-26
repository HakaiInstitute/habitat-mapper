# Agent guide

## Testing

Every behaviour change comes with a test that goes red without the change. Test through the module's interface, and assert on values (whole rasters with `np.testing.assert_array_equal`, exact classes, exact messages) rather than on the code running without errors.

Where tests go:

- **Segmentation output**: `tests/test_processing_golden.py`. Its `PixelwiseModel` scores each pixel from that pixel's value alone, so the expected class of every output pixel can be computed directly, however tiles overlap.
- **End to end**: `tests/helpers.py` `write_tiny_model` builds a real ONNX model that runs on the CPU in milliseconds. `tests/test_segment_end_to_end.py` runs `hab segment` with it.
- **Desktop app**: logic goes in the Qt-free modules (`gui/batch.py`, `gui/jobs.py`, `gui/model_info.py`, `gui/preview.py`) and is tested there. CI installs no PySide6, so `gui/app.py` and `gui/worker.py` stay thin wiring that calls those modules.
- **Shared builders and fakes**: `tests/helpers.py`. Test modules import from it, never from each other.

When a new test exposes a bug, fix the bug in the same change.

CI's `coverage` job fails when:

- under 80% of the lines a PR changes are covered, or
- total coverage falls below `fail_under` in `pyproject.toml`. When a change raises coverage, raise `fail_under` with it.

Check locally:

```sh
uv run pytest --cov --cov-report=xml
uv run diff-cover coverage.xml --compare-branch=origin/dev
```

`uv run mutmut run` then `uv run mutmut results` lists mutants no test caught in the modules where a bug gives a wrong raster rather than an error. Each surviving mutant marks a missing assertion. It also runs weekly in `.github/workflows/mutation.yml`.
