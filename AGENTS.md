# Agent guide

## Testing

Every behaviour change comes with a test that goes red without the change. Test through the module's interface, and assert on values (whole rasters with `np.testing.assert_array_equal`, exact classes, exact messages) rather than on the code running without errors.

Where tests go:

- **Segmentation output**: `tests/test_processing_golden.py`. Its `PixelwiseModel` scores each pixel from that pixel's value alone, so the expected class of every output pixel can be computed directly, however tiles overlap.
- **End to end**: `tests/helpers.py` `write_tiny_model` builds a real ONNX model that runs on the CPU in milliseconds. `tests/test_segment_end_to_end.py` runs `hab segment` with it.
- **Desktop app**: rules go in the Qt-free modules (`gui/batch.py`, `gui/jobs.py`, `gui/model_info.py`, `gui/preview.py`) and are tested there, so they run in every CI job. `tests/test_gui_window.py` drives the window itself with pytest-qt. It needs the `gui` extra and `gui-tests` group, so only CI's coverage job runs it, and it skips elsewhere.
- **Shared builders and fakes**: `tests/helpers.py`. Test modules import from it, never from each other.

When a new test exposes a bug, fix the bug in the same change.

CI's `coverage` job fails when:

- under 80% of the lines a PR changes are covered, or
- total coverage falls below `fail_under` in `pyproject.toml`. When a change raises coverage, raise `fail_under` with it.

Check locally:

```sh
uv run --extra gui --group gui-tests pytest --cov --cov-report=xml
uv run diff-cover coverage.xml --compare-branch=origin/dev
```

`uv run mutmut run` then `uv run mutmut results` lists mutants no test caught in the modules where a bug gives a wrong raster rather than an error. Each surviving mutant marks a missing assertion. It also runs weekly in `.github/workflows/mutation.yml`.
