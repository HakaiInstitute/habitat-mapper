"""Tests for the window that weights each tile's scores before overlapping tiles are blended."""

from __future__ import annotations

import numpy as np
import pytest

from habitat_mapper.hann import BartlettHannKernel


@pytest.mark.parametrize("size", [8, 32, 512])
def test_bartlett_hann_follows_its_formula(size: int) -> None:
    # Ha and Pearce (1989), periodic form: w(n) = 0.62 - 0.48 |n/N - 1/2| - 0.38 cos(2 pi n/N)
    n = np.arange(size)
    expected = 0.62 - 0.48 * np.abs(n / size - 0.5) - 0.38 * np.cos(2 * np.pi * n / size)

    np.testing.assert_allclose(BartlettHannKernel(size).wi, expected)


@pytest.mark.parametrize("tiles", [1, 2, 5])
def test_weights_of_overlapping_tiles_add_up_to_one(tiles: int) -> None:
    # Tiles step half a tile across and down, as ImageProcessor lays them out, and keep full weight at the image's
    # edges. Every pixel's weights then add up to one, so blending never brightens or dims the scores
    size, half = 16, 8
    image = half * (tiles + 1)
    kernel = BartlettHannKernel(size)
    total = np.zeros((image, image))
    for row in range(0, image - half, half):
        for col in range(0, image - half, half):
            weights = kernel.get_kernel(
                top=row == 0, bottom=row + size >= image, left=col == 0, right=col + size >= image
            )
            total[row : row + size, col : col + size] += weights

    np.testing.assert_allclose(total, 1)


def test_the_kernel_peaks_in_the_middle_and_fades_to_the_edges() -> None:
    weights = BartlettHannKernel(16).get_kernel()

    assert weights.max() == weights[8, 8]
    assert weights[0, 0] < weights[4, 4] < weights[8, 8]
    np.testing.assert_allclose(weights, weights.T)
