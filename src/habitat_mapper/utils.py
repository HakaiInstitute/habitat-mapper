"""Utilities and helper functions for the Habitat-Mapper package."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


def is_url(uri: str) -> bool:
    """Check if a URI is a URL or a local file path.

    Args:
        uri: The URI to check

    Returns:
        True if it's a URL, False if it's a local path

    """
    return uri.startswith(("http://", "https://", "ftp://", "ftps://"))


def batched(iterable: Iterable[Any], n: int) -> Iterable[tuple[Any, ...]]:
    """Batch data from the iterable into tuples of length n. The last batch may be shorter.

    This function emulates itertools.batched() which was introduced in Python 3.12.

    Args:
        iterable: Any iterable (list, string, generator, etc.)
        n: Batch size (positive integer)

    Yields:
        Tuples containing up to n elements from the iterable

    Raises:
        ValueError: If n is less than 1

    Examples:
        >>> list(batched('ABCDEFG', 3))
        [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

        >>> list(batched([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
        [(1, 2, 3, 4), (5, 6, 7, 8), (9,)]

        >>> list(batched(range(10), 2))
        [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    """
    if n < 1:
        raise ValueError("n must be at least one")

    iterator = iter(iterable)
    while True:
        batch = tuple(itertools.islice(iterator, n))
        if not batch:
            break
        yield batch


def softmax(x: np.ndarray, axis: int = 0, keepdims: bool = False) -> np.ndarray:
    """Apply softmax function to ndarray x.

    Args:
        x: A numpy array
        axis: Axis along which to apply the softmax function
        keepdims: Whether to keep dimensions or not

    Returns:
        A numpy array representing the softmax output
    """
    # Subtracting the largest score changes nothing but keeps exp from overflowing
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex / np.sum(ex, axis=axis, keepdims=keepdims)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the sigmoid activation function to array x.

    Args:
        x: A numpy array

    Returns:
        A numpy array of the sigmoid output
    """
    # Written with tanh, as exp overflows for large scores
    return 0.5 * (1 + np.tanh(0.5 * x))
