"""Shared numerical and random-stream contracts for the Python port.

MATLAB arrays are 1-based and column-major. NumPy uses 0-based indexing and
usually row-major traversal; use the explicit helpers here where translating
MATLAB linear indexing. MATLAB numeric arrays have value semantics. Use
``copy_array`` at state/process ownership boundaries to avoid aliasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def copy_array(value: Any) -> np.ndarray:
    """Return an independent NumPy array without squeezing or reshaping it."""
    return np.array(value, copy=True, subok=False)


def matlab_linear_index(array: Any, one_based_index: int) -> Any:
    """Read one element by MATLAB's 1-based, column-major linear index."""
    arr = np.asarray(array)
    if not isinstance(one_based_index, (int, np.integer)):
        raise TypeError("MATLAB linear index must be an integer")
    if one_based_index < 1 or one_based_index > arr.size:
        raise IndexError("MATLAB linear index out of bounds")
    return arr.ravel(order="F")[int(one_based_index) - 1]


def matlab_linear_indices(array: Any, one_based_indices: Any) -> np.ndarray:
    """Gather 1-based MATLAB linear indices, preserving the index array shape."""
    arr = np.asarray(array)
    idx = np.asarray(one_based_indices)
    if not np.issubdtype(idx.dtype, np.integer):
        raise TypeError("MATLAB linear indices must be integers")
    if np.any(idx < 1) or np.any(idx > arr.size):
        raise IndexError("MATLAB linear index out of bounds")
    return arr.ravel(order="F")[idx.astype(np.intp, copy=False) - 1]


@dataclass
class NumpyRNG:
    """Configurable NumPy RNG adapter; it is not MATLAB mcg16807 compatible."""

    seed: int | None = None

    def __post_init__(self) -> None:
        self.generator = np.random.default_rng(self.seed)

    def random(self, size: Any = None) -> Any:
        return self.generator.random(size)

    def integers(
        self, low: int, high: int | None = None, size: Any = None
    ) -> Any:
        return self.generator.integers(low, high=high, size=size)

    def snapshot(self) -> dict[str, Any]:
        """Copyable bit-generator checkpoint for Python-side restarts."""
        return {
            "bit_generator": type(self.generator.bit_generator).__name__,
            "state": self.generator.bit_generator.state,
        }

    def restore(self, checkpoint: dict[str, Any]) -> None:
        if (
            checkpoint.get("bit_generator")
            != type(self.generator.bit_generator).__name__
        ):
            raise ValueError(
                "checkpoint bit generator does not match this adapter"
            )
        self.generator.bit_generator.state = checkpoint["state"]
