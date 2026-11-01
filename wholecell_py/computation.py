"""Small numerical contracts from the original ComputationUtil."""

import numpy as np


def invert_composition_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(1.0 / np.sum(matrix, axis=0)) @ matrix.T
