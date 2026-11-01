import numpy as np

from wholecell_py.computation import invert_composition_matrix


def test_composition_inverse_is_source_column_normalization_not_pseudoinverse():
    matrix = np.array([[1.0, 1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(
        invert_composition_matrix(matrix), [[1.0, 0.0], [0.5, 0.5]]
    )
