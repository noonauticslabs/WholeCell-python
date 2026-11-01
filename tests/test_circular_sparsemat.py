import numpy as np


def test_constructor_wraps_without_mutating_input_coordinates():
    from wholecell_py.circular_sparsemat import CircularSparseMat

    coords = np.array([[0, 1], [11, 2]], dtype=np.int64)
    before = coords.copy()
    matrix = CircularSparseMat(coords, [3, 4], [10, 2], 1)
    np.testing.assert_array_equal(coords, before)
    np.testing.assert_array_equal(
        matrix.get_coords([[10, 1], [1, 2]]).ravel(), [3, 4]
    )


import pytest

from wholecell_py.circular_sparsemat import CircularSparseMat


def test_empty_and_constructor_wraps_only_circular_dimensions():
    empty = CircularSparseMat()
    assert empty.siz.tolist() == [0, 2]
    assert empty.circularDims.tolist() == []

    mat = CircularSparseMat([[0, 2], [8, 3], [2, 4]], [4, 5, 6], [3, 4], [1])
    assert mat.subs.tolist() == [[3, 2], [2, 3], [2, 4]]
    assert mat.vals[:, 0].tolist() == [4, 5, 6]
    assert mat.circularDims.tolist() == [1]


def test_dense_cast_and_copy_keep_circular_metadata():
    mat = CircularSparseMat(np.array([[1, 0], [0, 2]]), [2])
    copied = CircularSparseMat(mat)
    assert copied.circularDims.tolist() == [2]
    assert copied.isequal(mat)


def test_coordinate_and_range_indexing_wrap():
    mat = CircularSparseMat([[1, 1], [2, 2]], [10, 20], [3, 2], [1])
    np.testing.assert_array_equal(mat[[[4, 1], [-1, 2]]][:, 0], [10, 20])
    np.testing.assert_array_equal(mat[(np.array([4, -1]), 2)][:, 0], [0, 20])


def test_assignment_wraps_and_preserves_sparse_coordinates():
    mat = CircularSparseMat([], [], [4, 3], [1])
    mat[[[5, 2], [-3, 3]]] = [7, 9]
    assert mat.subs.tolist() == [[1, 2], [1, 3]]
    assert mat.vals[:, 0].tolist() == [7, 9]
    mat[[[-3, 3]]] = 0
    assert mat.nnz() == 1
    assert mat.subs.tolist() == [[1, 2]]


def test_rejects_invalid_circular_dimensions():
    with pytest.raises(ValueError):
        CircularSparseMat(np.zeros((2, 2)), [0])
    with pytest.raises(ValueError):
        CircularSparseMat([[1, 1]], [1], [2, 2], [1, 1])


def test_normalize_maps_circular_dimensions_after_squeezing():
    mat = CircularSparseMat([[1, 2, 1]], [3], [4, 5, 1], [2])
    normalized = mat.normalize(True, True, True, [1, 2, 0])
    assert normalized.siz.tolist() == [4, 5]
    assert normalized.subs.tolist() == [[1, 2]]
    assert normalized.circularDims.tolist() == [2]


def test_reshape_returns_circular_type_and_clears_axis_metadata():
    mat = CircularSparseMat([[2, 3]], [7], [2, 3], [1])

    reshaped = mat.reshape(3, 2, 1)

    assert isinstance(reshaped, CircularSparseMat)
    assert reshaped.siz.tolist() == [3, 2]
    assert reshaped.circularDims.tolist() == []
    assert reshaped.get_coords([[3, 2]]).item() == 7


def test_permute_remaps_circular_axes_by_original_dimension():
    mat = CircularSparseMat([[2, 3, 4]], [9], [2, 3, 4], [1, 3])

    permuted = mat.permute([3, 1, 2])

    assert isinstance(permuted, CircularSparseMat)
    assert permuted.siz.tolist() == [4, 2, 3]
    assert permuted.circularDims.tolist() == [1, 2]
    assert permuted.get_coords([[4, 2, 3]]).item() == 9


def test_squeeze_drops_singleton_axes_and_remaps_remaining_circular_axes():
    mat = CircularSparseMat([[1, 2, 1, 4]], [11], [1, 3, 1, 4], [2, 3, 4])

    squeezed = mat.squeeze()

    assert isinstance(squeezed, CircularSparseMat)
    assert squeezed.siz.tolist() == [3, 4]
    assert squeezed.circularDims.tolist() == [1, 2]
    assert squeezed.get_coords([[2, 4]]).item() == 11


def test_sum_uses_wholecell_remaining_dimensions_for_circular_metadata():
    mat = CircularSparseMat([[1, 1, 1], [2, 2, 3]], [2, 5], [2, 3, 4], [1])

    reduced = mat.sum(2)

    assert isinstance(reduced, CircularSparseMat)
    assert reduced.siz.tolist() == [2, 1, 4]
    # SparseMat.m:2559 supplies all remaining dimensions to the constructor.
    assert reduced.circularDims.tolist() == [1, 3]
    np.testing.assert_array_equal(
        reduced.full().ravel(order="F"), [2, 0, 0, 0, 0, 5, 0, 0]
    )
    untouched = mat.sum(4)
    assert isinstance(untouched, CircularSparseMat)
    assert untouched.circularDims.tolist() == [1]


def test_tprod_contracts_and_preserves_uncontracted_circular_axes():
    left = CircularSparseMat(np.array([[1, 2], [3, 4]]), [1])
    right = CircularSparseMat(np.array([[5, 6], [7, 8]]), [2])
    product = left.tprod(right, 2, 1)
    np.testing.assert_array_equal(
        product.full(), np.array([[19, 22], [43, 50]])
    )
    assert product.circularDims.tolist() == [1, 2]


def test_tprod_handles_nan_and_infinity_times_implicit_zero():
    from wholecell_py.sparsemat import SparseMat

    a = SparseMat([[1, 1], [2, 1]], [np.inf, 2.0], [2, 1])
    b = SparseMat([[1, 1]], [3.0], [1, 2])
    result = a.tprod(b, 2, 1)
    assert np.isinf(result.get_coords([[1, 1]]).item())
    assert np.isnan(result.get_coords([[1, 2]]).item())


def test_tprod_retains_right_operand_circular_axis_when_left_is_plain():
    from wholecell_py.sparsemat import SparseMat

    left = SparseMat(np.array([[2, 0], [0, 3]]))
    right = CircularSparseMat(np.array([[1, 4], [0, 2]]), [2])
    result = left.tprod(right, 2, 1)
    assert isinstance(result, CircularSparseMat)
    assert result.circularDims.tolist() == [2]
