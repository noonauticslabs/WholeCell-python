import numpy as np
import pytest

from wholecell_py.sparsemat import SparseMat


def test_constructor_last_write_zero_removal_and_copy():
    a = SparseMat([[2, 1], [1, 2], [2, 1]], [4, 3, 9], [2, 2])
    assert a.nnz() == 2
    assert a.get_coords([[2, 1], [1, 2]]).ravel().tolist() == [9, 3]
    b = a.copy()
    b[[2, 1]] = 0
    assert a.nnz() == 2 and b.nnz() == 1


def test_dense_column_major_linear_indices_and_shape():
    a = SparseMat(np.array([[1, 0, 2], [0, 3, 0]]))
    assert a.size().tolist() == [2, 3]
    np.testing.assert_array_equal(a.sub2ind(a.subs).ravel(), [1, 4, 5])
    np.testing.assert_array_equal(
        a.ind2sub([1, 4, 5]), [[1, 1], [2, 2], [1, 3]]
    )
    np.testing.assert_array_equal(a.full(), [[1, 0, 2], [0, 3, 0]])


def test_ranges_reshape_permute_sum_and_value_semantics():
    a = SparseMat([[1, 1, 1], [2, 1, 2]], [2, 5], [2, 2, 2])
    np.testing.assert_array_equal(
        a.take([1, 2], [1], [1, 2]), [[2], [0], [0], [5]]
    )
    b = a.reshape(4, 2)
    assert b.get_coords([[1, 1], [2, 2]]).ravel().tolist() == [2, 5]
    c = a.permute([3, 2, 1])
    assert c.size().tolist() == [2, 2, 2]
    assert a.sum(1).full().sum() == 7


def test_invalid_coordinates_are_rejected():
    with pytest.raises(ValueError):
        SparseMat([[0, 1]], [1], [2, 2])


def test_comparisons_include_equal_implicit_zeros_like_source_sparsemat():
    a = SparseMat([[1, 1]], [2.0], [2, 2])
    b = SparseMat([[1, 1], [2, 2]], [2.0, 3.0], [2, 2])
    np.testing.assert_array_equal(
        (a == b).full(), [[True, True], [True, False]]
    )
    np.testing.assert_array_equal((a <= b).full(), np.ones((2, 2), bool))
    np.testing.assert_array_equal(
        (a == 0).full(), [[False, True], [True, True]]
    )


def test_nonzero_or_nan_implicit_arithmetic_matches_source_sparsemat():
    a = SparseMat([[1, 1]], [2.0], [2, 2])
    b = SparseMat([[1, 1]], [4.0], [2, 2])
    np.testing.assert_array_equal((a + 3).full(), [[5, 3], [3, 3]])
    np.testing.assert_allclose(
        (a / b).full(), [[0.5, np.nan], [np.nan, np.nan]], equal_nan=True
    )
    np.testing.assert_allclose(
        (a * np.inf).full(),
        [[np.inf, np.nan], [np.nan, np.nan]],
        equal_nan=True,
    )


def test_sparse_order_matches_matlab_linear_indices_in_four_dimensions():
    rng = np.random.default_rng(41)
    shape = np.array([11, 4, 3, 2])
    coords = rng.integers(1, shape + 1, size=(100, 4))
    values = np.arange(1, 101)
    matrix = SparseMat(coords, values, shape)
    expected = {tuple(c): v for c, v in zip(coords, values)}
    stride = np.cumprod(np.r_[1, shape[:-1]])
    order = sorted(
        expected, key=lambda c: sum((x - 1) * s for x, s in zip(c, stride))
    )
    actual_coords, actual_values = matrix.find()
    assert actual_coords.tolist() == [list(c) for c in order]
    assert actual_values.ravel().tolist() == [expected[c] for c in order]


def test_sparse_binary_coordinate_union_matches_dense_arithmetic():
    rng = np.random.default_rng(123)
    for shape in ((17, 4), (7, 3, 2)):
        a = rng.integers(-4, 5, size=shape)
        b = rng.integers(-4, 5, size=shape)
        a[rng.random(shape) < 0.8] = 0
        b[rng.random(shape) < 0.8] = 0
        sa, sb = SparseMat(a), SparseMat(b)
        np.testing.assert_array_equal((sa + sb).full(), a + b)
        np.testing.assert_array_equal((sa - sb).full(), a - b)
        np.testing.assert_array_equal((sa * sb).full(), a * b)
        empty = SparseMat(np.zeros(shape))
        np.testing.assert_array_equal((empty + sa).full(), a)


def test_coordinate_reads_reflect_public_array_mutations_and_invalid_rows():
    matrix = SparseMat([[1, 2], [2, 1]], [4, 7], [2, 2])
    # Keep public-array mutability coherent: value updates are immediately
    # visible, and a noncanonical coordinate order still uses last-write wins.
    matrix.vals[0, 0] = 9
    matrix.subs[:] = [[2, 1], [1, 2]]
    np.testing.assert_array_equal(
        matrix.get_coords([[1, 2], [2, 1]]).ravel(), [4, 9]
    )
    with pytest.raises(IndexError, match="out of range"):
        matrix.get_coords([[3, 1]])


def test_coordinate_reads_handle_duplicate_public_coordinates_last_wins():
    matrix = SparseMat([[1, 1], [2, 2]], [4, 7], [2, 2])
    matrix.subs[1] = matrix.subs[0]
    np.testing.assert_array_equal(matrix.get_coords([[1, 1]]).ravel(), [7])


def test_coordinate_read_dtype_matches_numpy_list_inference():
    matrix = SparseMat([[1, 1]], np.asarray([2], dtype=np.float32), [2, 2])
    assert matrix.get_coords([[1, 1]]).dtype == np.dtype(np.float32)
    assert matrix.get_coords([[1, 1], [2, 2]]).dtype == np.dtype(np.float64)
    assert matrix.get_coords([[2, 2]]).dtype == np.dtype(np.int64)


def test_coordinate_inputs_reject_fractional_and_nonfinite_indices():
    matrix = SparseMat([[1, 1]], [2.0], [2, 2])
    with pytest.raises(ValueError, match="finite integers"):
        matrix.get_coords([[1.5, 1]])
    with pytest.raises(ValueError, match="finite integers"):
        SparseMat([[np.inf, 1]], [2], [2, 2])


def test_boolean_sum_counts_true_values_numerically():
    matrix = SparseMat(np.asarray([[True, False], [True, True]], dtype=bool))

    result = matrix.sum(1)

    np.testing.assert_array_equal(result.full(), [[2.0, 1.0]])
    assert result.vals.dtype == np.dtype(np.float64)


def test_sum_beyond_ndims_returns_independent_value_copy():
    matrix = SparseMat([[1, 2]], [3], [2, 2])

    result = matrix.sum(3)

    assert result is not matrix
    np.testing.assert_array_equal(result.siz, matrix.siz)
    np.testing.assert_array_equal(result.subs, matrix.subs)
    np.testing.assert_array_equal(result.vals, matrix.vals)
    result.vals[0, 0] = 9
    assert matrix.vals[0, 0] == 3


def test_collapse_without_dimension_sums_only_stored_values():
    matrix = SparseMat([[1, 1], [2, 2]], [4, 7], [3, 3])

    assert matrix.collapse() == 11


def test_squeeze_singleton_leading_dimension_pads_coordinates_once():
    matrix = SparseMat([[1, 1], [1, 4]], [2, 5], [1, 4])

    result = matrix.squeeze()

    np.testing.assert_array_equal(result.siz, [4, 1])
    np.testing.assert_array_equal(result.subs, [[1, 1], [4, 1]])
    np.testing.assert_array_equal(result.full(), [[2], [0], [0], [5]])


def test_squeeze_retains_two_non_singleton_dimensions_after_middle_singleton():
    matrix = SparseMat([[2, 1, 3]], [8], [2, 1, 3])

    result = matrix.squeeze()

    np.testing.assert_array_equal(result.siz, [2, 3])
    np.testing.assert_array_equal(result.subs, [[2, 3]])
    assert result.get_coords([[2, 3]])[0, 0] == 8


def test_full_preserves_dtype_for_empty_logical_dense_input():
    matrix = SparseMat(np.zeros((2, 3), dtype=bool))
    assert matrix.full().dtype == np.dtype(bool)


def test_normalize_flag_behavior_and_value_semantics():
    matrix = SparseMat([[1, 1], [2, 2]], [2, 3], [2, 2])
    matrix.subs = np.array([[2, 1], [1, 2], [1, 1]])
    matrix.vals = np.array([[0], [3], [2]])

    result = matrix.normalize(False, True, True, [1, 2])

    np.testing.assert_array_equal(result.subs, [[1, 1], [1, 2]])
    np.testing.assert_array_equal(result.vals.ravel(), [2, 3])
    # Normalization returns an independent value, as the MATLAB class does.
    assert matrix.subs.tolist() == [[2, 1], [1, 2], [1, 1]]


def test_tprod_multidimensional_contract_matches_dense_reference():
    rng = np.random.default_rng(71)
    a = rng.integers(-2, 3, size=(2, 3, 4))
    b = rng.integers(-2, 3, size=(3, 4, 5))
    result = SparseMat(a).tprod(SparseMat(b), [2, 3], [1, 2])
    np.testing.assert_array_equal(
        result.full(), np.tensordot(a, b, axes=([1, 2], [0, 1]))
    )
