import numpy as np

from wholecell_py.randstream import RandStream


def test_randi_bound_one_does_not_advance_stream():
    a = RandStream(seed=8)
    b = RandStream(seed=8)
    assert np.array_equal(a.randi(1, 2, 3), np.ones((2, 3), dtype=int))
    assert np.array_equal(a.rand(3, 2), b.rand(3, 2))


def test_snapshot_reset_and_one_based_permutation():
    rng = RandStream(seed=13)
    state = rng.save()
    x = rng.rand(8)
    rng.restore(state)
    assert np.array_equal(x, rng.rand(8))
    rng.reset()
    assert np.array_equal(x, rng.rand(8))
    perm = rng.randperm(30, 7)
    assert len(set(perm)) == 7 and np.all((perm >= 1) & (perm <= 30))


def test_weighted_sampling_and_zero_weights():
    rng = RandStream(seed=5)
    draws = rng.randsample(4, 500, True, [0, 0, 3, 1])
    assert set(draws) == {3, 4}
    assert rng.randsample(4, 5, True, [0, 0, 0, 0]).size == 0
    sample = rng.randsample(6, 4, False, [1, 2, 3, 4, 5, 6])
    assert len(set(sample)) == 4


def test_weighted_without_replacement_does_not_mutate_input_weights():
    rng = RandStream(seed=91)
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    original = weights.copy()

    sample = rng.randsample(4, 3, False, weights)

    assert len(np.unique(sample)) == 3
    assert np.array_equal(weights, original)


def test_randsample_defaults_to_without_replacement():
    rng = RandStream(seed=92)

    sample = rng.randsample(4, 4)

    assert np.array_equal(np.sort(sample), np.arange(1, 5))


def test_randcounts_conserves_total_and_row_selection_is_sorted_one_based():
    rng = RandStream(seed=19)
    counts = np.array([10, 0, 20, 3])
    result = rng.randCounts(counts, 12)
    assert result.sum() == 12
    assert np.all(result <= counts)
    mat = np.arange(20).reshape(10, 2)
    rows, idx = rng.randomlySelectNRows(mat, 4)
    assert rows.shape == (4, 2)
    assert np.all(np.diff(idx) > 0)
    assert np.array_equal(rows, mat[idx - 1])


def test_stochastic_round_expected_bounds_and_matlab_shapes():
    rng = RandStream(seed=2)
    assert rng.rand(3).shape == (3, 3)
    values = rng.stochasticRound(np.array([1.2, 2.8]))
    assert np.all(np.isin(values, [1, 2, 3]))


def test_random_distribution_aliases_used_by_simulation():
    rng = RandStream(seed=17)
    assert rng.random("poiss", 4, 3, 1).shape == (3, 1)
    assert rng.random("binomial", np.full((2, 3), 5), 0.5).shape == (2, 3)
    assert rng.random("norm", 2, 0.1).shape == (1, 1)
    assert rng.random("exponential", 2, 4, 1).shape == (4, 1)


def test_mnrnd_preserves_row_distributions_and_does_not_renormalize_invalid_rows():
    rng = RandStream(seed=123)
    result = rng.mnrnd([20, 30], [[0.25, 0.75], [0.0, 1.0]])
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.sum(axis=1), [20, 30])
    assert np.isnan(rng.mnrnd(5, [0.2, 0.2])).all()


def test_mnrnd_bad_probability_rows_are_nan_per_matlab_contract():
    rng = RandStream(seed=1)
    result = rng.mnrnd([4, 6], [[0.5, 0.5], [0.2, 0.2]])
    assert np.isfinite(result[0]).all()
    assert np.isnan(result[1]).all()


def test_stochastic_round_preserves_scalar_vector_and_empty_shapes():
    rng = RandStream(seed=2)
    for shape in ((), (5,), (0,), (3, 1), (2, 3, 1)):
        values = np.full(shape, 2.5)
        result = rng.stochasticRound(values)
        assert result.shape == values.shape
        assert np.all((result == 2) | (result == 3))


def test_stochastic_round_preserves_infinite_resource_limits():
    rng = RandStream(seed=1)
    values = rng.stochasticRound(np.array([np.inf, -np.inf, np.nan, 3.0]))
    assert np.isposinf(values[0])
    assert np.isneginf(values[1])
    assert np.isnan(values[2])
    assert values[3] == 3


def test_mnrnd_scalar_trials_expand_across_probability_rows():
    from wholecell_py.randstream import RandStream

    actual = RandStream(seed=2).mnrnd(7, [[1, 0], [0, 1], [0.5, 0.5]])
    np.testing.assert_array_equal(actual[:2], [[7, 0], [0, 7]])
    np.testing.assert_array_equal(actual.sum(axis=1), [7, 7, 7])


def test_mnrnd_explicit_repeat_count_and_column_probability_vector():
    from wholecell_py.randstream import RandStream

    actual = RandStream(seed=2).mnrnd(5, [[0], [1]], 3)
    np.testing.assert_array_equal(actual, [[0, 5], [0, 5], [0, 5]])
