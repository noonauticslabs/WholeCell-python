import numpy as np
from wholecell_py.polymerize import polymerize
from wholecell_py.randstream import RandStream


def test_base_limited_allocation_preserves_usage_and_stops_starved_sequences():
    seq = np.array([list("ACG"), list("AAC")])
    progress, remaining, costs, energy, spent = polymerize(
        seq, np.array([2, 3, 3]), "ACG", " ", randStream=RandStream(seed=1)
    )
    np.testing.assert_array_equal(progress[:, 0], [3, 1])
    np.testing.assert_array_equal(costs[:, 0], [2, 1, 1])
    np.testing.assert_array_equal(remaining, [0, 2, 2])
    assert energy == spent == 0


def test_energy_limited_last_base_is_assigned_to_exactly_one_polymer():
    p, remaining, costs, energy, spent = polymerize(
        np.array([[1, 1], [1, 1]]),
        [10, 0],
        [1, 2],
        0,
        6,
        2,
        RandStream(seed=1),
    )
    assert sorted(p[:, 0]) == [1, 2]
    assert remaining[0] == 7 and costs[0, 0] == 3
    assert energy == 0 and spent == 6


def test_padding_and_negative_amount_early_exit():
    seq = np.array([[1, 0, 0], [1, 1, 0]])
    p, *_ = polymerize(seq, [10, 0], [1, 2], 0)
    np.testing.assert_array_equal(p[:, 0], [1, 2])
    p, remaining, costs, energy, spent = polymerize(seq, [-1], [1], 0, 10, 1)
    assert not p.any() and not costs.any() and remaining[0] == -1
    assert energy == 10 and spent == 0


def test_original_matlab_single_base_energy_edge_case_is_preserved():
    p, remaining, costs, energy, spent = polymerize(
        np.array([[1, 1], [1, 1]]), [10], [1], 0, 6, 2, RandStream(seed=1)
    )
    # Verified on licensed R2026b original util/polymerize.m, not a physical
    # correction: its MATLAB sum collapses the sole base row to a scalar.
    np.testing.assert_array_equal(p[:, 0], [2, 2])
    assert remaining[0] == 6 and costs[0, 0] == 4
    assert energy == -2 and spent == 8
