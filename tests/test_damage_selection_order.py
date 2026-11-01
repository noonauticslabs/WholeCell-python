import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome_damage_updates import (
    ChromosomeDamageUpdates,
)


def test_numeric_damage_candidates_follow_matlab_find_column_order():
    class Stream:
        def stochasticRound(self, count):
            return np.asarray(count)

        def randomlySelectNRows(self, positions, count):
            np.testing.assert_array_equal(positions, [[8, 1], [2, 2]])
            return positions[:count], np.arange(count) + 1

    state = ChromosomeDamageUpdates()
    state.damagedBases = CircularSparseMat(
        [[2, 2], [8, 1]], [7, 7], [10, 2], 1
    )
    state.damagedSites_nonRedundant = state.damagedBases.copy()
    state.randStream = Stream()
    selected, _ = state.setSiteDamaged(
        "damagedBases", 9, 1, 1, 7, "damagedBases"
    )
    np.testing.assert_array_equal(selected, [[8, 1]])
    assert state.damagedBases.get_coords([[8, 1]]).item() == 9
    assert state.damagedBases.get_coords([[2, 2]]).item() == 7


def test_zero_motif_includes_implicit_zeros_from_both_sparse_inputs():
    class Stream:
        def stochasticRound(self, count):
            assert count == 2
            return np.asarray(count)

        def randomlySelectNRows(self, positions, count):
            np.testing.assert_array_equal(positions, [[2, 1], [1, 2]])
            return positions, np.arange(len(positions)) + 1

    state = ChromosomeDamageUpdates()
    state.damagedBases = CircularSparseMat([[1, 1]], [7], [2, 2], 1)
    state.damagedSites_nonRedundant = CircularSparseMat(
        [[2, 2]], [8], [2, 2], 1
    )
    state.randStream = Stream()
    selected, _ = state.setSiteDamaged(
        "damagedBases", 9, 1, 2, 0, "damagedBases"
    )
    np.testing.assert_array_equal(selected, [[2, 1], [1, 2]])
