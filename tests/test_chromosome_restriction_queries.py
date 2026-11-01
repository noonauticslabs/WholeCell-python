import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome_restriction_queries import (
    ChromosomeRestrictionQueries,
)


class DR:
    RM_MunI_RecognitionSites = np.array([[2, 3, 4]])
    RM_MunI_MethylatedPositions = np.array([1, 2])
    RM_MunI_RestrictionPositions = np.array([1, 3])


class Metabolite:
    m6ADIndexs = 9


class State(ChromosomeRestrictionQueries):
    sequenceLen = 10
    nCompartments = 4
    dnaRepair = DR()
    metabolite = Metabolite()

    def isRegionPolymerized(self, coords, length, return_extent, *args):
        return np.ones(len(coords), dtype=bool), np.arange(1, len(coords) + 1)


def test_hemi_mun_i_returns_unmethylated_positions_across_four_strands():
    state = State()
    # First methyl position is methylated on strand 1 only; the site is hemi.
    state.damagedBases = CircularSparseMat([[2, 1]], [9], [10, 4], 1)
    result = state.calcHemiunmethylatedMunIRMSites()
    coords, vals = result.find()
    assert {tuple(x) for x in coords.tolist()} == {(3, 2)}
    assert np.asarray(vals).reshape(-1).tolist() == [1]


def test_restrictable_mun_i_preserves_matlab_break_mask_axis_order():
    state = State()
    # The source's linear logical assignment pairs this strand-2 cut break with
    # the first strand-3 damage element. Reversing the mask axes leaves that
    # damage uncleared and incorrectly removes the chromosome-2 cut sites.
    state.damagedBases = CircularSparseMat([], [], [10, 4], 1)
    state.damagedSites_shifted_incm6AD = CircularSparseMat(
        [[2, 3]], [5], [10, 4], 1
    )
    state.strandBreaks = CircularSparseMat([[4, 2]], [True], [10, 4], 1)

    coords, values = state.calcRestrictableMunIRMSites().find()

    assert {tuple(x) for x in coords.tolist()} == {
        (2, 1),
        (2, 3),
        (4, 2),
        (4, 4),
    }
    assert np.asarray(values, dtype=bool).all()
