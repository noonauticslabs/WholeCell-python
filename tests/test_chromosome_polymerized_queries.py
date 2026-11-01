import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome_derived import ChromosomeDerived
from wholecell_py.states.chromosome_mass_view import ChromosomeMassView
from wholecell_py.states.chromosome_polymerized_queries import (
    ChromosomePolymerizedQueries,
)


class State(
    ChromosomePolymerizedQueries, ChromosomeDerived, ChromosomeMassView
):
    sequenceLen = 12
    nCompartments = 4
    dnaStrandedness_ssDNA = 1
    dnaStrandedness_dsDNA = 2

    @property
    def singleStrandedRegions(self):
        return self.calcSingleStrandedRegions()

    @property
    def doubleStrandedRegions(self):
        return self.calcDoubleStrandedRegions()

    @property
    def unpolymerizedRegions(self):
        return self.calcUnpolymerizedRegions()


def make_state():
    state = State()
    state.polymerizedRegions = CircularSparseMat(
        [[10, 1], [2, 2], [10, 2], [2, 3], [10, 4]],
        [3, 3, 3, 3, 3],
        [12, 4],
        [1],
    )
    return state


def test_polymerized_intersections_and_ori_c_wrap():
    state = make_state()
    tf, idx, p, extent = state.isRegionPolymerized(
        [[11, 1], [10, 1], [4, 1]], [3, 2, 2], True
    )
    assert tf.tolist() == [True, True, True]
    assert idx.tolist() == [1, 2, 3]
    assert p.tolist() == [[11, 1], [10, 1], [4, 1]]
    assert extent.tolist() == [2, 2, 0]
    tf, idx, p, extent = state.isRegionPolymerized(
        [[11, 1], [10, 1], [4, 1]], [2, 2, 2]
    )
    assert tf.tolist() == [True, True, False]
    assert idx.tolist() == [1, 2]
    assert p.tolist() == [[11, 1], [10, 1]]
    assert extent.tolist() == [2, 2]


def test_ss_ds_and_not_polymerized_queries_use_overlap_regions():
    state = make_state()
    assert state.isRegionSingleStranded([[10, 4]], [2])[0].tolist() == [True]
    assert state.isRegionDoubleStranded([[2, 1]], [3])[0].tolist() == [False]
    assert state.isRegionDoubleStranded([[10, 2]], [3])[0].tolist() == [True]
    tf, _, _, extent = state.isRegionNotPolymerized(
        [[1, 1], [2, 1]], [3, 2], True
    )
    assert tf.tolist() == [True, True]
    assert extent.tolist() == [3, 2]


def test_sparse_genome_size_is_not_expanded():
    state = State()
    state.sequenceLen = 10**9
    state.polymerizedRegions = CircularSparseMat(
        [[999_999_999, 1], [1, 1]], [2, 1], [10**9, 4], [1]
    )
    tf, _, _, extent = state.isRegionPolymerized([[1, 1]], [1], True)
    assert tf.tolist() == [True]
    assert extent.tolist() == [1]
