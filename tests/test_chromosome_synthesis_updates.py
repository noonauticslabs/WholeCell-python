import numpy as np
import pytest

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.side_effect import SimulationStateSideEffect
from wholecell_py.states.chromosome_derived import ChromosomeDerived
from wholecell_py.states.chromosome_mass_view import ChromosomeMassView
from wholecell_py.states.chromosome_polymerized_queries import (
    ChromosomePolymerizedQueries,
)
from wholecell_py.states.chromosome_synthesis_updates import (
    ChromosomeSynthesisUpdates,
)


class State(
    ChromosomeSynthesisUpdates,
    ChromosomePolymerizedQueries,
    ChromosomeDerived,
    ChromosomeMassView,
):
    sequenceLen = 20
    nCompartments = 4
    dnaStrandedness_ssDNA = 1
    dnaStrandedness_dsDNA = 2
    strandIndexs_template = np.array([1, 4])
    strandIndexs_nonTemplate = np.array([2, 3])
    strandIndexs_ch1 = np.array([1, 2])
    strandIndexs_ch2 = np.array([3, 4])
    relaxedBasesPerTurn = 10

    @property
    def singleStrandedRegions(self):
        return self.calcSingleStrandedRegions()

    @property
    def doubleStrandedRegions(self):
        return self.calcDoubleStrandedRegions()

    @property
    def unpolymerizedRegions(self):
        return self.calcUnpolymerizedRegions()


def state():
    s = State()
    s.polymerizedRegions = CircularSparseMat([[1, 1]], [20], [20, 4], [1])
    s.linkingNumbers = CircularSparseMat([[1, 1]], [2], [20, 4], [1])
    for name in (
        "monomerBoundSites",
        "complexBoundSites",
        "gapSites",
        "abasicSites",
        "damagedSugarPhosphates",
        "damagedBases",
        "intrastrandCrossLinks",
        "strandBreaks",
        "hollidayJunctions",
    ):
        setattr(s, name, CircularSparseMat([], [], [20, 4], [1]))
    return s


def test_polymerization_adds_opposite_strand_and_linking_number():
    s = state()
    effects = s.setRegionPolymerized([[4, 1]], [3])
    assert isinstance(effects, SimulationStateSideEffect)
    assert s.polymerizedRegions.get_coords([[4, 2]])[0, 0] == 3
    assert s.linkingNumbers.get_coords([[4, 1], [4, 2]]).reshape(
        -1
    ).tolist() == pytest.approx([0.3, 0.3])


def test_polymerization_rejects_missing_template_without_mutation():
    s = state()
    s.polymerizedRegions = CircularSparseMat([], [], [20, 4], [1])
    before = s.polymerizedRegions.copy()
    with pytest.raises(ValueError, match="without a template"):
        s.setRegionPolymerized([[1, 1]], [3])
    assert s.polymerizedRegions.isequal(before)


def test_shift_strand_moves_only_selected_sparse_coordinates():
    matrix = CircularSparseMat(
        [[2, 1], [4, 1], [4, 2]], [7, 8, 9], [10, 4], [1]
    )
    shifted = State.shiftStrandToNewChromosome(matrix, 3, 2, 1, 3)
    assert shifted.find()[0].tolist() == [[2, 1], [4, 2], [4, 3]]
    assert shifted.find()[1].reshape(-1).tolist() == [7, 9, 8]


def test_unwinding_both_origin_ends_preserves_remaining_parent_strand():
    s = state()
    s.polymerizedRegions = CircularSparseMat(
        [[1, 1], [1, 2]], [20, 20], [20, 4], 1
    )
    s.linkingNumbers = CircularSparseMat([[1, 1], [1, 2]], [2, 2], [20, 4], 1)
    s.setRegionUnwound(20, -3)
    assert s.polymerizedRegions.get_coords([[1, 2]])[0, 0] == 17
    s.setRegionUnwound(1, 3)
    assert s.polymerizedRegions.get_coords([[4, 2]])[0, 0] == 14
    assert np.sum(s.polymerizedRegions.find()[1]) == 40
