import numpy as np
import pytest

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome_mass_view import ChromosomeMassView


class State(ChromosomeMassView):
    sequenceLen = 12
    strandIndexs_negative = np.array([[2], [4]])


def test_merge_adjacent_regions_uses_strand_and_time_coordinates():
    state = State()
    regions = CircularSparseMat(
        [[1, 2, 1], [4, 2, 1], [7, 4, 1], [9, 2, 2]],
        [3, 2, 2, 1],
        [12, 4, 2],
        [1],
    )
    result = state.mergeAdjacentRegions(regions)
    coords, lengths = result.find()
    assert coords.tolist() == [[1, 2, 1], [7, 4, 1], [9, 2, 2]]
    assert lengths.reshape(-1).tolist() == [5, 2, 1]


def test_merge_rejects_overlapping_same_strand_intervals():
    state = State()
    regions = CircularSparseMat([[1, 2], [3, 2]], [4, 2], [12, 4], [1])
    with pytest.raises(ValueError, match="corrupt"):
        state.mergeAdjacentRegions(regions)


def test_polymerized_strand_view_unions_overlap_and_reports_intersection():
    state = State()
    state.polymerizedRegions = CircularSparseMat(
        [[2, 2, 1], [5, 4, 1]], [5, 4], [12, 4, 1], [1]
    )
    result = state.getStrandView("polymerizedRegions")
    coords, lengths = result.find()
    assert coords.tolist() == [[2, 2], [5, 4]]
    assert lengths.reshape(-1).tolist() == [7, 2]


def test_merge_groups_strands_before_positions():
    from wholecell_py.states.chromosome_synthesis_updates import (
        ChromosomeSynthesisUpdates,
    )

    regions = CircularSparseMat(
        [[1, 3], [12, 3], [1, 4], [134, 4]],
        [11, 100, 133, 100],
        [580076, 4],
        [1],
    )
    for implementation in (ChromosomeMassView, ChromosomeSynthesisUpdates):
        result = implementation().mergeAdjacentRegions(regions)
        coords, lengths = result.find()
        assert coords.tolist() == [[1, 3], [1, 4]]
        assert lengths.ravel().tolist() == [111, 233]


class Sequence:
    def strand(self, strand):
        return "ACGT"

    def subsequenceBaseCounts(self, positions, strands=None):
        return np.array([1.0, 1.0, 1.0, 1.0])


class Metabolite:
    molecularWeights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    dnmpIndexs = np.array([1, 2, 3, 4])
    dr5pIndexs = np.array([5])
    waterIndexs = np.array([6])
    unmodifiedBaseIndexs = np.array([1, 2, 3, 4])


def _dry_weight_state(time_count):
    state = State()
    state.sequenceLen = 4
    state.sequence = Sequence()
    state.metabolite = Metabolite()
    state.polymerizedRegions = CircularSparseMat(
        [[1, 2, t] for t in range(1, time_count + 1)],
        [4] * time_count,
        [4, 4, time_count],
        [1],
    )
    for name in (
        "abasicSites",
        "gapSites",
        "damagedSugarPhosphates",
        "damagedBases",
        "intrastrandCrossLinks",
        "strandBreaks",
    ):
        setattr(
            state,
            name,
            CircularSparseMat([], [], [4, 4, time_count], [1]),
        )
    return state


def test_calc_dry_weight_preserves_matlab_single_time_shape():
    state = _dry_weight_state(1)

    assert state.calcDryWeight().shape == (1, 1)


def test_calc_dry_weight_preserves_matlab_multitime_permute_shape():
    state = _dry_weight_state(2)

    value = state.calcDryWeight()

    assert value.shape == (1, 1, 2)
    np.testing.assert_allclose(value[0, 0, 0], value[0, 0, 1])
