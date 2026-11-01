import numpy as np
import pytest

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome_derived import ChromosomeDerived
from wholecell_py.states.chromosome_mass_view import ChromosomeMassView
from wholecell_py.states.chromosome_regions import ChromosomeRegions


class State(ChromosomeDerived, ChromosomeRegions, ChromosomeMassView):
    sequenceLen = 12
    nCompartments = 4
    relaxedBasesPerTurn = 10.0
    equilibriumSuperhelicalDensity = -0.06
    supercoiledSuperhelicalDensityTolerance = 0.02


def state_with_regions():
    state = State()
    state.polymerizedRegions = CircularSparseMat(
        [[2, 1], [5, 2], [8, 1], [10, 2], [2, 3], [5, 4], [8, 3]],
        [5, 3, 2, 2, 5, 3, 2],
        [12, 4],
        [1],
    )
    return state


def test_unpolymerized_single_and_double_stranded_regions_sparse_circular():
    state = state_with_regions()
    up, ul = state.calcUnpolymerizedRegions().find()
    assert up.tolist() == [
        [1, 1],
        [7, 1],
        [10, 1],
        [1, 2],
        [8, 2],
        [12, 2],
        [1, 3],
        [7, 3],
        [10, 3],
        [1, 4],
        [8, 4],
    ]
    assert ul.reshape(-1).tolist() == [1, 1, 3, 4, 2, 1, 1, 1, 3, 4, 5]

    ss, sl = state.calcSingleStrandedRegions().find()
    assert ss.tolist() == [
        [2, 1],
        [8, 1],
        [7, 2],
        [10, 2],
        [2, 3],
        [8, 3],
        [7, 4],
    ]
    assert sl.reshape(-1).tolist() == [3, 2, 1, 2, 3, 2, 1]

    ds, dl = state.calcDoubleStrandedRegions().find()
    assert ds.tolist() == [[5, 1], [5, 2], [5, 3], [5, 4]]
    assert dl.reshape(-1).tolist() == [2, 2, 2, 2]


def test_polymerized_feature_copies_copy_counts_and_ploidy():
    state = state_with_regions()
    state.gene = type(
        "Gene",
        (),
        {
            "startCoordinates": np.array([2, 8]),
            "strands": np.array([1, 1]),
            "lengths": np.array([3, 2]),
        },
    )()
    state.transcriptionUnitStartCoordinates = np.array([2])
    state.transcriptionUnitStrands = np.array([1])
    state.transcriptionUnitLengths = np.array([3])
    np.testing.assert_array_equal(
        state.calcPolymerizedGenes(), [[True, True], [True, True]]
    )
    np.testing.assert_array_equal(
        state.calcPolymerizedTranscriptionUnits(), [[True, True]]
    )
    state.polymerizedGenes = np.array([[True, False], [True, True]])
    state.polymerizedTranscriptionUnits = np.array([[True, True]])
    np.testing.assert_array_equal(state.calcGeneCopyNumbers(), [[1], [2]])
    np.testing.assert_array_equal(
        state.getTranscriptionUnitCopyNumbers(), [[2]]
    )
    assert state.calcPloidy() == pytest.approx(22 / 24)


def test_accessible_counts_and_feature_coordinates_use_both_chromosome_copies():
    state = State()
    state.accessibleGenes = np.array([[True, False], [True, True]])
    state.accessibleTranscriptionUnits = np.array([[True, True]])
    np.testing.assert_array_equal(
        state.calcCopyNumbers_Accessible(), [[1], [2]]
    )
    np.testing.assert_array_equal(
        state.calcTranscriptionUnitCopyNumbers_Accessible(), [[2]]
    )
    state.gene = type(
        "Gene",
        (),
        {
            "startCoordinates": np.array([2, 8]),
            "strands": np.array([1, 2]),
            "lengths": np.array([3, 2]),
        },
    )()
    state.transcriptionUnitStartCoordinates = np.array([2, 8])
    state.transcriptionUnitStrands = np.array([1, 2])
    state.transcriptionUnitLengths = np.array([3, 2])
    calls = []

    def accessibility(coords, lengths, *args):
        calls.append((coords, lengths))
        return np.array([True, False, False, True]), None, None, None

    state.isRegionAccessible = accessibility
    np.testing.assert_array_equal(
        state.calcAccessibleGenes(), [[True, False], [False, True]]
    )
    np.testing.assert_array_equal(
        state.calcAccessibleTranscriptionUnits(),
        [[True, False], [False, True]],
    )
    for coords, lengths in calls:
        np.testing.assert_array_equal(coords, [[2, 1], [8, 2], [2, 3], [8, 4]])
        np.testing.assert_array_equal(lengths, [3, 2, 3, 2])


def test_linking_number_derived_sparse_fields():
    state = state_with_regions()
    state.doubleStrandedRegions = state.calcDoubleStrandedRegions()
    state.linkingNumbers = CircularSparseMat(
        [[5, 1], [5, 2], [5, 3], [5, 4]],
        [0.18, 0.21, 0.18, 0.21],
        [12, 4],
        [1],
    )
    min_free = state.calcLinkingNumbers_minFreeEnergy()
    p, v = min_free.find()
    assert p.tolist() == [[5, 1], [5, 2], [5, 3], [5, 4]]
    np.testing.assert_allclose(v.reshape(-1), [0.2, 0.2, 0.2, 0.2])
    state.linkingNumbers_minFreeEnergy = min_free
    delta = state.calcSupercoils()
    np.testing.assert_allclose(
        delta.find()[1].reshape(-1), [-0.02, 0.01, -0.02, 0.01]
    )
    rho = state.calcSuperhelicalDensity()
    np.testing.assert_allclose(
        rho.find()[1].reshape(-1), [-0.1, 0.05, -0.1, 0.05]
    )
