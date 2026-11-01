from types import SimpleNamespace

import numpy as np
import pytest

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.processes.replication_updates import ReplicationUpdates


@pytest.mark.parametrize(
    "direction,fragment_index,expected",
    [
        (0, 1, [[100, 3]]),  # first fragment on the left-moving fork
        (1, 1, [[100, 2]]),  # first fragment on the right-moving fork
        (0, 2, [[19, 3]]),  # prior primase location minus one
        (1, 2, [[20, 2]]),  # prior primase location
    ],
)
def test_terminate_okazaki_fragment_places_break_on_correct_strand(
    direction, fragment_index, expected
):
    chromosome = SimpleNamespace(
        sequenceLen=100,
        strandBreaks=CircularSparseMat([], [], [100, 3], 1),
    )
    locations = (
        [np.array([20, 40])]
        if fragment_index == 1
        else [np.array([20, 40, 80])]
    )
    process = SimpleNamespace(
        chromosome=chromosome,
        isAnyHelicaseBound=True,
        leadingStrandElongating=np.array([True, True]),
        okazakiFragmentIndex=np.array([0, 0]),
        okazakiFragmentProgress=np.array([0, 0]),
        okazakiFragmentLength=np.array([0, 0]),
        helicasePosition=np.array([1, 90]),
        enzymeDNAFootprints=np.array([1, 1, 1]),
        enzymeDNAFootprints3Prime=np.array([1, 1, 1]),
        enzymeDNAFootprints5Prime=np.array([1, 1, 1]),
        enzymeIndexs_helicase=1,
        enzymeIndexs_coreBetaClampPrimase=2,
        enzymeIndexs_core=3,
        areLaggingStrandSSBSitesBound=np.array([False, False]),
        laggingBackupBetaClampPosition=np.array([0, 0]),
        leadingPolymerasePosition=np.array([1, 90]),
        laggingPolymerasePosition=np.array([0, 0]),
        primaseBindingLocations=locations * 2,
        laggingStrandIndexs=np.array([2, 3]),
        leadingStrandIndexs=np.array([1, 4]),
        startingOkazakiLoopLength=10,
        enzymeIndexs_primase=4,
        enzymeIndexs_betaClamp=5,
        enzymeIndexs_betaClampMonomer=6,
        enzymeIndexs_coreBetaClampGammaComplex=7,
        enzymeIndexs_2coreBetaClampGammaComplexPrimase=8,
        enzymes=np.zeros(8, dtype=int),
        releaseProteinFromSites=lambda *args: None,
        bindProteinToChromosome=lambda *args: (np.array([True]),),
        modifyProteinOnChromosome=lambda *args: None,
    )
    process.okazakiFragmentIndex[direction] = fragment_index
    process.okazakiFragmentProgress[direction] = 1
    process.okazakiFragmentLength[direction] = 1
    process.areLaggingStrandSSBSitesBound[direction] = True
    if direction == 0:
        process.laggingBackupBetaClampPosition[direction] = (
            locations[0][fragment_index] + 1
        )
    else:
        process.laggingBackupBetaClampPosition[direction] = (
            locations[0][fragment_index] + 2
        )

    ReplicationUpdates.terminateOkazakiFragment(process)

    assert chromosome.strandBreaks.get_coords(expected).item() == 1
    other_strand = 2 if expected[0][1] == 3 else 3
    coordinate = expected[0][0]
    assert (
        chromosome.strandBreaks.get_coords([[coordinate, other_strand]]).item()
        == 0
    )


def test_terminate_replication_restores_enzyme_composition_as_column():
    complex_index = 5  # one-based enzyme index
    composition = np.zeros((13, 13), dtype=int)
    composition[[1, 4, 8], complex_index - 1] = [2, 1, 3]
    enzymes = np.arange(13, dtype=int).reshape(13, 1) + 10
    original_enzymes = enzymes.copy()
    chromosome = SimpleNamespace(
        strandBreaks=CircularSparseMat([], [], [100, 3], 1)
    )
    released = []

    def release(enzyme_index, rate, protected_positions, protected_lengths):
        released.append(enzyme_index)
        return np.array([[20, 2], [80, 3]])

    process = SimpleNamespace(
        leadingStrandElongating=np.array([True, True]),
        strandPolymerized=np.array([True, True]),
        laggingStrandElongating=np.array([False, False]),
        terCPosition=50,
        chromosome=chromosome,
        enzymeIndexs_helicase=2,
        enzymeIndexs_2coreBetaClampGammaComplexPrimase=complex_index,
        enzymeComposition=composition,
        enzymes=enzymes,
        releaseProteinFromChromosome=release,
    )

    ReplicationUpdates.terminateReplication(process)

    expected = original_enzymes + 2 * composition[:, [complex_index - 1]]
    expected[complex_index - 1, 0] -= 2
    np.testing.assert_array_equal(process.enzymes, expected)
    assert process.enzymes.shape == (13, 1)
    assert released == [2, complex_index]
    np.testing.assert_array_equal(
        chromosome.strandBreaks.get_coords([[50, 2], [50, 3]]), [[1], [1]]
    )
