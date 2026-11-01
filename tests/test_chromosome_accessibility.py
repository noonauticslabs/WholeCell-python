import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome import Chromosome


def state():
    c = Chromosome(
        sequenceLen=12,
        nCompartments=4,
        dnaStrandedness_dsDNA=2,
        dnaStrandedness_ssDNA=1,
        polymerizedRegions=CircularSparseMat(
            [[1, 1], [1, 2]], [12, 12], [12, 4], [1]
        ),
        monomerBoundSites=CircularSparseMat([], np.empty(0), [12, 4], [1]),
        complexBoundSites=CircularSparseMat([], np.empty(0), [12, 4], [1]),
        _testDamagedSites=CircularSparseMat([], np.empty(0), [12, 4], [1]),
        monomerDNAFootprints=np.array([3]),
        complexDNAFootprints=np.array([3]),
        monomerDNAFootprintBindingStrandedness=np.array([2]),
        complexDNAFootprintBindingStrandedness=np.array([2]),
        monomerDNAFootprintRegionStrandedness=np.array([2]),
        complexDNAFootprintRegionStrandedness=np.array([2]),
        reactionMonomerCatalysisMatrix=np.zeros((1, 1)),
        reactionComplexCatalysisMatrix=np.zeros((1, 1)),
        reactionThresholds=np.array([1]),
        reactionBoundMonomer=np.array([1]),
        reactionBoundComplex=np.array([1]),
    )
    c.calcDamagedSites = lambda: c._testDamagedSites
    return c


def test_footprint_and_empty_query_results():
    c = state()
    assert c.getDNAFootprint([], []) == (1, 0, 0, 2, 2)
    tf, idx, p, l = c.isRegionUndamaged([[2, 1]], [3], returnExtent=True)
    assert (
        tf.tolist() == [True]
        and idx.tolist() == [1]
        and p.tolist() == [[2, 1]]
        and l.tolist() == [3]
    )


def test_damage_blocks_region_and_accessibility_preserves_positions():
    c = state()
    c._testDamagedSites = CircularSparseMat([[4, 1]], [1], [12, 4], [1])
    tf, idx, p, l = c.isRegionUndamaged([[2, 1]], [5], returnExtent=True)
    assert tf.tolist() == [True] and idx.tolist() == [1] and l.tolist() == [2]
