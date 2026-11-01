import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.randstream import RandStream
from wholecell_py.states.chromosome_sampling import ChromosomeSampling


class SamplingHost(ChromosomeSampling):
    dnaStrandedness_ssDNA = 1
    dnaStrandedness_dsDNA = 2

    def __init__(self):
        self.randStream = RandStream(seed=2)
        self.sequenceLen = 20
        self.nCompartments = 4
        self.dnaStrandedness_xsDNA = 3
        self.polymerizedRegions = CircularSparseMat([[1, 1]], [20], [20, 4], 1)
        self.singleStrandedRegions = self.polymerizedRegions
        self.doubleStrandedRegions = CircularSparseMat(
            [[1, 1], [1, 2]], [20, 20], [20, 4], 1
        )
        self.supercoiled = self.doubleStrandedRegions
        self.monomerBoundSites = CircularSparseMat([], [], [20, 4], 1)
        self.complexBoundSites = CircularSparseMat([], [], [20, 4], 1)
        self.damagedSites = CircularSparseMat([], [], [20, 4], 1)
        self.monomerDNAFootprints = np.array([1])
        self.complexDNAFootprints = np.array([1])
        self.monomerDNAFootprintBindingStrandedness = np.array([1])
        self.complexDNAFootprintBindingStrandedness = np.array([1])

    def getDNAFootprint(self, monomers, complexes):
        return 1, 0, 0, self.dnaStrandedness_ssDNA, self.dnaStrandedness_ssDNA

    def getReleasableProteins(self, monomers, complexes):
        return np.array([], dtype=int), np.array([], dtype=int)

    def excludeRegions(
        self, positions, lengths, excluded_positions, excluded_lengths
    ):
        return np.asarray(positions, dtype=int).reshape(-1, 2), np.asarray(
            lengths, dtype=int
        ).reshape(-1)

    def isRegionAccessible(self, positions, lengths, *args):
        p = np.asarray(positions, dtype=int).reshape(-1, 2)
        return (
            np.ones(len(p), bool),
            np.arange(1, len(p) + 1),
            p,
            np.asarray(lengths).reshape(-1),
        )


def test_sample_accessible_regions_returns_sorted_one_based_indices():
    host = SamplingHost()
    ps = np.array([[3, 1], [8, 1], [15, 1]])
    flags, indices, selected, lengths = host.sampleAccessibleRegions(
        2, [], ps, [1, 1, 1], [], [], False, None, True, False
    )
    assert len(flags) == 3 and flags.sum() == 2
    np.testing.assert_array_equal(indices, np.flatnonzero(flags) + 1)
    np.testing.assert_array_equal(selected, ps[flags])
    np.testing.assert_array_equal(lengths, [1, 1])


def test_sample_accessible_sites_handles_empty_probability_and_zero_request():
    host = SamplingHost()
    np.testing.assert_array_equal(
        host.sampleAccessibleSites(None, 0, ""), np.empty((0, 2), int)
    )


def test_get_accessible_regions_preserves_one_based_sparse_regions():
    host = SamplingHost()
    positions, lengths = host.getAccessibleRegions([], [])
    np.testing.assert_array_equal(positions, [[1, 1]])
    np.testing.assert_array_equal(lengths, [20])


def test_ds_binding_checks_even_strand_supercoiling_flag():
    host = SamplingHost()
    host.getDNAFootprint = lambda monomers, complexes: (
        1,
        0,
        0,
        host.dnaStrandedness_dsDNA,
        host.dnaStrandedness_dsDNA,
    )
    host.supercoiled = CircularSparseMat([[1, 2]], [1], [20, 4], 1)

    positions, lengths = host.getAccessibleRegions(
        [], [], checkRegionSupercoiled=True
    )

    np.testing.assert_array_equal(positions, [[1, 1]])
    np.testing.assert_array_equal(lengths, [20])
