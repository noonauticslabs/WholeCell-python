import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome_binding_creation import (
    ChromosomeBindingCreation,
)


class Host(ChromosomeBindingCreation):
    dnaStrandedness_xsDNA = 1
    dnaStrandedness_dsDNA = 2

    def __init__(self):
        self.monomerBoundSites = CircularSparseMat([[5, 1]], [1], [20, 2], 1)
        self.complexBoundSites = CircularSparseMat([], [], [20, 2], 1)
        self.monomerDNAFootprints = np.array([3, 1])
        self.complexDNAFootprints = np.array([4])
        self.sample_args = None

    def sampleAccessibleRegions(self, *args):
        self.sample_args = args
        return (
            np.array([True]),
            np.array([[1]]),
            np.array([[10, 1]]),
            np.array([1]),
        )

    def getDNAFootprint(self, monomers, complexes):
        return 2, 1, 1, 2, 1

    def getReleasableProteins(self, monomers, complexes):
        return [], []

    def excludeRegions(self, pos, lengths, occupied, footprints):
        self.excluded = (pos, lengths, occupied, footprints)
        return pos, lengths

    def setRegionProteinUnbound(self, *args):
        self.unbound = args
        return np.zeros((2, 1)), np.zeros((1, 1)), np.empty((0, 1), object)


def test_stable_binding_samples_releases_and_binds_at_position():
    h = Host()
    result = h.setSiteProteinBound(
        [[10, 1]], 1, [1], [2], [], [1, 2], [1], True, False, [1], False, False
    )
    assert len(result) == 7
    np.testing.assert_array_equal(h.monomerBoundSites[[10, 1]], [[2]])
    assert h.unbound[5:7] == (True, False)


def test_modify_bound_protein_counts_identity_changes():
    h = Host()
    mon, comp, effects = h.modifyBoundProtein([[5, 1]], [2], [0], [1, 2], [1])
    np.testing.assert_array_equal(mon[:, 0], [1, -1])
    np.testing.assert_array_equal(comp[:, 0], [0])
    assert effects.shape == (0, 1)


def test_modify_multiple_bound_proteins_with_column_footprints():
    h = Host()
    h.monomerBoundSites = CircularSparseMat(
        [[5, 1], [6, 1]], [1, 2], [20, 2], 1
    )
    # Model the Nx1 shape of constants loaded from the native simulation.
    h.monomerDNAFootprints = np.array([[3], [1]])
    h.complexDNAFootprints = np.array([[4]])

    mon, comp, _ = h.modifyBoundProtein(
        [[5, 1], [6, 1]], [2, 2], [0, 0], [1, 2], [1]
    )

    np.testing.assert_array_equal(mon[:, 0], [1, -1])
    np.testing.assert_array_equal(comp[:, 0], [0])
    np.testing.assert_array_equal(h.monomerBoundSites[[5, 6], 1], [[2], [2]])


def test_modify_rejects_larger_footprint():
    h = Host()
    try:
        h.modifyBoundProtein([[5, 1]], [0], [1], [1, 2], [1])
    except ValueError as exc:
        assert "footprints" in str(exc)
    else:
        raise AssertionError("larger footprint was accepted")


def test_modify_multiple_sites_rejects_larger_footprints_before_mutation():
    h = Host()
    h.monomerBoundSites = CircularSparseMat(
        [[5, 1], [6, 1]], [1, 1], [20, 2], 1
    )
    try:
        h.modifyBoundProtein([[5, 1], [6, 1]], [0, 0], [1, 1], [1, 2], [1])
    except ValueError as exc:
        assert "footprints" in str(exc)
    else:
        raise AssertionError("larger footprints were accepted")
    np.testing.assert_array_equal(h.monomerBoundSites[[5, 6], 1], [[1], [1]])
    np.testing.assert_array_equal(h.complexBoundSites[[5, 6], 1], [[0], [0]])
