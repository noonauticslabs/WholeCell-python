import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome_binding_updates import (
    ChromosomeBindingUpdates,
)


class Compartment:
    cytosolIndexs = 1


class Chromosome(ChromosomeBindingUpdates):
    def __init__(self):
        self.sequenceLen = 20
        self.nCompartments = 2
        self.monomerBoundSites = CircularSparseMat(
            [[3, 1], [10, 2]], [2, 3], [20, 2], 1
        )
        self.complexBoundSites = CircularSparseMat([[19, 1]], [4], [20, 2], 1)
        self.monomerDNAFootprints = np.array([1, 2, 1])
        self.complexDNAFootprints = np.array([1, 1, 3, 2])
        self.monomerDNAFootprintBindingStrandedness = np.array([1, 1, 1])
        self.complexDNAFootprintBindingStrandedness = np.array([1, 1, 1, 1])
        self.dnaStrandedness_dsDNA = 2
        self.compartment = Compartment()
        self.complexIndexs_rnaPolymerase = np.array([4])
        self.released = None

    def updateExternalState(self, *args):
        self.external = args

    def getDNAFootprint(self, monomer, complex):
        return 2, 1, 1


def test_region_unbound_removes_overlapping_occupancies_and_returns_counts():
    chromosome = Chromosome()
    monomers, complexes, effects = chromosome.setRegionProteinUnbound(
        [[3, 1]], [1], [2, 3], [4], False, False, True, False
    )
    np.testing.assert_array_equal(monomers[:, 0], [1, 0])
    np.testing.assert_array_equal(complexes[:, 0], [0])
    np.testing.assert_array_equal(chromosome.monomerBoundSites.subs, [[10, 2]])
    assert effects.size == 0


def test_region_unbound_wraps_and_calls_external_state():
    chromosome = Chromosome()
    chromosome.setRegionProteinUnbound(
        [[20, 1]], [2], [], [], False, False, False, True
    )
    np.testing.assert_array_equal(
        chromosome.complexBoundSites.subs, np.empty((0, 2), dtype=int)
    )
    assert chromosome.external[-1] is True
    np.testing.assert_array_equal(chromosome.external[2], [[19, 1]])


def test_stochastic_unbinding_preserves_protected_sites():
    chromosome = Chromosome()
    chromosome.randStream = type(
        "R", (), {"rand": lambda self, *shape: np.zeros(shape)}
    )()
    positions, _ = chromosome.stochasticallySetProteinUnbound(
        [], 4, 1, [[19, 1]], [3], True, False
    )
    assert positions.shape == (0, 2)
    np.testing.assert_array_equal(chromosome.complexBoundSites.subs, [[19, 1]])
