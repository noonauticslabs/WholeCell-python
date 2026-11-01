import numpy as np
import pytest

from wholecell_py.chromosome_process_aspect import ChromosomeProcessAspect


class ChromosomeStub:
    sequenceLen = 100
    monomerDNAFootprints = np.array([0, 0, 0, 6])
    complexDNAFootprints = np.array([0, 0, 0, 0, 0, 0, 8])

    @staticmethod
    def calculateFootprintOverhangs(footprints):
        five_prime = np.ceil((footprints - 1) / 2)
        three_prime = footprints - 1 - five_prime
        return three_prime, five_prime

    def setSiteProteinBound(self, *args):
        return (
            np.array([[2.0]]),
            np.array([[1.0]]),
            np.array([["effect"]], object),
            np.array([True]),
            np.array([1]),
            np.array([[10, 1]]),
            np.array([1]),
        )

    def modifyBoundProtein(self, *args):
        return np.array([[0.0]]), np.array([[0.0]]), np.empty((0, 1), object)

    def stochasticallySetProteinUnbound(self, *args):
        return np.array([[10, 1], [20, 2]]), np.empty((0, 1), object)

    def setRegionProteinUnbound(self, *args):
        return np.array([[1.0]]), np.array([[1.0]]), np.empty((0, 1), object)


class Aspect(ChromosomeProcessAspect):
    def __init__(self):
        self.chromosome = ChromosomeStub()
        self.enzymeMonomerLocalIndexs = np.array([1])
        self.enzymeMonomerGlobalIndexs = np.array([4])
        self.enzymeComplexLocalIndexs = np.array([2])
        self.enzymeComplexGlobalIndexs = np.array([7])
        self.enzymeGlobalIndexs = np.array([4, 7])
        self.enzymes = np.array([3.0, 0.0])
        self.boundEnzymes = np.array([0.0, 0.0])
        self.simulationStateSideEffects = np.empty((0, 1), object)
        self.stepSizeSec = 0.5


def test_bind_delegates_and_preserves_source_outputs_and_counts():
    aspect = Aspect()
    tfs, idxs, positions, maximum, processivity = (
        aspect.bindProteinToChromosome([[10, 1]], 1)
    )
    np.testing.assert_array_equal(tfs, [True])
    np.testing.assert_array_equal(idxs, [1])
    np.testing.assert_array_equal(positions, [[10, 1]])
    assert maximum == 1
    np.testing.assert_array_equal(processivity, [1])
    np.testing.assert_array_equal(aspect.enzymes, [5, 1])
    np.testing.assert_array_equal(aspect.boundEnzymes, [-2, -1])
    assert aspect.simulationStateSideEffects[0, 0] == "effect"


def test_release_chromosome_updates_local_counts_and_uses_step_size():
    aspect = Aspect()
    aspect.enzymes[:] = [0, 0]
    aspect.boundEnzymes[:] = [2, 0]
    positions = aspect.releaseProteinFromChromosome(1, 2, [], [])
    np.testing.assert_array_equal(positions, [[10, 1], [20, 2]])
    np.testing.assert_array_equal(aspect.enzymes, [2, 0])
    np.testing.assert_array_equal(aspect.boundEnzymes, [0, 0])


def test_release_sites_updates_both_protein_kinds():
    aspect = Aspect()
    released = aspect.releaseProteinFromSites([[10, 1]])
    np.testing.assert_array_equal(released, [1, 1])
    np.testing.assert_array_equal(aspect.enzymes, [4, 1])
    np.testing.assert_array_equal(aspect.boundEnzymes, [-1, -1])


def test_store_references_appends_chromosome_to_states():
    aspect = Aspect()
    aspect.states = ["prior"]
    chrom = object()
    aspect.storeObjectReferences({"states": {"Chromosome": chrom}})
    assert aspect.chromosome is chrom
    assert aspect.states == ["prior", chrom]


def test_initialize_constants_ignores_inherited_kb_and_simulation_arguments():
    aspect = Aspect()
    aspect.enzymeWholeCellModelIDs = ["MON", "COM"]
    # MATLAB's aspect method ignores the KB/simulation arguments and consumes
    # mappings prepared by the enclosing process initializer.
    aspect.initializeConstants(object(), object())
    np.testing.assert_array_equal(aspect.enzymeDNAFootprints, [6, 8])
    np.testing.assert_array_equal(aspect.enzymeDNAFootprints3Prime, [2, 3])
    np.testing.assert_array_equal(aspect.enzymeDNAFootprints5Prime, [3, 4])


def test_stochastic_binding_rejects_failure_at_selected_accessible_site():
    from types import SimpleNamespace

    aspect = Aspect()
    aspect.enzymeDNAFootprints = [1, 1]
    region = (np.array([[1, 1]]), np.array([10]))
    aspect.chromosome.getAccessibleRegions = lambda *args: region
    aspect.chromosome.intersectRegions = lambda *args: region
    aspect.randStream = SimpleNamespace(
        randsample=lambda *args: [1], rand=lambda: 0.5
    )
    aspect.bindProteinToChromosome = lambda *args: (np.array([False]),)
    with pytest.raises(RuntimeError, match="Region should be accessible"):
        aspect.bindProteinToChromosomeStochastically(1, 1, *region)
