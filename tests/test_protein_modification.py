import numpy as np
import pytest

from wholecell_py.processes.protein_modification import ProteinModification


class Stream:
    def stochasticRound(self, x):
        return np.floor(x).astype(int)

    def randsample(self, n, k, replacement=True, w=None):
        return np.array([np.flatnonzero(w)[0] + 1])


def test_species_network_preserves_one_based_enzyme_indexes():
    p = ProteinModification(
        stepSizeSec=2,
        reactionModificationMatrix=np.array([[1, 0], [1, 1]]),
        reactionStoichiometryMatrix=np.array([[-1, 0], [0, -1]]),
        reactionCatalysisMatrix=np.array([[1], [0]]),
        enzymeBounds=np.array([[0, 2], [0, 1]]),
        monomerIndexs_modified=np.array([1, 2]),
        monomerIndexs_notmodified=np.array([3]),
        monomerCompartments=np.array([1, 1, 1]),
    )
    p.initializeConstants()
    assert p.speciesIndexs_enzymes.tolist() == [3]
    np.testing.assert_allclose(
        p.speciesReactantMatrix, [[1, 1, 0.25, 1, 0], [0, 1, 0, 0, 1]]
    )


def test_species_network_uses_modified_monomer_columns():
    from pathlib import Path

    from wholecell_py.runtime_registry import implemented_registry
    from wholecell_py.snapshot import load_snapshot

    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.ProteinModification"] = (
        ProteinModification
    )
    graph = load_snapshot(
        Path(__file__).resolve().parents[1] / "reference" / "snapshot_t0.json",
        class_registry=registry,
    )
    process = graph.processes["ProteinModification"]
    expected_reactants = np.asarray(process.speciesReactantMatrix).copy()
    expected_net = np.asarray(process.speciesReactantByproductMatrix).copy()

    process.initializeSpeciesNetwork()

    np.testing.assert_allclose(
        process.speciesReactantMatrix, expected_reactants
    )
    np.testing.assert_allclose(
        process.speciesReactantByproductMatrix, expected_net
    )


def test_evolve_modifies_available_protein_and_moves_plain_protein():
    p = ProteinModification(
        reactionModificationMatrix=np.array([[1]]),
        reactionStoichiometryMatrix=np.array([[-1]]),
        reactionCatalysisMatrix=np.array([[1]]),
        enzymeBounds=np.array([[0, 1]]),
        speciesReactantMatrix=np.array([[1, 1, 1]]),
        speciesReactantByproductMatrix=np.array([[1, 1, 1]]),
        speciesIndexs_enzymes=np.array([2]),
        monomerIndexs_modified=np.array([1]),
        monomerIndexs_notmodified=np.array([2]),
        randStream=Stream(),
        substrates=np.array([1]),
        enzymes=np.array([1]),
        unmodifiedMonomers=np.array([[2], [3]]),
        modifiedMonomers=np.zeros((2, 1)),
    )
    p.evolveState()
    np.testing.assert_array_equal(p.substrates, [0])
    np.testing.assert_array_equal(p.unmodifiedMonomers.ravel(), [1, 0])
    np.testing.assert_array_equal(p.modifiedMonomers.ravel(), [1, 3])


def test_initialize_constants_reports_missing_snapshot_network():
    with pytest.raises(NotImplementedError, match="missing"):
        ProteinModification().initializeConstants()
