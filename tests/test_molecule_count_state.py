import numpy as np
import pytest

from wholecell_py.molecule_count_state import MoleculeCountState
from wholecell_py.snapshot import load_snapshot


def test_actual_protein_monomer_snapshot_fields_restore_in_base():
    graph = load_snapshot(
        __file__.replace(
            "tests/test_molecule_count_state.py", "reference/snapshot_t0.json"
        )
    )
    source = graph.states["ProteinMonomer"].fields
    state = MoleculeCountState(snapshot=source)
    for field in (
        "wholeCellModelIDs",
        "names",
        "molecularWeights",
        "baseCounts",
        "lengths",
        "halfLives",
        "compartments",
        "counts",
    ):
        actual, expected = getattr(state, field), source[field]
        if isinstance(expected, np.ndarray):
            assert actual.shape == expected.shape
            if actual.dtype == object:
                for index in np.ndindex(actual.shape):
                    left, right = actual[index], expected[index]
                    if isinstance(right, np.ndarray):
                        np.testing.assert_array_equal(left, right)
                    else:
                        assert left == right
            else:
                np.testing.assert_array_equal(actual, expected)
            if actual.ndim > 1:
                assert actual.flags.f_contiguous
        else:
            assert actual == expected
    np.testing.assert_array_equal(
        state.wholeCellModelID, source["wholeCellModelID"]
    )
    np.testing.assert_array_equal(state.name, source["name"])


def test_references_allocation_and_initialization_keep_matlab_dimensions():
    state = MoleculeCountState(snapshot={"wholeCellModelIDs": ["a", "b"]})
    compartment = type("Compartment", (), {"count": 3})()
    simulation = type("Simulation", (), {"compartment": compartment})()
    state.storeObjectReferences(simulation)
    state.allocateMemory(4)
    assert state.counts.shape == (2, 3, 4)
    assert state.counts.flags.f_contiguous
    np.testing.assert_array_equal(state.counts, 0)
    state.initialize()
    assert state.counts.shape == (2, 3, 1)
    assert state.compartment is compartment


def test_index_decay_rate_and_dry_weight_getters():
    state = MoleculeCountState(
        snapshot={
            "wholeCellModelIDs": ["a", "b"],
            "molecularWeights": np.array([10.0, 20.0]),
            "halfLives": np.array([2.0, 4.0]),
            "counts": np.array([[1, 2], [3, 4]]),
        }
    )
    np.testing.assert_array_equal(
        state.getIndexs(["b", "missing", "a"]), [2, 0, 1]
    )
    assert state.getIndexs("b") == 2
    np.testing.assert_allclose(state.decayRates, np.log(2) / [2, 4])
    state.decayRates = [np.log(2) / 5, np.log(2) / 10]
    np.testing.assert_allclose(state.halfLives, [5, 10])
    np.testing.assert_allclose(
        state.dryWeight, np.array([[70, 100]]) / state.nAvogadro
    )


def test_dry_weight_multiple_time_points_and_singleton_2d_counts():
    state = MoleculeCountState(
        molecularWeights=[2, 3],
        counts=np.array([[[1, 4], [2, 5]], [[3, 6], [4, 7]]]),
    )
    expected = np.array([[[11, 26], [16, 31]]]) / state.nAvogadro
    np.testing.assert_allclose(state.calcDryWeight(), expected)
    state.counts = np.array([[1, 2], [3, 4]])
    np.testing.assert_allclose(
        state.calcDryWeight(), np.array([[11, 16]]) / state.nAvogadro
    )


def test_missing_knowledge_base_or_state_is_an_explicit_error():
    state = MoleculeCountState("id", "name")
    with pytest.raises(ValueError, match="knowledgeBase is required"):
        state.initializeConstants(None)
    with pytest.raises(ValueError, match="missing from knowledgeBase"):
        state.initializeConstants({"states": []})


def test_initialize_constants_copies_cellstate_simulation_options():
    state = MoleculeCountState("id", "name")
    kb = {"states": [{"wholeCellModelID": "id", "parameters": []}]}
    simulation = type("Simulation", (), {"verbosity": 4, "seed": 29})()

    state.initializeConstants(kb, simulation)

    assert state.verbosity == 4
    assert state.seed == 29


@pytest.mark.parametrize("count", [-1, 1.5, "2"])
def test_allocate_memory_validates_time_point_count(count):
    state = MoleculeCountState(
        wholeCellModelIDs=[],
        compartment=type("Compartment", (), {"count": 1})(),
    )
    with pytest.raises((TypeError, ValueError)):
        state.allocateMemory(count)
