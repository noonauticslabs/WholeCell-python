from pathlib import Path

import numpy as np
import pytest

from wholecell_py.snapshot import load_snapshot
from wholecell_py.sparsemat import SparseMat
from wholecell_py.states.metabolite import Metabolite


SNAPSHOT = (
    Path(__file__).resolve().parents[1] / "reference" / "snapshot_t0.json"
)


def test_actual_snapshot_restores_metabolite_state_and_derived_weights():
    graph = load_snapshot(SNAPSHOT)
    source = graph.states["Metabolite"].fields
    state = Metabolite(snapshot=source)

    for field in (
        "wholeCellModelIDs",
        "names",
        "molecularWeights",
        "baseCounts",
        "lengths",
        "halfLives",
        "compartments",
        "counts",
        "biomassComposition",
        "biomassProduction",
        "byproducts",
        "setCounts",
        "waterIndexs",
        "processWholeCellModelIDs",
        "processRequirements",
        "processAllocations",
        "processUsages",
    ):
        actual, expected = getattr(state, field), source[field]
        if isinstance(expected, np.ndarray):
            if expected.dtype != object:
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual.shape == expected.shape
            assert actual.flags.f_contiguous
        else:
            assert actual == expected

    np.testing.assert_array_equal(
        state.wholeCellModelID, source["wholeCellModelID"]
    )
    np.testing.assert_array_equal(state.name, source["name"])
    assert Metabolite.optionNames == ("verbosity", "seed")
    assert Metabolite.stateNames == (
        "counts",
        "processRequirements",
        "processAllocations",
        "processUsages",
    )
    assert Metabolite.dependentStateNames == ()
    assert state.wetWeight.shape == (1,) + state.counts.shape[1:]
    np.testing.assert_allclose(state.dryWeight, state.calcDryWeight())
    np.testing.assert_allclose(
        state.metabolismProduction,
        np.asarray(state.biomassProduction) - np.asarray(state.byproducts),
    )


def test_snapshot_fields_can_be_overridden_by_constructor_keywords():
    state = Metabolite(
        snapshot={"wholeCellModelID": "old", "counts": [[1]]},
        wholeCellModelID="Metabolite",
        counts=[[4]],
    )
    assert state.wholeCellModelID == "Metabolite"
    np.testing.assert_array_equal(state.counts, [[4]])


def test_compositions_and_weights_use_matlab_one_based_indices():
    state = Metabolite(
        snapshot={
            "waterIndexs": [[3]],
            "nmpIndexs": [[1], [2]],
            "aminoAcidIndexs": [[1], [2]],
            "experimentalBiomassCompositionMolFractions": [
                [0.1],
                [0.2],
                [0.7],
            ],
            "biomassComposition": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "molecularWeights": [[10.0], [20.0], [30.0]],
            "counts": np.array([[1, 2], [3, 4], [5, 6]]),
        }
    )
    np.testing.assert_allclose(
        state.experimentalNMPComposition, [[1 / 3], [2 / 3]]
    )
    np.testing.assert_allclose(
        state.experimentalAAComposition, [[1 / 3], [2 / 3]]
    )
    np.testing.assert_allclose(state.nmpComposition, [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(state.aaComposition, [[3 / 10], [7 / 10]])
    np.testing.assert_allclose(
        state.wetWeight, np.array([[5, 6]]) * 30 / state.nAvogadro
    )
    np.testing.assert_allclose(
        state.dryWeight, np.array([[70, 100]]) / state.nAvogadro
    )


def test_allocation_creates_process_sparse_arrays_with_matlab_dimensions():
    state = Metabolite(
        snapshot={
            "wholeCellModelIDs": ["a", "b"],
            "processWholeCellModelIDs": ["p1", "p2"],
        }
    )
    state.storeObjectReferences(
        type(
            "Simulation",
            (),
            {
                "compartment": type("Compartment", (), {"count": 3})(),
            },
        )()
    )
    state.allocateMemory(4)
    assert state.counts.shape == (2, 3, 4)
    assert state.counts.flags.f_contiguous
    for field in (
        "processRequirements",
        "processAllocations",
        "processUsages",
    ):
        value = getattr(state, field)
        assert isinstance(value, SparseMat)
        assert value.siz.tolist() == [6, 2, 4]
        assert value.nnz() == 0


def test_unavailable_initialization_and_dependent_data_are_explicit():
    state = Metabolite("Metabolite", "Metabolites")
    with pytest.raises(ValueError, match="knowledgeBase is required"):
        state.initializeConstants(None)
    with pytest.raises(ValueError, match="biomassProduction"):
        _ = state.metabolismProduction
    with pytest.raises(ValueError, match="counts"):
        _ = state.wetWeight
