from pathlib import Path

import numpy as np
import pytest

from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.protein_complex import ProteinComplex

SNAPSHOT = (
    Path(__file__).resolve().parents[1] / "reference" / "snapshot_t0.json"
)


def test_actual_snapshot_restores_protein_complex_and_getters():
    graph = load_snapshot(SNAPSHOT)
    source = graph.states["ProteinComplex"].fields
    state = ProteinComplex(snapshot=source)
    for field in (
        "wholeCellModelIDs",
        "names",
        "molecularWeights",
        "baseCounts",
        "lengths",
        "halfLives",
        "compartments",
        "counts",
        "proteinComplexComposition",
        "formationProcesses",
        "nascentIndexs",
        "matureIndexs",
        "inactivatedIndexs",
        "boundIndexs",
        "misfoldedIndexs",
        "damagedIndexs",
        "ribosome30SIndexs",
        "ribosome50SIndexs",
        "ribosome70SIndexs",
        "translationFactorIndexs",
        "ftsZGTPIndexs",
        "ftsZGDPIndexs",
        "rnaPolymeraseIndexs",
        "replisomeIndexs",
        "dnaPolymeraseIndexs",
        "dnaAPolymerIndexs",
        "minimumAverageExpression",
    ):
        actual, expected = getattr(state, field), source[field]
        if isinstance(expected, np.ndarray):
            if expected.dtype == object:
                assert actual.shape == expected.shape
                for index in np.ndindex(expected.shape):
                    if isinstance(expected[index], np.ndarray):
                        np.testing.assert_array_equal(
                            actual[index], expected[index]
                        )
                    else:
                        assert actual[index] == expected[index]
            else:
                np.testing.assert_array_equal(actual, expected)
            if actual.ndim > 1:
                assert actual.flags.f_contiguous
        else:
            assert actual == expected
    assert ProteinComplex.optionNames == ("verbosity", "seed")
    assert ProteinComplex.stateNames == ("counts",)
    assert state.dryWeight.shape == (1, 6)
    np.testing.assert_allclose(state.dryWeight, state.calcDryWeight())
    np.testing.assert_allclose(state.decayRates, np.log(2) / state.halfLives)


def test_mature_id_lookup_and_explicit_kb_gap():
    state = ProteinComplex(
        "ProteinComplex",
        snapshot={
            "wholeCellModelIDs": np.array(["a", "b", "a", "b"]),
            "matureIndexs": np.array([[3], [4]]),
        },
    )
    np.testing.assert_array_equal(
        state.getIndexs(["b", "missing", "a"]), [2, 0, 1]
    )
    assert state.getIndexs("a") == 1
    with pytest.raises(ValueError, match="knowledgeBase is required"):
        state.initializeConstants(None)
    with pytest.raises(NotImplementedError, match="proteinComplexs records"):
        state.initializeConstants(
            {
                "states": [
                    {"wholeCellModelID": "ProteinComplex", "parameters": []}
                ],
                "numProteinComplexs": 1,
            }
        )


def test_kb_initialization_builds_complex_species_and_composition_blocks():
    ids = [
        "RIBOSOME_30S",
        "RIBOSOME_50S",
        "RIBOSOME_70S",
        "MG_089_DIMER",
        "MG_224_9MER_GTP",
    ]
    records = []
    for i, model_id in enumerate(ids):
        records.append(
            {
                "wholeCellModelID": model_id,
                "name": f"complex {i}",
                "baseCount": np.full(4, i + 1),
                "matureBaseCount": np.full(4, i + 11),
                "molecularWeight": 100 + i,
                "matureMolecularWeight": 200 + i,
                "sequenceLength": 10 + i,
                "matureSequenceLength": 20 + i,
                "halfLife": 1000 + i,
                "matureHalfLife": 2000 + i,
                "complexFormationProcess": {"idx": i + 1},
            }
        )
    state = ProteinComplex("ProteinComplex", "Protein complexes")
    state.compartment = type(
        "Compartment",
        (),
        {"cytosolIndexs": [1], "terminalOrganelleCytosolIndexs": [3]},
    )()
    kb = {
        "states": [{"wholeCellModelID": "ProteinComplex", "parameters": []}],
        "numProteinComplexs": len(records),
        "proteinComplexs": records,
        "proteinComplexCompartments": [1, 2, 1, 3, 2],
        "proteinComplexAllRNAComposition": np.zeros((4, 2, 1)),
        "proteinComplexAllMonomerComposition": np.array([[[7], [8]]]),
        "mRNAGenes": [{"idx": 2}],
    }

    state.initializeConstants(kb)

    np.testing.assert_array_equal(state.nascentIndexs.ravel(), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(state.matureIndexs.ravel(), [6, 7, 8, 9, 10])
    np.testing.assert_array_equal(
        state.molecularWeights.ravel(),
        [100, 101, 102, 103, 104] + [200, 201, 202, 203, 204] * 5,
    )
    np.testing.assert_array_equal(state.baseCounts[5], [11, 11, 11, 11])
    np.testing.assert_array_equal(
        state.proteinComplexComposition[1, :, 0], [7, 8]
    )
    np.testing.assert_array_equal(
        state.formationProcesses.ravel(), [1, 2, 3, 4, 5] * 6
    )
    assert state.ribosome30SIndexs == 1
    assert state.ribosome50SIndexs == 2
    assert state.ribosome70SIndexs == 3
    np.testing.assert_array_equal(state.translationFactorIndexs, [4, 0, 0])
    assert state.halfLives[1, 0] == np.inf


def test_protected_group_cancels_all_deltas_when_any_member_decreases():
    from types import SimpleNamespace

    state = ProteinComplex(
        matureIndexs=[1, 2],
        boundIndexs=[3, 4],
        compartment=SimpleNamespace(cytosolIndexs=1),
        dnaAPolymerIndexs=[1, 2],
        replisomeIndexs=[],
        ribosome70SIndexs=[],
        translationFactorIndexs=[],
        ftsZGTPIndexs=[],
        ftsZGDPIndexs=[],
        rnaPolymeraseIndexs=[],
    )
    delta = np.array([[0], [0], [-1], [2]])
    with pytest.warns(RuntimeWarning, match="DnaA"):
        result = state.updateExternalState(delta, True)
    np.testing.assert_array_equal(result, [[0], [0], [1], [-2]])
