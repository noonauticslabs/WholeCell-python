from pathlib import Path

import numpy as np
import pytest

from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.protein_monomer import ProteinMonomer


SNAPSHOT = (
    Path(__file__).resolve().parents[1] / "reference" / "snapshot_t0.json"
)


def test_actual_snapshot_restores_protein_monomer_and_dry_weight_getters():
    graph = load_snapshot(SNAPSHOT)
    source = graph.states["ProteinMonomer"].fields
    state = ProteinMonomer(snapshot=source)

    for field in (
        "wholeCellModelIDs",
        "names",
        "molecularWeights",
        "baseCounts",
        "lengths",
        "halfLives",
        "compartments",
        "counts",
        "nascentIndexs",
        "processedIIndexs",
        "processedIIIndexs",
        "signalSequenceIndexs",
        "foldedIndexs",
        "matureIndexs",
        "inactivatedIndexs",
        "boundIndexs",
        "misfoldedIndexs",
        "damagedIndexs",
        "translationFactorIndexs",
        "minimumAverageExpression",
        "macromoleculeStateInitializationVariation",
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

    np.testing.assert_array_equal(
        state.wholeCellModelID, source["wholeCellModelID"]
    )
    np.testing.assert_array_equal(state.name, source["name"])
    assert ProteinMonomer.optionNames == ("verbosity", "seed")
    assert ProteinMonomer.stateNames == ("counts",)
    assert state.dryWeight.shape == (1, 6)
    np.testing.assert_allclose(state.dryWeight, state.calcDryWeight())
    np.testing.assert_allclose(state.decayRates, np.log(2) / state.halfLives)


def test_snapshot_override_and_mature_id_index_lookup():
    state = ProteinMonomer(
        snapshot={
            "wholeCellModelIDs": np.array(["a", "b", "a", "b"]),
            "matureIndexs": np.array([[3], [4]]),
        },
        wholeCellModelIDs=np.array(["x", "y", "a", "b"]),
    )
    np.testing.assert_array_equal(
        state.getIndexs(["b", "missing", "a"]), [2, 0, 1]
    )
    assert state.getIndexs("a") == 1


def test_knowledge_base_constant_derivation_gap_is_explicit():
    state = ProteinMonomer("ProteinMonomer", "Protein monomers")
    with pytest.raises(ValueError, match="knowledgeBase is required"):
        state.initializeConstants(None)
    with pytest.raises(NotImplementedError, match="proteinMonomers records"):
        state.initializeConstants(
            {
                "states": [
                    {"wholeCellModelID": "ProteinMonomer", "parameters": []}
                ],
                "numProteinMonomers": 1,
            }
        )


def test_kb_initialization_builds_source_ordered_monomer_species_blocks():
    records = []
    for i, model_id in enumerate(("MG_A", "MG_026_MONOMER")):
        records.append(
            {
                "wholeCellModelID": model_id,
                "name": f"protein {i}",
                "molecularWeight": 100 + i,
                "processedISequenceMolecularWeight": 90 + i,
                "processedIISequenceMolecularWeight": 80 + i,
                "signalSequenceMolecularWeight": 70 + i,
                "foldedSequenceMolecularWeight": 60 + i,
                "matureSequenceMolecularWeight": 50 + i,
                "baseCount": np.full(21, 1 + i),
                "processedISequenceBaseCount": np.full(21, 2 + i),
                "processedIISequenceBaseCount": np.full(21, 3 + i),
                "signalSequenceBaseCount": np.full(21, 4 + i),
                "foldedSequenceBaseCount": np.full(21, 5 + i),
                "matureSequenceBaseCount": np.full(21, 6 + i),
                "sequenceLength": 10 + i,
                "processedISequenceLength": 9 + i,
                "processedIISequenceLength": 8 + i,
                "signalSequenceLength": 7 + i,
                "foldedSequenceLength": 6 + i,
                "matureSequenceLength": 5 + i,
                "halfLife": 100 + i,
                "processedISequenceHalfLife": 90 + i,
                "processedIISequenceHalfLife": 80 + i,
                "signalSequenceHalfLife": 70 + i,
                "foldedSequenceHalfLife": 60 + i,
                "matureSequenceHalfLife": 50 + i,
            }
        )
    state = ProteinMonomer("ProteinMonomer", "Protein monomer")
    state.compartment = type(
        "Compartment",
        (),
        {"cytosolIndexs": [1], "terminalOrganelleCytosolIndexs": [3]},
    )()
    kb = {
        "states": [{"wholeCellModelID": "ProteinMonomer", "parameters": []}],
        "numProteinMonomers": 2,
        "proteinMonomers": records,
        "proteinMonomerCompartments": [1, 2],
    }

    state.initializeConstants(kb)

    np.testing.assert_array_equal(state.nascentIndexs.ravel(), [1, 2])
    np.testing.assert_array_equal(state.matureIndexs.ravel(), [11, 12])
    assert state.wholeCellModelIDs == ["MG_A", "MG_026_MONOMER"] * 10
    np.testing.assert_array_equal(
        state.molecularWeights.ravel(),
        [100, 101, 90, 91, 80, 81, 70, 71, 60, 61] + [50, 51] * 5,
    )
    assert state.baseCounts.shape == (20, 21)
    assert state.compartments[6, 0] == 1  # Signal peptides are cytosolic.
    assert np.isinf(
        state.halfLives[1, 0]
    )  # Noncytosolic proteins do not decay.
    assert state.halfLives[6, 0] == 0  # Signal peptide lifetime is zero.
    np.testing.assert_array_equal(state.translationFactorIndexs.ravel(), [2])
