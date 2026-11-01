from pathlib import Path

import numpy as np
import pytest

from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.rna import Rna


SNAPSHOT = Path(__file__).parents[1] / "reference" / "snapshot_t0.json"


def test_actual_snapshot_restores_rna_fields_and_dry_weight_getters():
    graph = load_snapshot(SNAPSHOT)
    source = graph.states["Rna"].fields
    state = Rna(snapshot=source)
    for field in (
        "wholeCellModelIDs",
        "names",
        "molecularWeights",
        "baseCounts",
        "lengths",
        "halfLives",
        "compartments",
        "counts",
        "expression",
        "nascentIndexs",
        "processedIndexs",
        "intergenicIndexs",
        "matureIndexs",
        "boundIndexs",
        "misfoldedIndexs",
        "damagedIndexs",
        "aminoacylatedIndexs",
        "matureMRNAIndexs",
        "matureRRNAIndexs",
        "matureSRNAIndexs",
        "matureTRNAIndexs",
        "matureRibosomalRRNAIndexs",
        "matureTMRNAIndexs",
        "matureRNAGeneComposition",
        "expectedGeneHalfLives",
        "expectedGeneExpression",
        "weightFractionMRNA",
        "weightFractionRRNA5S",
        "weightFractionRRNA16S",
        "weightFractionRRNA23S",
        "weightFractionTRNA",
        "weightFractionSRNA",
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
    assert Rna.optionNames == ("verbosity", "seed")
    assert Rna.stateNames == ("counts",)
    assert state.dryWeight.shape == (1, 6)
    np.testing.assert_allclose(state.dryWeight, state.calcDryWeight())
    np.testing.assert_allclose(state.decayRates, np.log(2) / state.halfLives)
    np.testing.assert_allclose(
        state.geneDecayRates, np.log(2) / state.geneHalfLives
    )
    np.testing.assert_allclose(
        state.expectedGeneDecayRates, np.log(2) / state.expectedGeneHalfLives
    )
    np.testing.assert_allclose(state.geneExpression.sum(), 1)
    np.testing.assert_allclose(state.weightFractions.sum(), 1)
    np.testing.assert_allclose(state.expectedWeightFractions.sum(), 1)


def test_mature_index_lookup_requires_a_complete_kb_for_initialization():
    state = Rna(
        "Rna",
        wholeCellModelIDs=np.array(["pre", "mature-a", "mature-b"]),
        matureIndexs=np.array([[2], [3]]),
    )
    np.testing.assert_array_equal(
        state.getIndexs(["mature-b", "missing", "mature-a"]), [2, 0, 1]
    )
    assert state.getIndexs("mature-a") == 1
    with pytest.raises(ValueError, match="knowledgeBase is required"):
        state.initializeConstants(None)


def test_kb_initialization_builds_rna_species_and_literal_prefit_expression():
    def group(ids, lengths):
        n = len(ids)
        return {
            "wholeCellModelIDs": ids,
            "names": [f"name-{x}" for x in ids],
            "lengths": lengths,
            "baseCounts": np.arange(n * 4).reshape(n, 4),
            "molecularWeights": np.arange(n) + 100.0,
        }

    genes = [
        {"idx": i + 1, "halfLife": 10 + i, "expression": [1 + i, 4 + i]}
        for i in range(4)
    ]
    mature_ids = ["mRNA", "rRNA", "sRNA", "MG_0004"]
    mature_gene_composition = np.eye(4).reshape(4, 4, 1)
    state = Rna("Rna", "RNA")
    kb = {
        "states": [{"wholeCellModelID": "Rna", "parameters": []}],
        "genes": genes,
        "numGenes": 4,
        "transcriptionUnitComposition_Genes": np.ones((4, 1, 1)),
        "transcriptionUnitComposition_ProcessedRNAs": np.ones((1, 1, 1)),
        "processedRNAComposition_Genes": mature_gene_composition,
        "transcriptionUnitComposition_IntergenicRNAs": np.ones((1, 1, 1)),
        "nascentRNAs": group(["nascent"], [11]),
        "processedRNAs": group(["processed"], [10]),
        "intergenicRNAs": group(["intergenic"], [2]),
        "matureRNAs": group(mature_ids, [9, 8, 7, 6]),
        "aminoacylatedRNAs": group(["charged-A", "charged-B"], [5, 4]),
        "mRNAGenes": [genes[0]],
        "rRNAGenes": [genes[1]],
        "sRNAGenes": [genes[2]],
        "tRNAGenes": [genes[3]],
        "ribosomalRRNAIndexs": [2],
    }

    state.initializeConstants(kb)

    np.testing.assert_array_equal(state.nascentIndexs.ravel(), [1])
    np.testing.assert_array_equal(state.processedIndexs.ravel(), [2])
    np.testing.assert_array_equal(state.intergenicIndexs.ravel(), [3])
    np.testing.assert_array_equal(state.matureIndexs.ravel(), [4, 5, 6, 7])
    np.testing.assert_array_equal(state.aminoacylatedIndexs.ravel(), [20, 21])
    assert len(state.wholeCellModelIDs) == 21
    # Rna.expression is a precursor initializer fitted later by FitConstants;
    # preserve source's missing intergenic zero block rather than snapshot shape.
    assert state.expression.shape == (20, 1)
    np.testing.assert_array_equal(state.matureMRNAIndexs.ravel(), [1])
    np.testing.assert_array_equal(state.matureRRNAIndexs.ravel(), [2])
    np.testing.assert_array_equal(state.matureSRNAIndexs.ravel(), [3])
    np.testing.assert_array_equal(state.matureTRNAIndexs.ravel(), [4])
    np.testing.assert_array_equal(state.matureRibosomalRRNAIndexs.ravel(), [2])
    assert state.matureTMRNAIndexs == 4
