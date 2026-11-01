"""Snapshot contract tests for Translation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wholecell_py.processes.translation import Translation
from wholecell_py.process_base import Process
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot

ROOT = Path(__file__).parents[1]


def test_translation_kb_initializer_maps_tm_rna_and_mature_trnas(monkeypatch):
    process = Translation()
    process.rna = SimpleNamespace(
        wholeCellModelIDs=[
            "tRNA1",
            "tRNA2",
            "tmRNA",
            "aa-tRNA1",
            "aa-tRNA2",
            "aa-tmRNA",
        ],
        matureIndexs=[1, 2, 3],
        matureTRNAIndexs=[1, 2],
        aminoacylatedIndexs=[4, 5, 6],
    )
    process.enzymeRNALocalIndexs = [14]
    process.enzymeRNAGlobalIndexs = [3]
    process.ribosomeElongationRate = 2
    process.tmRNABindingProbability = 0.5
    monkeypatch.setattr(
        Process,
        "initializeConstants",
        lambda self, kb, sim, options=None: None,
    )

    process.initializeConstants(object(), object())

    assert process.tmRNAGlobalIndex == 3
    np.testing.assert_array_equal(
        process.freeTRNAWholeCellModelIDs.ravel(), ["tRNA1", "tRNA2"]
    )
    np.testing.assert_array_equal(
        process.aminoacylatedTRNAWholeCellModelIDs.ravel(),
        ["aa-tRNA1", "aa-tRNA2"],
    )
    assert process.aminoacylatedTMRNAWholeCellModelID == "aa-tmRNA"


def test_translation_initialize_state_binds_ribosomes_and_preserves_matlab_half_ties():
    from types import SimpleNamespace

    class Draws:
        def __init__(self):
            self.weights = []

        def randsample(self, n, k, replacement=False, weights=None):
            if weights is not None:
                self.weights.append(np.asarray(weights).copy())
            return np.array([1])

        def rand(self):
            return 0.5

    p = Translation()
    p.randStream = Draws()
    p.ribosome = SimpleNamespace(
        boundMRNAs=np.zeros((0, 1), int),
        mRNAPositions=np.zeros((0, 1), int),
        tmRNAPositions=np.zeros((0, 1), int),
        notExistValue=-1,
        activeValue=2,
    )
    p.polypeptide = SimpleNamespace(
        boundMRNAs=np.zeros((0, 1), int),
        nascentMonomerLengths=np.zeros((0, 1), int),
        proteolysisTagLengths=np.zeros((0, 1), int),
        monomerLengths=np.array([10, 20]),
        dryWeight=1,
    )
    p.mRNAs = np.array([[2], [1]])
    p.ribosomeElongationRate = np.array([[2]])
    p.enzymes = np.zeros((16, 1), int)
    p.boundEnzymes = np.zeros((16, 1), int)
    p.enzymes[10 - 1] = 2
    p.enzymes[11 - 1] = 1
    p.enzymes[12 - 1] = 2
    p.enzymes[3:7, 0] = 2
    p.monomer = SimpleNamespace(
        counts=np.array([[2], [0]]),
        dryWeight=np.array([[2.0]]),
        molecularWeights=np.array([6.0221417930e23, 6.0221417930e23]),
    )
    p.complex = SimpleNamespace(
        counts=np.zeros((2, 1), int),
        dryWeight=np.array([[3.0]]),
        molecularWeights=np.array([1.0, 1.0]),
        matureIndexs=np.array([1, 2]),
    )
    p.copyToState = lambda: None
    p.copyFromState = lambda: None

    p.initializeState()

    assert p.ribosome.states.shape == (4, 1)
    np.testing.assert_array_equal(p.ribosome.states[:2, 0], [2, 2])
    np.testing.assert_array_equal(p.polypeptide.boundMRNAs[:2, 0], [1, 1])
    np.testing.assert_array_equal(
        p.polypeptide.nascentMonomerLengths[:2, 0], [2, 8]
    )
    np.testing.assert_array_equal(p.enzymes[[9, 11], 0], [1, 0])
    np.testing.assert_array_equal(p.boundEnzymes[3:7, 0], [2, 2, 2, 2])
    np.testing.assert_array_equal(p.monomer.counts, [[1], [0]])
    # MATLAB samples with replacement from unchanged mRNA-length weights.
    np.testing.assert_array_equal(p.randStream.weights[1], [20, 20])


def test_translation_growth_appends_full_source_count_to_both_states(
    monkeypatch,
):
    import wholecell_py.polymerize as polymerize_module
    from wholecell_py.runner import load_model

    graph = load_model(ROOT / "reference" / "snapshot_t0.json", seed=1)
    p = graph.processes["Translation"]
    p.copyFromState()
    p.ribosome.states = np.full((6, 1), p.ribosome.notExistValue)
    p.ribosome.boundMRNAs = np.zeros((6, 1), int)
    p.ribosome.mRNAPositions = np.zeros((6, 1), int)
    p.ribosome.tmRNAPositions = np.zeros((6, 1), int)
    p.polypeptide.boundMRNAs = np.zeros((6, 1), int)
    p.polypeptide.nascentMonomerLengths = np.zeros((6, 1), int)
    p.polypeptide.proteolysisTagLengths = np.zeros((6, 1), int)
    p.mRNAs = np.zeros_like(p.mRNAs)
    p.enzymes.reshape(-1)[9:13] = [10, 0, 10, 0]
    p.enzymes.reshape(-1)[2] = 3
    p.boundEnzymes.reshape(-1)[12] = 4

    def no_polymerization(
        sequence, aminoacylated, trna_ids, pad, energy, cost, stream
    ):
        return (
            np.zeros(len(sequence), int),
            aminoacylated,
            np.zeros(4, int),
            energy,
            None,
        )

    monkeypatch.setattr(polymerize_module, "polymerize", no_polymerization)
    p.evolveState()

    # Four bound 70S ribosomes plus three newly assembled 30S-IF3 particles
    # cause MATLAB to append seven rows to the six existing rows.
    assert p.ribosome.states.shape == (13, 1)
    assert p.polypeptide.boundMRNAs.shape == (13, 1)
    assert p.polypeptide.nascentMonomerLengths.shape == (13, 1)
    assert p.polypeptide.proteolysisTagLengths.shape == (13, 1)


def test_translation_dry_weight_includes_aminoacylated_trna_and_tmrna():
    _, p = snapshot()
    p.monomers = np.zeros_like(p.monomers)
    p.freeTRNAs = np.zeros_like(p.freeTRNAs)
    p.aminoacylatedTRNAs = np.zeros_like(p.aminoacylatedTRNAs)
    p.aminoacylatedTMRNA = np.zeros_like(p.aminoacylatedTMRNA)
    base = np.asarray(p.getDryWeight()).reshape(-1)

    mature_trna = int(np.asarray(p.rna.matureTRNAIndexs).reshape(-1)[0])
    aminoacylated_trna = int(
        np.asarray(p.rna.aminoacylatedIndexs).reshape(-1)[mature_trna - 1]
    )
    tmrna = int(np.asarray(p.tmRNAGlobalIndex).reshape(-1)[0])
    aminoacylated_tmrna = int(
        np.asarray(p.rna.aminoacylatedIndexs).reshape(-1)[tmrna - 1]
    )
    p.aminoacylatedTRNAs.reshape(-1)[0] = 5
    p.aminoacylatedTMRNA.reshape(-1)[0] = 7
    actual = np.asarray(p.getDryWeight()).reshape(-1) - base
    expected = (
        5 * p.rna.molecularWeights[aminoacylated_trna - 1]
        + 7 * p.rna.molecularWeights[aminoacylated_tmrna - 1]
    ) / 6.0221417930e23
    np.testing.assert_allclose(actual, expected)


def test_translation_lifecycle_reports_peptide_bond_water_byproduct():
    from types import SimpleNamespace

    _, p = snapshot()
    lengths = np.asarray(p.polypeptide.monomerLengths).reshape(-1)
    production = np.zeros_like(lengths, dtype=float)
    production[0] = 1
    states = SimpleNamespace(
        monomerProductions=production,
        monomerProductions0=production,
    )
    _, byproducts, _, _ = p.calcResourceRequirements_LifeCycle(states)
    np.testing.assert_allclose(byproducts[24], max(0, lengths[0] - 1))


def snapshot():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.Translation"] = Translation
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json", class_registry=registry
    )
    return graph, graph.processes["Translation"]


def test_snapshot_translation_requirements_and_state_projection():
    graph, p = snapshot()
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["Translation"]
    p.copyFromState()
    actual = p.calcResourceRequirements_Current().reshape(
        expected["shape"], order="F"
    )
    np.testing.assert_allclose(
        actual,
        np.asarray(expected["data"]).reshape(expected["shape"], order="F"),
        rtol=1e-12,
    )
    assert p.mRNAs.shape[0] == len(p.gene.mRNAIndexs)
    assert p.freeTRNAs.shape[0] == len(p.rna.matureTRNAIndexs)
    assert p.ribosome is graph.states["Ribosome"]
    assert p.polypeptide is graph.states["Polypeptide"]


def test_knowledge_base_initialization_fails_explicitly_without_derivation_inputs():
    try:
        Translation().initializeConstants()
    except NotImplementedError as exc:
        assert "KnowledgeBase.computeTRNASequences" in str(exc)
    else:
        raise AssertionError(
            "Translation must not invent missing sequence mappings"
        )


def test_translation_termination_does_not_consume_catalytic_factors():
    from wholecell_py.runner import load_model

    graph = load_model(ROOT / "reference/snapshot_t0.json", seed=1)
    p = graph.processes["Translation"]
    p.copyFromState()
    active = np.flatnonzero(
        np.asarray(p.ribosome.states).ravel() == p.ribosome.activeValue
    )
    row = active[0]
    monomer = int(np.asarray(p.polypeptide.boundMRNAs).ravel()[row]) - 1
    length = np.asarray(p.polypeptide.monomerLengths).ravel()[monomer]
    p.polypeptide.nascentMonomerLengths.reshape(-1)[row] = length
    p.ribosome.mRNAPositions.reshape(-1)[row] = length
    before_factors = np.asarray(p.enzymes).ravel()[[7, 8]].copy()
    before_monomers = np.asarray(p.monomers).sum()
    p.evolveState()
    assert np.asarray(p.monomers).sum() > before_monomers
    np.testing.assert_array_equal(
        np.asarray(p.enzymes).ravel()[[7, 8]], before_factors
    )
