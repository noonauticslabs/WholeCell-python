import json
from pathlib import Path
import numpy as np
from wholecell_py.randstream import RandStream
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot
from wholecell_py.processes.macromolecular_complexation import (
    MacromolecularComplexation,
)
from types import SimpleNamespace as NS

ROOT = Path(__file__).parents[1]


def snapshot():
    registry = implemented_registry()
    registry[
        "edu.stanford.covert.cell.sim.process.MacromolecularComplexation"
    ] = MacromolecularComplexation
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json", class_registry=registry
    )
    return graph, graph.processes["MacromolecularComplexation"]


def test_snapshot_requirements_evolve_and_writeback():
    graph, p = snapshot()
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["MacromolecularComplexation"]
    p.copyFromState()
    np.testing.assert_array_equal(
        p.calcResourceRequirements_Current().reshape(
            expected["shape"], order="F"
        ),
        np.asarray(expected["data"]).reshape(expected["shape"], order="F"),
    )
    before = np.asarray(p.substrates).copy()
    p.randStream = RandStream(seed=1)
    p.evolveState()
    assert np.all(p.complexs >= 0) and np.all(p.substrates >= 0)
    # The t0 snapshot has no complete stoichiometric set in either network;
    # completion correctly takes the source's no-reaction return branch.
    np.testing.assert_array_equal(p.substrates, before)
    assert not np.any(p.complexs)
    p.copyToState()
    gi = np.asarray(p.complexGlobalIndexs, dtype=int).ravel() - 1
    ci = np.asarray(p.complexCompartmentIndexs, dtype=int).ravel() - 1
    counts = np.asarray(p.complex.counts)
    got = (
        counts.ravel(order="F")[gi + ci * counts.shape[0]]
        if counts.ndim == 2
        else counts[gi, ci, 0]
    )
    np.testing.assert_array_equal(got, np.asarray(p.complexs).reshape(-1))


def test_formed_complex_preserves_column_substrate_counts():
    p = MacromolecularComplexation()
    p.substrates = np.array([[4.0], [3.0]])
    p.complexs = np.zeros((1, 1))
    p.substrates2complexNetworks = np.array([1, 1])
    p.complexs2complexNetworks = np.array([1])
    p.complexNetworks = [np.array([[2.0], [1.0]])]
    p.complexComposition = np.array([[2.0], [1.0]])
    p.evolveState()
    np.testing.assert_array_equal(p.substrates, [[0], [1]])
    np.testing.assert_array_equal(p.complexs, [[2]])


def test_initialize_networks_matches_exported_matlab_partition():
    _, p = snapshot()
    expected_matrix = np.asarray(p.complexComposition).copy()
    expected_substrate_map = np.asarray(p.substrates2complexNetworks).copy()
    expected_complex_map = np.asarray(p.complexs2complexNetworks).copy()
    expected_networks = []
    for value in np.asarray(p.complexNetworks, dtype=object).ravel(order="F"):
        block = np.asarray(value)
        if block.dtype == object and block.size == 1:
            block = np.asarray(block.item())
        expected_networks.append(block.copy())

    p.initializeComplexNetworks()

    np.testing.assert_array_equal(p.complexComposition, expected_matrix)
    np.testing.assert_array_equal(
        p.substrates2complexNetworks, expected_substrate_map
    )
    np.testing.assert_array_equal(
        p.complexs2complexNetworks, expected_complex_map
    )
    assert len(p.complexNetworks) == len(expected_networks)
    for actual, expected in zip(p.complexNetworks, expected_networks):
        np.testing.assert_array_equal(actual, expected)


def test_kb_initializer_builds_gene_subunits_and_selected_complex_network():
    p = MacromolecularComplexation()
    p.compartment = NS(count=1, cytosolIndexs=1)
    p.gene = NS(
        wholeCellModelIDs=["gM", "gR", "gS", "gT"],
        mRNAIndexs=[1],
        rRNAIndexs=[2],
        sRNAIndexs=[3],
        tRNAIndexs=[4],
    )
    p.stimulus = NS(wholeCellModelIDs=[], names=[])
    p.metabolite = NS(
        wholeCellModelIDs=[],
        names=[],
        molecularWeights=[],
        hydrophobicIndexs=[],
    )
    p.monomer = NS(
        wholeCellModelIDs=["M"],
        names=["monomer"],
        matureIndexs=[1],
        nascentIndexs=[1],
        boundIndexs=[1],
        compartments=[1],
        molecularWeights=[2.0],
    )
    p.rna = NS(
        wholeCellModelIDs=["R", "S", "T"],
        names=["rRNA", "sRNA", "tRNA"],
        matureIndexs=[1, 2, 3],
        matureRRNAIndexs=[1],
        matureSRNAIndexs=[2],
        matureTRNAIndexs=[3],
        boundIndexs=[1, 2, 3],
        compartments=[1, 1, 1],
        molecularWeights=[1.0, 1.0, 1.0],
    )
    p.complex = NS(
        wholeCellModelIDs=["C1", "C2"],
        names=["complex1", "complex2"],
        matureIndexs=[1, 2],
        boundIndexs=[1, 2],
        compartments=[1, 1],
        molecularWeights=[5.0, 7.0],
    )
    process_record = NS(
        wholeCellModelID=p.wholeCellModelID, idx=9, parameters=[]
    )
    kb_rna = np.zeros((4, 2, 1))
    kb_monomer = np.zeros_like(kb_rna)
    kb_rna[1, 0, 0] = 1
    kb_rna[2, 0, 0] = 1
    kb_monomer[0, 0, 0] = 1
    kb = NS(
        processes=[process_record],
        proteinComplexs=[
            NS(
                complexFormationProcess=NS(wholeCellModelID=p.wholeCellModelID)
            ),
            NS(complexFormationProcess=NS(wholeCellModelID="Process_Other")),
        ],
        proteinComplexRNAComposition=kb_rna,
        proteinComplexMonomerComposition=kb_monomer,
    )
    sim = NS(stepSizeSec=1, verbosity=0, seed=3)

    p.initializeConstants(kb, sim)

    np.testing.assert_array_equal(
        p.substrateWholeCellModelIDs.ravel(), ["M", "R", "S"]
    )
    np.testing.assert_array_equal(p.complexWholeCellModelIDs.ravel(), ["C1"])
    np.testing.assert_array_equal(p.complexComposition.ravel(), [1, 1, 1])
    np.testing.assert_array_equal(
        p.complexGlobalCompartmentIndexs.ravel(), [1]
    )
