import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wholecell_py.processes.ribosome_assembly import RibosomeAssembly
from wholecell_py.randstream import RandStream
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot

ROOT = Path(__file__).parents[1]


def from_snapshot():
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=implemented_registry(),
    )
    # The runtime registry hydrates this process in place, preserving snapshot
    # references while providing its executable methods.
    return graph, graph.processes["RibosomeAssembly"]


def test_resource_requirements_match_matlab_t0_reference():
    _, process = from_snapshot()
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["RibosomeAssembly"]
    np.testing.assert_array_equal(
        process.calcResourceRequirements_Current(),
        np.asarray(expected["data"]).reshape(expected["shape"], order="F"),
    )


def test_copy_evolve_and_copy_to_state_roundtrip():
    graph, process = from_snapshot()
    sim = graph.simulation
    snapshot_rnas = np.asarray(process.RNAs).copy()
    process.compartment, process.gene = sim.compartment, sim.gene
    process.geometry, process.metabolite, process.rna = (
        graph.states["Geometry"],
        graph.states["Metabolite"],
        graph.states["Rna"],
    )
    process.monomer, process.complex = (
        graph.states["ProteinMonomer"],
        graph.states["ProteinComplex"],
    )
    process.copyFromState()
    np.testing.assert_array_equal(process.RNAs[:, :, 0], snapshot_rnas)
    process.randStream = RandStream(seed=11)
    before = np.asarray(process.substrates).copy()
    process.evolveState()
    assert np.all(np.asarray(process.substrates) >= 0)
    change = np.asarray(process.substrates) - before
    np.testing.assert_array_equal(
        change[:, 0],
        [
            change[0, 0],
            -change[0, 0],
            -change[0, 0],
            change[0, 0],
            -change[0, 0],
        ],
    )
    process.copyToState()
    rna_idx = (
        np.asarray(process.rna.matureIndexs)
        .reshape(-1)
        .astype(int)[
            np.asarray(process.rna.matureRRNAIndexs)
            .reshape(-1)
            .astype(int)[
                np.asarray(process.rnaGlobalIndexs).astype(int).ravel() - 1
            ]
            - 1
        ]
    )
    np.testing.assert_array_equal(
        process.rna.counts[
            rna_idx - 1,
            np.asarray(process.compartment.cytosolIndexs).astype(int).ravel()
            - 1,
            0,
        ],
        process.RNAs[:, 0, 0],
    )


def test_lifecycle_requirements_and_initialize_state():
    _, process = from_snapshot()
    states = SimpleNamespace(
        rnaProductions=np.zeros(
            (int(np.max(process.rna.matureRRNAIndexs)), 1)
        ),
        monomerProductions=np.zeros((len(process.monomer.matureIndexs), 1)),
    )
    bm, by, minimum, maximum = process.calcResourceRequirements_LifeCycle(
        None, states
    )
    assert bm.shape == (5, 1) and by.shape == (5, 1)
    np.testing.assert_array_equal(minimum.ravel(), [2, 2, 2, 2, 2, 2])
    assert np.isinf(maximum).all()
    process.initializeState()
    assert process.complexs.shape == (2, 1)


def test_initialize_constants_reports_unavailable_knowledge_base_behavior():
    with np.testing.assert_raises_regex(NotImplementedError, "knowledge-base"):
        RibosomeAssembly().initializeConstants()


def test_kb_initializer_populates_fixed_substrate_indices():
    from types import SimpleNamespace

    graph, process = from_snapshot()
    process.wholeCellModelID = "Process_RibosomeAssembly"
    n_genes = len(process.gene.wholeCellModelIDs)
    n_complexes = int(np.max(process.complexGlobalIndexs))
    n_compartments = int(np.asarray(process.compartment.count).item())
    rna = np.zeros((n_genes, n_complexes, n_compartments))
    monomer = np.zeros_like(rna)
    cyt = int(np.asarray(process.compartment.cytosolIndexs).item()) - 1
    complex_columns = process.complexGlobalIndexs.ravel().astype(int) - 1
    rna_genes = np.asarray(process.gene.rRNAIndexs).ravel().astype(int) - 1
    monomer_genes = np.asarray(process.gene.mRNAIndexs).ravel().astype(int) - 1
    rna[rna_genes[:2], complex_columns, cyt] = 1
    monomer[monomer_genes[:2], complex_columns, cyt] = 1
    kb = SimpleNamespace(
        processes=[
            SimpleNamespace(
                wholeCellModelID="Process_RibosomeAssembly", parameters=[]
            )
        ],
        proteinComplexRNAComposition=rna,
        proteinComplexMonomerComposition=monomer,
    )
    process.initializeConstants(kb, graph.simulation)
    assert [
        process.substrateIndexs_gtp,
        process.substrateIndexs_gdp,
        process.substrateIndexs_phosphate,
        process.substrateIndexs_water,
        process.substrateIndexs_hydrogen,
    ] == [1, 2, 3, 4, 5]
    assert process.complexationCatalysisMatrix.shape == (6, 2)
    assert process.proteinComplexRNAComposition.shape == (2, 2)
    assert process.proteinComplexMonomerComposition.shape == (2, 2)
