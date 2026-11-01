"""Structural and rejection-policy tests, not biological equivalence tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from wholecell_py import initialization as init
from wholecell_py.process_base import Process


def test_base_allocation_uses_mapping_compartments_and_clears_pools():
    p = Process()
    for prefix, ids, count in (
        ("stimulus", "stimuliWholeCellModelIDs", 2),
        ("substrate", "substrateWholeCellModelIDs", 3),
        ("enzyme", "enzymeWholeCellModelIDs", 4),
    ):
        setattr(p, ids, np.full((count, 1), "x", dtype=object))
        for kind in ("Stimulus", "Metabolite", "RNA", "Monomer", "Complex"):
            setattr(p, prefix + kind + "GlobalIndexs", np.zeros((0, 2)))
    p.boundEnzymes = np.ones((4, 2))
    p.allocateMemoryForState(1)
    assert p.stimuli.shape == (2, 2)
    assert p.substrates.shape == (3, 2)
    assert p.enzymes.shape == (4, 2)
    assert not p.boundEnzymes.any()


def test_runtime_dimensions_preserve_nontrailing_axes():
    obj = SimpleNamespace(a=np.zeros((3, 1, 1)), b=np.zeros((3, 1, 2)))
    init._runtime_dimensions(obj)
    assert obj.a.shape == (3, 1)
    assert obj.b.shape == (3, 1, 2)


def fake_graph(growth):
    reaction = SimpleNamespace(
        growth=growth, meanInitialGrowthRate=1.0, initialGrowthFilterWidth=0.1
    )
    sim = SimpleNamespace(
        state=lambda _: reaction,
        macromoleculeStateInitialization="multinomial",
        randStream=SimpleNamespace(randi=lambda _: np.array([[42]])),
    )
    return SimpleNamespace(
        simulation=sim,
        states={"Metabolite": SimpleNamespace(counts=np.ones((1, 1)))},
    )


def test_rejection_reloads_and_records_attempt_seeds(tmp_path, monkeypatch):
    template = tmp_path / "template"
    template.write_text("model constants")
    seeds = []

    def load(_, seed):
        seeds.append(seed)
        return fake_graph(0.0 if len(seeds) == 1 else 1.0)

    monkeypatch.setattr(init, "load_model", load)
    monkeypatch.setattr(init, "_initialize_once", lambda sim: None)
    graph = init.initialize_model(template, seed=7, max_attempts=2)
    assert seeds == [7, 41]
    assert len(graph.initialization_provenance["attempts"]) == 2


def test_exhaustion_and_implementation_failure_are_not_accepted(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(init, "load_model", lambda *a, **k: fake_graph(0.0))
    monkeypatch.setattr(init, "_initialize_once", lambda sim: None)
    with pytest.raises(RuntimeError, match="rejected all 2"):
        init.initialize_model(tmp_path / "unused", max_attempts=2)

    def fail(sim):
        raise ValueError("implementation defect")

    monkeypatch.setattr(init, "_initialize_once", fail)
    with pytest.raises(ValueError, match="implementation defect"):
        init.initialize_model(tmp_path / "unused")


def test_dna_initial_methylation_uses_one_based_sparse_coordinates():
    from wholecell_py.circular_sparsemat import CircularSparseMat
    from wholecell_py.processes.dna_repair import DNARepair

    p = DNARepair()
    p.chromosome = SimpleNamespace(
        damagedBases=CircularSparseMat([], [], [20, 4], 1)
    )
    p.RM_MunI_RecognitionSites = np.array([[1, 2, 3], [10, 11, 12]])
    p.RM_MunI_MethylatedPositions = np.array([[1, 3]])
    p.substrateGlobalIndexs = np.array([[17]])
    p.substrateIndexs_m6AD = np.array([[1]])
    p.initializeState()
    coordinates, values = p.chromosome.damagedBases.find()
    assert sorted(map(tuple, coordinates.tolist())) == [
        (1, 1),
        (3, 2),
        (10, 1),
        (12, 2),
    ]
    assert np.all(values == 17)


def test_molecule_ids_accept_exported_matlab_cell_strings():
    from wholecell_py.molecule_count_state import MoleculeCountState

    ids = np.empty((2, 1), dtype=object)
    ids[0, 0] = np.array([list("ATP")])
    ids[1, 0] = np.array([list("m6dAMP")])
    state = MoleculeCountState(wholeCellModelIDs=ids)
    assert state.getIndexs("m6dAMP") == 2
    assert state.getIndexs("absent") == 0


def test_ribosome_initialization_does_not_broadcast_species_across_time():
    from wholecell_py.processes.ribosome_assembly import RibosomeAssembly

    p = RibosomeAssembly()
    p.complexWholeCellModelIDs = ["A", "B"]
    p.RNAs = np.array([12.0, 7.0, 20.0]).reshape(3, 1, 1)
    p.monomers = np.array([10.0, 8.0]).reshape(2, 1, 1)
    p.complexs = np.zeros((2, 1, 1))
    p.proteinComplexRNAComposition = np.array([[2, 0], [0, 1], [0, 1]])
    p.proteinComplexMonomerComposition = np.eye(2)
    p.initializeState()
    assert p.RNAs.shape == (3, 1, 1)
    assert p.monomers.shape == (2, 1, 1)
    np.testing.assert_array_equal(p.RNAs.ravel(), [0, 0, 13])
    np.testing.assert_array_equal(p.monomers.ravel(), [4, 1])
    np.testing.assert_array_equal(p.complexs.ravel(), [6, 7])
