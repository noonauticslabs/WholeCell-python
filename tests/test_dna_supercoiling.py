import numpy as np
from types import SimpleNamespace as NS
import json
from pathlib import Path
from types import SimpleNamespace

from wholecell_py.processes.dna_supercoiling import DNASupercoiling
from wholecell_py.process_base import Process
from wholecell_py.chromosome_process_aspect import ChromosomeProcessAspect


def test_kb_initializer_composes_process_chromosome_and_tu_maps(monkeypatch):
    process = DNASupercoiling()
    process.gene = NS(wholeCellModelIDs=["MG_003", "MG_203", "MG_122"])
    process.rna = NS(
        nascentRNAGeneComposition=np.array(
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
        )
    )
    process.chromosome = NS(
        transcriptionUnitStartCoordinates=np.array([10, 20, 30, 40])
    )
    constants = dict(
        gyraseMeanDwellTime=1,
        gyraseSigmaLimit=-0.1,
        topoISigmaLimit=0.1,
        topoIVSigmaLimit=-0.1,
        gyraseDeltaLK=-1,
        topoIDeltaLK=1,
        topoIVDeltaLK=-1,
        gyraseActivityRate=2,
        topoIActivityRate=3,
        topoIVActivityRate=4,
        gyraseATPCost=2,
        topoIATPCost=0,
        topoIVATPCost=0,
        topoILogisiticConst=1,
        gyrLogisiticConst=1,
        foldChangeSlopes=np.ones(2),
        foldChangeIntercepts=np.zeros(2),
        foldChangeLowerSigmaLimit=-1,
        foldChangeUpperSigmaLimit=1,
        enzymeDNAFootprints=np.array([3, 5, 7]),
        enzymeGlobalIndexs=np.array([1, 2, 3]),
        enzymeMonomerLocalIndexs=np.array([1, 2, 3]),
        enzymeComplexLocalIndexs=np.array([], int),
    )
    monkeypatch.setattr(
        Process,
        "initializeConstants",
        lambda self, kb, sim, options=None: None,
    )
    monkeypatch.setattr(
        ChromosomeProcessAspect,
        "initializeConstants",
        lambda self, kb, sim: None,
    )

    process.initializeConstants(object(), object(), **constants)

    np.testing.assert_array_equal(process.tuIndexs, [1, 2, 3, 4])
    np.testing.assert_array_equal(process.tuCoordinates, [10, 20, 30, 40])
    assert len(process.enzymeProperties) == 3


from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.chromosome import Chromosome


def process(**overrides):
    defaults = dict(
        gyraseMeanDwellTime=45,
        gyraseSigmaLimit=-0.1,
        topoISigmaLimit=0,
        topoIVSigmaLimit=0,
        gyraseDeltaLK=-2,
        topoIDeltaLK=1,
        topoIVDeltaLK=-2,
        gyraseActivityRate=1.2,
        topoIActivityRate=1.0,
        topoIVActivityRate=2.5,
        gyraseATPCost=2,
        topoIATPCost=0,
        topoIVATPCost=2,
        topoILogisiticConst=100,
        gyrLogisiticConst=-100,
    )
    defaults.update(overrides)
    p = DNASupercoiling(**defaults)
    p.chromosome = type("C", (), {"equilibriumSuperhelicalDensity": -0.06})()
    p.enzymeProperties = p.buildEnzymeProperties()
    return p


def test_source_enzyme_activity_functions_and_constants():
    p = process()
    props = p.enzymeProperties
    assert [x["deltaLK"] for x in props] == [-2, -2, 1]
    assert [x["activityRate"] for x in props] == [1.2, 2.5, 1.0]
    assert np.isclose(props[0]["probOfActivityFunc"](-0.06, p.chromosome), 0.5)
    assert props[1]["probOfActivityFunc"](-0.2, p.chromosome) == 1
    assert np.isclose(props[2]["probOfActivityFunc"](-0.06, p.chromosome), 0.5)


def test_resource_current_uses_enzyme_and_bound_gyrase():
    p = process()
    p.chromosome.superhelicalDensity = np.array([[-0.06]])
    p.substrates = np.zeros((5, 1))
    p.enzymes = np.array([[2.0], [0.0], [0.0]])
    p.boundEnzymes = np.array([[1.0], [0.0], [0.0]])
    result = p.calcResourceRequirements_Current()
    assert np.array_equal(result[:, 0], [4, 0, 0, 4, 0])


def test_fold_change_uses_clamped_sigma_and_chromosome_copy():
    p = process(
        foldChangeSlopes=np.array([2.0]),
        foldChangeIntercepts=np.array([1.0]),
        foldChangeLowerSigmaLimit=-0.08,
        foldChangeUpperSigmaLimit=0.07,
        tuIndexs=np.array([1]),
        tuCoordinates=np.array([5]),
    )
    c = p.chromosome
    c.transcriptionUnitStartCoordinates = np.array([5, 10])
    c.relaxedBasesPerTurn = 10
    result = p.calcRNAPolymeraseBindingProbFoldChange(
        np.array([[1, 1]]), np.array([10]), np.array([0.92])
    )
    assert np.allclose(result, [[0.84, 1.0], [1.0, 1.0]])


def test_snapshot_current_requirements_match_reference_values():
    root = Path(__file__).parents[1]
    registry = {
        "edu.stanford.covert.cell.sim.process.DNASupercoiling": DNASupercoiling,
        "edu.stanford.covert.cell.sim.state.Chromosome": Chromosome,
    }
    graph = load_snapshot(
        root / "reference/snapshot_t0.json",
        class_registry=registry,
        runtime_values=True,
    )
    p = graph.processes["DNASupercoiling"]
    p.chromosome = graph.states["Chromosome"]
    p.enzymeProperties = p.buildEnzymeProperties()
    actual = p.calcResourceRequirements_Current().reshape(-1)
    expected = np.asarray(
        json.loads((root / "reference/requirements_t0.json").read_text())[
            "DNASupercoiling"
        ]["data"]
    )
    assert np.allclose(actual, expected)


class EmptySparseMatrix:
    def find(self):
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)


class SparseMatrix:
    def __init__(self, positions, values):
        self.positions = np.asarray(positions, dtype=int).reshape((-1, 2))
        self.values = np.asarray(values, dtype=int).reshape(-1)

    def find(self):
        return self.positions, self.values


class ScriptedRand:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def rand(self, *shape):
        self.calls.append(shape)
        return np.full(shape, self.value)


def test_two_full_chromosome_topoi_rounding_uses_source_random_tie_break():
    p = process()
    p.chromosome = SimpleNamespace(
        monomerBoundSites=EmptySparseMatrix(),
        complexBoundSites=EmptySparseMatrix(),
        damagedSites=EmptySparseMatrix(),
        monomerDNAFootprints=np.array([1.0]),
        complexDNAFootprints=np.array([1.0]),
    )
    p.enzymeDNAFootprints = np.array([1.0, 1.0, 1.0])
    p.enzymes = np.array([[0.0], [0.0], [3.0]])
    lengths = np.array([100, 100])

    p.randStream = ScriptedRand(0.25)
    first_copy_up = p._transient_topoi_binding_complete_chromosomes(lengths, 2)
    p.randStream = ScriptedRand(0.75)
    second_copy_up = p._transient_topoi_binding_complete_chromosomes(
        lengths, 2
    )

    # MATLAB DNASupercoiling.m:405-425 allocates 1.5 to each copy then uses
    # one rand draw to choose which copy receives the half-up tie.
    assert np.array_equal(first_copy_up, [2.0, 1.0])
    assert np.array_equal(second_copy_up, [1.0, 2.0])
    assert p.randStream.calls == [(1, 1)]


def test_one_full_chromosome_topoi_uses_all_bound_sites_without_random_rounding():
    p = process()
    p.chromosome = SimpleNamespace(
        monomerBoundSites=SparseMatrix([[1, 1]], [1]),
        complexBoundSites=EmptySparseMatrix(),
        damagedSites=EmptySparseMatrix(),
        monomerDNAFootprints=np.array([100.0]),
        complexDNAFootprints=np.array([1.0]),
    )
    p.enzymeDNAFootprints = np.array([1.0, 1.0, 1.0])
    p.enzymes = np.array([[0.0], [0.0], [3.0]])
    p.randStream = ScriptedRand(0.25)

    allocation = p._transient_topoi_binding_complete_chromosomes(
        np.array([100]), 2
    )

    assert np.array_equal(allocation, [0.0])
    assert p.randStream.calls == []


def test_two_full_chromosome_groups_compartments_as_source_columns_1_2_and_3_4():
    p = process()
    p.chromosome = SimpleNamespace(
        monomerBoundSites=SparseMatrix(
            [[1, 1], [1, 2], [1, 3], [1, 4]], [1, 2, 3, 4]
        ),
        complexBoundSites=EmptySparseMatrix(),
        damagedSites=SparseMatrix(
            [[2, 1], [2, 2], [2, 3], [2, 4]], [1, 1, 1, 1]
        ),
        monomerDNAFootprints=np.array([1.0, 5.0, 3.0, 3.0]),
        complexDNAFootprints=np.array([1.0]),
    )
    p.enzymeDNAFootprints = np.array([1.0, 1.0, 20.0])
    p.enzymes = np.array([[0.0], [0.0], [9.0]])
    lengths = np.array([100, 100])

    p.randStream = ScriptedRand(0.75)
    second_copy_up = p._transient_topoi_binding_complete_chromosomes(
        lengths, 2
    )

    assert np.array_equal(second_copy_up, [4.0, 5.0])


def test_two_full_chromosome_rounding_matches_source_for_negative_half_ties():
    p = process()
    p.chromosome = SimpleNamespace(
        monomerBoundSites=SparseMatrix([[1, 1], [1, 3]], [1, 2]),
        complexBoundSites=EmptySparseMatrix(),
        damagedSites=EmptySparseMatrix(),
        monomerDNAFootprints=np.array([101.5, 101.5]),
        complexDNAFootprints=np.array([1.0]),
    )
    p.enzymeDNAFootprints = np.array([1.0, 1.0, 1.0])
    p.enzymes = np.array([[0.0], [0.0], [3.0]])
    p.randStream = ScriptedRand(0.25)

    allocation = p._transient_topoi_binding_complete_chromosomes(
        np.array([100, 100]), 2
    )

    # ComputationUtil.roundHalfDown uses floor + (rem > .5); for -1.5,
    # rem(-1.5, 1) is -.5, so both copies round down to -2.
    assert np.array_equal(allocation, [-2.0, -2.0])


def test_fragmented_topoi_complex_sites_use_source_monomer_footprint_lookup():
    p = process()
    p.chromosome = SimpleNamespace(
        monomerDNAFootprints=np.array([2.0]),
        complexDNAFootprints=np.array([9.0]),
    )
    space = p._topoi_accessible_space(
        1,
        1,
        20,
        np.empty((0, 2), int),
        np.empty(0, int),
        np.array([[3, 1]]),
        np.array([1]),
        np.empty((0, 2), int),
    )

    # Source DNASupercoiling.m:453 uses monomerDNAFootprints for the complex
    # index here; complete-chromosome branches use complexDNAFootprints.
    assert space == 18


def test_fragmented_topoi_compartment_group_compares_to_raw_source_strand():
    p = process()
    p.chromosome = SimpleNamespace(
        monomerDNAFootprints=np.array([2.0]),
        complexDNAFootprints=np.array([9.0]),
    )
    space = p._topoi_accessible_space(
        1,
        3,
        20,
        np.empty((0, 2), int),
        np.empty(0, int),
        np.array([[3, 3]]),
        np.array([1]),
        np.empty((0, 2), int),
    )

    # MATLAB lines 434-448 map bound compartments to 1-based copy IDs 1/2,
    # but compare those IDs to raw dsPosStrands(:,2), which is 3 here.
    assert space == 20
