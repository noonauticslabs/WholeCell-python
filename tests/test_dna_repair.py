import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from wholecell_py.processes.dna_repair import DNARepair
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.chromosome import Chromosome
from wholecell_py.sparsemat import SparseMat

ROOT = Path(__file__).parents[1]


def snapshot():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.DNARepair"] = DNARepair
    registry["edu.stanford.covert.cell.sim.state.Chromosome"] = Chromosome
    return load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=registry,
        runtime_values=True,
    )


def test_t0_current_resource_requirements_match_matlab():
    graph = snapshot()
    process = graph.processes["DNARepair"]
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["DNARepair"]
    actual = process.calcResourceRequirements_Current()
    np.testing.assert_allclose(
        actual.reshape(expected["shape"], order="F"),
        np.asarray(expected["data"]).reshape(expected["shape"], order="F"),
        rtol=1e-12,
        atol=1e-12,
    )


def test_all_source_evolution_entry_points_exist():
    for method in DNARepair._runtime_methods:
        assert callable(getattr(DNARepair, method))


def test_lifecycle_repair_rates_include_dntp_polymerization_by_lesion_type():
    class PolymerizedRegions:
        def collapse(self, _):
            return 100

    process = DNARepair()
    process.chromosome = SimpleNamespace(
        sequenceGCContent=0.5,
        polymerizedRegions=PolymerizedRegions(),
    )
    process.dnaDamage = SimpleNamespace(
        reactionStoichiometryMatrix=np.zeros((1, 3)),
        reactionBounds=np.ones((3, 2)),
        reactionVulnerableMotifs=["A", "C", "G"],
        reactionVulnerableMotifTypes=["sequence"] * 3,
        reactionRadiationStoichiometryMatrix=np.zeros((0, 3)),
        reactionDamageTypes=[
            "abasicSites",
            "damagedBases",
            "intrastrandCrossLinks",
        ],
    )
    process.reactionWholeCellModelIDs = np.arange(17)
    process.NER_UvrABC_IncisionMargin5 = 1
    process.NER_UvrABC_IncisionMargin3 = 1
    process.NER_PcrA_StepSize = 2
    process.reactionIndexs_BER_aplyase = np.array([1])
    process.reactionIndexs_BER_drpesterase = np.array([2])
    process.reactionIndexs_BER_apendonuclease = np.array([3])
    process.reactionIndexs_BER_dr5plyase = np.array([4])
    process.reactionIndexs_BER_baseexcision = np.array([5])
    process.reactionIndexs_ligation = np.array([6])
    process.reactionIndexs_NER_incision = np.array([7])
    process.reactionIndexs_NER_excision = np.array([8])
    process.reactionIndexs_DNACleavage = np.array([9, 10, 11, 12])
    process.reactionIndexs_polymerization = np.array([13, 14, 15, 16])
    process.reactionIndexs_RM_MunI_Methylation = np.array([17])
    process.RM_MunI_RecognitionSites = np.empty((0, 2), int)
    process.stimulus = SimpleNamespace(values=np.empty(0))
    process.substrateStimulusGlobalCompartmentIndexs = np.empty(0, int)
    process.substrateStimulusLocalIndexs = np.empty(0, int)
    time = SimpleNamespace(
        replicationInitiationDuration=1,
        replicationDuration=1,
        cytokinesisDuration=1,
    )
    constants = SimpleNamespace(states=SimpleNamespace(Time=time))

    rates = process.calcExpectedReactionRates(constants)

    # Each motif has 25 vulnerable sites and integrated damage rate 112.5.
    # dNTP composition is 0.25 per base; crosslinks require two strands.
    np.testing.assert_allclose(rates[12:16], np.full(4, 112.5))


def test_lifecycle_vulnerable_damage_counts_only_nonredundant_sites():
    class PolymerizedRegions:
        def collapse(self, _):
            return 10

    process = DNARepair()
    process.chromosome = SimpleNamespace(
        sequenceGCContent=0.5,
        polymerizedRegions=PolymerizedRegions(),
        damagedBases=np.array([[1], [2]]),
        damagedSites_nonRedundant=np.array([[0], [2]]),
    )
    process.dnaDamage = SimpleNamespace(
        reactionStoichiometryMatrix=np.zeros((1, 1)),
        reactionBounds=np.ones((1, 2)),
        reactionVulnerableMotifs=[np.array([2])],
        reactionVulnerableMotifTypes=["damagedBases"],
        reactionRadiationStoichiometryMatrix=np.zeros((0, 1)),
        reactionDamageTypes=["strandBreaks"],
    )
    process.reactionWholeCellModelIDs = np.array(["ligation", "methylation"])
    process.reactionIndexs_ligation = np.array([1])
    process.reactionIndexs_RM_MunI_Methylation = np.array([2])
    process.RM_MunI_RecognitionSites = np.empty((0, 2), int)
    process.stimulus = SimpleNamespace(values=np.empty(0))
    process.substrateStimulusGlobalCompartmentIndexs = np.empty(0, int)
    process.substrateStimulusLocalIndexs = np.empty(0, int)
    constants = SimpleNamespace(
        states=SimpleNamespace(
            Time=SimpleNamespace(
                replicationInitiationDuration=1,
                replicationDuration=1,
                cytokinesisDuration=1,
            )
        )
    )

    rates = process.calcExpectedReactionRates(constants)

    # MATLAB intersects the selected lesion type with damagedSites_nonRedundant:
    # only coordinate 2 contributes, at 1 + 1.5 + 2 integrated exposure.
    np.testing.assert_allclose(rates, [4.5, 0])


def test_sparse_motif_site_count_intersects_coordinates_and_values():
    target = SparseMat([[1, 1], [2, 1]], [5, 5], [2, 1])
    nonredundant = SparseMat([[2, 1], [3, 1]], [5, 8], [3, 1])

    assert DNARepair._shared_motif_site_count(5, target, nonredundant) == 1


def test_lifecycle_radiation_rate_uses_matlab_one_based_stimulus_local_index():
    class PolymerizedRegions:
        def collapse(self, _):
            return 100

    process = DNARepair()
    process.chromosome = SimpleNamespace(
        sequenceGCContent=0.5,
        polymerizedRegions=PolymerizedRegions(),
    )
    process.dnaDamage = SimpleNamespace(
        reactionStoichiometryMatrix=np.zeros((1, 2)),
        reactionBounds=np.array([[1, 1], [1, 0]]),
        reactionVulnerableMotifs=["A", "A"],
        reactionVulnerableMotifTypes=["sequence", "sequence"],
        reactionRadiationStoichiometryMatrix=np.array([[1, 0], [0, 0]]),
        reactionDamageTypes=["strandBreaks", "strandBreaks"],
        substrateStimulusLocalIndexs=np.array([1]),
        substrateStimulusGlobalCompartmentIndexs=np.array([2]),
    )
    process.stimulus = SimpleNamespace(values=np.array([0, 3]))
    process.reactionWholeCellModelIDs = np.array(["radiated", "nonradiated"])
    process.reactionIndexs_ligation = np.array([1])
    process.reactionIndexs_RM_MunI_Methylation = np.array([2])
    process.RM_MunI_RecognitionSites = np.empty((0, 2), int)
    constants = SimpleNamespace(
        states=SimpleNamespace(
            Time=SimpleNamespace(
                replicationInitiationDuration=1,
                replicationDuration=1,
                cytokinesisDuration=1,
            )
        )
    )

    rates = process.calcExpectedReactionRates(constants)

    # MATLAB find returns the radiation row as index 1; its stimulus map is
    # also one-based and points to global stimulus slot 2, whose value is 3.
    np.testing.assert_allclose(rates, [337.5, 0])
