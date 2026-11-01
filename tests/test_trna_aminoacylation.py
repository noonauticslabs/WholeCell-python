from pathlib import Path
import json
from types import SimpleNamespace as NS
import numpy as np

from wholecell_py.randstream import RandStream
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot
from wholecell_py.processes.trna_aminoacylation import TRNAAminoacylation
from wholecell_py.process_base import Process
from wholecell_py.reaction_process import ReactionProcess


ROOT = Path(__file__).parents[1]


def test_kb_initializer_adds_glutamine_trnas_and_species_network(monkeypatch):
    process = TRNAAminoacylation()
    aa = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "FMET",
    ]
    base_substrates = aa + [
        "ATP",
        "AMP",
        "ADP",
        "PPI",
        "PI",
        "H2O",
        "H",
        "FTHF10",
        "THF",
    ]
    synthetases = [
        "MG_292_TETRAMER",
        "MG_378_MONOMER",
        "MG_036_DIMER",
        "MG_113_DIMER",
        "MG_253_MONOMER",
        "MG_462_MONOMER",
        "MG_251_DIMER",
        "MG_035_DIMER",
        "MG_345_MONOMER",
        "MG_266_MONOMER",
        "MG_136_DIMER",
        "MG_021_DIMER",
        "MG_194_195_TETRAMER",
        "MG_283_DIMER",
        "MG_005_DIMER",
        "MG_375_DIMER",
        "MG_126_DIMER",
        "MG_455_DIMER",
        "MG_334_MONOMER",
    ]
    enzyme_ids = synthetases + ["MG_098_099_100_TRIMER", "MG_365_MONOMER"]
    process.compartment = NS(wholeCellModelIDs=["cytosol"], cytosolIndexs=1)
    process.rna = NS(
        wholeCellModelIDs=["pre", "tRNA", "tmRNA", "aa-tRNA", "aa-tmRNA"],
        matureIndexs=np.array([2, 3]),
        matureTRNAIndexs=np.array([1]),
        matureRNAGeneComposition=np.eye(2),
        aminoacylatedIndexs=np.array([4, 5]),
    )
    process.reactionWholeCellModelIDs = np.array(
        [["MG502_Amidotransferase"], ["MG488_Formyltransferase"]], dtype=object
    )

    def component_index(self, identifiers, kind):
        available = np.asarray(
            getattr(self, f"{kind}WholeCellModelIDs"), dtype=object
        ).ravel()
        return np.asarray(
            [
                list(available).index(x) + 1
                for x in np.asarray(identifiers, dtype=object).ravel()
            ],
            dtype=int,
        ).reshape(-1, 1)

    def initialize_reaction(self, knowledge_base, simulation, options=None):
        self.compartment = process.compartment
        self.rna = process.rna
        self.substrateWholeCellModelIDs = np.asarray(
            base_substrates, dtype=object
        ).reshape(-1, 1)
        self.enzymeWholeCellModelIDs = np.asarray(
            enzyme_ids, dtype=object
        ).reshape(-1, 1)
        self.reactionWholeCellModelIDs = process.reactionWholeCellModelIDs
        self.reactionTypes = np.array(
            [["transfer"], ["transfer"]], dtype=object
        )
        self.reactionStoichiometryMatrix = np.zeros((len(base_substrates), 2))
        self.reactionCatalysisMatrix = np.ones((2, len(enzyme_ids)))
        self.reactionModificationMatrix = np.array(
            [[np.nan, 0.0], [0.0, np.nan]]
        )
        self.reactionCoenzymeMatrix = np.zeros((2, len(base_substrates)))
        self.enzymeBounds = np.ones((2, 2))
        self.reactionBounds = np.ones((2, 2))
        self.stepSizeSec = 1

    monkeypatch.setattr(Process, "componentIndexs", component_index)
    monkeypatch.setattr(
        ReactionProcess, "initializeConstants", initialize_reaction
    )
    monkeypatch.setattr(
        Process,
        "initializeConstants",
        lambda self, kb, sim, options=None: None,
    )

    process.initializeConstants(
        NS(proteinMonomerTRNACounts=np.zeros(2)), object()
    )

    assert process.substrateWholeCellModelIDs[-1, 0] == "GLN"
    assert process.substrateIndexs_glutamine.item() == len(base_substrates) + 1
    assert process.enzymeIndexs_tRNAs.item() == len(enzyme_ids) + 1
    assert process.enzymeIndexs_tmRNA.item() == len(enzyme_ids) + 2
    np.testing.assert_array_equal(process.aminoacylatedRNAGlobalIndexs, [1, 2])
    np.testing.assert_array_equal(
        process.reactionStoichiometryMatrix[-1, :], [-1, 0]
    )
    assert process.speciesReactantMatrix.shape == (
        2,
        len(base_substrates) + 1 + len(enzyme_ids) + 2,
    )


def test_snapshot_copy_evolve_copy_to_state_and_requirements():
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=implemented_registry(),
    )
    p = graph.processes["tRNAAminoacylation"]
    assert isinstance(p, TRNAAminoacylation)
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["tRNAAminoacylation"]
    p.copyFromState()
    assert p.freeRNAs.shape == (37, 1)
    assert len(p.substrates) == 30 and len(p.enzymes) == 58
    # Reference requirements are a reaction-count vector and must retain its
    # MATLAB singleton-column shape in the serialized artifact.
    assert expected["shape"] == [30, 1]
    np.testing.assert_array_equal(
        p.calcResourceRequirements_Current(),
        np.asarray(expected["data"]).ravel(),
    )
    p.randStream = RandStream(seed=1)
    p.evolveState()
    p.copyToState()
    assert np.asarray(p.rna.counts).ndim == 3


def test_evolution_constants_reflect_in_place_network_edits():
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=implemented_registry(),
    )
    p = graph.processes["tRNAAminoacylation"]
    first = p._evolution_constants()
    net_consumption = first[-1]
    assert not np.shares_memory(
        net_consumption, np.asarray(p.speciesReactantByproductMatrix)
    )

    # Public model matrices can be edited in place; derived masks must then
    # reflect the new values on the next process call.
    net = np.asarray(p.speciesReactantByproductMatrix)
    index = tuple(np.argwhere(net != 0)[0])
    before = first[-1][index]
    net[index] = abs(net[index]) + 2
    second = p._evolution_constants()
    assert second[-1][index] == abs(net[index])
    assert second[-1][index] != before

    enzyme_indexes = np.asarray(p.speciesIndexs_enzymes)
    previous = int(enzyme_indexes.flat[0])
    enzyme_indexes.flat[0] = previous + 1
    third = p._evolution_constants()
    assert third[5][0] == previous


class ScriptedRandStream:
    def __init__(self, rounded_limits, draws):
        self.rounded_limits = [
            np.asarray(value, dtype=float) for value in rounded_limits
        ]
        self.draws = [
            np.asarray(value, dtype=float).reshape(-1) for value in draws
        ]
        self.round_calls = []
        self.rand_shapes = []

    def stochasticRound(self, values):
        self.round_calls.append(np.asarray(values).copy())
        return self.rounded_limits.pop(0)

    def rand(self, *shape):
        self.rand_shapes.append(shape)
        values = self.draws.pop(0)
        return values.reshape(shape)


def make_greedy_process(free, enzymes, metabolite, rounded_limits, draws):
    free = np.asarray(free, dtype=float)
    enzymes = np.asarray(enzymes, dtype=float)
    net = np.zeros((3, 9), dtype=float)
    for reaction in range(3):
        net[reaction, 2] = 1
        net[reaction, 3 + reaction] = 1
        net[reaction, 6 + reaction] = 1
    need = np.zeros_like(net)
    need[np.arange(3), 6 + np.arange(3)] = 1
    stoichiometry = np.zeros((3, 3), dtype=float)
    stoichiometry[2, :] = -1
    process = TRNAAminoacylation(
        reactionStoichiometryMatrix=stoichiometry,
        reactionCatalysisMatrix=np.eye(3),
        reactionModificationMatrix=np.eye(3),
        speciesReactantMatrix=need,
        speciesReactantByproductMatrix=net,
        speciesIndexs_enzymes=np.array([[4], [5], [6]]),
        substrateIndexs_water=np.array([[1]]),
        substrateIndexs_hydrogen=np.array([[2]]),
        substrates=np.array([10.0, 10.0, float(metabolite)]),
        enzymes=enzymes,
        freeRNAs=free.copy(),
        aminoacylatedRNAs=np.zeros(3),
    )
    process.randStream = ScriptedRandStream(rounded_limits, draws)
    return process


def test_source_single_draw_branch_uses_histc_right_edge_and_updates_species():
    process = make_greedy_process(
        free=[1, 1, 1],
        enzymes=[1, 1, 1],
        metabolite=1,
        rounded_limits=[[1, 1, 1], [0, 0, 0]],
        draws=[[1 / 3]],
    )
    process.evolveState()

    # MATLAB histc assigns an exact interior edge to the bin on its right.
    np.testing.assert_array_equal(process.freeRNAs, [1, 0, 1])
    np.testing.assert_array_equal(process.aminoacylatedRNAs, [0, 1, 0])
    assert process.substrates[2] == 0
    assert process.randStream.rand_shapes == [(1, 1)]
    assert len(process.randStream.round_calls) == 2
    assert not process.randStream.draws


def test_source_batch_branch_counts_histc_bins_and_updates_species_once():
    process = make_greedy_process(
        free=[3, 3, 3],
        enzymes=[3, 3, 3],
        metabolite=9,
        rounded_limits=[[3, 3, 3], [0, 0, 0]],
        draws=[[0, 0.5, 0.999]],
    )
    process.evolveState()

    # With equal reaction limits the three bins are [0,1/3), [1/3,2/3),
    # [2/3,1.1); the final draw must stay in the last reaction's bin.
    np.testing.assert_array_equal(process.freeRNAs, [2, 2, 2])
    np.testing.assert_array_equal(process.aminoacylatedRNAs, [1, 1, 1])
    assert process.substrates[2] == 6
    assert process.randStream.rand_shapes == [(3, 1)]
    assert len(process.randStream.round_calls) == 2
    assert not process.randStream.draws


def test_source_batch_branch_preserves_final_reaction_edge_when_cdf_ends_at_one():
    process = make_greedy_process(
        free=[2, 2, 2],
        enzymes=[2, 2, 2],
        metabolite=6,
        rounded_limits=[[2, 2, 2], [0, 0, 0]],
        draws=[[0.999, 0.999]],
    )
    process.evolveState()

    np.testing.assert_array_equal(process.freeRNAs, [2, 2, 0])
    np.testing.assert_array_equal(process.aminoacylatedRNAs, [0, 0, 2])
    assert process.substrates[2] == 4
    assert process.randStream.rand_shapes == [(2, 1)]
    assert len(process.randStream.round_calls) == 2


def test_source_initial_infinite_limit_is_marked_inactive():
    process = make_greedy_process(
        free=[2, 2, 2],
        enzymes=[2, 2, 2],
        metabolite=6,
        rounded_limits=[[2, 2, 2]],
        draws=[[0.0]],
    )
    process.speciesReactantMatrix[:, :] = 0
    process.evolveState()

    # MATLAB's initial cleanup maps each all-Inf limit to zero, marking every
    # reaction inactive despite ample dynamic resources.
    np.testing.assert_array_equal(process.freeRNAs, [2, 2, 2])
    np.testing.assert_array_equal(process.aminoacylatedRNAs, [0, 0, 0])
    assert process.randStream.rand_shapes == []
    assert len(process.randStream.round_calls) == 1
    assert len(process.randStream.rounded_limits) == 0
