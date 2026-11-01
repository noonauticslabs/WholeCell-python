import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from wholecell_py.processes.dna_damage import DNADamage
from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.chromosome import Chromosome


def test_snapshot_current_requirements_match_reference_values():
    root = Path(__file__).parents[1]
    registry = {
        "edu.stanford.covert.cell.sim.process.DNADamage": DNADamage,
        "edu.stanford.covert.cell.sim.state.Chromosome": Chromosome,
    }
    graph = load_snapshot(
        root / "reference/snapshot_t0.json",
        class_registry=registry,
        runtime_values=True,
    )
    p = graph.processes["DNADamage"]
    p.chromosome = graph.states["Chromosome"]
    actual = p.calcResourceRequirements_Current().reshape(-1)
    expected = np.asarray(
        json.loads((root / "reference/requirements_t0.json").read_text())[
            "DNADamage"
        ]["data"]
    )
    assert np.allclose(actual, expected)


def test_vulnerable_motif_counts_and_expected_radiation_rate():
    p = DNADamage()
    p.chromosome = type(
        "Chromosome",
        (),
        {
            "polymerizedRegions": np.array([[100]]),
            "sequenceGCContent": 0.5,
            "damagedSites": np.zeros((100, 2)),
        },
    )()
    p.reactionStoichiometryMatrix = np.zeros((1, 2))
    p.reactionVulnerableMotifs = ["A", "GC"]
    p.reactionVulnerableMotifTypes = ["sequence", "sequence"]
    p.reactionBounds = np.array([[0, 2], [0, 3]])
    p.reactionRadiation = np.array([0, 1])
    p.substrates = np.array([4])
    assert np.allclose(p.calcNumberVulnerableSites(), [25, 6.25])
    assert np.allclose(p.calcExpectedReactionRates(), [50, 75])


def test_dna_damage_partition_moves_radicals_and_keeps_spontaneous_base_loss_free():
    process = DNADamage()
    process.reactionWholeCellModelIDs = np.array([["base-loss"]], dtype=object)
    process.reactionTypes = np.array([["spontaneous base loss"]], dtype=object)
    process.reactionStoichiometryMatrix = np.array(
        [[-1.0], [0], [0], [0], [0], [-1.0], [-2.0], [0], [0]]
    )
    process.reactionBounds = np.array([[3.0, 7.0]])
    process.substrateGlobalIndexs = np.arange(1, 10)
    process.substrateIndexs_unmodifiedNucleobases = np.array([1, 2, 3, 4])
    process.substrateIndexs_unmodifiedNucleosides = np.array([], dtype=int)
    process.substrateIndexs_unmodifiedDNMPs = np.array([], dtype=int)
    process.substrateIndexs_modifiedNucleobases = np.array([], dtype=int)
    process.substrateIndexs_modifiedNucleosides = np.array([], dtype=int)
    process.substrateIndexs_modifiedDNMPs = np.array([], dtype=int)
    process.substrateIndexs_DR5P = np.array([2])
    process.substrateIndexs_dRibose5P_dRibose5P = np.array([3])
    process.substrateIndexs_radiation = np.array([4, 5])
    process.substrateIndexs_hydrogen = np.array([8])
    process.substrateIndexs_hydroxylRadical = np.array([7])
    process.substrateIndexs_water = np.array([6])
    process.reactionIndexs_spontaneousBaseLoss = np.array([1])
    process.reactionIndexs_strandBreak = np.array([], dtype=int)

    process._initialize_reaction_partitions()

    assert process.reactionDNAStoichiometryMatrix[0, 0] == 0
    assert process.reactionSmallMoleculeStoichiometryMatrix[0, 0] == -1
    assert process.reactionSmallMoleculeStoichiometryMatrix[5, 0] == -3
    assert process.reactionSmallMoleculeStoichiometryMatrix[6, 0] == 0
    assert process.reactionSmallMoleculeStoichiometryMatrix[7, 0] == 2
    assert process.reactionBounds[0, 0] == 0
    assert process.reactionDamageTypes == ["abasicSites"]


def test_dna_damage_initializes_reaction_partition_from_small_kb():
    process = DNADamage()
    process.chromosome = NS(
        monomerDNAFootprints=np.empty(0),
        complexDNAFootprints=np.empty(0),
        calculateFootprintOverhangs=lambda values: (
            values.copy(),
            values.copy(),
        ),
    )
    process.compartment = NS(
        count=1, extracellularIndexs=1, cytosolIndexs=1, membraneIndexs=1
    )
    ids = [
        "AD",
        "CSN",
        "GN",
        "THY",
        "DA",
        "DC",
        "DG",
        "DT",
        "DAMP",
        "DCMP",
        "DGMP",
        "DTMP",
        "DR5P",
        "dRibose5P_dRibose5P",
        "UVB_radiation",
        "gamma_radiation",
        "H",
        "hydroxyl_radical",
        "H2O",
        "URA",
        "URI",
        "DUMP",
        "modified-base",
    ]
    categories = [""] * len(ids)
    categories[ids.index("URA")] = "modified nucleobase"
    categories[ids.index("URI")] = "modified deoxyribonucleoside"
    categories[ids.index("DUMP")] = (
        "modified deoxyribonucleotide monophosphate"
    )
    categories[ids.index("modified-base")] = "modified nucleobase"
    process.stimulus = NS(wholeCellModelIDs=[], names=[])
    process.metabolite = NS(
        wholeCellModelIDs=ids,
        names=ids,
        molecularWeights=np.ones(len(ids)),
        hydrophobicIndexs=[],
    )
    for name in ("rna", "monomer", "complex"):
        setattr(
            process,
            name,
            NS(
                wholeCellModelIDs=[],
                names=[],
                matureIndexs=[],
                boundIndexs=[],
                molecularWeights=[],
                compartments=[],
            ),
        )
    process.gene = NS(mRNAIndexs=[])
    process_record = NS(
        wholeCellModelID=process.wholeCellModelID,
        parameters=[],
        reactions=[
            NS(
                idx=1,
                wholeCellModelID="base-loss",
                name="Base loss",
                type="spontaneous base loss",
            )
        ],
    )
    kb = NS(
        processes=[process_record],
        metabolites=[
            NS(wholeCellModelID=i, category=c) for i, c in zip(ids, categories)
        ],
        reactionStimuliStoichiometryMatrix=np.zeros((0, 1, 1)),
        reactionMetaboliteStoichiometryMatrix=np.zeros((len(ids), 1, 1)),
        reactionRNAStoichiometryMatrix=np.zeros((0, 1, 1)),
        reactionProteinMonomerStoichiometryMatrix=np.zeros((0, 1, 1)),
        reactionProteinComplexStoichiometryMatrix=np.zeros((0, 1, 1)),
        reactionCatalysisMatrix=np.zeros((1, 0, 1)),
        reactionModificationMatrix=np.zeros((1, 0, 1)),
        reactionCoenzymeMatrix=np.zeros((1, len(ids), 1)),
        enzymeBounds=np.zeros((1, 2)),
        reactionBounds=np.array([[0.0, 1.0]]),
    )
    # Keep the required substrate IDs in the reaction network, as the MATLAB
    # initializer only force-adds DA/DC/DG/DT and gets other IDs from reactions.
    for identifier in (
        "AD",
        "UVB_radiation",
        "gamma_radiation",
        "H",
        "hydroxyl_radical",
        "H2O",
        "DR5P",
        "dRibose5P_dRibose5P",
    ):
        kb.reactionMetaboliteStoichiometryMatrix[
            ids.index(identifier), 0, 0
        ] = -1
    simulation = NS(stepSizeSec=1, verbosity=0, seed=1)
    simulation.state = lambda name: (
        process.metabolite if name == "Metabolite" else process.chromosome
    )

    process.initializeConstants(kb, simulation)

    assert process.reactionIndexs_spontaneousBaseLoss.tolist() == [1]
    assert (
        process.reactionSmallMoleculeStoichiometryMatrix[
            int(process.substrateIndexs_unmodifiedNucleobases[0]) - 1, 0
        ]
        == -1
    )
    assert (
        process.reactionDNAStoichiometryMatrix[
            int(process.substrateIndexs_unmodifiedNucleobases[0]) - 1, 0
        ]
        == 0
    )
    assert process.reactionDamageTypes == ["abasicSites"]
