import json
from pathlib import Path

import numpy as np

from wholecell_py.snapshot import load_snapshot
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.processes.protein_decay import ProteinDecay
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]


def test_protein_decay_runtime_requirements_match_registry():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.ProteinDecay"] = (
        ProteinDecay
    )
    graph = load_snapshot(
        ROOT / "reference/snapshot_t0.json",
        class_registry=registry,
        runtime_values=True,
    )
    process = graph.processes["ProteinDecay"]
    actual = np.asarray(process.calcResourceRequirements_Current())
    expected = json.loads(
        (ROOT / "reference/requirements_t0.json").read_text()
    )["ProteinDecay"]
    expected = np.asarray(expected["data"]).reshape(
        expected["shape"], order="F"
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_refolding_preserves_species_compartment_and_atp_products():
    from types import SimpleNamespace
    from wholecell_py.randstream import RandStream

    p = ProteinDecay()
    p.compartment = SimpleNamespace(
        cytosolIndexs=1, terminalOrganelleCytosolIndexs=2
    )
    p.monomer = SimpleNamespace(
        misfoldedIndexs=np.array([3, 4]), matureIndexs=np.array([1, 2])
    )
    p.complex = SimpleNamespace(
        misfoldedIndexs=np.array([2]), matureIndexs=np.array([1])
    )
    p.monomers = np.zeros((4, 2))
    p.monomers[2, 1] = 1
    p.complexs = np.zeros((2, 2))
    p.substrates = np.array([[2.0], [2.0], [0.0], [0.0], [0.0]])
    for name, index in zip(
        ("atp", "water", "adp", "phosphate", "hydrogen"), range(1, 6)
    ):
        setattr(p, "substrateIndexs_" + name, index)
    p.enzymes = np.array([[1.0]])
    p.enzymeIndexs_clpBProtease = 1
    p.randStream = RandStream(seed=1)
    p.evolveState_RefoldProteins()
    expected = np.zeros((4, 2))
    expected[0, 1] = 1
    np.testing.assert_array_equal(p.monomers, expected)
    np.testing.assert_array_equal(p.substrates, np.ones((5, 1)))


def test_aborted_polypeptide_water_requirement_matches_matlab_sequence_length_minus_one():
    from types import SimpleNamespace

    p = ProteinDecay()
    p.compartment = SimpleNamespace(
        cytosolIndexs=1, terminalOrganelleCytosolIndexs=1
    )
    p.monomer = SimpleNamespace(decayRates=np.empty(0))
    p.complex = SimpleNamespace(decayRates=np.empty(0))
    p.monomers = np.empty((0, 1))
    p.complexs = np.empty((0, 1))
    p.substrates = np.zeros((5, 1))
    for name, index in zip(
        ("atp", "water", "adp", "phosphate", "hydrogen"), range(1, 6)
    ):
        setattr(p, "substrateIndexs_" + name, index)
    p.complexDecayReactions = np.empty((5, 0))
    p.monomerDecayReactions = np.empty((5, 0))
    p.monomerLonProteaseCleavages = np.empty(0)
    p.enzymes = np.array([0.0])
    p.enzymeIndexs_clpBProtease = 1
    p.enzymeIndexs_peptidases = np.array([1])
    p.enzymeIndexs_lonProtease = 1
    p.enzymeIndexs_ftsHProtease = 1
    p.lonProteaseEnergyCost = 0
    p.lonProteaseSpecificRate = 0
    p.ftsHProteaseEnergyCost = 0
    p.ftsHProteaseSpecificRate = 0
    p.ftsHProteaseFragmentLength = 10
    p.polypeptide = SimpleNamespace(
        abortedPolypeptides=np.ones((2, 3)),
        abortedSequenceLengths=np.array([0, 4]),
    )

    actual = p.calcResourceRequirements_Current()

    # ProteinDecay.m:527 is sum(abortedSequenceLengths - 1), including zero lengths.
    assert actual[1, 0] == 2  # (0 - 1) + (4 - 1)


def test_kb_initializer_builds_decay_matrices_and_converts_complex_gtp_hydrolysis():
    p = ProteinDecay()
    enzyme_ids = list(p.enzymeWholeCellModelIDs)
    metabolite_ids = sorted(
        set(
            [
                "GTP",
                "GDP",
                "H2O",
                "PI",
                "H",
                "ATP",
                "ADP",
                "NH3",
                "FOR",
                "ALA",
                "ARG",
                "ASN",
                "ASP",
                "CYS",
                "GLN",
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
        )
    )
    met_index = {value: i for i, value in enumerate(metabolite_ids)}
    p.compartment = NS(
        count=1, cytosolIndexs=1, terminalOrganelleCytosolIndexs=1
    )
    p.stimulus = NS(wholeCellModelIDs=[], names=[])
    p.metabolite = NS(
        wholeCellModelIDs=metabolite_ids,
        names=metabolite_ids,
        molecularWeights=np.ones(len(metabolite_ids)),
        hydrophobicIndexs=[],
    )
    p.gene = NS(
        rRNAIndexs=[10],
        sRNAIndexs=[11],
        mRNAIndexs=np.arange(1, 10),
        wholeCellModelIDs=[f"g{i}" for i in range(11)],
    )
    p.rna = NS(
        wholeCellModelIDs=["rRNA", "sRNA"],
        matureIndexs=[1, 2],
        matureRRNAIndexs=[1],
        matureSRNAIndexs=[2],
        damagedIndexs=[1, 2],
        boundIndexs=[1, 2],
    )
    p.monomer = NS(
        wholeCellModelIDs=enzyme_ids * 10,
        names=enzyme_ids * 10,
        matureIndexs=np.arange(
            5 * len(enzyme_ids) + 1, 6 * len(enzyme_ids) + 1
        ),
        nascentIndexs=np.arange(1, len(enzyme_ids) + 1),
        processedIIndexs=np.arange(
            len(enzyme_ids) + 1, 2 * len(enzyme_ids) + 1
        ),
        processedIIIndexs=np.arange(
            2 * len(enzyme_ids) + 1, 3 * len(enzyme_ids) + 1
        ),
        signalSequenceIndexs=np.arange(
            3 * len(enzyme_ids) + 1, 4 * len(enzyme_ids) + 1
        ),
        foldedIndexs=np.arange(
            4 * len(enzyme_ids) + 1, 5 * len(enzyme_ids) + 1
        ),
        inactivatedIndexs=np.arange(
            6 * len(enzyme_ids) + 1, 7 * len(enzyme_ids) + 1
        ),
        boundIndexs=np.arange(
            7 * len(enzyme_ids) + 1, 8 * len(enzyme_ids) + 1
        ),
        misfoldedIndexs=np.arange(
            8 * len(enzyme_ids) + 1, 9 * len(enzyme_ids) + 1
        ),
        damagedIndexs=np.arange(
            9 * len(enzyme_ids) + 1, 10 * len(enzyme_ids) + 1
        ),
        lengths=np.arange(1, 10 * len(enzyme_ids) + 1),
        molecularWeights=np.ones(10 * len(enzyme_ids)),
        compartments=np.ones(10 * len(enzyme_ids)),
    )
    p.complex = NS(
        wholeCellModelIDs=["C"] * 6,
        names=["complex"] * 6,
        nascentIndexs=[1],
        matureIndexs=[2],
        inactivatedIndexs=[3],
        boundIndexs=[4],
        misfoldedIndexs=[5],
        damagedIndexs=[6],
        molecularWeights=np.ones(6),
        compartments=np.ones(6),
    )
    n_met = len(metabolite_ids)
    zeros = np.zeros(n_met)
    monomers = []
    for i in range(len(enzyme_ids)):
        decay = zeros.copy()
        if i == 0:
            for amino_acid in (
                "ALA",
                "ARG",
                "ASN",
                "ASP",
                "CYS",
                "GLN",
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
            ):
                decay[met_index[amino_acid]] = -1
        monomers.append(
            NS(
                decayReaction=decay,
                processedISequenceDecayReaction=zeros.copy(),
                processedIISequenceDecayReaction=zeros.copy(),
                signalSequenceDecayReaction=zeros.copy(),
                foldedSequenceDecayReaction=zeros.copy(),
                matureSequenceDecayReaction=zeros.copy(),
            )
        )
    complex_decay = zeros.copy()
    complex_decay[[met_index[x] for x in ("GTP", "H2O", "PI", "H")]] = [
        1,
        1,
        -1,
        -1,
    ]
    kb = NS(
        processes=[NS(wholeCellModelID=p.wholeCellModelID, parameters=[])],
        proteinMonomers=monomers,
        proteinComplexs=[
            NS(decayReaction=complex_decay, matureDecayReaction=zeros.copy())
        ],
        proteinComplexAllRNAComposition=np.zeros((11, 1, 1)),
        proteinComplexAllMonomerComposition=np.zeros((11, 1, 1)),
    )
    kb.proteinComplexAllRNAComposition[9, 0, 0] = 1
    kb.proteinComplexAllMonomerComposition[0, 0, 0] = 1
    p.lonProteaseFragmentLength = 20

    p.initializeConstants(kb, NS(stepSizeSec=1, verbosity=0, seed=5))

    assert p.substrateIndexs_aminoAcids.size == 21
    assert p.monomerDecayReactions.shape == (
        len(p.substrateWholeCellModelIDs),
        90,
    )
    assert p.complexDecayReactions.shape == (
        len(p.substrateWholeCellModelIDs),
        6,
    )
    assert (
        p.monomerDecayReactions[int(p.substrateIndexs(["ALA"]).item()) - 1, 0]
        == -1
    )
    assert (
        p.complexDecayReactions[int(p.substrateIndexs(["GDP"]).item()) - 1, 0]
        == 1
    )
    assert "GTP" not in p.substrateWholeCellModelIDs.ravel().tolist()
    assert p.monomerLonProteaseCleavages.shape == (90,)
