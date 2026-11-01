from types import SimpleNamespace as NS

import numpy as np

from wholecell_py.processes.protein_processing_i import ProteinProcessingI
from wholecell_py.processes.protein_processing_ii import ProteinProcessingII
from wholecell_py.processes.rna_modification import RNAModification
from wholecell_py.processes.rna_decay import RNADecay
from wholecell_py.process_base import Process


def _attach_process_states(
    process, metabolites, monomer_ids, complex_ids=None
):
    process.compartment = NS(
        count=4,
        extracellularIndexs=1,
        cytosolIndexs=2,
        membraneIndexs=3,
        terminalOrganelleCytosolIndexs=4,
        terminalOrganelleMembraneIndexs=3,
    )
    process.stimulus = NS(wholeCellModelIDs=[], names=[])
    process.metabolite = NS(
        wholeCellModelIDs=metabolites,
        names=metabolites,
        molecularWeights=np.ones(len(metabolites)),
        hydrophobicIndexs=[],
    )
    process.rna = NS(
        wholeCellModelIDs=["rna-pre", "rna"],
        names=["rna-pre", "rna"],
        matureIndexs=[2],
        boundIndexs=[1],
        molecularWeights=[1, 1],
        compartments=[2, 2],
    )
    process.monomer = NS(
        wholeCellModelIDs=monomer_ids,
        names=monomer_ids,
        matureIndexs=np.arange(1, len(monomer_ids) + 1),
        boundIndexs=np.arange(1, len(monomer_ids) + 1),
        molecularWeights=np.ones(len(monomer_ids)),
        compartments=np.full(len(monomer_ids), 2),
        nascentIndexs=np.array([1]),
        processedIIndexs=np.array([2]),
        processedIIIndexs=np.array([3]),
        signalSequenceIndexs=np.array([4]),
    )
    complex_ids = complex_ids or ["complex-pre", "complex"]
    process.complex = NS(
        wholeCellModelIDs=complex_ids,
        names=complex_ids,
        matureIndexs=np.arange(2, len(complex_ids) + 1),
        boundIndexs=np.ones(len(complex_ids), dtype=int),
        molecularWeights=np.ones(len(complex_ids)),
        compartments=np.full(len(complex_ids), 2),
    )


def test_protein_processing_i_initializes_from_kb_and_state_indexes():
    process = ProteinProcessingI()
    _attach_process_states(
        process,
        ["H2O", "H", "MET", "FOR"],
        ["nascent", "processed", "other", "MG_172_MONOMER"],
        ["complex-pre", "complex", "MG_106_DIMER"],
    )
    knowledge_base = NS(
        processes=[
            NS(wholeCellModelID=process.wholeCellModelID, parameters=[])
        ],
        proteinMonomerNTerminalMethionineCleavages=np.array(
            [False, True, False]
        ),
    )

    process.initializeConstants(
        knowledge_base, NS(stepSizeSec=1, verbosity=0, seed=2)
    )

    np.testing.assert_array_equal(
        process.unprocessedMonomerWholeCellModelIDs, ["nascent"]
    )
    np.testing.assert_array_equal(
        process.processedMonomerWholeCellModelIDs, ["processed"]
    )
    np.testing.assert_array_equal(
        process.nascentMonomerNTerminalMethionineCleavages,
        [False, True, False],
    )


def test_protein_processing_ii_initializes_classes_and_remaps_compartments():
    process = ProteinProcessingII()
    ids = [
        "MG_210_MONOMER",
        "MG_086_MONOMER",
        "MG_A_MONOMER",
        "MG_B_MONOMER",
        "MG_C_MONOMER",
        "MG_D_MONOMER",
        "MG_E_MONOMER",
        "MG_F_MONOMER",
    ]
    _attach_process_states(
        process, ["H2O", "diacylglycerolCys", "PG160", "SNGLYP", "H"], ids
    )
    process.monomer.processedIIndexs = np.array([1, 2, 3])
    process.monomer.processedIIIndexs = np.array([2, 3])
    process.monomer.signalSequenceIndexs = np.array([4])
    protein_records = [
        NS(signalSequenceType="lipoprotein"),
        NS(signalSequenceType="secretory"),
    ] + [NS(signalSequenceType="none") for _ in ids[2:]]
    compartments = np.array([4, 3, 2, 2, 2, 2, 2, 2])
    knowledge_base = NS(
        processes=[
            NS(wholeCellModelID=process.wholeCellModelID, parameters=[])
        ],
        proteinMonomers=protein_records,
        numProteinMonomers=len(ids),
        proteinMonomerCompartments=compartments,
    )

    process.initializeConstants(
        knowledge_base, NS(stepSizeSec=1, verbosity=0, seed=3)
    )

    assert process.lipoproteinMonomerIndexs.ravel().tolist() == [1]
    assert process.secretedMonomerIndexs.ravel().tolist() == [2]
    assert process.unprocessedMonomerIndexs.ravel().tolist() == [
        3,
        4,
        5,
        6,
        7,
        8,
    ]
    np.testing.assert_array_equal(
        process.monomerCompartments, [2, 3, 2, 2, 2, 2, 2, 2]
    )


def test_rna_modification_initializes_reaction_and_mature_rna_maps_from_kb():
    process = RNAModification()
    substrate_ids = [
        "AMET",
        "FTHF5",
        "CYS",
        "GLY",
        "LYS",
        "ATP",
        "H2O",
        "H",
        "AMP",
        "CMP",
        "GMP",
        "UMP",
        "mNMP",
    ]
    enzyme_ids = [
        "MG_463_MONOMER",
        "MG_380_MONOMER",
        "MG_252_DIMER",
        "MG_346_DIMER",
        "MG_209_MONOMER",
        "MG_370_MONOMER",
        "MG_445_DIMER",
        "MG_347_DIMER",
        "MG_084_TETRAMER",
        "MG_182_DIMER",
        "MG_295_MONOMER",
        "MG_372_DIMER",
        "MG_008_379_TETRAMER",
    ]
    _attach_process_states(process, substrate_ids, enzyme_ids)
    process.rna.wholeCellModelIDs = [
        "precursor-1",
        "precursor-2",
        "mature-1",
        "mature-2",
    ]
    process.rna.names = process.rna.wholeCellModelIDs
    process.rna.processedIndexs = np.array([1, 2])
    process.rna.matureIndexs = np.array([3, 4])
    process.rna.boundIndexs = np.array([1, 2])
    process.rna.molecularWeights = np.ones(4)
    process.rna.compartments = np.full(4, 2)
    composition = np.zeros((len(enzyme_ids), 2))
    composition[0, 0] = 1
    composition[1, 1] = 1
    process.rna.matureRNAGeneComposition = composition
    process.gene = NS(mRNAIndexs=np.arange(1, len(enzyme_ids) + 1))
    process.monomer.matureIndexs = np.arange(1, len(enzyme_ids) + 1)
    process.monomer.compartments = np.full(len(enzyme_ids), 2)
    process.monomer.molecularWeights = np.ones(len(enzyme_ids))

    sto = np.zeros((len(substrate_ids), 1, 2))
    sto[:, 0, 1] = -1
    rna_sto = np.zeros((2, 1, 2))
    rna_sto[0, 0, 1] = -1
    cat = np.ones((1, len(enzyme_ids), 2))
    mod = np.zeros((1, len(enzyme_ids), 2))
    mod[0, 0, 1] = 1
    reaction = NS(
        idx=1,
        wholeCellModelID="rna-reaction",
        name="RNA modification",
        type="modification",
    )
    knowledge_base = NS(
        processes=[
            NS(
                wholeCellModelID=process.wholeCellModelID,
                reactions=[reaction],
                parameters=[],
            )
        ],
        reactionStimuliStoichiometryMatrix=np.zeros((0, 1, 2)),
        reactionMetaboliteStoichiometryMatrix=sto,
        reactionRNAStoichiometryMatrix=rna_sto,
        reactionProteinMonomerStoichiometryMatrix=np.zeros(
            (len(enzyme_ids), 1, 2)
        ),
        reactionProteinComplexStoichiometryMatrix=np.zeros((2, 1, 2)),
        reactionCatalysisMatrix=cat,
        reactionModificationMatrix=mod,
        reactionCoenzymeMatrix=np.zeros((1, len(substrate_ids), 2)),
        enzymeBounds=np.array([[0.0, 2.0]]),
        reactionBounds=np.array([[0.0, 1.0]]),
        metabolites=[
            NS(
                category=(
                    "modified ribonucleotide monophosphate"
                    if x == "mNMP"
                    else "other"
                )
            )
            for x in substrate_ids
        ],
    )

    process.initializeConstants(
        knowledge_base, NS(stepSizeSec=1, verbosity=0, seed=4)
    )

    np.testing.assert_array_equal(process.substrateIndexs_modifiedNMP, [[13]])
    np.testing.assert_array_equal(
        process.modifiedRNAWholeCellModelIDs.ravel(), ["mature-1", "mature-2"]
    )
    np.testing.assert_array_equal(process.rnaMask_modified, [True, False])
    assert process.speciesReactantMatrix.shape == (
        1,
        len(substrate_ids) + 1 + len(enzyme_ids) + 1,
    )


def test_rna_decay_kb_initializer_maps_decay_reactions_to_rna_states(
    monkeypatch,
):
    process = RNADecay()
    fixed = [
        "H",
        "H2O",
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
    metabolite_ids = sorted(fixed + ["AMP", "CMP", "GMP", "UMP", "X"])
    process.metabolite = NS(
        wholeCellModelIDs=metabolite_ids,
        names=metabolite_ids,
        molecularWeights=np.ones(len(metabolite_ids)),
        hydrophobicIndexs=[],
    )
    process.compartment = NS(
        count=2, extracellularIndexs=1, cytosolIndexs=2, membraneIndexs=1
    )
    process.stimulus = NS(wholeCellModelIDs=[], names=[])
    process.gene = NS(tRNAIndexs=np.array([1]))
    process.rna = NS(
        wholeCellModelIDs=["nascent", "mature-a", "mature-b"],
        names=["nascent", "mature-a", "mature-b"],
        matureIndexs=np.array([2, 3]),
        nascentIndexs=np.array([1]),
        processedIndexs=np.array([1]),
        intergenicIndexs=np.array([1]),
        aminoacylatedIndexs=np.array([2, 3]),
        boundIndexs=np.array([1]),
        misfoldedIndexs=np.array([], dtype=int),
        damagedIndexs=np.array([], dtype=int),
        matureRNAGeneComposition=np.array([[1, 0], [0, 1]]),
        getIndexs=lambda ids: np.array([2]),
    )
    process.monomer = NS(
        wholeCellModelIDs=["m"],
        names=["m"],
        matureIndexs=[1],
        boundIndexs=[1],
        molecularWeights=[1],
        compartments=[2],
    )
    process.complex = NS(
        wholeCellModelIDs=["c"],
        names=["c"],
        matureIndexs=[1],
        boundIndexs=[1],
        molecularWeights=[1],
        compartments=[2],
    )

    def decay(offset):
        values = np.zeros((offset, len(metabolite_ids)))
        values[:, metabolite_ids.index("H2O")] = 1
        for nmp in ("AMP", "CMP", "GMP", "UMP"):
            values[:, metabolite_ids.index(nmp)] = 1
        return values

    kb = NS(
        nascentRNAs=NS(decayReactions=decay(1)),
        processedRNAs=NS(decayReactions=decay(1)),
        intergenicRNAs=NS(decayReactions=decay(1)),
        matureRNAs=NS(decayReactions=decay(2)),
        aminoacylatedRNAs=NS(decayReactions=decay(2)),
    )
    monkeypatch.setattr(
        Process,
        "initializeConstants",
        lambda self, kb, sim, options=None: setattr(
            self,
            "substrateMetaboliteGlobalIndexs",
            np.asarray(
                [
                    metabolite_ids.index(x) + 1
                    for x in self.substrateWholeCellModelIDs.ravel()
                ]
            ),
        ),
    )

    process.initializeConstants(kb, NS(stepSizeSec=1, verbosity=0, seed=8))

    assert process.decayReactions.shape == (
        3,
        len(process.substrateWholeCellModelIDs),
    )
    np.testing.assert_array_equal(
        process.substrateIndexs_water,
        [[list(process.substrateWholeCellModelIDs.ravel()).index("H2O") + 1]],
    )
    np.testing.assert_array_equal(process.matureTRNAIndexs, [[1]])
    np.testing.assert_array_equal(process.matureTMRNAIndexs, [[2]])
    water_index = process.substrateIndexs_water.item() - 1
    assert process.decayReactions[0, water_index] == 1
