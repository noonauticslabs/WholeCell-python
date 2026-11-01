from types import SimpleNamespace as NS
import numpy as np
from wholecell_py.processes.protein_folding import ProteinFolding
from wholecell_py.processes.protein_activation import ProteinActivation
from wholecell_py.processes.protein_translocation import ProteinTranslocation


def make(kind):
    p = kind()
    p.compartment = NS(
        count=5,
        cytosolIndexs=1,
        membraneIndexs=2,
        extracellularIndexs=3,
        terminalOrganelleCytosolIndexs=4,
        terminalOrganelleMembraneIndexs=5,
    )
    mids = ["ADP", "ATP", "FE2", "H", "H2O", "K", "MG", "MN", "NA", "PI", "ZN"]
    p.stimulus = NS(wholeCellModelIDs=[], names=[])
    p.rna = NS(
        wholeCellModelIDs=[],
        names=[],
        matureIndexs=[],
        boundIndexs=[],
        molecularWeights=[],
    )
    p.metabolite = NS(
        wholeCellModelIDs=mids,
        names=mids,
        molecularWeights=np.ones(11),
        hydrophobicIndexs=[],
    )
    monids = [
        "u1",
        "u2",
        "MG_238_MONOMER",
        "MG_305_MONOMER",
        "f1",
        "f2",
        "b1",
        "b2",
    ]
    cids = [
        "u1",
        "u2",
        "u3",
        "MG_019_DIMER",
        "MG_201_DIMER",
        "MG_392_393_21MER",
        "b1",
        "b2",
        "b3",
    ]
    p.monomer = NS(
        wholeCellModelIDs=monids,
        names=monids,
        matureIndexs=[3, 4],
        processedIIndexs=[1, 2],
        processedIIIndexs=[1, 2],
        foldedIndexs=[5, 6],
        boundIndexs=[7, 8],
        inactivatedIndexs=[7, 8],
        molecularWeights=np.ones(8),
        compartments=np.ones(8),
        lengths=np.arange(8) + 10,
    )
    p.complex = NS(
        wholeCellModelIDs=cids,
        names=cids,
        matureIndexs=[4, 5, 6],
        nascentIndexs=[1, 2, 3],
        boundIndexs=[7, 8, 9],
        inactivatedIndexs=[7, 8, 9],
        molecularWeights=np.ones(9),
        compartments=np.ones(9),
    )
    kb = NS(
        processes=[NS(wholeCellModelID=p.wholeCellModelID, parameters=[])],
        proteinMonomerCompartments=np.array([[4], [5]]),
        proteinComplexCompartments=np.array([[1], [2], [3]]),
    )
    sim = NS(stepSizeSec=1, verbosity=0, seed=4)
    return p, kb, sim


def test_folding_builds_chaperone_and_species_network_without_mutating_kb():
    p, kb, sim = make(ProteinFolding)
    kb.proteinProstheticGroupMatrix = np.zeros((5, 11, 5))
    kb.proteinProstheticGroupMatrix[0, :, 0] = -1
    kb.proteinChaperoneMatrix = np.zeros((5, 5, 5))
    kb.proteinChaperoneMatrix[1, 4, 0] = 2
    before = kb.proteinProstheticGroupMatrix.copy()
    p.initializeConstants(kb, sim)
    assert p.initialized
    np.testing.assert_array_equal(kb.proteinProstheticGroupMatrix, before)
    np.testing.assert_array_equal(p.monomerCompartments, [[1], [2]])
    np.testing.assert_array_equal(
        p.unfoldedMonomerCompartmentIndexs, [[1], [10]]
    )
    for name in ("dnaK", "dnaJ", "grpE"):
        assert (
            p.proteinChaperoneMatrix[
                4, int(getattr(p, "enzymeIndexs_" + name).item()) - 1
            ]
            == 2
        )
    assert p.speciesReactantMatrix.shape[0] == 3
    np.testing.assert_array_equal(kb.proteinMonomerCompartments, [[4], [5]])


def test_translocation_uses_kb_monomer_count_not_all_state_forms():
    p, kb, sim = make(ProteinTranslocation)
    p.substrateWholeCellModelIDs = []
    p.enzymeWholeCellModelIDs = []
    kb.proteinMonomers = [
        NS(signalSequenceType="lipoprotein"),
        NS(signalSequenceType="none"),
    ]
    p.translocaseSpecificRate = 1
    p.preproteinTranslocase_aaTranslocatedPerATP = 2
    p.SRP_GTPUsedPerMonomer = 3
    p.initializeConstants(kb, sim)
    np.testing.assert_array_equal(p.monomerSRPPathways, [[True], [False]])
    np.testing.assert_array_equal(p.monomerLengths, [[10], [11]])
    np.testing.assert_array_equal(p.monomerIndexs_translocating, [[2]])
    np.testing.assert_array_equal(kb.proteinMonomerCompartments, [[4], [5]])


def test_activation_builds_sorted_regulators_and_all_compartment_inactive_maps():
    p, kb, sim = make(ProteinActivation)
    regulated = NS(
        wholeCellModelID="MG_238_MONOMER",
        activationRule="!ATP",
        stimuliRegulators=[],
        metaboliteRegulators=[NS(wholeCellModelID="ATP")],
        proteinMonomerRegulators=[],
        proteinComplexRegulators=[],
    )
    kb.proteinMonomers = [regulated]
    kb.proteinComplexs = []
    p.initializeConstants(kb, sim)
    np.testing.assert_array_equal(p.activationRules, [["~ATP"]])
    np.testing.assert_array_equal(p.stimuliWholeCellModelIDs, [["ATP"]])
    np.testing.assert_array_equal(
        p.inactivatedSubstrateMonomerGlobalCompartmentIndexs,
        [[7, 15, 23, 31, 39]],
    )
