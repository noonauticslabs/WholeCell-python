from types import SimpleNamespace as NS

import numpy as np
import pytest

from wholecell_py.reaction_process import (
    ReactionProcess,
    _active_rows,
    _ids,
    _reaction_rows,
    _select_compartment,
)
from wholecell_py.sparsemat import SparseMat
from wholecell_py.processes.protein_modification import ProteinModification
from wholecell_py.processes.rna_modification import RNAModification
from wholecell_py.processes.terminal_organelle_assembly import (
    TerminalOrganelleAssembly,
)


def test_reaction_process_uses_local_reaction_ids_and_matlab_one_based_indices():
    process = ReactionProcess("P")
    process.reactionWholeCellModelIDs = np.array(
        [["r1"], ["r2"]], dtype=object
    )
    np.testing.assert_array_equal(
        process.reactionIndexs(["r2", "r1"]), [[2], [1]]
    )
    with pytest.raises(ValueError):
        process.reactionIndexs(["not-local"])


def test_reaction_process_mature_id_selection_uses_active_row_subset():
    state = NS(
        wholeCellModelIDs=["immature", "mature-a", "mature-b"],
        matureIndexs=[2, 3],
    )
    np.testing.assert_array_equal(
        _ids(state, [0], mature=True), [["mature-a"]]
    )
    np.testing.assert_array_equal(
        _ids(state, [1], mature=True), [["mature-b"]]
    )


def test_reaction_process_empty_active_rows_keep_the_nonreduced_axis():
    assert _active_rows(np.zeros((0, 2, 3)), (0, 2)).shape == (2,)
    assert _active_rows(np.zeros((3, 0, 2)), (1, 2)).shape == (3,)


def test_reaction_process_compartment_collapse_uses_explicit_component_axis():
    # Equal reaction/component counts make shape-based axis inference ambiguous.
    matrix = np.arange(2 * 2 * 3).reshape(2, 2, 3)
    actual = _select_compartment(matrix, [3, 1], axis=2, component_axis=1)
    np.testing.assert_array_equal(actual, [[2, 3], [8, 9]])


def test_reaction_rows_supports_wholecell_coordinate_sparse_tensors():
    dense = np.arange(2 * 3 * 2).reshape(2, 3, 2)
    sparse = SparseMat(dense)
    actual = _reaction_rows(sparse, [3, 1], axis=1)
    np.testing.assert_array_equal(actual, dense[:, [2, 0], :])


def test_reaction_process_initializes_a_small_dense_knowledge_base():
    process = ReactionProcess("P")
    process.compartment = NS(
        count=2,
        extracellularIndexs=1,
        cytosolIndexs=2,
        membraneIndexs=1,
    )
    process.stimulus = NS(wholeCellModelIDs=["light"], names=["Light"])
    process.metabolite = NS(
        wholeCellModelIDs=["water", "unused"],
        names=["Water", "Unused"],
        molecularWeights=[18, 10],
        hydrophobicIndexs=[],
    )
    for name, ids in (
        ("rna", ["pre-rna", "rna"]),
        ("monomer", ["pre-protein", "protein"]),
        ("complex", ["pre-complex", "complex"]),
    ):
        setattr(
            process,
            name,
            NS(
                wholeCellModelIDs=ids,
                names=ids,
                matureIndexs=[2],
                boundIndexs=[1],
                molecularWeights=[0, 5],
                compartments=[2, 2],
            ),
        )
    process.gene = NS(mRNAIndexs=[1])

    rna_sto = np.zeros((2, 1, 2))
    rna_sto[0, 0, 1] = -1
    monomer_sto = np.zeros((2, 1, 2))
    monomer_sto[0, 0, 1] = -1
    complex_sto = np.zeros((2, 1, 2))
    complex_sto[0, 0, 1] = -1
    cat = np.zeros((1, 2, 2))
    cat[0, 0, 1] = 1
    cat[0, 1, 1] = 1
    mod = np.zeros((1, 2, 2))
    coenzyme = np.zeros((1, 2, 2))
    coenzyme[0, 0, 1] = 1
    reaction = NS(
        idx=1, wholeCellModelID="rx1", name="Reaction 1", type="test"
    )
    kb = NS(
        processes=[
            NS(wholeCellModelID="P", reactions=[reaction], parameters=[])
        ],
        reactionStimuliStoichiometryMatrix=np.array([[[-1.0, 0.0]]]),
        reactionMetaboliteStoichiometryMatrix=np.zeros((2, 1, 2)),
        reactionRNAStoichiometryMatrix=rna_sto,
        reactionProteinMonomerStoichiometryMatrix=monomer_sto,
        reactionProteinComplexStoichiometryMatrix=complex_sto,
        reactionCatalysisMatrix=cat,
        reactionModificationMatrix=mod,
        reactionCoenzymeMatrix=coenzyme,
        enzymeBounds=np.array([[0.0, 1.0]]),
        reactionBounds=np.array([[0.0, 1.0]]),
    )
    simulation = NS(stepSizeSec=1.0, verbosity=0, seed=7)

    process.initializeConstants(kb, simulation)

    np.testing.assert_array_equal(
        process.substrateWholeCellModelIDs.ravel(),
        ["light", "water", "rna", "protein", "complex"],
    )
    np.testing.assert_array_equal(
        process.enzymeWholeCellModelIDs.ravel(), ["protein", "complex"]
    )
    np.testing.assert_array_equal(
        process.reactionStoichiometryMatrix[:, 0], [-1, 0, -1, -1, -1]
    )
    np.testing.assert_array_equal(process.reactionCoenzymeMatrix[:, 1], [1])
    np.testing.assert_array_equal(process.reactionIndexs(["rx1"]), [[1]])


def test_reaction_derived_processes_inherit_reaction_process():
    assert issubclass(ProteinModification, ReactionProcess)
    assert issubclass(RNAModification, ReactionProcess)
    assert issubclass(TerminalOrganelleAssembly, ReactionProcess)


@pytest.mark.parametrize("retain_compartments", [False, True])
def test_override_reactions_reorders_tensors_and_trims_unused_components(
    retain_compartments,
):
    process = ReactionProcess("P")
    process.reactionWholeCellModelIDs = np.array(
        [["r0"], ["r1"], ["r2"]], object
    )
    process.reactionNames = np.array([["R0"], ["R1"], ["R2"]], object)
    process.reactionTypes = np.array([["T0"], ["T1"], ["T2"]], object)
    process.substrateWholeCellModelIDs = np.array(
        [["s0"], ["s1"], ["s2"]], object
    )
    process.enzymeWholeCellModelIDs = np.array(
        [["e0"], ["e1"], ["e2"]], object
    )
    tail = (2,) if retain_compartments else ()
    stoich = np.zeros((3, 3) + tail)
    coenzyme = np.zeros((3, 3) + tail)
    catalysis = np.zeros((3, 3) + tail)
    if retain_compartments:
        stoich[0, 0, 0] = -1
        stoich[2, 2, 1] = -1
        coenzyme[1, 1, 0] = 1  # Only the removed reaction uses s1.
        catalysis[0, 0, 0] = 1
        catalysis[2, 1, 1] = 1
        catalysis[1, 2, 0] = 1  # Reaction r1 is removed.
    else:
        stoich[0, 0] = -1
        stoich[2, 2] = -1
        coenzyme[1, 1] = 1  # Only the removed reaction uses s1.
        catalysis[0, 0] = 1
        catalysis[2, 1] = 1
        catalysis[1, 2] = 1  # Reaction r1 is removed.
    process.reactionStoichiometryMatrix = stoich
    process.reactionCoenzymeMatrix = coenzyme
    process.reactionCatalysisMatrix = catalysis
    process.reactionModificationMatrix = np.arange(3 * 2).reshape(3, 2)
    process.reactionBounds = np.arange(6).reshape(3, 2)
    process.enzymeBounds = np.arange(6).reshape(3, 2)

    def override_substrates(ids):
        old = process.substrateWholeCellModelIDs.ravel().tolist()
        keep = [old.index(identifier) for identifier in ids]
        process.reactionStoichiometryMatrix = (
            process.reactionStoichiometryMatrix[keep]
        )
        process.reactionCoenzymeMatrix = process.reactionCoenzymeMatrix[
            :, keep
        ]
        process.substrateWholeCellModelIDs = np.asarray(
            ids, dtype=object
        ).reshape(-1, 1)

    def override_enzymes(ids):
        old = process.enzymeWholeCellModelIDs.ravel().tolist()
        keep = [old.index(identifier) for identifier in ids]
        process.reactionCatalysisMatrix = process.reactionCatalysisMatrix[
            :, keep
        ]
        process.enzymeWholeCellModelIDs = np.asarray(
            ids, dtype=object
        ).reshape(-1, 1)

    process.initializeConstants_overrideSubstrates = override_substrates
    process.initializeConstants_overrideEnzymes = override_enzymes
    process.initializeConstants_overrideReactions(None, ["r2", "r0"])

    np.testing.assert_array_equal(
        process.reactionWholeCellModelIDs.ravel(), ["r2", "r0"]
    )
    np.testing.assert_array_equal(process.reactionNames.ravel(), ["R2", "R0"])
    np.testing.assert_array_equal(
        process.substrateWholeCellModelIDs.ravel(), ["s0", "s2"]
    )
    np.testing.assert_array_equal(
        process.enzymeWholeCellModelIDs.ravel(), ["e0", "e1"]
    )
    np.testing.assert_array_equal(process.reactionBounds[:, 0], [4, 0])
