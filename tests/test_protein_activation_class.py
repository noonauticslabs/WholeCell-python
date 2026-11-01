import numpy as np
import json
from pathlib import Path
from types import SimpleNamespace

from wholecell_py.processes.protein_activation import ProteinActivation
from wholecell_py.randstream import RandStream
from wholecell_py.snapshot import load_snapshot


def test_activation_writeback_updates_shared_monomer_count():
    p = ProteinActivation(
        activationRules=["REG > 0"], stimuliWholeCellModelIDs=["REG"]
    )
    p.substrates = np.array([[5.0]])
    p.inactivatedSubstrates = np.array([[0.0]])
    p.stimuli = np.array([[0.0]])
    p.compartment = SimpleNamespace(extracellularIndexs=np.array([1]))
    p.geometry = SimpleNamespace(volume=1.0)
    p.nAvogadro = 6.022e23
    p.stimulusStimulusLocalIndexs = np.array([], dtype=int)
    p.substrateWholeCellModelIDs = ["P"]
    p.enzymeWholeCellModelIDs = []
    p.substrateMolecularWeights = np.array([1.0])
    p.substrateMonomerLocalIndexs = np.array([1])
    p.substrateMonomerGlobalIndexs = np.array([1])
    p.substrateMonomerCompartmentIndexs = np.array([1])
    p.substrateMonomerGlobalCompartmentIndexs = np.array([1])
    p.substrateComplexLocalIndexs = np.array([], dtype=int)
    p.substrateComplexGlobalIndexs = np.array([], dtype=int)
    p.substrateComplexCompartmentIndexs = np.array([], dtype=int)
    p.substrateComplexGlobalCompartmentIndexs = np.array([], dtype=int)
    p.inactivatedSubstrateMonomerGlobalCompartmentIndexs = np.array([2])
    p.inactivatedSubstrateComplexGlobalCompartmentIndexs = np.array(
        [], dtype=int
    )
    state = {"counts": np.array([[0.0], [0.0]])}
    p.monomer = state
    p.complex = {"counts": np.zeros((0, 1))}
    p.stimulus = {"values": np.zeros((0, 1))}
    p.metabolite = {"counts": np.zeros((0, 1))}
    p.rna = {"counts": np.zeros((0, 1))}
    p.substrateStimulusLocalIndexs = p.substrateMetaboliteLocalIndexs = (
        p.substrateRNALocalIndexs
    ) = np.array([], dtype=int)
    p.substrateStimulusGlobalIndexs = p.substrateMetaboliteGlobalIndexs = (
        p.substrateRNAGlobalIndexs
    ) = np.array([], dtype=int)
    p.substrateStimulusCompartmentIndexs = (
        p.substrateMetaboliteCompartmentIndexs
    ) = p.substrateRNACompartmentIndexs = np.array([], dtype=int)
    p.substrateStimulusGlobalCompartmentIndexs = (
        p.substrateMetaboliteGlobalCompartmentIndexs
    ) = p.substrateRNAGlobalCompartmentIndexs = np.array([], dtype=int)
    p.enzymeStimulusLocalIndexs = p.enzymeMetaboliteLocalIndexs = (
        p.enzymeRNALocalIndexs
    ) = p.enzymeMonomerLocalIndexs = p.enzymeComplexLocalIndexs = np.array(
        [], dtype=int
    )
    p.enzymeStimulusGlobalIndexs = p.enzymeMetaboliteGlobalIndexs = (
        p.enzymeRNAGlobalIndexs
    ) = p.enzymeMonomerGlobalIndexs = p.enzymeComplexGlobalIndexs = np.array(
        [], dtype=int
    )
    p.enzymeStimulusCompartmentIndexs = p.enzymeMetaboliteCompartmentIndexs = (
        p.enzymeRNACompartmentIndexs
    ) = p.enzymeMonomerCompartmentIndexs = p.enzymeComplexCompartmentIndexs = (
        np.array([], dtype=int)
    )
    p.enzymeStimulusGlobalCompartmentIndexs = (
        p.enzymeMetaboliteGlobalCompartmentIndexs
    ) = p.enzymeRNAGlobalCompartmentIndexs = (
        p.enzymeMonomerGlobalCompartmentIndexs
    ) = p.enzymeComplexGlobalCompartmentIndexs = np.array([], dtype=int)
    p.substrateComplexGlobalCompartmentIndexs = np.array([], dtype=int)
    p.enzymeBoundMonomerGlobalCompartmentIndexs = (
        p.enzymeBoundComplexGlobalCompartmentIndexs
    ) = p.enzymeBoundRNAGlobalCompartmentIndexs = np.array([], dtype=int)
    p.evolveState()
    p.copyToState()
    np.testing.assert_array_equal(state["counts"].reshape(2), [0.0, 5.0])


def test_snapshot_roundtrip_uses_matlab_linear_indices_and_requirements():
    root = Path(__file__).parents[1]
    graph = load_snapshot(root / "reference" / "snapshot_t0.json")
    process = ProteinActivation(**graph.processes["ProteinActivation"].fields)
    process.randStream = RandStream(seed=1)

    process.copyFromState()
    assert process.inactivatedSubstrates.shape == (10, 6)
    expected = json.loads(
        (root / "reference" / "requirements_t0.json").read_text()
    )["ProteinActivation"]
    requirement = process.calcResourceRequirements_Current()
    assert list(requirement.shape) == expected["shape"]
    np.testing.assert_array_equal(
        requirement.ravel(order="F"), expected["data"]
    )

    process.evolveState()
    process.copyToState()
    for state, linear_indices, local_indices in (
        (
            process.monomer,
            process.inactivatedSubstrateMonomerGlobalCompartmentIndexs,
            process.substrateMonomerLocalIndexs,
        ),
        (
            process.complex,
            process.inactivatedSubstrateComplexGlobalCompartmentIndexs,
            process.substrateComplexLocalIndexs,
        ),
    ):
        counts = state["counts"] if isinstance(state, dict) else state.counts
        actual = np.asarray(counts).ravel(order="F")[
            np.asarray(linear_indices, dtype=int).ravel(order="F") - 1
        ]
        expected_inactive = process.inactivatedSubstrates[
            np.asarray(local_indices, dtype=int).ravel() - 1, :
        ].ravel(order="F")
        np.testing.assert_array_equal(actual, expected_inactive)


def test_activation_negation_precedes_comparison_as_in_matlab():
    from wholecell_py.processes.protein_activation import _eval_rule

    assert not _eval_rule("~A > 1", {"A": 0})
    assert _eval_rule("~(A > 1)", {"A": 0})
    assert _eval_rule("A ~= 2", {"A": 1})
    assert _eval_rule("A > -1 & B == 1", {"A": 0, "B": 1})
    assert _eval_rule("3 < 2 < 1", {})  # (3 < 2) < 1
