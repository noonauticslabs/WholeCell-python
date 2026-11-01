import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wholecell_py.processes.transcriptional_regulation import (
    TranscriptionalRegulation,
)
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.chromosome import Chromosome
from wholecell_py.process_base import Process
from wholecell_py.chromosome_process_aspect import ChromosomeProcessAspect

ROOT = Path(__file__).parents[1]


def snapshot():
    registry = implemented_registry()
    registry[
        "edu.stanford.covert.cell.sim.process.TranscriptionalRegulation"
    ] = TranscriptionalRegulation
    registry["edu.stanford.covert.cell.sim.state.Chromosome"] = Chromosome
    return load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=registry,
        runtime_values=True,
    )


def test_snapshot_current_resource_vector_matches_reference():
    graph = snapshot()
    process = graph.processes["TranscriptionalRegulation"]
    assert isinstance(process, TranscriptionalRegulation)
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["TranscriptionalRegulation"]
    np.testing.assert_allclose(
        process.calcResourceRequirements_Current().reshape(
            expected["shape"], order="F"
        ),
        np.asarray(expected["data"]).reshape(expected["shape"], order="F"),
    )


def test_activity_fold_change_and_runtime_evolution():
    graph = snapshot()
    process = graph.processes["TranscriptionalRegulation"]
    process.enzymes[:] = 0
    process.enzymes[int(process.tfIndexs[0, 0]) - 1] = 1
    bound = np.zeros_like(process.tfIndexs, dtype=bool)
    bound[0, 0] = True
    actual = process.calcBindingProbabilityFoldChange(bound)
    tu = int(process.tuIndexs[0, 0]) - 1
    expected = np.ones_like(actual)
    expected[tu, 0] = process.tfActivities[0, 0]
    np.testing.assert_allclose(actual, expected)

    process.randStream = (
        None  # No binding event is possible with the zeroed enzyme vector.
    )
    process.enzymes[:] = 0
    existing_bound = process.tfBoundPromoters
    process.evolveState()
    np.testing.assert_allclose(
        process.rnaPolymerase.transcriptionFactorBindingProbFoldChange,
        process.calcBindingProbabilityFoldChange(existing_bound),
    )


def test_kb_initializer_derives_binding_coordinates_and_compartment_activity(
    monkeypatch,
):
    tf = SimpleNamespace(idx=2, wholeCellModelID="TF_MONOMER")
    tu = SimpleNamespace(
        wholeCellModelID="TU_1",
        transcriptionFactorProteinMonomers=[tf],
        transcriptionFactorProteinComplexs=[],
        transcriptionFactorProteinMonomerCompartments=[SimpleNamespace(idx=1)],
        transcriptionFactorProteinComplexCompartments=[],
        transcriptionFactorProteinMonomerActivitys=[0.5],
        transcriptionFactorProteinComplexActivitys=[],
        transcriptionFactorProteinMonomerBindingSiteStartCoordinates=[10],
        transcriptionFactorProteinMonomerBindingSiteLengths=[4],
        transcriptionFactorProteinComplexBindingSiteStartCoordinates=[],
        transcriptionFactorProteinComplexBindingSiteLengths=[],
    )
    kb = SimpleNamespace(transcriptionUnits=[tu])
    process = TranscriptionalRegulation()

    def init_base(self, knowledge_base, simulation, options=None):
        self.enzymeMonomerGlobalIndexs = np.array([2])
        self.enzymeMonomerLocalIndexs = np.array([1])
        self.enzymeMonomerCompartmentIndexs = np.array([1])
        self.enzymeComplexGlobalIndexs = np.array([], dtype=int)
        self.enzymeComplexLocalIndexs = np.array([], dtype=int)
        self.enzymeComplexCompartmentIndexs = np.array([], dtype=int)
        self.enzymeWholeCellModelIDs = np.array(["TF_MONOMER"], dtype=object)

    monkeypatch.setattr(Process, "initializeConstants", init_base)
    monkeypatch.setattr(
        ChromosomeProcessAspect,
        "initializeConstants",
        lambda self, kb, sim: setattr(
            self,
            "chromosome",
            SimpleNamespace(
                sequenceLen=100,
                monomerDNAFootprints=np.array([1, 3]),
                complexDNAFootprints=np.array([]),
            ),
        ),
    )
    process.enzymeIndexs = lambda ids: np.array([1])

    process.initializeConstants(kb, object())

    np.testing.assert_array_equal(process.tuIndexs, [[1, 1]])
    np.testing.assert_array_equal(process.tfIndexs, [[1, 1]])
    np.testing.assert_allclose(process.tfAffinities, [[0.5, 0.5]])
    np.testing.assert_allclose(process.tfActivities, [[2.0, 2.0]])
    np.testing.assert_array_equal(
        process.tfPositionStrands, [[11, 1], [11, 3]]
    )
    np.testing.assert_allclose(process.otherActivities, [[1.0]])
