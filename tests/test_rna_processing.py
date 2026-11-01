import json
from pathlib import Path
import numpy as np

from wholecell_py.snapshot import load_snapshot
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.processes.rna_processing import RNAProcessing
from wholecell_py.randstream import RandStream

ROOT = Path(__file__).resolve().parents[1]


def graph_with_rna_process():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.RNAProcessing"] = (
        RNAProcessing
    )
    return load_snapshot(
        ROOT / "reference/snapshot_t0.json", class_registry=registry
    )


def test_snapshot_rna_processing_resource_demand_and_state_roundtrip():
    graph = graph_with_rna_process()
    process = graph.processes["RNAProcessing"]
    process.initializeConstants()
    process.copyFromState()
    expected = json.loads(
        (ROOT / "reference/requirements_t0.json").read_text()
    )["RNAProcessing"]
    observed = np.asarray(process.calcResourceRequirements_Current()).reshape(
        -1, order="F"
    )
    target = (
        np.asarray(expected["data"], dtype=float)
        .reshape(expected["shape"], order="F")
        .reshape(-1, order="F")
    )
    np.testing.assert_allclose(observed, target, rtol=1e-12, atol=1e-12)
    before = np.asarray(process.rna.counts).copy()
    process.copyToState()
    np.testing.assert_array_equal(
        np.asarray(process.rna.counts).reshape(before.shape), before
    )
    process.randStream = RandStream(seed=1)
    process.evolveState()
    process.copyToState()


def test_missing_knowledge_base_data_is_explicit():
    process = RNAProcessing()
    try:
        process.initializeConstants()
    except NotImplementedError as exc:
        assert "RNA" in str(exc)
    else:
        raise AssertionError(
            "RNAProcessing accepted missing initialized RNA data"
        )
