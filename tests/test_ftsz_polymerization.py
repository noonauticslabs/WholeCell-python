import json
from pathlib import Path

import numpy as np

from wholecell_py.randstream import RandStream
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot
from wholecell_py.processes.ftsz_polymerization import FtsZPolymerization

ROOT = Path(__file__).resolve().parents[1]


def snapshot_registry():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.FtsZPolymerization"] = (
        FtsZPolymerization
    )
    return registry


def test_snapshot_copy_requirements_and_roundtrip():
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=snapshot_registry(),
    )
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )
    process = graph.processes["FtsZPolymerization"]
    process.copyFromState()
    np.testing.assert_array_equal(
        process.calcResourceRequirements_Current().ravel(order="F"),
        expected["FtsZPolymerization"]["data"],
    )
    before_s, before_e, before_b = (
        process.substrates.copy(),
        process.enzymes.copy(),
        process.boundEnzymes.copy(),
    )
    process.copyToState()
    process.copyFromState()
    np.testing.assert_array_equal(process.substrates, before_s)
    np.testing.assert_array_equal(process.enzymes, before_e)
    np.testing.assert_array_equal(process.boundEnzymes, before_b)


def test_snapshot_evolve_and_writeback_with_seeded_runtime_stream():
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=snapshot_registry(),
    )
    process = graph.processes["FtsZPolymerization"]
    process.randStream = RandStream(seed=1)
    process.copyFromState()
    process.evolveState()
    assert np.all(process.enzymes >= 0)
    assert np.all(process.substrates >= 0)
    process.copyToState()
    process.copyFromState()
    assert np.all(process.enzymes >= 0)
