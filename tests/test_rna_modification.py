import json
from pathlib import Path

import numpy as np

from wholecell_py.randstream import RandStream
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot
from wholecell_py.processes.rna_modification import RNAModification

ROOT = Path(__file__).parents[1]


def snapshot():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.RNAModification"] = (
        RNAModification
    )
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json", class_registry=registry
    )
    return graph, graph.processes["RNAModification"]


def test_snapshot_demands_copy_evolve_copy_to_state():
    graph, p = snapshot()
    assert isinstance(p, RNAModification)
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["RNAModification"]
    assert expected["shape"] == [29, 1]
    p.copyFromState()
    np.testing.assert_array_equal(
        p.calcResourceRequirements_Current().reshape(
            expected["shape"], order="F"
        ),
        np.asarray(expected["data"]).reshape(expected["shape"], order="F"),
    )
    before = np.asarray(p.unmodifiedRNAs).copy()
    p.randStream = RandStream(seed=1)
    p.evolveState()
    assert np.all(np.asarray(p.unmodifiedRNAs) >= 0)
    assert np.all(np.asarray(p.modifiedRNAs) >= 0)
    assert np.asarray(p.unmodifiedRNAs).sum() <= before.sum()
    p.copyToState()
    cyt = int(np.asarray(p.compartment.cytosolIndexs).ravel()[0]) - 1
    proc = np.asarray(p.rna.processedIndexs, dtype=int).ravel() - 1
    mature = np.asarray(p.rna.matureIndexs, dtype=int).ravel() - 1
    np.testing.assert_array_equal(
        p.rna.counts[proc, cyt, 0], np.asarray(p.unmodifiedRNAs).reshape(-1)
    )
    np.testing.assert_array_equal(
        p.rna.counts[mature, cyt, 0], np.asarray(p.modifiedRNAs).reshape(-1)
    )


def test_initialize_constants_reports_unavailable_kb_behavior():
    try:
        RNAModification().initializeConstants()
    except NotImplementedError as exc:
        assert "snapshot lacks initialized network fields" in str(exc)
    else:
        raise AssertionError(
            "missing initialized network fields must be reported"
        )


def test_current_requirements_are_limited_by_enzyme_throughput():
    process = RNAModification(
        reactionStoichiometryMatrix=np.array([[-1.0, -2.0]]),
        reactionCatalysisMatrix=np.array([[1.0], [3.0]]),
        reactionModificationMatrix=np.array([[1.0], [1.0]]),
        unmodifiedRNAs=np.array([[10.0]]),
        enzymes=np.array([1.0]),
        stepSizeSec=1.0,
    )

    # Enzyme throughput is [1, 3], RNA availability is [10, 10], and the
    # substrate cost is [1, 2] per reaction. The original implementation
    # incorrectly reported the unconstrained demand of 30.
    np.testing.assert_array_equal(
        process.calcResourceRequirements_Current(), [7.0]
    )
