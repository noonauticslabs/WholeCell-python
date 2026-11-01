import json
from pathlib import Path

import numpy as np

from wholecell_py.randstream import RandStream
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot
from wholecell_py.processes.rna_decay import RNADecay

ROOT = Path(__file__).parents[1]


def snapshot():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.RNADecay"] = RNADecay
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json", class_registry=registry
    )
    return graph, graph.processes["RNADecay"]


def test_snapshot_resource_vector_and_copy_evolve_writeback():
    graph, p = snapshot()
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["RNADecay"]
    assert expected["shape"] == [39, 1]
    p.copyFromState()
    actual = p.calcResourceRequirements_Current().reshape(
        expected["shape"], order="F"
    )
    np.testing.assert_allclose(
        actual,
        np.asarray(expected["data"]).reshape(expected["shape"], order="F"),
        rtol=1e-12,
    )
    before = np.asarray(p.RNAs).copy()
    p.randStream = RandStream(seed=1)
    p.evolveState()
    assert np.all(np.asarray(p.RNAs) >= 0)
    assert np.asarray(p.RNAs).sum() <= before.sum()
    p.copyToState()
    cyt = int(np.asarray(p.compartment.cytosolIndexs).ravel()[0]) - 1
    np.testing.assert_array_equal(
        p.rna.counts[:, cyt, 0], np.asarray(p.RNAs).reshape(-1)
    )


def test_initialize_constants_reports_unavailable_kb_derivation():
    try:
        RNADecay().initializeConstants()
    except NotImplementedError as exc:
        assert "snapshot lacks initialized network fields" in str(exc)
    else:
        raise AssertionError(
            "missing initialized network fields must be reported"
        )


def test_aborted_transcript_hydrolysis_updates_nucleotides_water_and_protons():
    from pathlib import Path
    import numpy as np
    from wholecell_py.snapshot import load_snapshot
    from wholecell_py.runtime_registry import implemented_registry
    from wholecell_py.randstream import RandStream

    graph = load_snapshot(
        Path(__file__).resolve().parents[1] / "reference/snapshot_t0.json",
        class_registry=implemented_registry(),
    )
    process = graph.processes["RNADecay"]
    process.randStream = RandStream(seed=1)
    process.copyFromState()
    process.RNAs = np.zeros_like(process.RNAs)
    process.transcripts.abortedTranscripts = np.array([[1, 4]])
    sequence = process.transcripts.abortedSequences[0]
    expected = np.asarray(process.substrates).ravel().copy()
    expected[
        np.asarray(process.substrateIndexs_nmps).ravel().astype(int) - 1
    ] += [sequence.count(base) for base in "ACGU"]
    expected[int(np.asarray(process.substrateIndexs_water).item()) - 1] -= 3
    expected[int(np.asarray(process.substrateIndexs_hydrogen).item()) - 1] += 3
    process.evolveState()
    np.testing.assert_array_equal(
        np.asarray(process.substrates).ravel(), expected
    )
    assert process.transcripts.abortedTranscripts.shape == (0, 2)
