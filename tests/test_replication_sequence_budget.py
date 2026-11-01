"""Regression coverage for sequence assignment in DNA replication."""

from pathlib import Path

import numpy as np

from wholecell_py.runner import load_model


ROOT = Path(__file__).parents[1]


def test_unwind_passes_real_chromosome_letters_to_polymerase(monkeypatch):
    graph = load_model(ROOT / "reference" / "snapshot_t60.json", seed=1)
    process = graph.processes["Replication"]
    original_unwind = process.unwindAndPolymerizeDNA
    sequence = process.chromosome.sequence
    sequence_reads = []
    in_unwind = False
    captured = {}

    class ReachedPolymerize(Exception):
        pass

    sequence_type = type(sequence)
    original_getitem = sequence_type.__getitem__

    def recording_getitem(instance, indices):
        result = original_getitem(instance, indices)
        if (
            in_unwind
            and isinstance(indices, tuple)
            and isinstance(indices[0], np.ndarray)
        ):
            sequence_reads.append("".join(np.asarray(result).ravel().tolist()))
        return result

    monkeypatch.setattr(sequence_type, "__getitem__", recording_getitem)

    def recording_unwind():
        nonlocal in_unwind
        in_unwind = True
        try:
            return original_unwind()
        finally:
            in_unwind = False

    monkeypatch.setattr(process, "unwindAndPolymerizeDNA", recording_unwind)
    import wholecell_py.polymerize as polymerize_module

    original_polymerize = polymerize_module.polymerize

    def capture_active_sequence(seq, *args, **kwargs):
        values = np.asarray(seq)
        if in_unwind and np.any(values != " "):
            captured["seq"] = values.copy()
            raise ReachedPolymerize
        return original_polymerize(seq, *args, **kwargs)

    monkeypatch.setattr(
        polymerize_module, "polymerize", capture_active_sequence
    )
    try:
        for _ in range(20):
            graph.simulation.evolveState()
    except ReachedPolymerize:
        pass
    else:
        raise AssertionError(
            "replication did not reach polymerize with active bases"
        )

    actual = captured["seq"]
    nonempty = [read for read in sequence_reads if read]
    assert nonempty
    assert len(nonempty) == np.count_nonzero(np.any(actual != " ", axis=1))
    row = 0
    for expected in sequence_reads:
        if not expected:
            continue
        while not np.any(actual[row] != " "):
            row += 1
        np.testing.assert_array_equal(
            actual[row, : len(expected)],
            np.asarray(list(expected), dtype="<U1"),
        )
        assert (
            len(set(expected)) > 1
        ), "snapshot segment should contain varied DNA bases"
        row += 1
