from pathlib import Path

import numpy as np
import pytest

from wholecell_py.snapshot import DataObject, load_snapshot
from wholecell_py.states.rna_polymerase import RNAPolymerase


def test_snapshot_restoration_and_state_counts():
    graph = load_snapshot(
        Path(__file__).resolve().parents[1] / "reference" / "snapshot_t0.json"
    )
    data = graph.states["RNAPolymerase"]
    state = RNAPolymerase(snapshot=data.fields)
    assert state.states.shape == data.states.shape
    assert state.positionStrands.shape == data.positionStrands.shape
    expected = [
        np.sum(state.states >= 1),
        np.sum(state.states == -3),
        np.sum(state.states == -1),
        np.sum(state.states == -2),
    ]
    assert [
        state.nActive,
        state.nSpecificallyBound,
        state.nNonSpecificallyBound,
        state.nFree,
    ] == expected
    np.testing.assert_allclose(
        state.stateOccupancies[:, 0, 0], np.asarray(expected) / sum(expected)
    )


def test_allocation_and_degradation():
    state = RNAPolymerase(states=np.array([[-2], [-2], [-1], [1]]))
    state.degradeFreePolymerase(1)
    np.testing.assert_array_equal(state.states[:, 0], [0, -2, -1, 1])
    state.chromosome = DataObject(
        "c",
        "Chromosome",
        {"transcriptionUnitWholeCellModelIDs": np.array([1, 2])},
    )
    state.allocateMemory(3)
    assert state.states.shape == (0, 1, 3)
    assert state.positionStrands.shape == (0, 2, 3)
    assert state.transcriptionFactorBindingProbFoldChange.shape == (2, 2, 3)
    assert state.supercoilingBindingProbFoldChange.shape == (2, 2, 3)


def test_release_requires_references():
    state = RNAPolymerase(
        states=np.array([[-2]]), positionStrands=np.zeros((1, 2))
    )
    with pytest.raises(RuntimeError, match="storeObjectReferences"):
        state.releasePolymerase([[1, 1]], False)


@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("degraded", [False, True])
def test_release_appends_scalar_pair_to_existing_aborted_transcripts(
    column, degraded
):
    from types import SimpleNamespace

    def vector(values):
        a = np.array(values)
        return a[:, None] if column else a

    state = RNAPolymerase(
        states=vector([5, 1, -2]),
        positionStrands=np.array([[10, 1], [20, 1], [30, 1]]),
    )
    state.chromosome = SimpleNamespace(sequenceLen=100, nCompartments=4)
    state.transcripts = SimpleNamespace(
        abortedTranscripts=np.array([[7, 12]]),
        boundTranscriptProgress=vector([5, 1, 0]),
        boundTranscriptionUnits=vector([9, 10, 0]),
        boundTranscriptChromosome=vector([1, 1, 0]),
        nullTranscriptValue=0,
    )
    state.releasePolymerase([[10, 1], [20, 1]], degraded)
    np.testing.assert_array_equal(
        state.transcripts.abortedTranscripts, [[7, 12], [9, 4]]
    )
    np.testing.assert_array_equal(
        state.states.ravel(), [0 if degraded else -2] * 2 + [-2]
    )
    np.testing.assert_array_equal(
        state.transcripts.boundTranscriptProgress.ravel(), [0, 0, 0]
    )
    np.testing.assert_array_equal(
        state.positionStrands, [[0, 0], [0, 0], [30, 1]]
    )
