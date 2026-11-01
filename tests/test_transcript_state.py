import numpy as np

from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.transcript import Transcript


def test_snapshot_restore_and_dependent_values():
    graph = load_snapshot("reference/snapshot_t0.json")
    data = graph.states["Transcript"]
    state = Transcript(snapshot=data.fields)
    assert (
        state.boundTranscriptionUnits.shape
        == data.boundTranscriptionUnits.shape
    )
    assert state.transcriptionUnitSequences.shape == (335, 1)
    assert state.rnaBoundRNAPolymerases.shape == (335, 1, 1)
    assert state.rnaMaxRNAPolymeraseState.shape == (335, 1, 1)
    assert state.totalBaseCounts.shape == (1, 4)
    assert state.dryWeight >= 0


def test_sequences_base_counts_and_dry_weight():
    state = Transcript(
        transcriptionUnitSequences=np.array(["ACGUAC"]),
        transcriptionUnitLengths=np.array([[6]]),
        nmpMolecularWeights=np.array([[100.0], [110.0], [120.0], [130.0]]),
        boundTranscriptionUnits=np.array([[1], [0]]),
        boundTranscriptProgress=np.array([[4], [0]]),
        boundTranscriptChromosome=np.array([[1], [0]]),
        abortedTranscripts=np.array([[1, 2], [0, 0]]),
    )
    assert state.abortedSequences == ["AC", ""]
    np.testing.assert_array_equal(state.totalBaseCounts, [[2, 2, 1, 0]])
    # Active sequence is ACG; aborted sequence is AC. The two strands contain three bonds in total.
    expected = (
        2 * 100 + 2 * 110 + 120 - 3 * (1.0079 + 15.9994)
    ) / 6.0221417930e23
    np.testing.assert_allclose(state.dryWeight, expected)
    np.testing.assert_array_equal(state.rnaBoundRNAPolymerases[:, 0, 0], [1])
    np.testing.assert_array_equal(state.rnaMaxRNAPolymeraseState[:, 0, 0], [6])


def test_allocate_and_initialize_shapes():
    state = Transcript()
    state.allocateMemory(4)
    assert state.boundTranscriptionUnits.shape == (0, 1, 4)
    assert state.boundTranscriptProgress.shape == (0, 1, 4)
    assert state.boundTranscriptChromosome.shape == (0, 1, 4)
    assert state.abortedTranscripts.shape == (0, 2, 4)
    state.initialize()
    assert state.abortedTranscripts.shape == (0, 2)
