from types import SimpleNamespace

import numpy as np
import pytest

from wholecell_py.processes.transcription import (
    Transcription,
    _active_transcription_sequence_matrix,
    _specific_binding_coordinates,
)
from wholecell_py.process_base import Process
from wholecell_py.chromosome_process_aspect import ChromosomeProcessAspect


class TinyChromosome:
    transcriptionUnitStartCoordinates = np.array([10, 30])
    transcriptionUnitLengths = np.array([4, 4])

    class Sites:
        def find(self):
            # DnaA ATP complex 7 is bound on the plus strand of TU 1.
            return np.array([[11, 1]]), np.array([7])

    complexBoundSites = Sites()


def test_transition_probabilities_use_matlab_state_column_convention():
    p = Transcription(
        rnaPolymerases=SimpleNamespace(
            activelyTranscribingIndex=1,
            specificallyBoundIndex=2,
            nonSpecificallyBoundIndex=3,
            freeIndex=4,
            stateExpectations=np.array([0.5, 0.2, 0.2, 0.1]),
        ),
        rna=SimpleNamespace(
            nascentIndexs=np.array([2, 4]), lengths=np.array([0, 6, 0, 10])
        ),
    )
    matrix = p.calcStateTransitionProbabilities(10, 20, np.array([0.01, 0.02]))
    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(matrix.sum(axis=0), 1)
    assert matrix[0, 1] == 1
    assert matrix[2, 1] == 0
    assert matrix[1, 2] == pytest.approx(0.1)
    np.testing.assert_array_equal(matrix[:, 3], matrix[:, 2])


def test_binding_probabilities_protect_dnaa_occupied_tu_with_one_based_ids():
    p = Transcription(
        transcriptionUnitBindingProbabilities=np.ones((2, 2)),
        transcriptionUnitIndexs_DnaAR12345Boxes=np.array([1]),
        complexIndexs_DnaA_ATP=np.array([7]),
        chromosome=TinyChromosome(),
        rnaPolymerases=SimpleNamespace(
            transcriptionFactorBindingProbFoldChange=np.ones((2, 2)),
            supercoilingBindingProbFoldChange=np.ones((2, 2)),
        ),
    )
    probabilities = p.computeRNAPolymeraseTUBindingProbabilities()
    np.testing.assert_array_equal(probabilities, [[0, 1], [1, 1]])


def test_runtime_evolution_rejects_incomplete_chromosome_state():
    p = Transcription(
        rnaPolymerases=SimpleNamespace(
            states=np.array([[-2]]), positionStrands=np.zeros((1, 2))
        ),
        chromosome=object(),
    )
    with pytest.raises(AttributeError, match="sequenceLen"):
        p.evolveState()


def _legacy_active_sequence_matrix(
    sequences, tu_indices, states, active, rate
):
    values = np.asarray(sequences, dtype=object).reshape(-1)
    remaining = [
        "".join(np.asarray(values[tu - 1]).ravel(order="F"))[
            int(states[i]) - 1 :
        ]
        for i, tu in zip(active, tu_indices)
    ]
    width = min(int(rate), max(map(len, remaining), default=0))
    result = np.full((len(remaining), width), " ", dtype="<U1")
    for row, sequence in enumerate(remaining):
        result[row, : min(width, len(sequence))] = list(sequence[:width])
    return result


def test_active_sequence_extraction_matches_string_and_fortran_array_baseline():
    column_major = np.array(list("ACGUAC"), dtype="U1").reshape(
        (2, 3), order="F"
    )
    backing = np.array(list("UGCAUGCA"), dtype="U1").reshape((2, 4), order="F")
    noncontiguous = backing[:, ::2]
    sequences = np.empty((4, 1), dtype=object)
    sequences[:, 0] = [
        "AACGUU",
        column_major,
        noncontiguous,
        np.asarray("GCAU"),
    ]
    tu_indices = np.array([1, 2, 3, 4])
    states = np.array([2, 3, 2, 1])
    active = np.array([0, 1, 2, 3])
    expected = _legacy_active_sequence_matrix(
        sequences, tu_indices, states, active, 3
    )
    actual = _active_transcription_sequence_matrix(
        sequences, tu_indices, states, active, 3
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape[1] == 3


def test_active_sequence_extraction_reads_mutated_source_arrays_each_call():
    sequence = np.array(list("ACGUAC"), dtype="U1")
    sequence_values = np.empty((1, 1), dtype=object)
    sequence_values[0, 0] = sequence
    args = (sequence_values, [1], np.array([2]), [0], 3)
    first = _active_transcription_sequence_matrix(*args)
    sequence[1] = "U"
    second = _active_transcription_sequence_matrix(*args)
    np.testing.assert_array_equal(first, [["C", "G", "U"]])
    np.testing.assert_array_equal(second, [["U", "G", "U"]])


def test_specific_binding_coordinates_match_matlab_direction_transform():
    # MATLAB stores TU direction as 0/1, then computes tuDirs = 2*direction-1.
    actual = _specific_binding_coordinates([10, 30], [0, 1])

    np.testing.assert_array_equal(actual, [11, 29])


def test_transcription_kb_initializer_derives_base_counts_and_dnaa_boxes(
    monkeypatch,
):
    process = Transcription(
        transcripts=SimpleNamespace(
            transcriptionUnitLengths=np.array([10, 20])
        )
    )
    process.substrateMetaboliteGlobalIndexs = np.array([3, 1])
    kb = SimpleNamespace(
        transcriptionUnits=[
            SimpleNamespace(baseCount=np.array([10, 20, 30])),
            SimpleNamespace(baseCount=np.array([40, 50, 60])),
        ],
        proteinComplexs=[
            SimpleNamespace(wholeCellModelID="MG_469_1MER_ATP"),
            SimpleNamespace(wholeCellModelID="other"),
        ],
        genomeFeatures=[
            SimpleNamespace(
                name="Functional box R1",
                transcriptionUnits=[
                    SimpleNamespace(idx=2),
                    SimpleNamespace(idx=1),
                ],
            ),
            SimpleNamespace(
                name="Functional box R2",
                transcriptionUnits=[SimpleNamespace(idx=2)],
            ),
        ],
    )
    for name in (
        "enzymeDNAFootprints",
        "enzymeDNAFootprints3Prime",
        "enzymeDNAFootprints5Prime",
    ):
        setattr(process, name, np.array([1]))
    process.rnaPolymeraseElongationRate = 5
    monkeypatch.setattr(
        Process,
        "initializeConstants",
        lambda self, kb, sim, options=None: None,
    )
    monkeypatch.setattr(
        ChromosomeProcessAspect,
        "initializeConstants",
        lambda self, kb, sim: None,
    )

    process.initializeConstants(kb, object())

    np.testing.assert_array_equal(
        process.transcriptionUnitBaseCounts, [[30, 10], [60, 40]]
    )
    np.testing.assert_array_equal(
        process.transcriptionUnitBindingProbabilities, [0, 0]
    )
    np.testing.assert_array_equal(process.complexIndexs_DnaA_ATP, [1])
    np.testing.assert_array_equal(
        process.transcriptionUnitIndexs_DnaAR12345Boxes, [1, 2]
    )


def test_transcription_copy_from_state_preserves_multiple_timepoints(
    monkeypatch,
):
    counts = np.arange(12).reshape((3, 2, 2))
    process = Transcription(
        rna=SimpleNamespace(counts=counts, nascentIndexs=[1, 3]),
        compartment=SimpleNamespace(cytosolIndexs=2),
    )
    monkeypatch.setattr(Process, "copyFromState", lambda self: None)

    process.copyFromState()

    np.testing.assert_array_equal(
        process.RNAs, counts[[0, 2], 1, :].reshape(2, 1, 2)
    )


def test_transcription_copy_to_state_preserves_multiple_timepoints(
    monkeypatch,
):
    counts = np.zeros((3, 2, 2), dtype=int)
    process = Transcription(
        rna=SimpleNamespace(counts=counts, nascentIndexs=[1, 3]),
        compartment=SimpleNamespace(cytosolIndexs=2),
    )
    process.RNAs = np.array([[[1, 4]], [[2, 5]]])
    monkeypatch.setattr(Process, "copyToState", lambda self: None)

    process.copyToState()

    np.testing.assert_array_equal(counts[[0, 2], 1, :], [[1, 4], [2, 5]])


def test_transcription_initialize_state_preserves_free_polymerase_and_transcript_counts():
    from types import SimpleNamespace

    class Draws:
        def randsample(self, n, k, replacement=False, weights=None):
            return np.array([4])

    process = Transcription()
    process.randStream = Draws()
    process.rnaPolymerases = SimpleNamespace(
        stateExpectations=np.array([0.0, 0.0, 0.0, 1.0]),
        activelyTranscribingIndex=1,
        specificallyBoundIndex=2,
        nonSpecificallyBoundIndex=3,
        freeIndex=4,
        activelyTranscribingValue=1,
        specificallyBoundValue=-3,
        nonSpecificallyBoundValue=-1,
        freeValue=-2,
        notExistValue=0,
    )
    process.transcripts = SimpleNamespace(
        transcriptionUnitDirections=np.array([1]),
        transcriptionUnitLengths=np.array([10]),
        transcriptionUnitFivePrimeCoordinates=np.array([5]),
        nullTranscriptValue=0,
        dryWeight=0,
    )
    process.enzymeWholeCellModelIDs = [
        "sigma",
        "ef",
        "term",
        "anti",
        "core",
        "holo",
    ]
    process.enzymeIndexs_transcriptionFactors = np.arange(1, 5)
    process.enzymeIndexs_sigmaFactor = 1
    process.enzymeIndexs_elongationFactor = 2
    process.enzymeIndexs_rnaPolymerase = 5
    process.enzymeIndexs_rnaPolymeraseHoloenzyme = 6
    process.enzymes = np.array([[1], [1], [0], [0], [1], [0]])
    process.boundEnzymes = np.zeros((6, 1), int)
    process.enzymeDNAFootprints5Prime = np.ones(6)
    process.enzymeDNAFootprints3Prime = np.ones(6)
    process.enzymeDNAFootprints = np.ones(6)
    process.computeRNAPolymeraseTUBindingProbabilities = lambda: np.zeros(
        (1, 2)
    )
    process.chromosome = SimpleNamespace()
    process.rna = SimpleNamespace(
        counts=np.array([[2], [3]]), dryWeight=np.array([10.0])
    )
    process.compartment = SimpleNamespace(cytosolIndexs=np.array([1]))
    process.copyToState = lambda: None
    process.copyFromState = lambda: None

    process.initializeState()

    np.testing.assert_array_equal(process.rnaPolymerases.states, [[-2], [0]])
    np.testing.assert_array_equal(
        process.rnaPolymerases.positionStrands, [[0, 0], [0, 0]]
    )
    np.testing.assert_array_equal(
        process.transcripts.boundTranscriptionUnits, [[0], [0]]
    )
    np.testing.assert_array_equal(process.enzymes[4:6, 0], [1, 0])
    np.testing.assert_array_equal(process.boundEnzymes[4:6, 0], [0, 0])
    np.testing.assert_array_equal(process.rna.counts, [[2], [3]])


def test_transcription_initialize_state_binds_polymerase_and_sets_source_progress():
    from types import SimpleNamespace

    class Draws:
        def randsample(self, n, k, replacement=False, weights=None):
            return np.array([1])

    process = Transcription()
    process.randStream = Draws()
    process.rnaPolymerases = SimpleNamespace(
        stateExpectations=np.array([1.0, 0.0, 0.0, 0.0]),
        activelyTranscribingIndex=1,
        specificallyBoundIndex=2,
        nonSpecificallyBoundIndex=3,
        freeIndex=4,
        activelyTranscribingValue=1,
        specificallyBoundValue=-3,
        nonSpecificallyBoundValue=-1,
        freeValue=-2,
        notExistValue=0,
    )
    process.transcripts = SimpleNamespace(
        transcriptionUnitDirections=np.array([1]),
        transcriptionUnitLengths=np.array([10]),
        transcriptionUnitFivePrimeCoordinates=np.array([5]),
        nullTranscriptValue=0,
        dryWeight=0,
    )
    process.enzymeWholeCellModelIDs = [
        "sigma",
        "ef",
        "term",
        "anti",
        "core",
        "holo",
    ]
    process.enzymeIndexs_transcriptionFactors = np.arange(1, 5)
    process.enzymeIndexs_sigmaFactor = 1
    process.enzymeIndexs_elongationFactor = 2
    process.enzymeIndexs_rnaPolymerase = 5
    process.enzymeIndexs_rnaPolymeraseHoloenzyme = 6
    process.enzymes = np.array([[1], [1], [0], [0], [1], [0]])
    process.boundEnzymes = np.zeros((6, 1), int)
    process.enzymeDNAFootprints5Prime = np.ones(6)
    process.enzymeDNAFootprints3Prime = np.ones(6)
    process.enzymeDNAFootprints = np.ones(6)
    process.computeRNAPolymeraseTUBindingProbabilities = lambda: np.array(
        [[1.0, 0.0]]
    )
    process.bindProteinToChromosomeStochastically = (
        lambda enzyme, n, positions, extent: (1, np.array([[6, 1]]))
    )
    process.chromosome = SimpleNamespace()
    process.rna = SimpleNamespace(
        counts=np.array([[2], [3]]), dryWeight=np.array([10.0])
    )
    process.compartment = SimpleNamespace(cytosolIndexs=np.array([1]))
    process.copyToState = lambda: None
    process.copyFromState = lambda: None

    process.initializeState()

    np.testing.assert_array_equal(process.rnaPolymerases.states, [[3], [0]])
    np.testing.assert_array_equal(
        process.rnaPolymerases.positionStrands, [[7, 1], [0, 0]]
    )
    np.testing.assert_array_equal(
        process.transcripts.boundTranscriptionUnits, [[1], [0]]
    )
    np.testing.assert_array_equal(
        process.transcripts.boundTranscriptProgress, [[3], [0]]
    )
    np.testing.assert_array_equal(
        process.transcripts.boundTranscriptChromosome, [[1], [0]]
    )
    np.testing.assert_array_equal(process.enzymes[4:6, 0], [0, 0])
    np.testing.assert_array_equal(process.boundEnzymes[4:6, 0], [1, 0])
