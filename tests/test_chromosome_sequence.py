import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from wholecell_py.chromosome_sequence import ChromosomeSequence
from wholecell_py.snapshot import ChromosomeSequenceData, load_snapshot


def test_strands_are_complemented_without_reversing_and_columns_wrap():
    chromosome = ChromosomeSequence("ACGT")
    assert chromosome.positive_strand == "ACGT"
    assert chromosome.negative_strand == "TGCA"
    assert chromosome[1, 1] == "A"
    assert chromosome[1, 2] == "T"
    assert chromosome[4, 5] == "T"


def test_two_subscript_indexing_uses_matlab_cartesian_product():
    chromosome = ChromosomeSequence("ACGT")
    np.testing.assert_array_equal(
        chromosome[[1, 3], [1, 2]], np.array([["A", "T"], ["G", "C"]])
    )
    assert chromosome[[1, 3], 2].ravel().tolist() == ["T", "C"]


def test_two_subscript_nonscalar_results_keep_matlab_matrix_shape():
    chromosome = ChromosomeSequence("ACGT")

    row = chromosome[1, [1, 2, 4]]
    column = chromosome[[1, 3], 2]

    assert row.shape == (1, 3)
    assert column.shape == (2, 1)
    np.testing.assert_array_equal(row, [["A", "T", "T"]])
    np.testing.assert_array_equal(column, [["T"], ["C"]])


@pytest.mark.parametrize("indices", [1.5, np.nan, [1, 2.5], [1, np.inf]])
def test_linear_indexing_rejects_noninteger_or_nonfinite_indices(indices):
    with pytest.raises(IndexError, match="finite integers"):
        ChromosomeSequence("ACGT")[indices]


def test_constructor_rejects_character_matrices_like_matlab_vector_contract():
    with pytest.raises(ValueError, match="character vector"):
        ChromosomeSequence(np.array([["A", "C"], ["G", "T"]]))


def test_find_subsequence_wraps_and_reports_motif_anchor_on_both_strands():
    chromosome = ChromosomeSequence("AAGT")
    np.testing.assert_array_equal(chromosome.findSubsequence("TA", 2), [1])
    np.testing.assert_array_equal(
        ChromosomeSequence("ATGC").findSubsequence("AT", 1, True),
        [[1, 1], [1, 2]],
    )


def test_find_subsequence_supports_iupac_motif_symbols():
    chromosome = ChromosomeSequence("TTACCCCGTCA")
    np.testing.assert_array_equal(chromosome.findSubsequence("TTA", 1), [1])
    np.testing.assert_array_equal(chromosome.findSubsequence("TTN", 1), [1])


def test_circular_positions_and_subsequence_strand_parity():
    chromosome = ChromosomeSequence("AACG")
    assert chromosome.circularposition([-1, 1, 5, 9]).tolist() == [3, 1, 1, 1]
    assert chromosome.subsequence([4, 1, 5], [1, 2, 4]) == "GTT"
    assert chromosome.subsequence([[4, 2], [1, 1]]) == "CA"


def test_base_counts_match_matlab_acgt_order_and_strand():
    chromosome = ChromosomeSequence("AACCGT")
    np.testing.assert_array_equal(chromosome.basecount(1), [2, 2, 1, 1])
    np.testing.assert_array_equal(chromosome.basecount(2), [1, 1, 2, 2])
    np.testing.assert_array_equal(chromosome.basecount(), [3, 3, 3, 3])


def test_snapshot_exported_sequence_constructs_and_preserves_bases():
    snapshot = Path(__file__).parents[1] / "reference" / "snapshot_t0.json"
    raw = json.loads(snapshot.read_text())
    record = next(o for o in raw["objects"] if o["id"] == "state:Chromosome")
    decoded_record = load_snapshot(snapshot).states["Chromosome"]
    exported = decoded_record.sequence
    assert isinstance(exported, ChromosomeSequenceData)
    chromosome = ChromosomeSequence(exported.positive_strand)
    assert (
        chromosome.positive_strand
        == record["fields"]["sequence"]["positive_strand"]
    )
    assert len(chromosome) == len(
        record["fields"]["sequence"]["positive_strand"]
    )


@pytest.mark.parametrize("sequence", ["ACNT", "", "acgt"])
def test_rejects_invalid_sequences(sequence):
    with pytest.raises(ValueError):
        ChromosomeSequence(sequence)


def test_empty_subsequence_counts_are_zero_for_both_strands():
    sequence = ChromosomeSequence("ACGT")
    for strand in (1, 2, 3, 4):
        np.testing.assert_array_equal(
            sequence.subsequenceBaseCounts([], strand), np.zeros((4, 1))
        )


def test_assigning_positive_strand_refreshes_derived_sequences_and_counts():
    sequence = ChromosomeSequence("AACG")
    sequence.positive_strand = "TTGC"
    assert sequence.negative_strand == "AACG"
    assert sequence.subsequence([1, 2, 3, 4], [1, 2, 3, 4]) == "TAGG"
    np.testing.assert_array_equal(sequence.basecount(1), [0, 1, 1, 2])
    assert sequence.getGCContent() == 0.5
    np.testing.assert_array_equal(
        sequence.subsequenceBaseCounts([1, 2], [1, 2]).ravel(), [1, 0, 0, 1]
    )


def test_basecount_results_do_not_expose_the_cached_counts():
    sequence = ChromosomeSequence("AACG")
    counts = sequence.basecount(1)
    counts[:] = 99
    np.testing.assert_array_equal(sequence.basecount(1), [2, 1, 1, 0])


def test_unpickles_legacy_sequence_state_without_derived_cache_fields():
    # Construct the __dict__ shape written before complement/count caches
    # existed, then exercise the actual pickle restore path.
    legacy = ChromosomeSequence.__new__(ChromosomeSequence)
    legacy.__dict__.update(
        {
            "_positive_strand": "AACG",
            "_positive_bases": np.asarray(list("AACG"), dtype="U1"),
        }
    )
    restored = pickle.loads(pickle.dumps(legacy))
    assert restored.negative_strand == "TTGC"
    assert restored.subsequence([1, 2, 3, 4], 2) == "TTGC"
    np.testing.assert_array_equal(restored.basecount(1), [2, 1, 1, 0])
