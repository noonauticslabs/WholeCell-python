import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome_breaks import ChromosomeBreaks


class State(ChromosomeBreaks):
    pass


def test_strand_break_classification_counts_only_fully_polymerized_segments():
    state = State()
    state.sequenceLen = 8
    state.nCompartments = 4
    state.strandBreakClassification_segmentLength = 4
    state.strandBreakClassification_doubleStrandBreakSeparation = 3
    state.strandBreaks = CircularSparseMat(
        [[1, 1, 1], [5, 1, 1], [6, 2, 1]],
        [True, True, True],
        [8, 4, 1],
        [1],
    )
    # Both chromosome copies have two complete segments.
    state.polymerizedRegions = CircularSparseMat(
        [[1, strand, 1] for strand in range(1, 5)],
        [8, 8, 8, 8],
        [8, 4, 1],
        [1],
    )

    result = state.calcStrandBreakClassification()
    np.testing.assert_array_equal(result[:, 0, 0], [2, 1, 0, 0, 1, 0, 0])


def test_strand_break_classification_wraps_partial_origin_segment():
    state = State()
    state.sequenceLen = 7
    state.nCompartments = 2
    state.strandBreakClassification_segmentLength = 4
    state.strandBreakClassification_doubleStrandBreakSeparation = 3
    state.strandBreaks = CircularSparseMat([[1, 1, 1]], [True], [7, 2, 1], [1])
    # The duplicated origin base completes segment two, as in Chromosome.m's
    # padded sparse matrix before classification.
    state.polymerizedRegions = CircularSparseMat(
        [[1, 1, 1], [1, 2, 1]], [7, 7], [7, 2, 1], [1]
    )
    result = state.calcStrandBreakClassification()
    np.testing.assert_array_equal(result[:, 0, 0], [0, 2, 0, 0, 0, 0, 0])


def test_strand_break_classification_preserves_matlab_find_run_order_for_two_copies():
    state = State()
    state.sequenceLen = 8
    state.nCompartments = 4
    state.strandBreakClassification_segmentLength = 4
    state.strandBreakClassification_doubleStrandBreakSeparation = 3
    state.strandBreaks = CircularSparseMat(
        [[1, 1, 1], [1, 3, 1], [3, 1, 1], [3, 3, 1]],
        [True] * 4,
        [8, 4, 1],
        [1],
    )
    state.polymerizedRegions = CircularSparseMat(
        [[1, strand, 1] for strand in range(1, 5)],
        [8] * 4,
        [8, 4, 1],
        [1],
    )

    # Chromosome.m groups contiguous rows from find(permute(...)); at each
    # damaged base the two chromosome-copy pairs interleave, yielding four
    # one-site SSB groups instead of coalescing per copy across the segment.
    result = state.calcStrandBreakClassification()
    np.testing.assert_array_equal(result[:, 0, 0], [0, 4, 0, 0, 0, 0, 0])


def test_strand_break_classification_matches_exported_matlab_snapshots():
    from pathlib import Path

    from wholecell_py.snapshot import load_snapshot
    from wholecell_py.states.chromosome import Chromosome

    root = Path(__file__).resolve().parents[1]
    for name in ("snapshot_t0.json", "snapshot_seed22_t4000.json"):
        graph = load_snapshot(root / "reference" / name)
        fields = graph.states["Chromosome"].fields
        state = Chromosome(snapshot=fields)
        np.testing.assert_array_equal(
            state.calcStrandBreakClassification().reshape(-1),
            np.asarray(fields["strandBreakClassification"]).reshape(-1),
        )
