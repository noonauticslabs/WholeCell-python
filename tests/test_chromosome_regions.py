import numpy as np

from wholecell_py.states.chromosome_regions import ChromosomeRegions


class Regions(ChromosomeRegions):
    sequenceLen = 20
    nCompartments = 4


def test_oriC_split_join_and_empty_shapes():
    x = Regions()
    p, l = x.splitOverOriC([[18, 1]], [5])
    np.testing.assert_array_equal(p, [[18, 1], [1, 1]])
    np.testing.assert_array_equal(l, [3, 2])
    p, l = x.joinSplitOverOriCRegions([[1, 1], [18, 1], [4, 2]], [2, 3, 2])
    np.testing.assert_array_equal(p, [[18, 1], [4, 2]])
    np.testing.assert_array_equal(l, [5, 2])
    p, l = x.splitOverOriC(np.empty((0, 2), int), [])
    assert p.shape == (0, 2) and l.shape == (0,)


def test_split_regions_direction_and_truncation():
    p, l = Regions().splitRegions([[4, 2], [10, 3]], [11, -9], 4)
    np.testing.assert_array_equal(p, [[4, 2], [8, 2], [10, 3], [6, 3]])
    np.testing.assert_array_equal(l, [4, 4, 4, 4])


def test_join_regions_merges_overlap_adjacency_and_oriC_per_strand():
    p, l = Regions().joinSplitRegions(
        [[5, 1], [1, 1], [8, 2], [8, 2]], [3, 3, 2, 4]
    )
    np.testing.assert_array_equal(p, [[1, 1], [5, 1], [8, 2]])
    np.testing.assert_array_equal(l, [3, 3, 4])


def test_intersection_respects_strand_and_oriC_boundary():
    p, l = Regions().intersectRegions(
        [[18, 1]], [5], [[20, 1], [1, 1], [2, 2]], [3, 3, 4]
    )
    np.testing.assert_array_equal(p, [[20, 1]])
    np.testing.assert_array_equal(l, [3])


def test_exclusion_preserves_nonoverlap_and_carves_matching_strand():
    p, l = Regions().excludeRegions([[3, 1], [3, 2]], [10, 10], [[6, 1]], [3])
    np.testing.assert_array_equal(p, [[3, 1], [9, 1], [3, 2]])
    np.testing.assert_array_equal(l, [3, 4, 10])


def test_exclude_overlapping_keeps_nonoverlap_candidates():
    x = Regions()
    kept, new = x.excludeOverlappingRegions(
        [], [1, 2], [[5, 1], [7, 1], [15, 1]], [3, 3, 2], 1, 0, 0, False, False
    )
    np.testing.assert_array_equal(kept, [])
    np.testing.assert_array_equal(new, [1])


def test_region_outputs_use_matlab_column_major_coordinate_order():
    from wholecell_py.states.chromosome import Chromosome

    c = Chromosome(sequenceLen=100, nCompartments=4)
    positions = np.array([[2, 3], [80, 1], [10, 2]])
    for operation in (
        lambda: c.joinSplitRegions(positions, [3, 3, 3]),
        lambda: c.excludeRegions(
            positions, [3, 3, 3], np.empty((0, 2), int), []
        ),
    ):
        actual, lengths = operation()
        np.testing.assert_array_equal(actual, [[80, 1], [10, 2], [2, 3]])
        np.testing.assert_array_equal(lengths, [3, 3, 3])
