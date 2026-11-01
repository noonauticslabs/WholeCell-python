import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome_damage_views import ChromosomeDamageViews


def mat(coords, values, size=8):
    return CircularSparseMat(coords, values, [size, 2], 1)


def state():
    class Metabolite:
        m6ADIndexs = 99

    class State(ChromosomeDamageViews):
        pass

    s = State()
    s.sequenceLen = 8
    s.metabolite = Metabolite()
    s.gapSites = mat([], [])
    s.abasicSites = mat([], [])
    s.damagedSugarPhosphates = mat([], [])
    s.damagedBases = mat([[1, 1], [2, 2]], [99, 7])
    s.intrastrandCrossLinks = mat([], [])
    s.strandBreaks = mat([], [])
    s.hollidayJunctions = mat([], [])
    for name in (
        "intrastrandCrossLinks5",
        "intrastrandCrossLinks3",
        "strandBreaks5",
        "strandBreaks3",
        "hollidayJunctions5",
        "hollidayJunctions3",
    ):
        setattr(s, name, mat([], []))
    return s


def test_damaged_sites_filters_m6ad_and_counts_overlapping_damage_values():
    s = state()
    s.gapSites = mat([[2, 2]], [3])
    sites = s.getDamagedSites(True, False, False, False, False, False, False)
    coords, values = sites.find()
    assert coords.tolist() == [[2, 2]]
    assert values[:, 0].tolist() == [10]
    with_m6ad = s.getDamagedSites(
        True, False, False, False, False, False, True
    )
    assert with_m6ad.find()[1][:, 0].tolist() == [99, 10]


def test_strand_shifts_wrap_circular_coordinates_and_preserve_values():
    source = mat([[1, 1], [8, 2]], [4, 5])
    shifted = ChromosomeDamageViews.shiftCircularSparseMatBase3Prime(source)
    coords, values = shifted.find()
    assert coords.tolist() == [[2, 1], [7, 2]]
    assert sorted(values[:, 0].tolist()) == [4, 5]
    # Bond 3-prime's source rule shifts only even-numbered strands.
    bond = ChromosomeDamageViews.shiftCircularSparseMatBond3Prime(source)
    assert bond.find()[0].tolist() == [[1, 1], [7, 2]]


def test_strand_shift_preserves_time_coordinates_including_empty_storage():
    source = CircularSparseMat([[8, 1, 1], [1, 2, 2]], [4, 5], [8, 2, 2], 1)
    shifted = ChromosomeDamageViews.shiftCircularSparseMatBase3Prime(source)
    np.testing.assert_array_equal(shifted.find()[0], [[1, 1, 1], [8, 2, 2]])
    np.testing.assert_array_equal(shifted.find()[1].ravel(), [4, 5])
    empty = CircularSparseMat([], [], [8, 2, 2], 1)
    actual = ChromosomeDamageViews.shiftCircularSparseMatBase3Prime(empty)
    np.testing.assert_array_equal(actual.siz, [8, 2, 2])
    assert actual.nnz() == 0


def test_vector_lengths_apply_to_each_source_parity_subset_as_written():
    positions = np.array([[10, 1], [20, 2], [30, 3], [40, 4]])
    lengths = np.array([[5], [7]])

    shifted = ChromosomeDamageViews.shiftPositionsStrandsBase3Prime(
        positions, lengths
    )

    # MATLAB's odd/even assignments each add the whole two-element `lengths`
    # vector to that parity subset; lengths are not indexed by original row.
    np.testing.assert_array_equal(
        shifted, [[15, 1], [15, 2], [37, 3], [33, 4]]
    )
    np.testing.assert_array_equal(
        positions, [[10, 1], [20, 2], [30, 3], [40, 4]]
    )


def test_vector_lengths_must_fit_each_nonempty_parity_assignment():
    positions = np.array([[10, 1], [20, 2], [30, 3]])

    import pytest

    with pytest.raises(
        ValueError, match="match each nonempty strand-parity subset"
    ):
        ChromosomeDamageViews.shiftPositionsStrandsBase3Prime(
            positions, [5, 7]
        )


def test_bond_shift_validates_only_the_parity_changed_by_source():
    positions = np.array([[10, 1], [20, 2], [30, 3]])

    shifted = ChromosomeDamageViews.shiftPositionsStrandsBond3Prime(
        positions, [7]
    )

    np.testing.assert_array_equal(shifted, [[10, 1], [13, 2], [30, 3]])


def test_damage_cache_tracks_content_and_returns_independent_values():
    s = state()
    flags = (True, False, False, False, False, False, True)
    first = s.getDamagedSites(*flags)
    first[1, 1] = 0
    assert s.getDamagedSites(*flags).get_coords([[1, 1]]).item() == 99
    s.damagedBases.vals[0, 0] = 12
    assert s.getDamagedSites(*flags).get_coords([[1, 1]]).item() == 12
    s.damagedBases = s.damagedBases.set_coords([[3, 1]], 4)
    actual = s.getDamagedSites(*flags)
    expected = s._computeDamagedSites(*flags)
    assert actual.isequal(expected)
    s.metabolite.m6ADIndexs = 12
    without_methyl = s.getDamagedSites(
        True, False, False, False, False, False, False
    )
    assert without_methyl.get_coords([[1, 1]]).item() == 0


def test_damage_cache_tracks_explicit_shifted_views():
    s = state()
    flags = (False, True, False, True, False, False, False)
    assert s.getDamagedSites(*flags).nnz() == 0
    s.strandBreaks5 = mat([[2, 1]], [1])
    assert s.getDamagedSites(*flags).get_coords([[2, 1]]).item() == 1
