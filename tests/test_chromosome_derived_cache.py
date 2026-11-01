import numpy as np
import pytest

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.states.chromosome import Chromosome


def make_chromosome():
    return Chromosome(
        sequenceLen=12,
        nCompartments=4,
        relaxedBasesPerTurn=10.0,
        equilibriumSuperhelicalDensity=-0.06,
        supercoiledSuperhelicalDensityTolerance=0.02,
        polymerizedRegions=CircularSparseMat(
            [[2, 1], [5, 2], [8, 1], [10, 2], [2, 3], [5, 4], [8, 3]],
            [5, 3, 2, 2, 5, 3, 2],
            [12, 4],
            [1],
        ),
    )


def assert_sparse_equal(actual, expected):
    ap, av = actual.find()
    ep, ev = expected.find()
    np.testing.assert_array_equal(ap, ep)
    np.testing.assert_array_equal(av, ev)
    np.testing.assert_array_equal(actual.siz, expected.siz)
    np.testing.assert_array_equal(actual.circularDims, expected.circularDims)


@pytest.mark.parametrize(
    "property_name,calculation_name",
    [
        ("unpolymerizedRegions", "calcUnpolymerizedRegions"),
        ("singleStrandedRegions", "calcSingleStrandedRegions"),
        ("doubleStrandedRegions", "calcDoubleStrandedRegions"),
    ],
)
def test_sparse_derived_getters_reuse_and_return_independent_values(
    property_name, calculation_name
):
    chromosome = make_chromosome()
    original = getattr(chromosome, calculation_name)
    calls = []

    def counted():
        calls.append(1)
        return original()

    setattr(chromosome, calculation_name, counted)
    first = getattr(chromosome, property_name)
    expected = first.copy()
    second = getattr(chromosome, property_name)
    assert len(calls) == 1
    assert_sparse_equal(second, expected)

    if second.nnz():
        second.vals[0] = 999
    third = getattr(chromosome, property_name)
    assert len(calls) == 1
    assert_sparse_equal(third, expected)


def test_sparse_derived_cache_detects_in_place_sparse_and_dimension_edits():
    chromosome = make_chromosome()
    calls = []
    original = chromosome.calcUnpolymerizedRegions

    def counted():
        calls.append(1)
        return original()

    chromosome.calcUnpolymerizedRegions = counted
    first = chromosome.unpolymerizedRegions
    assert len(calls) == 1

    # Mutations to each public sparse source field must invalidate by content.
    chromosome.polymerizedRegions.subs[0, 0] = 3
    expected = original()
    assert_sparse_equal(chromosome.unpolymerizedRegions, expected)
    assert len(calls) == 2

    chromosome.polymerizedRegions.vals[0] += 1
    expected = original()
    assert_sparse_equal(chromosome.unpolymerizedRegions, expected)
    assert len(calls) == 3

    chromosome.polymerizedRegions.siz[0] = 14
    chromosome.sequenceLen = 14
    expected = original()
    assert_sparse_equal(chromosome.unpolymerizedRegions, expected)
    assert len(calls) == 4

    chromosome.polymerizedRegions.circularDims[0] = 2
    expected = original()
    assert_sparse_equal(chromosome.unpolymerizedRegions, expected)
    assert len(calls) == 5

    chromosome.polymerizedRegions.siz[1] = 5
    chromosome.nCompartments = 5
    expected = original()
    assert_sparse_equal(chromosome.unpolymerizedRegions, expected)
    assert len(calls) == 6


def test_sparse_derived_cache_rebuilds_without_private_checkpoint_cache():
    chromosome = make_chromosome()
    expected = chromosome.calcSingleStrandedRegions()
    chromosome.__dict__.pop("_derived_sparse_cache", None)
    assert_sparse_equal(chromosome.singleStrandedRegions, expected)


def test_sparse_derived_cache_tracks_calculation_and_merge_methods():
    chromosome = make_chromosome()
    calculation_calls = []
    original_calculation = chromosome.calcDoubleStrandedRegions

    def counted_calculation():
        calculation_calls.append(1)
        return original_calculation()

    chromosome.calcDoubleStrandedRegions = counted_calculation
    chromosome.doubleStrandedRegions
    chromosome.doubleStrandedRegions
    assert len(calculation_calls) == 1

    merge_calls = []
    original_merge = chromosome.mergeAdjacentRegions

    def counted_merge(*args, **kwargs):
        merge_calls.append(1)
        return original_merge(*args, **kwargs)

    chromosome.mergeAdjacentRegions = counted_merge
    expected = original_calculation()
    actual = chromosome.doubleStrandedRegions
    assert_sparse_equal(actual, expected)
    assert len(calculation_calls) == 2
    assert (
        len(merge_calls) == 2
    )  # one direct calculation plus one cache rebuild

    chromosome.doubleStrandedRegions
    assert len(calculation_calls) == 2
    assert len(merge_calls) == 2
