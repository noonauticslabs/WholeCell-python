import numpy as np

from wholecell_py.sparsemat import SparseMat
from wholecell_py.states.chromosome_accessibility import (
    ChromosomeAccessibility,
)


def state():
    c = ChromosomeAccessibility()
    c.sequenceLen = 10
    c.nCompartments = 4
    c.dnaStrandedness_dsDNA = 2
    c.monomerDNAFootprints = np.array([3, 2])
    c.complexDNAFootprints = np.array([2])
    c.monomerDNAFootprintBindingStrandedness = np.array([2, 1])
    c.complexDNAFootprintBindingStrandedness = np.array([1])
    c.monomerBoundSites = SparseMat([[9, 1], [4, 3]], [1, 2], [10, 4])
    c.complexBoundSites = SparseMat([[4, 2]], [1], [10, 4])
    c.monomerIndexs_replisome = np.array([], dtype=int)
    c.complexIndexs_replisome = np.array([], dtype=int)
    return c


def test_nonempty_bindings_wrap_and_both_strands():
    c = state()
    positions = np.array([[9, 1], [9, 2], [4, 3]])
    flags, idx, kept, ext, monomers, complexes = c.isRegionProteinFree(
        positions, [3, 3, 2], True, [], [], True
    )
    np.testing.assert_array_equal(flags, [True, True, True])
    np.testing.assert_array_equal(idx, [1, 2, 3])
    np.testing.assert_array_equal(kept, positions)
    np.testing.assert_array_equal(ext, [0, 0, 0])
    np.testing.assert_array_equal(
        monomers.full()[:, :3], [[1, 1, 0], [1, 1, 0], [2, 0, 0]]
    )
    np.testing.assert_array_equal(
        complexes.full()[:, :2], [[0, 0], [0, 0], [0, 0]]
    )


def test_strand_selection_and_release_extent():
    c = state()
    # The dsDNA monomer bound on strand 1 occupies either strand when the
    # query is single-strand mode. A ssDNA monomer stays on strand 3.
    positions = np.array([[9, 2], [4, 3]])
    flags, _, _, ext, monomers, _ = c.isRegionProteinFree(
        positions, [3, 2], True, [1], [], False
    )
    np.testing.assert_array_equal(ext, [3, 0])
    np.testing.assert_array_equal(
        monomers.full()[:, :3], [[1, 1, 0], [2, 0, 0]]
    )
    flags, idx, kept, lengths, _, _ = c.isRegionProteinFree(
        positions, [3, 2], False, [1], [], False
    )
    np.testing.assert_array_equal(flags, [True, False])
    np.testing.assert_array_equal(idx, [1])
    np.testing.assert_array_equal(kept, positions[:1])
    np.testing.assert_array_equal(lengths, [3])


def test_negative_length_uses_reverse_region_coordinates():
    c = state()
    _, _, _, extent, monomers, _ = c.isRegionProteinFree(
        [[1, 1]], [-4], True, [], [], True
    )
    # The source clips the wrapped query at the origin before measuring the
    # reverse offset; the overlapping footprint therefore truncates to zero.
    np.testing.assert_array_equal(extent, [0])
    np.testing.assert_array_equal(monomers.full()[0, :4], [1, 1, 0, 0])
