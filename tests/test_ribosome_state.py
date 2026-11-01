from pathlib import Path

import numpy as np
import pytest

from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.ribosome import Ribosome


SNAPSHOT = Path(__file__).parents[1] / "reference" / "snapshot_t0.json"


def test_snapshot_ribosome_fields_and_occupancies():
    graph = load_snapshot(SNAPSHOT)
    source = graph.states["Ribosome"]
    ribosome = Ribosome(snapshot=source.fields)

    assert ribosome.states.shape == (132, 1)
    assert ribosome.boundMRNAs.shape == (132, 1)
    assert ribosome.states.flags.f_contiguous
    np.testing.assert_array_equal(
        ribosome.nActive, [[np.count_nonzero(ribosome.states == 1)]]
    )
    np.testing.assert_array_equal(
        ribosome.nNotExist, [[np.count_nonzero(ribosome.states == 0)]]
    )
    np.testing.assert_array_equal(
        ribosome.nStalled, [[np.count_nonzero(ribosome.states == -1)]]
    )
    np.testing.assert_allclose(
        ribosome.stateOccupancies[:, 0, 0],
        [np.count_nonzero(ribosome.states == v) / 132 for v in (1, 0, -1)],
    )


def test_n_mrnas_bound_retains_one_based_gene_indices_and_time_dimension():
    ribosome = Ribosome(
        states=np.ones((3, 1, 2)),
        boundMRNAs=np.array([[[2, 0]], [[4, 3]], [[2, 4]]]),
    )
    ribosome.polypeptide = type(
        "PolypeptideData", (), {"monomerLengths": np.arange(4)}
    )()

    np.testing.assert_array_equal(ribosome.nMRNAsBound[:, 0, 0], [0, 2, 0, 1])
    np.testing.assert_array_equal(ribosome.nMRNAsBound[:, 0, 1], [0, 0, 1, 1])


def test_release_reports_snapshot_only_rng_dependency():
    graph = load_snapshot(SNAPSHOT)
    ribosome = Ribosome(snapshot=graph.states["Ribosome"].fields)
    assert not hasattr(ribosome.randStream, "randomlySelectNRows")
    with pytest.raises(NotImplementedError, match="runtime RandStream"):
        ribosome.releaseRibosome(1, 0, np.zeros((0, 1)))


def test_allocate_memory_uses_matlab_column_and_time_layout():
    ribosome = Ribosome()
    ribosome.allocateMemory(4)
    for name in ribosome.stateNames:
        value = getattr(ribosome, name)
        assert value.shape == (0, 1, 4)
        assert value.flags.f_contiguous


def test_release_skips_empty_peptides_but_releases_all_selected_ribosomes():
    # Original Ribosome.m:168-181: zero peptide AND zero tag must not
    # append an aborted record; tag-only stalled peptides must be retained.
    from types import SimpleNamespace

    class SelectAll:
        def randomlySelectNRows(self, rows, n):
            return rows[:n], None

        def rand(self, *shape):
            raise AssertionError("no additional draws expected")

    ribosome = Ribosome(
        states=np.array([[1], [1], [-1]]),
        boundMRNAs=np.array([[1], [2], [3]]),
        mRNAPositions=np.array([[0], [7], [0]]),
        tmRNAPositions=np.array([[0], [0], [4]]),
        randStream=SelectAll(),
    )
    poly = SimpleNamespace(
        abortedPolypeptides=np.array([[9, 3, 0]]),
        boundMRNAs=ribosome.boundMRNAs.copy(),
        nascentMonomerLengths=ribosome.mRNAPositions.copy(),
        proteolysisTagLengths=ribosome.tmRNAPositions.copy(),
    )
    ribosome.polypeptide = poly
    ribosome.rna = SimpleNamespace(
        boundIndexs=np.array([2]),
        matureIndexs=np.array([1]),
        matureTMRNAIndexs=np.array([1]),
        compartment=SimpleNamespace(cytosolIndexs=1),
        counts=np.array([[0], [1]]),
    )
    ribosome.releaseRibosome(3, 0, np.zeros((0, 1)))
    np.testing.assert_array_equal(
        poly.abortedPolypeptides, [[9, 3, 0], [2, 7, 0], [3, 0, 4]]
    )
    for name in ribosome.stateNames:
        np.testing.assert_array_equal(
            getattr(ribosome, name), np.zeros((3, 1))
        )
    np.testing.assert_array_equal(ribosome.rna.counts, [[1], [0]])
