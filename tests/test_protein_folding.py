from types import SimpleNamespace

import numpy as np
import pytest

from wholecell_py.processes.protein_folding import ProteinFolding
from wholecell_py.randstream import RandStream
from wholecell_py.snapshot import load_snapshot


def process():
    return ProteinFolding(
        substrates=np.array([10.0, 10.0, 0.0, 10.0, 10.0]),
        enzymes=np.array([10.0, 10.0]),
        unfoldedMonomers=np.array([3.0, 2.0]),
        unfoldedComplexs=np.array([1.0]),
        foldedMonomers=np.zeros(2),
        foldedComplexs=np.zeros(1),
        proteinProstheticGroupMatrix=np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
        proteinChaperoneMatrix=np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
        monomerComplexIndexs_folded=np.array([1, 3]),
        monomerIndexs_folding=np.array([1]),
        complexIndexs_folding=np.array([1]),
        monomerIndexs_notFolding=np.array([2]),
        complexIndexs_notFolding=np.array([], dtype=int),
        speciesIndexs_monomers=np.array([1]),
        speciesIndexs_complexs=np.array([2]),
        speciesReactantByproductMatrix=np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        speciesReactantMatrix=np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        substrateIndexs_water=4,
        substrateIndexs_hydrogen=5,
        randStream=RandStream(seed=2),
    )


@pytest.mark.parametrize("column_vectors", [False, True])
def test_folds_requirements_and_pass_through(column_vectors):
    p = process()
    if column_vectors:
        for name in ("substrates", "enzymes", *p._LOCAL_STATES):
            setattr(p, name, np.asarray(getattr(p, name)).reshape(-1, 1))
    # One monomer and one complex have folding reactions; the second monomer is outside the folding network.
    p.evolveState()
    assert p.unfoldedMonomers.sum() == 2
    assert p.foldedMonomers.sum() == 3
    assert p.foldedComplexs.sum() == 1
    assert p.substrates[0] == 7 and p.substrates[1] == 9


def test_initialize_complexes_and_lifecycle_requirements():
    p = process()
    p.initializeState()
    assert p.unfoldedComplexs.sum() == 0 and p.foldedComplexs.sum() == 1
    p.substrateWholeCellModelIDs = ("a", "b", "c", "d", "e")
    p.enzymeWholeCellModelIDs = (
        "MG_238_MONOMER",
        "MG_305_MONOMER",
        "MG_019_DIMER",
        "MG_201_DIMER",
        "MG_392_393_21MER",
        "additional_chaperone",
    )
    bm, by, lo, hi = p.calcResourceRequirements_LifeCycle(
        None,
        SimpleNamespace(monomerProductions=[1, 2], complexProductions=[3]),
    )
    np.testing.assert_array_equal(bm, [1, 3, 0, 0, 0])
    np.testing.assert_array_equal(by, np.zeros(5))
    np.testing.assert_array_equal(lo, [2, 2, 2, 2, 2, 0])
    assert np.all(np.isinf(hi))


def test_initialize_constants_reports_missing_export_dependencies():
    with pytest.raises(
        NotImplementedError, match="initialized knowledge-base"
    ):
        ProteinFolding().initializeConstants()


def test_snapshot_t0_folding_integration():
    from pathlib import Path

    g = load_snapshot(
        Path(__file__).resolve().parents[1] / "reference/snapshot_t0.json"
    )
    p = ProteinFolding(**g.processes["ProteinFolding"].fields)
    p.randStream = RandStream(seed=1)
    p.copyFromState()
    p.calcResourceRequirements_Current()
    p.evolveState()
    p.copyToState()
    dry_weight = p.getDryWeight()
    assert np.asarray(dry_weight).size == 1
    assert np.isfinite(dry_weight).all()
