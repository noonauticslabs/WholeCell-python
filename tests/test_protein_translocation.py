from types import SimpleNamespace

import numpy as np

from wholecell_py.processes.protein_translocation import ProteinTranslocation
from wholecell_py.randstream import RandStream


def configured():
    return ProteinTranslocation(
        substrates=np.array([20.0, 10.0, 0.0, 0.0, 0.0, 50.0, 0.0]),
        enzymes=np.array([10.0, 10.0, 3.0, 3.0]),
        monomers=np.array([[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]]),
        monomerLengths=np.array([36.0, 70.0, 50.0]),
        monomerCompartments=np.array([2, 2, 2]),
        monomerIndexs_translocating=np.array([1, 2, 3]),
        monomerIndexs_nontranslocating=np.array([], dtype=int),
        monomerSRPPathways=np.array([True, False, True]),
        monomerWholeCellModelIDs=["a", "b", "c"],
        preproteinTranslocase_aaTranslocatedPerATP=35.0,
        SRP_GTPUsedPerMonomer=2.0,
        translocaseSpecificRate=150.0,
        compartment=SimpleNamespace(count=2, cytosolIndexs=1),
        randStream=RandStream(seed=4),
        stepSizeSec=1.0,
    )


def test_lifecycle_resource_requirements():
    p = configured()
    s = SimpleNamespace(
        monomerProductions=np.array([1.0, 2.0, 3.0]),
        monomerProductions0=np.array([1.0, 2.0, 3.0]),
    )
    bm, by, lo, hi = p.calcResourceRequirements_LifeCycle(None, s)
    np.testing.assert_array_equal(bm, [12, 8, 0, 0, 0, 20, 0])
    np.testing.assert_array_equal(by, [0, 0, 12, 8, 20, 0, 20])
    np.testing.assert_allclose(lo, [8, 8, 0.16, 0.16])
    assert np.isinf(hi).all()


def test_current_requirements_and_evolution():
    p = configured()
    np.testing.assert_array_equal(
        p.calcResourceRequirements_Current(), [12, 10, 0, 0, 0, 22, 0]
    )
    p.evolveState()
    # All six substrates are translocated when catalytic and substrate capacity suffice.
    assert p.monomers[:, 0].sum() == 0
    assert p.monomers[:, 1].sum() == 6
    np.testing.assert_array_equal(p.substrates, [8, 0, 12, 10, 22, 28, 22])


def test_initialize_requires_snapshot_constants():
    p = ProteinTranslocation()
    try:
        p.initializeConstants()
    except NotImplementedError as exc:
        assert "monomerWholeCellModelIDs" in str(exc)
    else:
        raise AssertionError("missing initialization data must be reported")
