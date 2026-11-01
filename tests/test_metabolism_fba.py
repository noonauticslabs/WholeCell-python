from pathlib import Path
import numpy as np
from wholecell_py.processes.metabolism_fba import _blocks, _null_active


def test_network_blocks_and_rational_null_support():
    r, c = _blocks(np.array([[1, 0, 1], [0, 0, 0], [1, 0, 0]]))
    np.testing.assert_array_equal(r, [2, 1, 2])
    np.testing.assert_array_equal(c, [2, 1, 2])
    np.testing.assert_array_equal(
        _null_active([[1, -1, 0], [0, 0, 1]]), [True, True, False]
    )
    np.testing.assert_array_equal(_null_active(np.eye(3)), [False] * 3)


def test_rebuilt_fba_matches_exported_matlab_network():
    from wholecell_py.runner import load_model

    p = load_model(
        Path(__file__).resolve().parents[1] / "reference/snapshot_t60.json",
        seed=4,
    ).processes["Metabolism"]
    fields = [
        name
        for name in vars(p)
        if name.startswith(
            (
                "fba",
                "substrateIndexs_",
                "reactionIndexs_fba",
                "metabolismNew",
                "metabolismRecycling",
            )
        )
        and isinstance(getattr(p, name), np.ndarray)
    ]
    expected = {name: getattr(p, name).copy() for name in fields}
    p.formulateFBA()
    for name, value in expected.items():
        actual = np.asarray(getattr(p, name))
        assert actual.shape == value.shape, name
        np.testing.assert_allclose(
            actual, value, rtol=1e-12, atol=1e-10, equal_nan=True, err_msg=name
        )
