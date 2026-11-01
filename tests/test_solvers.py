import numpy as np
import pytest
import json
from pathlib import Path

from wholecell_py.solvers import solve_lp


def test_nontrivial_maximization_equality_lp_matches_primal_and_duals():
    # max 3x + 2y; x+y<=4, x<=2, y<=3, x,y>=0
    result = solve_lp(
        {
            "backend": "highs",
            "maximize": "maximize",
            "f": [3, 2],
            "A": [[1, 1], [1, 0], [0, 1]],
            "b": [4, 2, 3],
            "lb": [0, 0],
            "ub": [np.inf, np.inf],
            "constraint_types": "UUU",
            "variable_types": "C",
        }
    )
    assert result["status"] == "optimal"
    assert result["solver"].startswith("scipy-highs")
    np.testing.assert_allclose(result["x"], [2, 2])
    assert result["fopt"] == pytest.approx(10)
    np.testing.assert_allclose(
        result["lambda"]["shadowPrices"], [2, 1, 0], atol=1e-8
    )
    assert not result["errorFlag"]


def test_matlab_constraint_codes_and_maximize_sign_conversion():
    # Equality fixes x+y=5, lower row forces x>=2, free row ignored, D row
    # enforces |x-y|<=1. Minimize x+2y => (4,1).
    result = solve_lp(
        {
            "backend": "highs",
            "maximize": False,
            "f": [1, 2],
            "A": [[1, 1], [1, 0], [0, 0], [1, -1]],
            "b": [5, 2, 100, 1],
            "lb": [0, 0],
            "ub": [10, 10],
            "constraint_types": ["S", "L", "F", "D"],
            "variable_types": ["C", "C"],
        }
    )
    assert result["status"] == "optimal"
    np.testing.assert_allclose(result["x"], [3, 2])
    assert result["fopt"] == pytest.approx(7)
    assert result["lambda"]["shadowPrices"].shape == (4,)


def test_integer_and_binary_variables_are_respected():
    result = solve_lp(
        {
            "backend": "highs",
            "maximize": "maximize",
            "f": [3, 2],
            "A": [[2, 1]],
            "b": [3],
            "lb": [0, 0],
            "ub": [np.inf, np.inf],
            "constraint_types": "U",
            "variable_types": ["I", "B"],
        }
    )
    assert result["status"] == "optimal"
    np.testing.assert_allclose(result["x"], [1, 1])
    assert result["fopt"] == pytest.approx(5)
    assert np.isnan(result["lambda"]["shadowPrices"]).all()


def test_infeasible_problem_returns_matlab_shaped_failure_result():
    result = solve_lp(
        {
            "backend": "highs",
            "maximize": "maximize",
            "f": [1],
            "A": [[1], [1]],
            "b": [2, 1],
            "lb": [0],
            "ub": [1],
            "constraint_types": ["L", "U"],
            "variable_types": "C",
        }
    )
    assert result["status"] == "infeasible"
    assert result["errorFlag"]
    assert result["x"].shape == (1,)
    assert result["lambda"]["shadowPrices"].shape == (2,)


def test_glpk_backend_returns_primal_dual_and_status():
    result = solve_lp(
        {
            "backend": "glpk",
            "maximize": "maximize",
            "f": [3, 2],
            "A": [[1, 1], [1, 0], [0, 1]],
            "b": [4, 2, 3],
            "lb": [0, 0],
            "ub": [np.inf, np.inf],
            "constraint_types": "UUU",
            "variable_types": "C",
        }
    )
    assert result["status"] == "optimal"
    assert result["solver"] == "swiglpk/GLPK"
    np.testing.assert_allclose(result["x"], [2, 2], atol=1e-8)
    assert result["fopt"] == pytest.approx(10)
    np.testing.assert_allclose(
        result["lambda"]["shadowPrices"], [2, 1, 0], atol=1e-8
    )
    assert not result["errorFlag"]


def test_glpk_supports_matlab_double_and_free_rows_and_mip():
    result = solve_lp(
        {
            "backend": "glpk",
            "maximize": "maximize",
            "f": [3, 2],
            "A": [[2, 1], [1, -1], [0, 0]],
            "b": [3, 1, 9],
            "lb": [0, 0],
            "ub": [np.inf, np.inf],
            "constraint_types": ["U", "D", "F"],
            "variable_types": ["I", "B"],
        }
    )
    assert result["status"] == "optimal"
    np.testing.assert_allclose(result["x"], [1, 1], atol=1e-8)
    assert result["fopt"] == pytest.approx(5)
    assert np.isnan(result["lambda"]["shadowPrices"]).all()


def test_glpk444_reproduces_cached_matlab_metabolism_lp_vectors():
    fixture_path = (
        Path(__file__).resolve().parents[1] / "reference" / "matlab_seed1.json"
    )
    fixture = json.loads(fixture_path.read_text())
    cases = [
        case
        for case in fixture["cases"]
        if case["process"] == "linear_programming"
    ]
    assert (
        cases
    ), "fixture must contain at least one cached MATLAB metabolic LP"
    for case in cases:
        result = solve_lp({**case["inputs"], "backend": "glpk444"})
        expected = case["expected"]
        assert result["status"] == "optimal", case["name"]
        assert result["solver"] == "GLPK 4.44 (standalone)"
        assert result["fopt"] == pytest.approx(
            expected["fopt"], abs=1e-15, rel=1e-12
        )
        np.testing.assert_allclose(
            result["x"], expected["x"], rtol=1e-12, atol=1e-9
        )


@pytest.mark.parametrize("order", ["C", "F"])
def test_glpk444_sparse_export_preserves_mex_insertion_order(
    monkeypatch, order
):
    """Storage order must not alter GLPK row-major insertion or numeric text."""
    import wholecell_py.solvers as solvers

    original_run = solvers.subprocess.run
    captured = []

    def capture_run(args, **kwargs):
        captured.append(Path(args[1]).read_text().splitlines())
        return original_run(args, **kwargs)

    monkeypatch.setattr(solvers.subprocess, "run", capture_run)
    result = solve_lp(
        {
            "backend": "glpk444",
            "maximize": True,
            "f": [0, 0, 0],
            "A": np.array([[0.0, 2.0, -0.0], [-3.0, 0.0, 4.0]], order=order),
            "b": [0, 0],
            "lb": [0, 0, 0],
            "ub": [1, 1, 1],
            "constraint_types": "S",
            "variable_types": "C",
        }
    )
    assert result["status"] == "optimal"
    assert captured[0][-4:] == ["3", "1 2 2", "2 1 -3", "2 3 4"]


def test_solver_rejects_matrix_vectors_and_complex_inputs():
    from wholecell_py.solvers import _vector

    for value in (
        np.ones((1, 2)),
        np.ones((2, 2)),
        np.ones((2, 1, 1)),
        np.array([1 + 1j]),
    ):
        with pytest.raises(ValueError):
            _vector(value, "b")
    np.testing.assert_array_equal(_vector([[1], [2]], "b"), [1, 2])


def test_invalid_bounds_keep_requested_solver_provenance():
    r = solve_lp(
        dict(A=[[1]], b=[1], f=[1], lb=[2], ub=[1], backend="glpk444")
    )
    assert r["status"] == "invalid_bounds"
    assert r["solver"] == "GLPK 4.44 (standalone)"


def test_scalar_array_type_codes_expand_like_matlab_scalars():
    from wholecell_py.solvers import _codes

    np.testing.assert_array_equal(
        _codes(np.array([["c"]]), 3, "variable_types", "CIB"), ["C", "C", "C"]
    )


def test_default_backend_is_original_glpk444():
    result = solve_lp(
        dict(A=[[1]], b=[2], f=[1], lb=[0], ub=[3], maximize=True)
    )
    assert result["solver"] == "GLPK 4.44 (standalone)"
    np.testing.assert_allclose(result["x"], [2])
