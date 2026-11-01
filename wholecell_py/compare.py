"""Run MATLAB-exported JSON cases against the supported Python ports.

Fixture schema: ``{schema_version: 1, source_commit, matlab_version,
cases: [{process, inputs, parameters, expected}]}``. Values are JSON scalars,
lists, and objects. This is a focused regression harness, not a WholeCell model
runner or MATLAB interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .processes import chromosome_segregation, protein_activation
from .solvers import solve_lp

ATOL = 1e-10
RTOL = 1e-9


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _compare_value(
    actual: Any, expected: Any, path: str
) -> list[dict[str, Any]]:
    """Return leaf mismatches; numeric arrays require identical shape."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [
                {
                    "path": path,
                    "expected_type": "object",
                    "actual": _jsonable(actual),
                }
            ]
        errors = []
        for key, val in expected.items():
            if key not in actual:
                errors.append(
                    {"path": f"{path}.{key}", "error": "missing actual key"}
                )
            else:
                errors.extend(
                    _compare_value(actual[key], val, f"{path}.{key}")
                )
        return errors

    # Lists are shape-significant numeric arrays when their members are numbers.
    exp_arr = np.asarray(expected)
    act_arr = np.asarray(actual)
    numeric_expected = np.issubdtype(
        exp_arr.dtype, np.number
    ) and exp_arr.dtype != np.dtype(bool)
    numeric_actual = np.issubdtype(
        act_arr.dtype, np.number
    ) and act_arr.dtype != np.dtype(bool)
    if numeric_expected and numeric_actual:
        if act_arr.shape != exp_arr.shape:
            return [
                {
                    "path": path,
                    "error": "shape mismatch",
                    "expected_shape": list(exp_arr.shape),
                    "actual_shape": list(act_arr.shape),
                }
            ]
        try:
            close = np.isclose(
                act_arr, exp_arr, atol=ATOL, rtol=RTOL, equal_nan=True
            )
        except (TypeError, ValueError):
            close = np.zeros(exp_arr.shape or (), dtype=bool)
        if not bool(np.all(close)):
            delta = np.abs(act_arr.astype(float) - exp_arr.astype(float))
            max_abs = float(np.nanmax(delta)) if delta.size else float(delta)
            return [
                {
                    "path": path,
                    "error": "numeric mismatch",
                    "max_abs_error": max_abs,
                    "expected": _jsonable(expected),
                    "actual": _jsonable(actual),
                }
            ]
        return []
    if isinstance(expected, (list, tuple, np.ndarray)) and isinstance(
        actual, (list, tuple, np.ndarray)
    ):
        if act_arr.shape != exp_arr.shape:
            return [
                {
                    "path": path,
                    "error": "shape mismatch",
                    "expected_shape": list(exp_arr.shape),
                    "actual_shape": list(act_arr.shape),
                }
            ]
        if not np.array_equal(act_arr, exp_arr):
            return [
                {
                    "path": path,
                    "error": "value mismatch",
                    "expected": _jsonable(expected),
                    "actual": _jsonable(actual),
                }
            ]
        return []
    if type(expected) is not type(actual) or expected != actual:
        return [
            {
                "path": path,
                "error": "value mismatch",
                "expected": _jsonable(expected),
                "actual": _jsonable(actual),
            }
        ]
    return []


def _lp_feasibility(problem: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
    A = np.asarray(problem["A"], dtype=float)
    b = np.asarray(problem["b"], dtype=float).reshape(-1)
    lb = np.asarray(
        problem.get("lb", np.full(A.shape[1], -np.inf)), dtype=float
    ).reshape(-1)
    ub = np.asarray(
        problem.get("ub", np.full(A.shape[1], np.inf)), dtype=float
    ).reshape(-1)
    if x.shape != (A.shape[1],) or not np.all(np.isfinite(x)):
        return {
            "feasible": False,
            "max_violation": None,
            "reason": "no finite solution vector",
        }
    codes = problem.get("constraint_types", "U")
    if isinstance(codes, str):
        codes = [codes] * A.shape[0] if len(codes) == 1 else list(codes)
    else:
        codes = [str(c) for c in np.asarray(codes).reshape(-1).tolist()]
    violations = [np.maximum(lb - x, 0), np.maximum(x - ub, 0)]
    ax = A @ x
    for i, code in enumerate(codes):
        code = code.upper()
        if code == "U":
            violations.append(np.asarray([max(ax[i] - b[i], 0)]))
        elif code == "L":
            violations.append(np.asarray([max(b[i] - ax[i], 0)]))
        elif code == "S":
            violations.append(np.asarray([abs(ax[i] - b[i])]))
        elif code == "D":
            violations.append(np.asarray([max(abs(ax[i]) - b[i], 0)]))
    maximum = max(
        (float(np.max(v)) for v in violations if v.size), default=0.0
    )
    return {
        "feasible": maximum <= 1e-6,
        "max_violation": maximum,
        "source_solver_tolerance": 1e-6,
        "strict_1e_10": maximum <= ATOL,
    }


def _run_lp(case: dict[str, Any]) -> dict[str, Any]:
    problem = case["inputs"].get("problem", case["inputs"])
    result = solve_lp(problem)
    expected = case["expected"]
    x = np.asarray(result["x"], dtype=float).reshape(-1)
    feasibility = _lp_feasibility(problem, x)
    f = np.asarray(problem["f"], dtype=float).reshape(-1)
    objective = float(np.dot(f, x)) if np.all(np.isfinite(x)) else None
    target_objective = expected.get("fopt", expected.get("objective"))
    objective_error = (
        abs(objective - float(target_objective))
        if objective is not None and target_objective is not None
        else None
    )
    flux_expected = expected.get("x", expected.get("fluxes"))
    flux_error = None
    if flux_expected is not None and np.all(np.isfinite(x)):
        exp_x = np.asarray(flux_expected, dtype=float)
        if exp_x.shape == x.shape:
            flux_error = float(np.max(np.abs(x - exp_x))) if x.size else 0.0
    status_expected = expected.get("status")
    status_match = (
        status_expected is None or result["status"] == status_expected
    )
    objective_match = (
        target_objective is None
        or objective_error is not None
        and objective_error <= ATOL + RTOL * abs(float(target_objective))
    )
    ok = (
        not result["errorFlag"]
        and feasibility["feasible"]
        and objective_match
        and status_match
    )
    return {
        "passed": bool(ok),
        "status": result["status"],
        "status_expected": status_expected,
        "status_match": bool(status_match),
        "solver": result["solver"],
        "feasibility": feasibility,
        "objective": objective,
        "expected_objective": target_objective,
        "objective_abs_error": objective_error,
        "objective_match": bool(objective_match),
        "flux_max_abs_difference": flux_error,
        "flux_difference_is_diagnostic_only": True,
        "flux_exact_match": (
            flux_error == 0.0 if flux_error is not None else None
        ),
        "reference_feasibility": (
            _lp_feasibility(
                problem, np.asarray(flux_expected, dtype=float).reshape(-1)
            )
            if flux_expected is not None
            else None
        ),
        "error_message": result["errorMsg"] if result["errorFlag"] else None,
    }


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    process = case["process"]
    if process == "ProteinActivation":
        actual = protein_activation.run_case(
            case["inputs"], case["parameters"]
        )
    elif process == "ChromosomeSegregation":
        actual = chromosome_segregation.run_case(
            case["inputs"], case["parameters"]
        )
    elif process == "linear_programming":
        return _run_lp(case)
    else:
        raise ValueError(f"unsupported fixture process {process!r}")
    mismatches = _compare_value(actual, case["expected"], "expected")
    return {"passed": not mismatches, "mismatches": mismatches}


def compare_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    if fixture.get("schema_version") != 1:
        raise ValueError("fixture schema_version must be 1")
    cases = fixture.get("cases")
    if not isinstance(cases, list):
        raise ValueError("fixture cases must be a list")
    results = []
    for index, case in enumerate(cases):
        for required in ("process", "inputs", "parameters", "expected"):
            if required not in case:
                raise ValueError(f"case {index} missing {required!r}")
        try:
            result = run_case(case)
        except Exception as exc:
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        results.append(
            {
                "case_index": index,
                "name": case.get("name"),
                "process": case["process"],
                **result,
            }
        )
    passed = sum(bool(r["passed"]) for r in results)
    return {
        "schema_version": 1,
        "source_commit": fixture.get("source_commit"),
        "matlab_version": fixture.get("matlab_version"),
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Python ports with MATLAB fixture cases"
    )
    parser.add_argument("fixture", type=Path, help="fixture JSON path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write summary JSON here (default: stdout)",
    )
    parser.add_argument(
        "--lp-backend",
        choices=["highs", "glpk", "glpk444"],
        help="override LP backend for all LP cases",
    )
    args = parser.parse_args(argv)
    try:
        fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        if args.lp_backend:
            for case in fixture.get("cases", []):
                if case.get("process") == "linear_programming":
                    problem = case["inputs"].get("problem", case["inputs"])
                    problem["backend"] = args.lp_backend
        summary = compare_fixture(fixture)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
