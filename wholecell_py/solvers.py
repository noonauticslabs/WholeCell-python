"""Linear programming compatibility layer for the MATLAB ComputationUtil API.

The default backend is the original GLPK 4.44. ``backend="glpk444"`` uses the
isolated GLPK 4.44 driver matching the historical MEX; ``backend="glpk"``
uses installed swiglpk; ``backend="highs"`` uses SciPy HiGHS. Results label
the selected backend because solver versions can produce different optima
or different primal solutions for degenerate problems.
"""

from __future__ import annotations

from typing import Any
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint

try:  # Optional so installations without GLPK retain the HiGHS path.
    import swiglpk as _glpk
except ImportError:  # pragma: no cover - depends on optional package
    _glpk = None


def _vector(value: Any, name: str, size: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    if raw.ndim > 2 or (raw.ndim == 2 and raw.shape[1] != 1):
        raise ValueError(f"{name} must be a vector or MATLAB column vector")
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must contain {size} values; got {arr.size}")
    if not np.all(np.isfinite(arr) | np.isinf(arr)):
        raise ValueError(f"{name} contains NaN")
    return arr


def _codes(value: Any, count: int, name: str, allowed: str) -> np.ndarray:
    if isinstance(value, str):
        chars = [value] * count if len(value) == 1 else list(value)
    else:
        chars = np.asarray(value).reshape(-1).tolist()
        chars = [str(x.decode() if isinstance(x, bytes) else x) for x in chars]
        if len(chars) == 1:
            chars *= count
    chars = [x.upper() for x in chars]
    if len(chars) != count or any(
        len(x) != 1 or x not in allowed for x in chars
    ):
        raise ValueError(
            f"{name} must be a scalar or {count} codes from {allowed}"
        )
    return np.asarray(chars, dtype="U1")


def _failure(n: int, m: int, message: str, status: str, solver: str) -> dict:
    nan = np.full(n, np.nan)
    return {
        "x": nan,
        "lambda": {
            "reducedCosts": nan.copy(),
            "shadowPrices": np.full(m, np.nan),
        },
        "fopt": float("nan"),
        "errorFlag": True,
        "errorMsg": message,
        "extra": {},
        "status": status,
        "solver": solver,
    }


def _solve_glpk(A, b, f, lb, ub, ct, vt, is_max, options):
    """Solve via the GLPK C API exposed by swiglpk."""
    if _glpk is None:
        raise RuntimeError(
            "backend='glpk' requires the optional swiglpk package"
        )
    glp = _glpk
    glp.glp_term_out(
        glp.GLP_OFF
        if options.get("glpk_options", {}).get("msglev", 0) == 0
        else glp.GLP_ON
    )
    m, n = A.shape
    prob = glp.glp_create_prob()
    glp.glp_set_obj_dir(prob, glp.GLP_MAX if is_max else glp.GLP_MIN)
    glp.glp_add_rows(prob, m)
    glp.glp_add_cols(prob, n)
    for i, code in enumerate(ct, start=1):
        rhs = float(b[i - 1])
        if code == "F":
            typ, lo, hi = glp.GLP_FR, 0.0, 0.0
        elif code == "U":
            typ, lo, hi = glp.GLP_UP, 0.0, rhs
        elif code == "S":
            typ, lo, hi = glp.GLP_FX, rhs, rhs
        elif code == "L":
            typ, lo, hi = glp.GLP_LO, rhs, 0.0
        else:
            typ, lo, hi = glp.GLP_DB, -rhs, rhs
        glp.glp_set_row_bnds(prob, i, typ, lo, hi)
    for j in range(n):
        lo, hi = float(lb[j]), float(ub[j])
        if vt[j] == "B":
            lo, hi = max(lo, 0.0), min(hi, 1.0)
        if np.isneginf(lo) and np.isposinf(hi):
            typ, lo, hi = glp.GLP_FR, 0.0, 0.0
        elif np.isneginf(lo):
            typ, lo = glp.GLP_UP, 0.0
        elif np.isposinf(hi):
            typ, hi = glp.GLP_LO, 0.0
        elif lo == hi:
            typ = glp.GLP_FX
        else:
            typ = glp.GLP_DB
        glp.glp_set_col_bnds(prob, j + 1, typ, lo, hi)
        glp.glp_set_obj_coef(prob, j + 1, float(f[j]))
        if vt[j] == "I":
            glp.glp_set_col_kind(prob, j + 1, glp.GLP_IV)
        elif vt[j] == "B":
            glp.glp_set_col_kind(prob, j + 1, glp.GLP_BV)

    ia, ja, ar = [0], [0], [0.0]  # GLPK sparse indices are one-based.
    for i in range(m):
        for j in range(n):
            if A[i, j] != 0:
                ia.append(i + 1)
                ja.append(j + 1)
                ar.append(float(A[i, j]))
    ne = len(ar) - 1
    ia_arr, ja_arr = glp.intArray(ne + 1), glp.intArray(ne + 1)
    ar_arr = glp.doubleArray(ne + 1)
    for k in range(ne + 1):
        ia_arr[k], ja_arr[k], ar_arr[k] = ia[k], ja[k], ar[k]
    glp.glp_load_matrix(prob, ne, ia_arr, ja_arr, ar_arr)
    opts = options.get("glpk_options", {})
    parm = glp.glp_smcp()
    glp.glp_init_smcp(parm)
    parm.msg_lev = int(opts.get("msglev", 0))
    parm.tol_bnd = float(opts.get("tolbnd", 1e-6))
    # glpkcc's separate `dual` option selects primal, dual, or dual with
    # fallback; lpsolver=1 only selects the simplex algorithm.
    parm.meth = {0: glp.GLP_PRIMAL, 1: glp.GLP_DUAL, 2: glp.GLP_DUALP}.get(
        int(opts.get("dual", 0)), glp.GLP_PRIMAL
    )
    parm.presolve = glp.GLP_ON if opts.get("presol", 1) else glp.GLP_OFF
    # glpkcc skips explicit scaling when presolve is enabled with simplex.
    scale_methods = {
        0: glp.GLP_SF_SKIP,
        1: glp.GLP_SF_GM,
        2: glp.GLP_SF_EQ,
        3: glp.GLP_SF_AUTO,
        4: glp.GLP_SF_2N,
    }
    scale_mode = int(opts.get("scale", 1))
    if not (opts.get("presol", 1) and opts.get("lpsolver", 1) == 1):
        glp.glp_scale_prob(prob, scale_methods.get(scale_mode, glp.GLP_SF_GM))
    rc = glp.glp_simplex(prob, parm)
    raw_status = glp.glp_get_status(prob)
    integral = bool(np.any(np.isin(vt, ["I", "B"])))
    if rc == 0 and integral and raw_status in (glp.GLP_OPT, glp.GLP_FEAS):
        iparm = glp.glp_iocp()
        glp.glp_init_iocp(iparm)
        iparm.msg_lev = int(opts.get("msglev", 0))
        iparm.presolve = glp.GLP_ON if opts.get("presol", 1) else glp.GLP_OFF
        rc = glp.glp_intopt(prob, iparm)
        raw_status = glp.glp_mip_status(prob)
        x = (
            np.array([glp.glp_mip_col_val(prob, j + 1) for j in range(n)])
            if raw_status in (glp.GLP_OPT, glp.GLP_FEAS)
            else np.full(n, np.nan)
        )
        fopt = (
            float(glp.glp_mip_obj_val(prob))
            if np.all(np.isfinite(x))
            else float("nan")
        )
        dual = {
            "reducedCosts": np.full(n, np.nan),
            "shadowPrices": np.full(m, np.nan),
        }
    else:
        x = (
            np.array([glp.glp_get_col_prim(prob, j + 1) for j in range(n)])
            if raw_status in (glp.GLP_OPT, glp.GLP_FEAS)
            else np.full(n, np.nan)
        )
        fopt = float(np.dot(f, x)) if np.all(np.isfinite(x)) else float("nan")
        if not integral and raw_status in (glp.GLP_OPT, glp.GLP_FEAS):
            dual = {
                "reducedCosts": np.array(
                    [glp.glp_get_col_dual(prob, j + 1) for j in range(n)]
                ),
                "shadowPrices": np.array(
                    [glp.glp_get_row_dual(prob, i + 1) for i in range(m)]
                ),
            }
        else:
            dual = {
                "reducedCosts": np.full(n, np.nan),
                "shadowPrices": np.full(m, np.nan),
            }
    status = {
        glp.GLP_OPT: "optimal",
        glp.GLP_FEAS: "feasible",
        glp.GLP_INFEAS: "infeasible",
        glp.GLP_NOFEAS: "infeasible",
        glp.GLP_UNBND: "unbounded",
        glp.GLP_UNDEF: "undefined",
    }.get(raw_status, "solver_error")
    err = status not in ("optimal", "feasible") or rc != 0
    msg = {
        "optimal": "Solution is optimal",
        "feasible": "Solution is feasible",
        "infeasible": "No feasible solution exists",
        "unbounded": "Solution is unbounded",
        "undefined": "Solution is undefined",
    }.get(status, f"GLPK returned status {raw_status}")
    glp.glp_delete_prob(prob)
    return {
        "x": x,
        "lambda": dual,
        "fopt": fopt,
        "errorFlag": err,
        "errorMsg": msg,
        "extra": {"glpk_status": int(raw_status), "return_code": int(rc)},
        "status": status,
        "solver": "swiglpk/GLPK",
    }


def _solve_glpk444(A, b, f, lb, ub, ct, vt, is_max):
    """Call the prebuilt, isolated GLPK 4.44 diagnostic driver."""
    driver = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "diagnostic"
        / "glpk444_text"
    )
    if not driver.is_file():
        raise RuntimeError(
            f"GLPK 4.44 driver not found at {driver}; build it from vendor/diagnostic/glpk444_text.c"
        )
    m, n = A.shape
    fmt = lambda x: format(float(x), ".17g")
    rows = [f"{m} {n} {int(is_max)}"]
    rows.extend(" ".join(fmt(x) for x in vec) for vec in (f, lb, ub, b))
    rows.append(" ".join(ct.tolist()))
    rows.append(" ".join(vt.tolist()))
    # np.nonzero enumerates in row-major order, even for Fortran-layout input.
    # Preserve the original MEX's insertion order without visiting every zero
    # in Python. The solver, numeric formatting and protocol are unchanged.
    row_indices, col_indices = np.nonzero(A)
    rows.append(str(len(row_indices)))
    rows.extend(
        f"{i + 1} {j + 1} {fmt(A[i, j])}"
        for i, j in zip(row_indices, col_indices)
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lp44", encoding="ascii"
    ) as fixture:
        fixture.write("\n".join(rows) + "\n")
        fixture.flush()
        completed = subprocess.run(
            [str(driver), fixture.name],
            check=True,
            capture_output=True,
            text=True,
        )
    raw = json.loads(completed.stdout)
    raw_status = int(raw["status"])
    status = {
        5: "optimal",
        2: "feasible",
        3: "infeasible",
        4: "infeasible",
        6: "unbounded",
        1: "undefined",
    }.get(raw_status, "solver_error")
    solved = status in ("optimal", "feasible")

    def arr(key):
        return np.asarray(
            [np.nan if x is None else x for x in raw[key]], dtype=float
        )

    x = arr("x")
    dual = {
        "reducedCosts": arr("reducedCosts"),
        "shadowPrices": arr("shadowPrices"),
    }
    message = {
        "optimal": "Solution is optimal",
        "feasible": "Solution is feasible",
        "infeasible": "No feasible solution exists",
        "unbounded": "Solution is unbounded",
        "undefined": "Solution is undefined",
    }.get(status, f"GLPK 4.44 returned status {raw_status}")
    rc = int(raw["return_code"])
    return {
        "x": x,
        "lambda": dual,
        "fopt": (
            float(raw["objective"]) if raw["objective"] is not None else np.nan
        ),
        "errorFlag": not solved or rc != 0,
        "errorMsg": message,
        "extra": {
            "glpk_status": raw_status,
            "return_code": rc,
            "version": "4.44",
        },
        "status": status,
        "solver": "GLPK 4.44 (standalone)",
    }


def solve_lp(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve a MATLAB-style LP/MILP problem dictionary.

    Required keys are ``f``, ``A``, ``b``, ``lb``, ``ub``,
    ``constraint_types`` and ``variable_types``. ``maximize`` accepts a bool
    or the MATLAB strings ``'maximize'``/``'minimize'``. Bounds may contain
    infinities. Rows support F/U/S/L/D; columns support C/I/B.

    Returns MATLAB-compatible ``x``, ``lambda`` (reducedCosts and
    shadowPrices), ``fopt``, ``errorFlag``, ``errorMsg`` and ``extra``, plus
    explicit status and solver labels. Continuous duals are returned in the
    original row orientation. HiGHS does not provide meaningful duals for
    integer problems, so those are NaN.
    """
    A = np.asarray(problem["A"], dtype=float)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("A must be a nonempty two-dimensional matrix")
    m, n = A.shape
    f = _vector(problem["f"], "f", n)
    b = _vector(problem["b"], "b", m)
    lb = (
        _vector(problem.get("lb", []), "lb", n)
        if "lb" in problem
        else np.full(n, -np.inf)
    )
    ub = (
        _vector(problem.get("ub", []), "ub", n)
        if "ub" in problem
        else np.full(n, np.inf)
    )
    ct = _codes(
        problem.get("constraint_types", "U"), m, "constraint_types", "FUSLD"
    )
    vt = _codes(problem.get("variable_types", "C"), n, "variable_types", "CIB")
    backend = str(problem.get("backend", "glpk444")).lower()
    if backend not in ("highs", "scipy-highs", "glpk", "glpk444"):
        raise ValueError("backend must be 'glpk', 'glpk444', or 'highs'")
    if np.any(lb > ub):
        return _failure(
            n,
            m,
            "Invalid bounds: lower bound exceeds upper bound",
            "invalid_bounds",
            {"glpk444": "GLPK 4.44 (standalone)", "glpk": "swiglpk/GLPK"}.get(
                backend, "scipy-highs"
            ),
        )
    maximize = problem.get("maximize", problem.get("maximizeFlag", "minimize"))
    is_max = maximize is True or (
        isinstance(maximize, str) and maximize.lower() == "maximize"
    )
    sense = -1.0 if is_max else 1.0
    c = sense * f  # HiGHS minimizes.
    if backend == "glpk444":
        try:
            return _solve_glpk444(A, b, f, lb, ub, ct, vt, is_max)
        except Exception as exc:
            return _failure(
                n, m, str(exc), "solver_error", "GLPK 4.44 (standalone)"
            )
    if backend == "glpk":
        try:
            return _solve_glpk(A, b, f, lb, ub, ct, vt, is_max, problem)
        except Exception as exc:
            return _failure(n, m, str(exc), "solver_error", "swiglpk/GLPK")
    solver = "scipy-highs (fallback; not GLPK)"

    # Convert MATLAB GLPK row codes to one or two HiGHS rows while retaining
    # the map needed to reconstruct one shadow price per original row.
    aub, bub, aeq, beq = [], [], [], []
    row_map: list[list[tuple[str, float]]] = [[] for _ in range(m)]
    for i, code in enumerate(ct):
        if code == "F":
            continue
        if code == "S":
            aeq.append(A[i])
            beq.append(b[i])
            row_map[i].append(("eq", 1.0))
        elif code == "U":
            aub.append(A[i])
            bub.append(b[i])
            row_map[i].append(("ub", 1.0))
        elif code == "L":
            aub.append(-A[i])
            bub.append(-b[i])
            row_map[i].append(("ub", -1.0))
        else:  # D: -b <= A*x <= b
            aub.append(A[i])
            bub.append(b[i])
            row_map[i].append(("ub", 1.0))
            aub.append(-A[i])
            bub.append(b[i])
            row_map[i].append(("ub", -1.0))
    aub = np.asarray(aub) if aub else None
    bub = np.asarray(bub) if bub else None
    aeq = np.asarray(aeq) if aeq else None
    beq = np.asarray(beq) if beq else None
    bounds = list(zip(lb, ub))
    integer = np.isin(vt, ["I", "B"])
    if np.any(vt == "B"):
        lb = lb.copy()
        ub = ub.copy()
        lb[vt == "B"] = np.maximum(lb[vt == "B"], 0)
        ub[vt == "B"] = np.minimum(ub[vt == "B"], 1)
        bounds = list(zip(lb, ub))

    try:
        if np.any(integer):
            # milp uses ranged constraints directly.
            rows, lows, highs = [], [], []
            for i, code in enumerate(ct):
                if code == "F":
                    continue
                rows.append(A[i])
                lows.append(
                    b[i]
                    if code in ("S", "L")
                    else (-b[i] if code == "D" else -np.inf)
                )
                highs.append(b[i] if code in ("S", "U", "D") else np.inf)
            result = milp(
                c,
                integrality=integer.astype(int),
                bounds=Bounds(lb, ub),
                constraints=LinearConstraint(np.asarray(rows), lows, highs),
            )
            x = result.x if result.x is not None else np.full(n, np.nan)
            ok = result.status == 0
            status = {
                0: "optimal",
                1: "iteration_limit",
                2: "infeasible",
                3: "unbounded",
                4: "solver_error",
            }.get(result.status, "solver_error")
            dual = {
                "reducedCosts": np.full(n, np.nan),
                "shadowPrices": np.full(m, np.nan),
            }
        else:
            result = linprog(
                c,
                A_ub=aub,
                b_ub=bub,
                A_eq=aeq,
                b_eq=beq,
                bounds=bounds,
                method="highs",
            )
            ok = result.status == 0
            status = {
                0: "optimal",
                1: "iteration_limit",
                2: "infeasible",
                3: "unbounded",
                4: "solver_error",
            }.get(result.status, "solver_error")
            x = result.x if result.x is not None else np.full(n, np.nan)
            if ok:
                shadow = np.zeros(m)
                ub_idx = eq_idx = 0
                for i in range(m):
                    for kind, multiplier in row_map[i]:
                        marginal = (
                            result.ineqlin.marginals[ub_idx]
                            if kind == "ub"
                            else result.eqlin.marginals[eq_idx]
                        )
                        if kind == "ub":
                            ub_idx += 1
                        else:
                            eq_idx += 1
                        shadow[i] += sense * multiplier * marginal
                reduced = np.asarray(result.lower.marginals) + np.asarray(
                    result.upper.marginals
                )
                dual = {
                    "reducedCosts": sense * reduced,
                    "shadowPrices": shadow,
                }
            else:
                dual = {
                    "reducedCosts": np.full(n, np.nan),
                    "shadowPrices": np.full(m, np.nan),
                }
        fopt = float(np.dot(f, x)) if np.all(np.isfinite(x)) else float("nan")
        return {
            "x": np.asarray(x),
            "lambda": dual,
            "fopt": fopt,
            "errorFlag": not ok,
            "errorMsg": (
                "Optimization terminated successfully"
                if ok
                else str(result.message)
            ),
            "extra": {},
            "status": status,
            "solver": solver,
        }
    except Exception as exc:
        return _failure(n, m, str(exc), "solver_error", solver)
