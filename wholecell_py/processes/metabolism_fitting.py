"""Enzyme fitting from Metabolism.m:771-1180; indices at the API are one-based."""

import warnings
import numpy as np

from .metabolism import _dense, _idx, _matlab_options
from ..solvers import solve_lp


def effective_network(self, n=4):
    objective = np.zeros_like(np.asarray(self.fbaObjective, dtype=float))
    objective.flat[_idx(self.fbaReactionIndexs_biomassProduction)] = 1
    matrix = _dense(self.fbaReactionStoichiometryMatrix).copy()
    si = _idx(self.substrateIndexs_fba)
    production = (
        -np.asarray(self.metabolismNewProduction).ravel(order="F")[si]
        - n
        * np.asarray(self.metabolismRecyclingProduction).ravel(order="F")[si]
    )
    matrix[
        np.ix_(
            _idx(self.fbaSubstrateIndexs_substrates),
            _idx(self.fbaReactionIndexs_biomassProduction),
        )
    ] = production[:, None]
    bounds = np.asarray(self.fbaReactionBounds).copy()
    bounds[_idx(self.fbaReactionIndexs_metaboliteInternalExchange), :] = 0
    return objective, matrix, bounds


def limited_reactions(self, substrates, enzymes, objective, matrix, bounds):
    limited = self.calcFluxBounds(
        substrates, enzymes, bounds, self.fbaEnzymeBounds
    )
    unlimited = self.calcFluxBounds(
        substrates, enzymes, bounds, self.fbaEnzymeBounds, False
    )
    flux = self.calcGrowthRate(limited, objective, matrix, return_full=True)[2]
    ix = _idx(self.fbaReactionIndexs_metabolicConversion)
    forward = ix[(flux[ix] > 0) & (limited[ix, 1] < unlimited[ix, 1])]
    reverse = ix[(flux[ix] < 0) & (limited[ix, 0] > unlimited[ix, 0])]
    return (forward + 1).reshape(-1, 1), (reverse + 1).reshape(-1, 1)


def _minimum(self, enzymes):
    result = np.zeros(enzymes.size)
    result[_idx(self.enzymeMonomerLocalIndexs)] = np.asarray(
        self.monomer.minimumAverageExpression
    ).item()
    result[_idx(self.enzymeComplexLocalIndexs)] = np.asarray(
        self.complex.minimumAverageExpression
    ).item()
    return result


def _optimize(self, objective, matrix, lo, hi, maximize):
    opts = _matlab_options(getattr(self, "linearProgrammingOptions", {}))
    solver_opts = _matlab_options(opts.get("solverOptions", {}))
    return solve_lp(
        dict(
            f=np.asarray(objective).reshape(-1),
            A=matrix,
            b=np.asarray(self.fbaRightHandSide).reshape(-1),
            lb=lo,
            ub=hi,
            constraint_types="S",
            variable_types="C",
            maximize=maximize,
            backend=opts.get("backend", "glpk444"),
            glpk_options=_matlab_options(solver_opts.get("glpk", {})),
        )
    )


def _growth(self, substrates, enzymes, objective, matrix, bounds):
    flux_bounds = self.calcFluxBounds(
        substrates, enzymes, bounds, self.fbaEnzymeBounds
    )
    return self.calcGrowthRate(flux_bounds, objective, matrix)[0]


def fit_enzymes(self, growth0, substrates, enzymes):
    original_shape = np.asarray(enzymes).shape
    enzymes = np.asarray(enzymes, dtype=float).ravel(order="F").copy()
    growth0 = np.asarray(growth0).item()
    weights = np.asarray(self.enzymeMolecularWeights).ravel(order="F")
    minimum = _minimum(self, enzymes)
    cat = _dense(self.fbaReactionCatalysisMatrix)
    low = (enzymes < minimum) & np.any(cat != 0, axis=0)
    high = enzymes > minimum
    need = (minimum[low] - enzymes[low]) @ weights[low]
    available = (enzymes[high] - minimum[high]) @ weights[high]
    if need < available:
        enzymes[high] = (
            minimum[high]
            + (enzymes[high] - minimum[high]) * (available - need) / available
        )
        enzymes[low] = minimum[low]
    else:
        warnings.warn(
            "Cannot ensure minimum enzyme expression", RuntimeWarning
        )
    initial = enzymes.copy()
    objective, matrix, bounds = self.calcEffectiveFBANetwork()
    flux_bounds = self.calcFluxBounds(
        substrates, enzymes, bounds, self.fbaEnzymeBounds
    )
    growth, _, old_flux, *_ = self.calcGrowthRate(
        flux_bounds, objective, matrix, return_full=True
    )
    tolerance = np.asarray(self.tolerance).item()
    if (
        np.abs(growth - growth0) / growth0 < tolerance
        or abs(growth - growth0) < 1e-12
    ):
        return enzymes.reshape(original_shape, order="F")
    no_kinetics = self.calcFluxBounds(
        substrates, enzymes, bounds, self.fbaEnzymeBounds, False
    )
    if (
        growth < growth0
        and growth0 > self.calcGrowthRate(no_kinetics, objective, matrix)[0]
    ):
        raise RuntimeError("Cannot fit enzyme expression to match growth rate")
    forward, reverse = self.calcEnzymeKineticallyLimitedReactions(
        substrates, enzymes, objective, matrix, bounds
    )
    forward, reverse = _idx(forward), _idx(reverse)
    active = np.r_[forward, reverse]
    unconstrained = np.flatnonzero(~np.any(cat[active, :] != 0, axis=0))
    cost = np.zeros(np.asarray(objective).size)
    cost[active] = 1 / old_flux[active]
    lo, hi = flux_bounds[:, 0].copy(), flux_bounds[:, 1].copy()
    if growth < growth0:
        lo[forward] = old_flux[forward]
        hi[forward] = np.minimum(
            no_kinetics[forward, 1],
            old_flux[forward] * (1 + tolerance) * growth0 / growth,
        )
        lo[reverse] = np.maximum(
            no_kinetics[reverse, 0],
            old_flux[reverse] * (1 + tolerance) * growth0 / growth,
        )
        hi[reverse] = old_flux[reverse]
    else:
        lo[forward], hi[forward] = 0, old_flux[forward]
        lo[reverse], hi[reverse] = old_flux[reverse], 0
    biomass = _idx(self.fbaReactionIndexs_biomassProduction)
    lo[biomass] = hi[biomass] = growth0
    lo = np.maximum(lo, -np.asarray(self.realmax).item())
    hi = np.minimum(hi, np.asarray(self.realmax).item())
    result = _optimize(self, cost, matrix, lo, hi, growth >= growth0)
    if result["errorFlag"]:
        raise RuntimeError("Unable to optimize fluxes: " + result["errorMsg"])
    flux = np.maximum(np.minimum(result["x"], hi), lo)
    demand = np.zeros_like(flux)
    kinetic = np.asarray(self.fbaEnzymeBounds)
    demand[forward] = flux[forward] / kinetic[forward, 1]
    demand[reverse] = flux[reverse] / kinetic[reverse, 0]
    enzymes = np.max(cat[active, :] * demand[active, None], axis=0)
    enzymes[unconstrained] = initial[unconstrained]
    enzymes = np.asarray(
        self.adjustEnzymeExpressionSparsely(
            growth0, substrates, enzymes, initial
        )
    ).reshape(-1)
    enzymes = np.maximum(minimum, enzymes)
    if growth < growth0:
        inc = np.flatnonzero(enzymes > initial)
        dec = np.setdiff1d(
            np.union1d(np.flatnonzero(enzymes < initial), unconstrained), inc
        )
        enzymes = np.maximum(initial, enzymes)
        donor = (initial[dec] - minimum[dec]) @ weights[dec]
        enzymes[dec] = (
            minimum[dec]
            + (initial[dec] - minimum[dec])
            * (donor - (enzymes[inc] - initial[inc]) @ weights[inc])
            / donor
        )
    else:
        inc = np.union1d(np.flatnonzero(enzymes > initial), unconstrained)
        dec = np.setdiff1d(np.flatnonzero(enzymes < initial), inc)
        enzymes = np.minimum(initial, enzymes)
        donor = (initial[inc] - minimum[inc]) @ weights[inc]
        enzymes[inc] = (
            minimum[inc]
            + (initial[inc] - minimum[inc])
            * (donor - (enzymes[dec] - initial[dec]) @ weights[dec])
            / donor
        )
    if (
        np.abs(
            _growth(self, substrates, enzymes, objective, matrix, bounds)
            - growth0
        )
        / growth0
        > tolerance
    ):
        raise RuntimeError("Growth rate should not change")
    if (
        np.abs((enzymes @ weights - initial @ weights) / (initial @ weights))
        > tolerance
    ):
        raise RuntimeError("Protein mass should not change")
    return enzymes.reshape(original_shape, order="F")


def min_max_enzymes(self, substrates, enzymes):
    shape = np.asarray(enzymes).shape
    enzymes = np.asarray(enzymes, dtype=float).ravel(order="F")
    minimum = _minimum(self, enzymes)
    objective, matrix, bounds = self.calcEffectiveFBANetwork()
    flux_bounds = self.calcFluxBounds(
        substrates, enzymes, bounds, self.fbaEnzymeBounds
    )
    growth, _, old_flux, *_ = self.calcGrowthRate(
        flux_bounds, objective, matrix, return_full=True
    )
    forward, reverse = self.calcEnzymeKineticallyLimitedReactions(
        substrates, enzymes, objective, matrix, bounds
    )
    forward, reverse = _idx(forward), _idx(reverse)
    active = np.r_[forward, reverse]
    if not active.size:
        warnings.warn("Unable to fit growth rate", RuntimeWarning)
        return enzymes.reshape(shape).copy(), enzymes.reshape(shape).copy()
    cat = _dense(self.fbaReactionCatalysisMatrix)
    other = np.setdiff1d(np.arange(cat.shape[0]), active)
    partly = np.any(cat[other, :] != 0, axis=0)
    unconstrained = ~np.any(cat[active, :] != 0, axis=0)
    kinetic = np.asarray(self.fbaEnzymeBounds)
    weights = np.asarray(self.enzymeMolecularWeights).reshape(-1)
    cost = np.zeros(np.asarray(objective).size)
    cost[forward] = (cat[forward, :] @ weights) / kinetic[forward, 1]
    cost[reverse] = (cat[reverse, :] @ weights) / kinetic[reverse, 0]
    results = []
    for maximize in (False, True):
        lo, hi = flux_bounds[:, 0].copy(), flux_bounds[:, 1].copy()
        biomass = _idx(self.fbaReactionIndexs_biomassProduction)
        lo[biomass] = hi[biomass] = growth
        if maximize:
            unlimited = self.calcFluxBounds(
                substrates, enzymes, bounds, self.fbaEnzymeBounds, False
            )
            lo[forward], hi[forward] = old_flux[forward], unlimited[forward, 1]
            lo[reverse], hi[reverse] = unlimited[reverse, 0], old_flux[reverse]
        else:
            lo[forward], hi[forward] = 0, old_flux[forward]
            lo[reverse], hi[reverse] = old_flux[reverse], 0
        lo = np.maximum(lo, -np.asarray(self.realmax).item())
        hi = np.minimum(hi, np.asarray(self.realmax).item())
        result = _optimize(self, cost, matrix, lo, hi, maximize)
        if result["errorFlag"]:
            warnings.warn(result["errorMsg"], RuntimeWarning)
            fitted = enzymes.copy()
        else:
            flux = np.maximum(np.minimum(result["x"], hi), lo)
            demand = np.zeros_like(flux)
            demand[forward] = flux[forward] / kinetic[forward, 1]
            demand[reverse] = flux[reverse] / kinetic[reverse, 0]
            if maximize:
                # Literal source 0*min(...), including its surprising zero factor.
                selected = cat[active, :].astype(float, copy=True)
                selected[selected == 0] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    fitted = 0 * np.nanmin(
                        selected * demand[active, None], axis=0
                    )
                fitted[unconstrained] = np.finfo(float).max
                fitted[partly] = np.fmax(fitted[partly], 1)
                fitted = np.fmax(enzymes, fitted)
            else:
                fitted = np.max(cat[active, :] * demand[active, None], axis=0)
                fitted[unconstrained] = 1
                fitted[partly] = np.maximum(fitted[partly], 1)
                fitted = np.maximum(minimum, np.minimum(enzymes, fitted))
        invalid_order = (
            np.any(fitted < enzymes) if maximize else np.any(fitted > enzymes)
        )
        if (
            np.abs(
                _growth(self, substrates, fitted, objective, matrix, bounds)
                - growth
            )
            / growth
            > np.asarray(self.tolerance).item()
            or invalid_order
        ):
            warnings.warn(
                "Unable to calculate "
                + ("maximum" if maximize else "minimum")
                + " enzymes",
                RuntimeWarning,
            )
            fitted = enzymes.copy()
        results.append(fitted.reshape(shape, order="F"))
    return tuple(results)
