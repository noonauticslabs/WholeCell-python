"""Source-faithful sparse enzyme-expression adjustment helpers.

These free functions let Metabolism wire the MATLAB helper behavior without
coupling this algorithm to the larger FBA port.
"""

from __future__ import annotations

import numpy as np


def _growth_matches(self, growth0, substrates, enzymes):
    fba_obj, fba_smat, reaction_bounds = self.calcEffectiveFBANetwork()
    flux_bounds = self.calcFluxBounds(
        substrates, enzymes, reaction_bounds, self.fbaEnzymeBounds
    )
    result = self.calcGrowthRate(flux_bounds, fba_obj, fba_smat)
    growth = result[0] if isinstance(result, tuple) else result
    relative_change = np.abs(
        np.asarray(growth) - np.asarray(growth0)
    ) / np.asarray(growth0)
    return bool(np.all(relative_change <= self.tolerance))


def adjustEnzymeExpressionSparsely(
    self, growth0, substrates, enzymes, initEnzymes
):
    """Restore as many original enzyme levels as the MATLAB search permits."""
    enzymes = np.asarray(enzymes).copy()
    initial = np.asarray(initEnzymes)
    helper = getattr(self, "adjustEnzymeExpressionSparselyHelper", None)
    if callable(helper):
        tfs = helper(
            growth0,
            substrates,
            enzymes,
            initial,
            np.ones(enzymes.shape, dtype=bool),
        )
    else:
        tfs = adjustEnzymeExpressionSparselyHelper(
            self,
            growth0,
            substrates,
            enzymes,
            initial,
            np.ones(enzymes.shape, dtype=bool),
        )
    enzymes[tfs] = initial[tfs]
    indices = np.flatnonzero(~tfs)

    for index in indices:
        candidate = enzymes.copy()
        candidate[index] = initial[index]
        if _growth_matches(self, growth0, substrates, candidate):
            enzymes = candidate
            break
    else:
        return enzymes

    try:
        n = len(indices)
        n_combinations = 1 << n
        # MATLAB's dec2bin(0:2^n-1) materializes this complete matrix inside
        # a try/catch. Preserve its fallback when the exact matrix cannot fit.
        patterns = np.asarray(
            [
                [bool((value >> bit) & 1) for bit in range(n - 1, -1, -1)]
                for value in range(n_combinations)
            ],
            dtype=bool,
        )
        order = np.argsort(-patterns.sum(axis=1), kind="stable")
        patterns = patterns[order]
    except (MemoryError, OverflowError, ValueError):
        return enzymes

    for pattern in patterns:
        candidate = enzymes.copy()
        selected = indices[pattern]
        candidate[selected] = initial[selected]
        if _growth_matches(self, growth0, substrates, candidate):
            return candidate
    return enzymes


def adjustEnzymeExpressionSparselyHelper(
    self, growth0, substrates, enzymes, initEnzymes, tfs
):
    """Recursively find subsets whose reversion preserves fitted growth."""
    tfs = np.asarray(tfs, dtype=bool).copy()
    candidate = np.asarray(enzymes).copy()
    initial = np.asarray(initEnzymes)
    candidate[tfs] = initial[tfs]
    if not np.any(tfs) or _growth_matches(
        self, growth0, substrates, candidate
    ):
        return tfs
    indices = np.flatnonzero(tfs)
    if len(indices) == 1:
        tfs[:] = False
        return tfs

    midpoint = int(np.ceil(len(indices) / 2))
    left = tfs.copy()
    right = tfs.copy()
    left[indices[:midpoint]] = False
    # MATLAB indexes from idxs(ceil(end/2))+1 through end, where idxs contains
    # one-based enzyme coordinates and the final end refers to the mask.
    right[indices[midpoint - 1] + 1 :] = False
    helper = getattr(self, "adjustEnzymeExpressionSparselyHelper", None)
    if callable(helper):
        left_result = helper(growth0, substrates, enzymes, initEnzymes, left)
        right_result = helper(growth0, substrates, enzymes, initEnzymes, right)
    else:
        left_result = adjustEnzymeExpressionSparselyHelper(
            self, growth0, substrates, enzymes, initEnzymes, left
        )
        right_result = adjustEnzymeExpressionSparselyHelper(
            self, growth0, substrates, enzymes, initEnzymes, right
        )
    return left_result | right_result


adjust_enzyme_expression_sparsely = adjustEnzymeExpressionSparsely
adjust_enzyme_expression_sparsely_helper = adjustEnzymeExpressionSparselyHelper
