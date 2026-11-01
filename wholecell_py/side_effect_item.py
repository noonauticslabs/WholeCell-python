"""Simulation state mutations emitted by a process side effect."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any

import numpy as np


def _get(obj: Any, name: str) -> Any:
    """Read an actual source field, without inventing missing-field defaults."""
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _state(simulation: Any, name: str) -> Any:
    lookup = getattr(simulation, "state", None)
    if callable(lookup):
        return lookup(name)
    states = _get(simulation, "states")
    if name in states:
        return states[name]
    # Simulation's Python representation stores normalized lower-case keys.
    lowered = name.lower()
    if lowered in states:
        return states[lowered]
    raise KeyError(name)


def _text(value):
    return (
        value
        if isinstance(value, str)
        else "".join(np.asarray(value).ravel(order="F"))
    )


@dataclass
class SimulationStateSideEffectItem:
    """One-based MATLAB component/compartment update to a simulation state."""

    stateName: str = ""
    propertyName: str = ""
    indexName: str = ""
    componentIndex: int = 0
    compartmentIndex: int = 0
    deltaValue: float = 0

    def updateSimulationState(self, simulation: Any) -> Any:
        delta_mass = 0.0
        for item in np.asarray(self, dtype=object).ravel(order="F"):
            state = _state(simulation, _text(item.stateName))
            if _text(item.indexName):
                component_index = _get(state, _text(item.indexName))
                component_index = np.asarray(component_index).reshape(
                    -1, order="F"
                )[int(item.componentIndex) - 1]
            else:
                component_index = item.componentIndex
            # Translate the MATLAB 1-based pair to NumPy's 0-based pair.
            row, col = int(component_index) - 1, int(item.compartmentIndex) - 1
            prop = _get(state, _text(item.propertyName))
            if prop.ndim <= 2:
                # MATLAB's ordinary two-subscript indexing is unchanged for
                # matrices (and for arrays with a trailing singleton time
                # dimension).
                prop[row, col] += item.deltaValue
            else:
                # With fewer subscripts than dimensions, MATLAB folds all
                # trailing dimensions into the final subscript. Resolve that
                # collapsed column in Fortran order so one side effect changes
                # one component/compartment/time element, never every time.
                tail = np.unravel_index(col, prop.shape[1:], order="F")
                prop[(row, *tail)] += item.deltaValue
            weights = np.asarray(_get(state, "molecularWeights")).reshape(
                -1, order="F"
            )
            delta_mass += weights[row] * item.deltaValue
        if delta_mass != 0:
            warnings.warn(
                "Simulation state side effects should preserve mass (for example, "
                "each SimulationStateSideEffectItem array should represent a "
                "mass-balanced change to simulation state)",
                RuntimeWarning,
                stacklevel=2,
            )
        return simulation
