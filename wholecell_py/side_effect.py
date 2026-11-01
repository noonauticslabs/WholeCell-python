"""Ordered side-effect containers from SimulationStateSideEffect.m."""

import numpy as np

from .side_effect_item import SimulationStateSideEffectItem


class SimulationStateSideEffect:
    def __init__(self, items=None):
        self.items = np.empty((0, 1), dtype=object) if items is None else items
        if items is not None:
            values = np.asarray(items, dtype=object).ravel(order="F")
            if not all(
                isinstance(item, SimulationStateSideEffectItem)
                for item in values
            ):
                raise TypeError(
                    "items must contain SimulationStateSideEffectItem objects"
                )

    def updateSimulationState(self, simulation):
        effects = np.asarray(self, dtype=object).ravel(order="F")
        for effect in effects:
            items = np.asarray(effect.items, dtype=object).ravel(order="F")
            SimulationStateSideEffectItem.updateSimulationState(
                items, simulation
            )
        return simulation


def apply_side_effects(effects, simulation):
    if isinstance(effects, (list, tuple, np.ndarray)):
        for effect in np.asarray(effects, dtype=object).ravel(order="F"):
            effect.updateSimulationState(simulation)
    else:
        effects.updateSimulationState(simulation)
