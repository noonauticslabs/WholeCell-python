"""Metabolic reaction state translated from WholeCell's ``MetabolicReaction``."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class MetabolicReaction:
    optionNames = ("verbosity", "seed")
    fixedConstantNames: tuple[str, ...] = ()
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = ("growth", "fluxs")
    dependentStateNames = ("doublingTime",)
    dryWeight = 0

    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "reactionWholeCellModelIDs",
        "growth0",
        "initialGrowthFilterWidth",
        "meanInitialGrowthRate",
        "growth",
        "fluxs",
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        self.wholeCellModelID, self.name = wholeCellModelID, name
        self.reactionWholeCellModelIDs = None
        self.growth0 = self.initialGrowthFilterWidth = (
            self.meanInitialGrowthRate
        ) = None
        self.growth: np.ndarray | None = None
        self.fluxs: np.ndarray | None = None
        restored = dict(snapshot or {})
        restored.update(attributes)
        for key, value in restored.items():
            setattr(
                self,
                key,
                (
                    np.array(value, copy=True)
                    if isinstance(value, np.ndarray)
                    else value
                ),
            )

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        """Read the reaction IDs from the metabolism knowledge-base process."""
        processes = _field(knowledgeBase, "processes")
        metabolism = None
        for process in processes:
            if (
                _field(process, "wholeCellModelID", None)
                == "Process_Metabolism"
            ):
                metabolism = process
                break
        if metabolism is None:
            raise ValueError(
                "knowledge base has no Process_Metabolism process"
            )
        self.reactionWholeCellModelIDs = [
            _field(reaction, "wholeCellModelID")
            for reaction in _field(metabolism, "reactions")
        ]

    def allocateMemory(self, numTimePoints: int) -> None:
        if not isinstance(numTimePoints, (int, np.integer)):
            raise TypeError("numTimePoints must be an integer")
        if numTimePoints < 0:
            raise ValueError("numTimePoints must be non-negative")
        if self.reactionWholeCellModelIDs is None:
            raise ValueError(
                "reactionWholeCellModelIDs must be initialized before allocating fluxs"
            )
        self.growth = np.zeros((1, 1, int(numTimePoints)))
        self.fluxs = np.zeros(
            (len(self.reactionWholeCellModelIDs), 1, int(numTimePoints))
        )

    def initialize(self) -> None:
        # MATLAB source intentionally leaves already allocated state at zero.
        return None

    @property
    def doublingTime(self) -> Any:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1 / np.asarray(self.growth)


def _field(value: Any, name: str, default: Any = ...):
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not ...:
        return default
    raise AttributeError(f"{type(value).__name__!s} has no field {name!r}")
