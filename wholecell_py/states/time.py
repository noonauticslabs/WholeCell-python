"""Time state translated from ``cell.sim.state.Time``.

This state stores the simulation time points and the four time-related model
constants. Arrays retain MATLAB dimensions, including the 1-by-1-by-N layout
used by :meth:`allocateMemory`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class Time:
    optionNames = ("verbosity", "seed")
    fixedConstantNames = (
        "cellCycleLength",
        "replicationDuration",
        "cytokinesisDuration",
    )
    fittedConstantNames = ("replicationInitiationDuration",)
    stateNames = ("values",)
    dependentStateNames: tuple[str, ...] = ()
    dryWeight = 0

    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "cellCycleLength",
        "replicationInitiationDuration",
        "replicationDuration",
        "cytokinesisDuration",
        "values",
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        """Create a state, optionally restoring fields from a snapshot mapping.

        Snapshot fields can be supplied as a mapping, keyword arguments, or
        both (keyword arguments take precedence). Unknown fields are preserved
        as attributes to support surrounding snapshot metadata.
        """
        self.wholeCellModelID = wholeCellModelID
        self.name = name
        self.cellCycleLength = None
        self.replicationInitiationDuration = None
        self.replicationDuration = None
        self.cytokinesisDuration = None
        self.values: np.ndarray | None = None
        restored = dict(snapshot or {})
        restored.update(attributes)
        for key, value in restored.items():
            setattr(self, key, value)

    def allocateMemory(self, numTimePoints: int) -> None:
        """Allocate zeros with MATLAB's ``zeros(1, 1, N)`` shape."""
        if not isinstance(numTimePoints, (int, np.integer)):
            raise TypeError("numTimePoints must be an integer")
        if numTimePoints < 0:
            raise ValueError("numTimePoints must be non-negative")
        self.values = np.zeros((1, 1, int(numTimePoints)), dtype=float)

    def initialize(self) -> None:
        """Initialize state storage with one time point."""
        self.allocateMemory(1)
