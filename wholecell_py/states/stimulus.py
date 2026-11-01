"""Stimulus state translated from ``cell.sim.state.Stimulus``."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class Stimulus:
    optionNames = ("verbosity", "seed")
    fixedConstantNames: tuple[str, ...] = ()
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = ("values",)
    dependentStateNames: tuple[str, ...] = ()
    dryWeight = 0

    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "wholeCellModelIDs",
        "names",
        "setValues",
        "compartment",
        "values",
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        self.wholeCellModelID = wholeCellModelID
        self.name = name
        self.wholeCellModelIDs = None
        self.names = None
        self.setValues: np.ndarray | None = None
        self.compartment = None
        self.values: np.ndarray | None = None
        restored = dict(snapshot or {})
        restored.update(attributes)
        for key, value in restored.items():
            setattr(self, key, value)

    def storeObjectReferences(self, simulation: Any) -> None:
        self.compartment = simulation.compartment

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        """Build stimulus metadata and the six-column assignment table.

        Knowledge-base records may be mappings or attribute objects, matching
        the two common representations used by the Python loader.
        """
        records = _field(knowledgeBase, "stimulis")
        self.wholeCellModelIDs = [
            _field(record, "wholeCellModelID") for record in records
        ]
        self.names = [_field(record, "name") for record in records]
        rows: list[list[float]] = []
        for object_index, record in enumerate(records, start=1):
            compartments = _field(record, "compartments")
            indexes = np.broadcast_to(
                np.asarray(_field(record, "idx")).reshape(-1),
                (len(compartments),),
            )
            vals = _field(record, "values")
            initial = _field(record, "initialTimes")
            final = _field(record, "finalTimes")
            for compartment, idx, value, start, end in zip(
                compartments, indexes, vals, initial, final
            ):
                compartment_index = _field(compartment, "idx")
                # MATLAB sub2ind([numStimulis numCompartments], object, compartment)
                n_stimuli = int(
                    _field(knowledgeBase, "numStimulis", len(records))
                )
                n_compartments = int(_field(knowledgeBase, "numCompartments"))
                linear_index = int(
                    int(idx) + (int(compartment_index) - 1) * n_stimuli
                )
                rows.append(
                    [
                        float(idx),
                        float(compartment_index),
                        float(value),
                        float(start),
                        float(end),
                        float(linear_index),
                    ]
                )
        self.setValues = np.asarray(rows, dtype=float).reshape((-1, 6))
        if simulation is not None:
            self.storeObjectReferences(simulation)

    def allocateMemory(self, numTimePoints: int) -> None:
        if not isinstance(numTimePoints, (int, np.integer)):
            raise TypeError("numTimePoints must be an integer")
        if numTimePoints < 0:
            raise ValueError("numTimePoints must be non-negative")
        if self.wholeCellModelIDs is None:
            raise ValueError(
                "wholeCellModelIDs must be initialized before allocating stimulus values"
            )
        if self.compartment is None:
            raise ValueError(
                "compartment reference must be initialized before allocating stimulus values"
            )
        count = _field(self.compartment, "count")
        self.values = np.zeros(
            (len(self.wholeCellModelIDs), int(count), int(numTimePoints))
        )

    def initialize(self) -> None:
        self.allocateMemory(1)

    def getIndexs(self, wholeCellModelIDs: Any) -> Any:
        """Return MATLAB-style 1-based positions, with zero for missing IDs."""
        ids = (
            []
            if self.wholeCellModelIDs is None
            else list(self.wholeCellModelIDs)
        )
        requested = np.asarray(wholeCellModelIDs, dtype=object)
        flat = requested.reshape(-1)
        result = np.asarray(
            [ids.index(item) + 1 if item in ids else 0 for item in flat],
            dtype=int,
        )
        if requested.ndim == 0:
            return int(result[0])
        return result.reshape(requested.shape)


def _field(value: Any, name: str, default: Any = ...):
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not ...:
        return default
    raise AttributeError(f"{type(value).__name__!s} has no field {name!r}")
