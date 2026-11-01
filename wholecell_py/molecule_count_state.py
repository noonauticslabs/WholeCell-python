"""Common state implementation for molecule counts.

This is the Python counterpart of ``cell.sim.MoleculeCountState``. Arrays use
MATLAB dimensions and indices in the public interface remain one based.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class MoleculeCountState:
    optionNames = ("verbosity", "seed")
    fixedConstantNames: tuple[str, ...] = ()
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = ("counts",)
    dependentStateNames = ("dryWeight", "decayRates")
    nAvogadro = 6.0221417930e23

    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "wholeCellModelIDs",
        "names",
        "molecularWeights",
        "baseCounts",
        "lengths",
        "halfLives",
        "compartments",
        "counts",
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        self.wholeCellModelID, self.name = wholeCellModelID, name
        self.wholeCellModelIDs = self.names = None
        self.molecularWeights = self.baseCounts = self.lengths = None
        self.halfLives = self.compartments = self.counts = None
        self.compartment = None
        self.parameterNames = self.parameterIndexs = None
        self.randStream = None
        self.verbosity, self.seed = 0, None
        restored = dict(snapshot or {})
        restored.update(attributes)
        for key, value in restored.items():
            setattr(
                self,
                key,
                (
                    np.array(value, copy=True, order="F")
                    if isinstance(value, np.ndarray)
                    else value
                ),
            )

    def storeObjectReferences(self, simulation: Any) -> None:
        if simulation is None or not hasattr(simulation, "compartment"):
            raise ValueError(
                "simulation with a compartment reference is required"
            )
        self.compartment = simulation.compartment

    def allocateMemory(self, numTimePoints: int) -> None:
        if not isinstance(numTimePoints, (int, np.integer)):
            raise TypeError("numTimePoints must be an integer")
        if numTimePoints < 0:
            raise ValueError("numTimePoints must be non-negative")
        if self.wholeCellModelIDs is None:
            raise ValueError(
                "wholeCellModelIDs must be initialized before allocating counts"
            )
        if self.compartment is None:
            raise ValueError(
                "compartment reference must be initialized before allocating counts"
            )
        self.counts = np.zeros(
            (
                int(np.size(self.wholeCellModelIDs)),
                int(self.compartment.count),
                int(numTimePoints),
            ),
            order="F",
        )

    def initialize(self) -> None:
        self.allocateMemory(1)

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        """Initialize inherited CellState parameter metadata from a KB record.

        Molecule-specific constants are provided by subclasses. A missing KB
        cannot be treated as an empty/default state because that would create
        plausible but incorrect molecule metadata.
        """
        if knowledgeBase is None:
            raise ValueError(
                "knowledgeBase is required to initialize molecule-count state constants"
            )
        if simulation is not None:
            # CellState.initializeConstants copies these options before it
            # applies parameter defaults. Preserve that inherited behavior.
            self.verbosity = _field(simulation, "verbosity")
            self.seed = _field(simulation, "seed")
        records = _field(knowledgeBase, "states")
        state = next(
            (
                item
                for item in records
                if _field(item, "wholeCellModelID") == self.wholeCellModelID
            ),
            None,
        )
        if state is None:
            raise ValueError(
                f"state {self.wholeCellModelID!r} is missing from knowledgeBase.states"
            )
        from .kb_parameters import initialize_parameters

        initialize_parameters(self, _field(state, "parameters", []))

    def getIndexs(self, wholeCellModelIDs: Any) -> Any:
        """Return one-based positions, or zero where an ID is absent."""
        if self.wholeCellModelIDs is None:
            raise ValueError("wholeCellModelIDs are not initialized")

        def text(value):
            return (
                value
                if isinstance(value, str)
                else "".join(np.asarray(value).ravel(order="F").tolist())
            )

        ids = [
            text(value)
            for value in np.asarray(
                self.wholeCellModelIDs, dtype=object
            ).ravel(order="F")
        ]
        requested = np.asarray(wholeCellModelIDs, dtype=object)
        flat = requested.reshape(-1)
        result = np.asarray(
            [
                ids.index(text(item)) + 1 if text(item) in ids else 0
                for item in flat
            ],
            dtype=int,
        )
        if requested.ndim == 0:
            return int(result[0])
        return result.reshape(requested.shape)

    @property
    def decayRates(self) -> Any:
        if self.halfLives is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(2) / np.asarray(self.halfLives)

    @decayRates.setter
    def decayRates(self, value: Any) -> None:
        with np.errstate(divide="ignore", invalid="ignore"):
            self.halfLives = np.log(2) / np.asarray(value)

    @property
    def dryWeight(self) -> Any:
        return self.calcDryWeight()

    def calcDryWeight(self) -> Any:
        if self.counts is None or self.molecularWeights is None:
            raise ValueError(
                "counts and molecularWeights must be initialized to calculate dry weight"
            )
        weights = np.asarray(self.molecularWeights).reshape(-1, order="F")
        counts = np.asarray(self.counts)
        if counts.ndim == 1:
            counts = counts.reshape((-1, 1))
        if counts.ndim == 2 or counts.shape[2] == 1:
            matrix = counts.reshape((counts.shape[0], -1), order="F")
            result = weights.reshape((1, -1)) @ matrix
        else:
            # Contract the molecule dimension, retaining MATLAB's leading
            # singleton row dimension and the compartment/time dimensions.
            result = np.einsum("m,mct->ct", weights, counts)[None, :, :]
        return result / self.nAvogadro


def _field(value: Any, name: str, default: Any = ...):
    if isinstance(value, Mapping) and name in value:
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    if default is not ...:
        return default
    raise AttributeError(name)
