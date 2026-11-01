"""FtsZ ring state translated from WholeCell's ``FtsZRing``."""

from __future__ import annotations

from typing import Any, Mapping
import warnings

import numpy as np


class FtsZRing:
    optionNames = ("verbosity", "seed")
    fixedConstantNames = (
        "numFtsZSubunitsPerNm",
        "numFtsZSubunitsPerFilament",
        "filamentLengthInNm",
    )
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = (
        "numEdgesOneStraight",
        "numEdgesTwoStraight",
        "numEdgesTwoBent",
        "numResidualBent",
    )
    dependentStateNames = ("numEdges",)
    dryWeight = 0
    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "numFtsZSubunitsPerNm",
        "numFtsZSubunitsPerFilament",
        "filamentLengthInNm",
        "numEdgesOneStraight",
        "numEdgesTwoStraight",
        "numEdgesTwoBent",
        "numResidualBent",
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        self.wholeCellModelID, self.name = wholeCellModelID, name
        self.numFtsZSubunitsPerNm = None
        self.numFtsZSubunitsPerFilament = None
        self.filamentLengthInNm = None
        self.numEdgesOneStraight = None
        self.numEdgesTwoStraight = None
        self.numEdgesTwoBent = None
        self.numResidualBent = None
        self.geometry = None
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

    def storeObjectReferences(self, simulation: Any) -> None:
        self.geometry = simulation.state("Geometry")

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        # CellState's base implementation has no additional FtsZ-specific inputs.
        self.filamentLengthInNm = (
            self.numFtsZSubunitsPerFilament / self.numFtsZSubunitsPerNm
        )

    def allocateMemory(self, numTimePoints: int) -> None:
        if not isinstance(numTimePoints, (int, np.integer)):
            raise TypeError("numTimePoints must be an integer")
        if numTimePoints < 0:
            raise ValueError("numTimePoints must be non-negative")
        shape = (1, 1, int(numTimePoints))
        self.numEdgesOneStraight = np.zeros(shape)
        self.numEdgesTwoStraight = np.zeros(shape)
        self.numEdgesTwoBent = np.zeros(shape)
        self.numResidualBent = np.zeros(shape)

    def initialize(self) -> None:
        self.allocateMemory(1)

    @staticmethod
    def releaseFtsZ(nFtsZ: Any) -> Any:
        result = (
            np.array(nFtsZ, copy=True)
            if isinstance(nFtsZ, np.ndarray)
            else nFtsZ
        )
        if np.any(nFtsZ):
            warnings.warn("FtsZ not decayed", RuntimeWarning, stacklevel=2)
        return result

    @property
    def numEdges(self) -> Any:
        diameter = self.geometry.pinchedDiameter
        if np.all(np.asarray(diameter) == 0):
            return 0
        return self.calcNumEdges(diameter, self.filamentLengthInNm)

    @staticmethod
    def calcNumEdges(pinchedDiameter: Any, filamentLengthInNm: Any) -> Any:
        return np.floor(
            np.pi
            / np.arcsin(
                np.asarray(filamentLengthInNm)
                * 1e-9
                / np.asarray(pinchedDiameter)
            )
        )
