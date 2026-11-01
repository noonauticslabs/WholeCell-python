"""Cell geometry state translated from WholeCell's ``CellGeometry``."""

from __future__ import annotations

from typing import Any, Mapping
import warnings

import numpy as np


class Geometry:
    optionNames = ("verbosity", "seed")
    fixedConstantNames = ("density",)
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = ("width", "pinchedDiameter")
    dependentStateNames = (
        "volume",
        "cylindricalLength",
        "surfaceArea",
        "totalLength",
        "pinchedCircumference",
        "pinched",
        "chamberVolume",
    )
    dryWeight = 0
    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "density",
        "width",
        "pinchedDiameter",
        "volume",
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        self.wholeCellModelID, self.name = wholeCellModelID, name
        self.density = self.width = self.pinchedDiameter = self.volume = None
        self.mass = None
        restored = dict(snapshot or {})
        restored.update(attributes)
        for key, value in restored.items():
            setattr(self, key, value)

    def storeObjectReferences(self, simulation: Any) -> None:
        self.mass = simulation.state("Mass")

    def allocateMemory(self, numTimePoints: int) -> None:
        if not isinstance(numTimePoints, (int, np.integer)):
            raise TypeError("numTimePoints must be an integer")
        if numTimePoints < 0:
            raise ValueError("numTimePoints must be non-negative")
        shape = (1, 1, int(numTimePoints))
        self.width = np.zeros(shape)
        self.pinchedDiameter = np.zeros(shape)

    def initialize(self) -> None:
        self.calculateVolume()
        self.width = self.calculateWidth(self.volume)
        self.pinchedDiameter = np.array(self.width, copy=True)

    @staticmethod
    def calculateWidth(volume: Any) -> Any:
        return 2 * (np.asarray(volume) / 1000 * 3 / (4 * np.pi)) ** (1 / 3)

    def calculateVolume(self) -> None:
        # MATLAB sum uses the first nonsingleton dimension (compartments here).
        cell = np.asarray(self.mass.cell)
        self.volume = np.sum(cell, axis=1, keepdims=True) / self.density

    @property
    def width(self) -> Any:
        if self._width is None or self.pinchedDiameter is None:
            return self._width
        return np.where(np.asarray(self.pinchedDiameter) > 0, self._width, -1)

    @width.setter
    def width(self, value: Any) -> None:
        self._width = value

    @property
    def cylindricalLength(self) -> Any:
        return self.calcGeometry()[0]

    @property
    def surfaceArea(self) -> Any:
        return self.calcGeometry()[1]

    @property
    def totalLength(self) -> Any:
        return self.calcGeometry()[2]

    @property
    def pinchedCircumference(self) -> Any:
        return np.asarray(self.pinchedDiameter) * np.pi

    @property
    def pinched(self) -> Any:
        return np.asarray(self.pinchedDiameter) == 0

    @property
    def chamberVolume(self) -> Any:
        mass = self.mass
        return (
            mass.cellInitialDryWeight
            / (1 - mass.fractionWetWeight)
            / mass.initialBiomassConcentration
        )

    def calcGeometry(self) -> tuple[Any, Any, Any]:
        return self.calculateGeometry(
            self.width, self.pinchedDiameter, self.volume
        )

    @staticmethod
    def calculateGeometry(
        width: Any, pinchedDiameter: Any, volume: Any
    ) -> tuple[Any, Any, Any]:
        w, diameter, vol = np.broadcast_arrays(
            np.asarray(width), np.asarray(pinchedDiameter), np.asarray(volume)
        )
        s = (w - diameter) / 2
        active = diameter > 0
        cylindrical = (
            np.maximum(
                0,
                vol / 1000
                - np.pi * w**3 / 6
                - np.pi
                / 2
                * (
                    s * (8 * s**2 - 4 * s * w + w**2)
                    + 2 * s**2 * (-2 * s + w) * np.pi / 2
                    - 4 * s**3 / 3
                ),
            )
            * 4
            / (np.pi * w**2)
        )
        area = (
            np.pi * w**2
            + np.pi * w * cylindrical
            + np.pi * 4 * s * (w - 2 * s + s)
        )
        length = cylindrical + w + 2 * s
        if np.any(~active):
            warnings.warn(
                "Cell has divided. Cell shape undefined",
                RuntimeWarning,
                stacklevel=2,
            )
        cylindrical = np.where(active, cylindrical, -1)
        area = np.where(active, area, -1)
        length = np.where(active, length, -1)
        return _scalarize(cylindrical), _scalarize(area), _scalarize(length)


def _scalarize(value: np.ndarray) -> Any:
    return value.item() if value.ndim == 0 else value
