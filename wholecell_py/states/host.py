"""Host interaction state translated from ``cell.sim.state.Host``."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class Host:
    optionNames = ("verbosity", "seed")
    fixedConstantNames: tuple[str, ...] = ()
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = (
        "isBacteriumAdherent",
        "isTLRActivated",
        "isNFkBActivated",
        "isInflammatoryResponseActivated",
    )
    dependentStateNames: tuple[str, ...] = ()

    nTLRs = 3
    tlrIDs = ("TLR1", "TLR2", "TLR6")
    tlrIndexs_1 = 1
    tlrIndexs_2 = 2
    tlrIndexs_6 = 3
    dryWeight = 0

    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "isBacteriumAdherent",
        "isTLRActivated",
        "isNFkBActivated",
        "isInflammatoryResponseActivated",
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        self.wholeCellModelID, self.name = wholeCellModelID, name
        self.isBacteriumAdherent: np.ndarray | None = None
        self.isTLRActivated: np.ndarray | None = None
        self.isNFkBActivated: Any = None
        self.isInflammatoryResponseActivated: Any = None
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

    def allocateMemory(self, numTimePoints: int) -> None:
        if not isinstance(numTimePoints, (int, np.integer)):
            raise TypeError("numTimePoints must be an integer")
        if numTimePoints < 0:
            raise ValueError("numTimePoints must be non-negative")
        self.isBacteriumAdherent = np.zeros(
            (1, 1, int(numTimePoints)), dtype=bool
        )
        self.isTLRActivated = np.zeros(
            (int(np.asarray(self.nTLRs).item()), 1, int(numTimePoints)),
            dtype=bool,
        )

    def initialize(self) -> None:
        # The MATLAB implementation intentionally leaves state to HostInteraction.
        return None
