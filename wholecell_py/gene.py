"""Gene constants translated from WholeCell's ``constant.Gene``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


_FIELDS = (
    "wholeCellModelIDs",
    "names",
    "startCoordinates",
    "lengths",
    "strands",
    "essential",
    "mRNAIndexs",
    "rRNAIndexs",
    "sRNAIndexs",
    "tRNAIndexs",
    "ribosomalRRNAIndexs",
)


class Gene:
    """Immutable gene constants, preserving MATLAB's one-based indices."""

    def __init__(
        self, snapshot: Mapping[str, Any] | None = None, **attributes: Any
    ) -> None:
        for field in _FIELDS:
            setattr(self, field, None)
        restored = dict(snapshot or {})
        restored.update(attributes)
        for key, value in restored.items():
            if key in ("wholeCellModelIDs", "names", "essential"):
                value = np.asarray(value, dtype=object).copy()
                value = np.vectorize(_char_string, otypes=[object])(value)
            elif isinstance(value, (np.ndarray, list, tuple)):
                value = np.array(value, copy=True)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            setattr(self, key, value)

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        del simulation
        if knowledgeBase is None or not hasattr(knowledgeBase, "genes"):
            raise NotImplementedError(
                "Gene.initializeConstants requires knowledge-base gene records"
            )
        genes = knowledgeBase.genes
        fields = {
            "wholeCellModelIDs": [g.wholeCellModelID for g in genes],
            "names": [g.name for g in genes],
            "startCoordinates": [g.startCoordinate for g in genes],
            "lengths": [g.sequenceLength for g in genes],
            "strands": [2 - g.direction for g in genes],
            "essential": [g.essential for g in genes],
        }
        for name, records in (
            ("mRNAIndexs", knowledgeBase.mRNAGenes),
            ("rRNAIndexs", knowledgeBase.rRNAGenes),
            ("sRNAIndexs", knowledgeBase.sRNAGenes),
            ("tRNAIndexs", knowledgeBase.tRNAGenes),
        ):
            fields[name] = [record.idx for record in records]
        fields["ribosomalRRNAIndexs"] = knowledgeBase.ribosomalRRNAIndexs
        for name, values in fields.items():
            dtype = (
                object
                if name in ("wholeCellModelIDs", "names", "essential")
                else None
            )
            arr = np.asarray(values, dtype=dtype).reshape(-1, 1)
            arr.setflags(write=False)
            setattr(self, name, arr)

    def getIndexs(self, wholeCellModelIDs: Any) -> Any:
        ids = [_char_string(v) for v in _flat(self.wholeCellModelIDs)]
        requested = np.asarray(wholeCellModelIDs, dtype=object)
        result = np.asarray(
            [
                next(
                    (
                        i + 1
                        for i, item in enumerate(ids)
                        if item == _char_string(key)
                    ),
                    0,
                )
                for key in requested.reshape(-1, order="F")
            ],
            dtype=int,
        )
        return (
            int(result[0])
            if requested.ndim == 0
            else result.reshape(requested.shape, order="F")
        )

    def getIndexsByPosition(self, pos: Any) -> int | None:
        starts, lengths = _flat(self.startCoordinates), _flat(self.lengths)
        for i, (start, length) in enumerate(zip(starts, lengths)):
            if pos >= start and pos <= start + length - 1:
                return i + 1
        return None


def _flat(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Gene constants have not been initialized")
    return np.asarray(value, dtype=object).reshape(-1, order="F")


def _char_string(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return "".join(str(ch) for ch in value.reshape(-1, order="F"))
    return value
