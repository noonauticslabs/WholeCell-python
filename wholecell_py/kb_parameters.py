"""Shared Process/CellState parameter assignment from KB records."""

import numpy as np


def initialize_parameters(owner, records):
    def field(record, name):
        return (
            record[name] if isinstance(record, dict) else getattr(record, name)
        )

    records = np.asarray(records, dtype=object).ravel(order="F")
    owner.parameterNames = [field(p, "name") for p in records]
    owner.parameterIndexs = [field(p, "index") for p in records]
    pending = {}
    for p in records:
        name, index, value = (
            field(p, key) for key in ("name", "index", "defaultValue")
        )
        if (
            index is None
            or np.size(index) == 0
            or (isinstance(index, str) and not index)
        ):
            pending[name] = value
        elif hasattr(owner, index):
            rows = np.asarray(getattr(owner, index), dtype=int).ravel(
                order="F"
            )
            if np.any(rows < 1):
                raise IndexError("parameter indices must be positive")
            old = np.asarray(pending.get(name, np.zeros((0, 1))))
            if old.ndim == 1:
                old = old.reshape(-1, 1)
            dest = np.zeros(
                (max(old.shape[0], int(rows.max()) if rows.size else 0), 1)
            )
            dest[: old.shape[0]] = old
            dest[rows - 1, 0] = np.asarray(value).ravel(order="F")
            pending[name] = dest
    for name, value in pending.items():
        setattr(owner, name, value)
