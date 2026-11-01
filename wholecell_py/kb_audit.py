"""Load explicit MATLAB knowledge-base audit projections.

These exports contain data only; they do not recreate MATLAB KB methods. In
particular, incomplete exports retain their per-record getter errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .snapshot import DataObject, load_snapshot


_TOP_LEVEL_RECORD_FIELDS = {
    "states",
    "metabolites",
    "processes",
    "compartments",
    "proteinMonomers",
    "proteinComplexs",
    "mRNAGenes",
    "genes",
    "rRNAGenes",
    "sRNAGenes",
    "tRNAGenes",
}
_NESTED_RECORD_FIELDS = {
    "parameters",
    "mediaCompartments",
    "features",
    "aminoAcid",
}
_VECTOR_FIELDS = {
    "mediaConcentrations",
    "mediaInitialTimes",
    "mediaFinalTimes",
}


def _normalize_data(value: Any, *, scalarize: bool = True) -> Any:
    """Decode KB scalar strings/numbers while preserving ordinary array shapes."""
    if isinstance(value, np.ndarray) and value.dtype.kind == "U":
        return "".join(value.ravel(order="F").tolist())
    if (
        scalarize
        and isinstance(value, np.ndarray)
        and value.size == 1
        and value.dtype.kind in "biufc"
    ):
        return value.item()
    if isinstance(value, np.ndarray) and value.dtype == object:
        out = np.empty(value.shape, dtype=object, order="F")
        out.ravel(order="F")[:] = [
            _normalize_data(item) for item in value.ravel(order="F")
        ]
        return out
    if isinstance(value, dict):
        return {key: _normalize_data(item) for key, item in value.items()}
    return value


def _one_record(value: Any) -> Any:
    """Unwrap a packed scalar struct without collapsing its child cell arrays."""
    if (
        isinstance(value, np.ndarray)
        and value.dtype == object
        and value.size == 1
    ):
        item = value.ravel(order="F")[0]
        if isinstance(item, dict):
            return _normalize_record(item)
    return _normalize_data(value)


def _record_array(value: Any, *, flatten: bool) -> Any:
    if not isinstance(value, np.ndarray) or value.dtype != object:
        return _normalize_data(value)
    values = [_one_record(item) for item in value.ravel(order="F")]
    shape = (len(values),) if flatten else value.shape
    out = np.empty(shape, dtype=object, order="F")
    out.ravel(order="F")[:] = values
    return out


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in record.items():
        if key in _NESTED_RECORD_FIELDS:
            result[key] = _record_array(value, flatten=True)
        elif key in _VECTOR_FIELDS:
            vector = _normalize_data(value, scalarize=False)
            result[key] = np.asarray(vector).ravel(order="F")
        else:
            result[key] = _normalize_data(value)
    return result


def load_kb_audit(path: str | Path) -> DataObject:
    """Decode a ``export_kb_audit.m`` file without masking incompleteness.

    Returns a data-only ``KnowledgeBaseProjection``. Fields use NumPy arrays
    and dictionaries; ``complete`` and ``errors`` are copied from the exporter
    root so callers can reject incomplete initializer inputs explicitly.
    """
    graph = load_snapshot(path, allow_incomplete=True, allow_projection=True)
    projection = graph.objects.get("knowledgeBase")
    if not isinstance(projection, DataObject):
        raise ValueError(
            "KB audit file has no knowledgeBase projection object"
        )
    projection.fields = {
        key: (
            _record_array(value, flatten=True)
            if key in _TOP_LEVEL_RECORD_FIELDS
            else _normalize_data(value)
        )
        for key, value in projection.fields.items()
    }
    projection.complete = bool(graph.root.get("complete", False))
    projection.errors = graph.root.get("errors", [])
    projection.source_file = graph.root.get("source_file")
    projection.source_sha256 = graph.root.get("source_sha256")
    return projection
