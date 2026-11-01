"""Reader for portable JSON snapshots emitted by ``export_snapshot.m``.

The reader restores stored data and reference identity. Generic objects are data
containers only; this module never attempts to recreate MATLAB methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .sparsemat import SparseMat


@dataclass
class DataObject:
    id: str
    class_name: str
    fields: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, "fields")[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass
class ValueObjectData:
    """Stored value object whose MATLAB behavior is unavailable in Python."""

    class_name: str
    fields: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, "fields")[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class CircularSparseData(SparseMat):
    """Sparse data plus the exporter-provided circular dimension metadata."""

    def __init__(
        self, subs: Any, vals: Any, shape: Any, circular_dimensions: Any
    ):
        super().__init__(subs, vals, shape)
        self.circular_dimensions = circular_dimensions


@dataclass(frozen=True)
class ChromosomeSequenceData:
    positive_strand: Any


@dataclass(frozen=True)
class RNGData:
    type: Any
    seed: Any
    state: Any


@dataclass(frozen=True)
class MatlabFunctionExpression:
    expression: str


@dataclass
class SnapshotGraph:
    root: dict[str, Any]
    objects: dict[str, Any]
    states: dict[str, Any]
    processes: dict[str, Any]
    simulation: Any
    constants: dict[str, Any]


_DTYPES = {
    "double": np.float64,
    "single": np.float32,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "logical": np.bool_,
}


def _shape(value: Any) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(x, int) or x < 0 for x in value)
    ):
        raise ValueError(f"invalid MATLAB shape {value!r}")
    return tuple(value)


def _array(node: dict[str, Any]) -> np.ndarray:
    shape = _shape(node["shape"])
    dtype_name = node["dtype"]
    if dtype_name not in _DTYPES:
        raise ValueError(f"unsupported MATLAB array dtype {dtype_name!r}")
    size = int(np.prod(shape, dtype=np.int64))
    if "real" in node:
        real = np.asarray(node["real"], dtype=_DTYPES[dtype_name]).reshape(-1)
        imag = np.asarray(node["imag"], dtype=_DTYPES[dtype_name]).reshape(-1)
        if real.size != size or imag.size != size:
            raise ValueError("complex array data does not match shape")
        arr = (real + 1j * imag).reshape(shape, order="F")
    else:
        data = np.asarray(
            node.get("data", []), dtype=_DTYPES[dtype_name]
        ).reshape(-1)
        if data.size != size:
            raise ValueError("array data does not match shape")
        arr = data.reshape(shape, order="F")
    indices = node.get("nonfinite_indices", [])
    values = node.get("nonfinite_values", [])
    # MATLAB jsonencode collapses singleton arrays to scalars.
    if not isinstance(indices, list):
        indices = [indices]
    if not isinstance(values, list):
        values = [values]
    if len(indices) != len(values):
        raise ValueError("nonfinite index/value counts differ")
    flat = arr.ravel(order="F")
    for index, value in zip(indices, values):
        if not isinstance(index, int) or index < 1 or index > size:
            raise ValueError(f"invalid nonfinite MATLAB index {index!r}")
        specials = {"NaN": np.nan, "Inf": np.inf, "-Inf": -np.inf}
        if value not in specials:
            raise ValueError(f"unknown nonfinite value {value!r}")
        flat[index - 1] = specials[value]
    return arr


def load_snapshot(
    path: str | Path,
    *,
    allow_incomplete: bool = False,
    allow_projection: bool = False,
    class_registry: Mapping[str, Any] | None = None,
    strict_runtime: bool = False,
    runtime_values: bool = False,
    rng_factory: Any = None,
) -> SnapshotGraph:
    """Load an exporter JSON file, resolving object references in two passes.

    ``class_registry`` maps MATLAB class names to constructors. Constructors are
    called with no arguments, then stored fields are assigned as attributes.
    ``runtime_values`` restores executable sequence/circular-sparse objects.
    ``rng_factory`` must explicitly choose how to translate MATLAB RNG state;
    the decoder never silently restarts a stream from its initial seed.
    """
    with Path(path).open(encoding="utf-8") as stream:
        root = json.load(stream)
    if root.get("schema_version") != 2:
        raise ValueError(
            f"unsupported snapshot schema version {root.get('schema_version')!r}"
        )
    if root.get("complete") is False and not allow_incomplete:
        raise ValueError(
            f"snapshot is incomplete; exporter errors: {root.get('errors', [])!r}"
        )
    records = root.get("objects")
    if not isinstance(records, list):
        raise ValueError("snapshot objects must be a list")
    registry = class_registry or {}
    objects: dict[str, Any] = {}
    raw: dict[str, dict[str, Any]] = {}
    for record in records:
        oid, cls = record.get("id"), record.get("class")
        if not isinstance(oid, str) or not isinstance(cls, str) or oid in raw:
            raise ValueError(
                f"invalid or duplicate snapshot object record {record!r}"
            )
        raw[oid] = record
        if cls in registry:
            obj = registry[cls]()
            if not hasattr(obj, "__dict__"):
                raise TypeError(
                    f"registered class {cls!r} cannot hold snapshot fields"
                )
        elif strict_runtime:
            raise ValueError(
                f"no runtime class registered for MATLAB class {cls!r} ({oid})"
            )
        else:
            obj = DataObject(oid, cls)
        objects[oid] = obj

    def unpack(value: Any) -> Any:
        if isinstance(value, list):
            return [unpack(x) for x in value]
        if not isinstance(value, dict):
            return value
        kind = value.get("kind")
        if kind is None:
            return {k: unpack(v) for k, v in value.items()}
        if kind == "reference":
            oid = value["id"]
            if oid not in objects:
                raise ValueError(
                    f"reference to unknown snapshot object {oid!r}"
                )
            return objects[oid]
        if kind == "array":
            return _array(value)
        if kind == "char":
            shape = _shape(value["shape"])
            data = value.get("data", "")
            if len(data) != int(np.prod(shape)):
                raise ValueError("char data does not match shape")
            return np.asarray(list(data), dtype="U1").reshape(shape, order="F")
        if kind in ("cell", "struct"):
            shape = _shape(value["shape"])
            data = value.get("data", [])
            if len(data) != int(np.prod(shape)):
                raise ValueError(f"{kind} data does not match shape")
            out = np.empty(shape, dtype=object, order="F")
            decoded = [unpack(x) for x in data]
            out.ravel(order="F")[:] = decoded
            return out
        if kind == "value_objects":
            shape = _shape(value["shape"])
            cls = value.get("class")
            data = value.get("data", [])
            if not isinstance(cls, str) or len(data) != int(np.prod(shape)):
                raise ValueError(
                    "value object class/data does not match shape"
                )
            ctor = registry.get(cls)
            if strict_runtime and ctor is None:
                raise ValueError(
                    f"no runtime class registered for MATLAB value class {cls!r}"
                )
            out = np.empty(shape, dtype=object, order="F")
            for i, record in enumerate(data):
                if not isinstance(record, dict):
                    raise ValueError(
                        f"value object data for {cls!r} must be a mapping"
                    )
                decoded = {name: unpack(item) for name, item in record.items()}
                if ctor is None:
                    obj = ValueObjectData(cls, decoded)
                else:
                    obj = ctor()
                    if not hasattr(obj, "__dict__"):
                        raise TypeError(
                            f"registered class {cls!r} cannot hold snapshot fields"
                        )
                    for name, item in decoded.items():
                        setattr(obj, name, item)
                out.ravel(order="F")[i] = obj
            return out
        if kind == "sparse":
            shape = _shape(value["shape"])
            subs = unpack(value["subs"])
            vals = unpack(value["vals"])
            if isinstance(subs, np.ndarray) and subs.ndim == 1:
                subs = subs.reshape(1, -1)
            if "circularDimensions" in value:
                if runtime_values or strict_runtime:
                    from .circular_sparsemat import CircularSparseMat

                    return CircularSparseMat(
                        subs, vals, shape, unpack(value["circularDimensions"])
                    )
                return CircularSparseData(
                    subs, vals, shape, unpack(value["circularDimensions"])
                )
            return SparseMat(subs, vals, shape)
        if kind == "chromosome_sequence":
            if runtime_values or strict_runtime:
                from .chromosome_sequence import ChromosomeSequence

                return ChromosomeSequence(unpack(value["positive_strand"]))
            return ChromosomeSequenceData(unpack(value["positive_strand"]))
        if kind == "rng":
            record = RNGData(
                unpack(value.get("type")),
                unpack(value.get("seed")),
                unpack(value.get("state")),
            )
            if rng_factory is not None:
                return rng_factory(record)
            if strict_runtime:
                raise ValueError(
                    "runtime RNG restoration requires an explicit rng_factory"
                )
            return record
        if kind == "matlab_function":
            return MatlabFunctionExpression(value["expression"])
        raise ValueError(f"unknown snapshot value kind {kind!r}")

    for oid, record in raw.items():
        fields = record.get("fields")
        if not isinstance(fields, dict):
            raise ValueError(f"fields for {oid!r} must be a mapping")
        obj = objects[oid]
        decoded = {name: unpack(value) for name, value in fields.items()}
        if isinstance(obj, DataObject):
            obj.fields.update(decoded)
        else:
            for name, value in decoded.items():
                # MATLAB index constants are doubles. Use the registered
                # runtime class's declared integer interface where available;
                # data containers and global mapping-array shapes stay intact.
                prototype = getattr(type(obj), name, None)
                if (
                    "Indexs" in name or name.endswith(("Index", "Value"))
                ) and isinstance(prototype, (int, np.integer)):
                    a = np.asarray(value)
                    if (
                        a.size != 1
                        or not np.isfinite(a.item())
                        or a.item() != int(a.item())
                    ):
                        raise ValueError(
                            f"invalid scalar index constant {name}"
                        )
                    value = int(a.item())
                setattr(obj, name, value)
    states = {
        key.split(":", 1)[1]: obj
        for key, obj in objects.items()
        if key.startswith("state:")
    }
    processes = {
        key.split(":", 1)[1]: obj
        for key, obj in objects.items()
        if key.startswith("process:")
    }
    constants = {
        key.split(":", 1)[1]: obj
        for key, obj in objects.items()
        if key.startswith("constant:")
    }
    if "simulation" not in objects and not allow_projection:
        raise ValueError("snapshot has no simulation object")
    for obj in objects.values():
        hook = getattr(obj, "restoreSnapshotRuntime", None)
        if callable(hook):
            hook()
    return SnapshotGraph(
        root, objects, states, processes, objects.get("simulation"), constants
    )
