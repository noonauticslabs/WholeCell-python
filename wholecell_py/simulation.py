"""One-step whole-cell simulation orchestration.

The process models remain responsible for their own biology. This module ports
the state, resource-allocation, ordering, and condition semantics of MATLAB's
``Simulation/evolveState.m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from .side_effect import apply_side_effects


def _one_based(values: Any, *, label: str) -> np.ndarray:
    idx = np.asarray(values)
    if (
        not np.issubdtype(idx.dtype, np.number)
        or not np.all(np.isfinite(idx))
        or np.any(idx != np.floor(idx))
    ):
        raise TypeError(f"{label} must contain integral MATLAB indices")
    if np.any(idx < 1):
        raise IndexError(f"{label} contains an index below 1")
    return idx.astype(np.intp, copy=False)


def _conditions(values: Any, conditions: Any, time: float) -> np.ndarray:
    """Apply active MATLAB six-column conditions to F-order linear positions."""
    result = np.array(values, copy=True, subok=False)
    rows = np.asarray(conditions)
    if rows.size == 0:
        return result
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError("conditions must be an N x 6 array")
    active = rows[(time >= rows[:, 3]) & (time <= rows[:, 4])]
    flat = result.ravel(order="F").copy()
    for row in active:
        pos = int(row[5])
        if pos < 1 or pos > flat.size:
            raise IndexError(
                "condition objectCompartmentIndex is out of bounds"
            )
        flat[pos - 1] = row[2]
    return flat.reshape(result.shape, order="F")


def _randperm(stream: Any, n: int) -> np.ndarray:
    if hasattr(stream, "randperm"):
        order = np.asarray(stream.randperm(n))
    elif hasattr(stream, "permutation"):
        order = np.asarray(stream.permutation(n)) + 1
    else:
        raise TypeError(
            "rand_stream must provide randperm(n) or permutation(n)"
        )
    if order.shape != (n,) or not np.array_equal(
        np.sort(order), np.arange(1, n + 1)
    ):
        raise ValueError(
            "random stream returned an invalid process permutation"
        )
    return order.astype(np.intp, copy=False)


def _f_order_coordinates(
    shape: tuple[int, ...], positions: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Convert MATLAB linear positions to NumPy coordinates once per scatter."""
    return np.unravel_index(positions, shape, order="F")


@dataclass
class Simulation:
    """Mutable state and process coordinator for a single evolution step.

    ``states`` maps ``Time``, ``Metabolite``, and ``Stimulus`` to state objects
    exposing camelCase MATLAB fields (``values``, ``counts``, ``setValues``,
    ``setCounts``). Process indexes, when supplied, use MATLAB's one-based
    indexing. Otherwise they are inferred from process IDs/names.
    """

    states: dict[str, Any]
    processes: list[Any]
    rand_stream: Any
    step_size_sec: float
    process_index_trna_aminoacylation: int | None = None
    process_index_translation: int | None = None

    def __post_init__(self) -> None:
        self.states = {str(k).lower(): v for k, v in self.states.items()}
        for key in ("time", "metabolite", "stimulus"):
            if key not in self.states:
                raise KeyError(f"missing required simulation state {key!r}")
        if not np.isscalar(self.step_size_sec):
            raise TypeError("step_size_sec must be scalar")
        self._infer_order_constraints()

    def _infer_order_constraints(self) -> None:
        aliases = {"trna": None, "translation": None}
        for i, proc in enumerate(self.processes, 1):
            text = " ".join(
                str(getattr(proc, a, ""))
                for a in (
                    "wholeCellModelID",
                    "whole_cell_model_id",
                    "id",
                    "name",
                )
            ).lower()
            if "translation" in text:
                aliases["translation"] = i
            if (
                "trna" in text
                or "trnaaminoacylation" in text
                or "trna aminoacylation" in text
            ):
                aliases["trna"] = i
        if self.process_index_trna_aminoacylation is None:
            self.process_index_trna_aminoacylation = aliases["trna"]
        if self.process_index_translation is None:
            self.process_index_translation = aliases["translation"]
        for name, idx in (
            ("tRNA aminoacylation", self.process_index_trna_aminoacylation),
            ("translation", self.process_index_translation),
        ):
            if idx is not None and not 1 <= idx <= len(self.processes):
                raise IndexError(f"{name} process index is out of bounds")

    def evolve_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time = self.states["time"]
        mets = self.states["metabolite"]
        stim = self.states["stimulus"]
        time.values = np.asarray(time.values) + self.step_size_sec
        now = float(np.asarray(time.values).reshape(-1)[0])
        stim.values = _conditions(stim.values, stim.setValues, now)

        procs = self.processes
        n = len(procs)
        n_counts = np.asarray(mets.counts).size
        requirements = np.zeros((n_counts, n), dtype=float)
        for col, proc in enumerate(procs):
            proc.copyFromState()
            demand = np.asarray(proc.calcResourceRequirements_Current())
            local = _one_based(
                proc.substrateMetaboliteLocalIndexs,
                label="local substrate indexes",
            )
            global_ = _one_based(
                proc.substrateMetaboliteGlobalCompartmentIndexs,
                label="global substrate indexes",
            )
            if demand.ndim == 0:
                demand = demand.reshape(1, 1)
            selected = demand[local - 1, ...]
            gathered = selected.reshape(-1, order="F")
            if gathered.size != global_.size:
                raise ValueError(
                    "resource demand shape does not match global substrate indexes"
                )
            if np.any(global_ > n_counts):
                raise IndexError("global substrate index is out of bounds")
            requirements[global_.reshape(-1, order="F") - 1, col] = gathered
        requirements = np.maximum(0, requirements)
        available = np.asarray(mets.counts).ravel(order="F")
        share = available / np.maximum(1, requirements.sum(axis=1))
        allocations = np.maximum(
            0, np.fix(requirements * share[:, None])
        ).astype(np.int64)

        if n:
            while True:
                order = _randperm(self.rand_stream, n)
                trna, translation = (
                    self.process_index_trna_aminoacylation,
                    self.process_index_translation,
                )
                if (
                    trna is None
                    or translation is None
                    or np.where(order == trna)[0][0]
                    < np.where(order == translation)[0][0]
                ):
                    break
        else:
            order = np.array([], dtype=np.intp)
        usages = np.zeros_like(allocations)
        for proc_index in order:
            proc = procs[int(proc_index) - 1]
            global_ = _one_based(
                proc.substrateMetaboliteGlobalCompartmentIndexs,
                label="global substrate indexes",
            )
            local = _one_based(
                proc.substrateMetaboliteLocalIndexs,
                label="local substrate indexes",
            )
            allocation = allocations[
                global_.reshape(-1, order="F") - 1, int(proc_index) - 1
            ].reshape(
                np.asarray(
                    proc.substrateMetaboliteGlobalCompartmentIndexs
                ).shape,
                order="F",
            )
            # Read live counts: earlier processes and side effects may update
            # molecules outside their allocated substrate mappings.
            positions = global_.reshape(-1, order="F") - 1
            state_counts = np.asarray(mets.counts)
            before = state_counts[
                _f_order_coordinates(state_counts.shape, positions)
            ].copy()
            proc.simulationStateSideEffects = []
            proc.copyFromState()
            target_shape = proc.substrates[local - 1, ...].shape
            proc.substrates[local - 1, ...] = allocation.reshape(
                target_shape, order="F"
            )
            proc.evolveState()
            proc.copyToState()
            substrate = np.asarray(proc.substrates)
            delta = substrate[local - 1, ...].reshape(
                -1, order="F"
            ) - allocation.reshape(-1, order="F")
            # Scheduler writeback has MATLAB value semantics: assign a fresh
            # F-order state array so existing Python references do not observe
            # the scheduler's mapped substrate update through aliasing.
            state_counts = np.array(
                mets.counts, copy=True, order="F", subok=False
            )
            state_counts[
                _f_order_coordinates(state_counts.shape, positions)
            ] = (before + delta)
            mets.counts = state_counts
            usages[positions, int(proc_index) - 1] = -delta
            effects = proc.simulationStateSideEffects
            if effects is not None and (
                not hasattr(effects, "__len__") or len(effects)
            ):
                apply_side_effects(effects, self)
            observer = getattr(self, "after_process", None)
            if observer is not None:
                observer(proc)
        mets.counts = _conditions(mets.counts, mets.setCounts, now)
        return requirements, allocations, usages
