"""Ribosome translation state translated from WholeCell's ``Ribosome``."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class Ribosome:
    optionNames = ("verbosity", "seed")
    fixedConstantNames: tuple[str, ...] = ()
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = ("states", "boundMRNAs", "mRNAPositions", "tmRNAPositions")
    dependentStateNames = (
        "stateOccupancies",
        "nActive",
        "nNotExist",
        "nStalled",
    )

    activeIndex, notExistIndex, stalledIndex = 1, 2, 3
    activeValue, notExistValue, stalledValue = 1, 0, -1
    dryWeight = 0
    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "states",
        "boundMRNAs",
        "mRNAPositions",
        "tmRNAPositions",
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        self.wholeCellModelID, self.name = wholeCellModelID, name
        self.states = self.boundMRNAs = self.mRNAPositions = (
            self.tmRNAPositions
        ) = None
        self.gene = self.polypeptide = self.chromosome = self.rna = (
            self.complex
        ) = None
        self.randStream = None
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
        self.gene = _simulation_value(simulation, "gene")
        self.polypeptide = _simulation_state(simulation, "Polypeptide")
        self.chromosome = _simulation_state(simulation, "Chromosome")
        self.rna = _simulation_state(simulation, "Rna")
        self.complex = _simulation_state(simulation, "ProteinComplex")

    def allocateMemory(self, numTimePoints: int) -> None:
        if (
            not isinstance(numTimePoints, (int, np.integer))
            or numTimePoints < 0
        ):
            raise ValueError("numTimePoints must be a non-negative integer")
        shape = (0, 1, int(numTimePoints))
        self.states = np.zeros(shape, order="F")
        self.boundMRNAs = np.zeros(shape, order="F")
        self.mRNAPositions = np.zeros(shape, order="F")
        self.tmRNAPositions = np.zeros(shape, order="F")

    def initialize(self) -> None:
        return None

    def releaseRibosome(self, nRibs: int, nTMRNA: int, mRNAs: Any) -> None:
        """Release selected ribosomes and any requested mRNA/tmRNA-bound ones."""
        poly = _require(self, "polypeptide")
        stream = _require(self, "randStream")
        if not hasattr(stream, "randomlySelectNRows") or not hasattr(
            stream, "rand"
        ):
            raise NotImplementedError(
                "releaseRibosome requires a runtime RandStream; snapshot RNGData stores state only"
            )
        states = _last_column(self.states)
        bound = _last_column(self.boundMRNAs)
        mrna_pos = _last_column(self.mRNAPositions)
        tmrna_pos = _last_column(self.tmRNAPositions)

        rib_idxs = np.empty(0, dtype=np.int64)
        if nRibs > 0:
            available = np.flatnonzero(states != self.notExistValue) + 1
            selected, _ = stream.randomlySelectNRows(available, nRibs)
            rib_idxs = np.asarray(selected, dtype=np.int64).reshape(-1)
            if rib_idxs.size < nRibs:
                raise RuntimeError("Unable release ribosomes")

        tm_idxs = np.empty(0, dtype=np.int64)
        if nTMRNA > 0:
            available = np.flatnonzero(states == self.stalledValue) + 1
            selected, _ = stream.randomlySelectNRows(available, nTMRNA)
            tm_idxs = np.asarray(selected, dtype=np.int64).reshape(-1)
            if tm_idxs.size < nTMRNA:
                raise RuntimeError("Unable release tmRNA")

        mrna_idxs = np.empty(0, dtype=np.int64)
        mRNAs = np.asarray(mRNAs)
        if np.any(mRNAs):
            rna, gene = _require(self, "rna"), _require(self, "gene")
            if not hasattr(rna, "matureRNAGeneComposition"):
                raise NotImplementedError(
                    "releaseRibosome requires Rna.matureRNAGeneComposition runtime behavior"
                )
            composition = np.asarray(rna.matureRNAGeneComposition)[
                np.ix_(
                    np.asarray(gene.mRNAIndexs, int).ravel() - 1,
                    np.asarray(rna.matureMRNAIndexs, int).ravel() - 1,
                )
            ]
            mrna_genes = np.asarray(composition @ mRNAs).reshape(-1)
            totals = np.asarray(
                composition
                @ np.asarray(rna.counts)[
                    _indices(rna.matureIndexs, rna.matureMRNAIndexs),
                    int(np.asarray(rna.compartment.cytosolIndexs).item()) - 1,
                ]
            ).reshape(-1)
            chosen = []
            for gene_idx in np.flatnonzero(mrna_genes):
                row_idxs = np.flatnonzero(bound == gene_idx + 1) + 1
                if row_idxs.size:
                    keep = (
                        np.asarray(stream.rand(row_idxs.size, 1)).reshape(-1)
                        <= mrna_genes[gene_idx] / totals[gene_idx]
                    )
                    chosen.extend(row_idxs[keep])
            mrna_idxs = np.asarray(chosen, dtype=np.int64)

        combined = np.concatenate((rib_idxs, mrna_idxs, tm_idxs))
        idxs = (
            np.unique(combined)
            if sum(x.size > 0 for x in (rib_idxs, mrna_idxs, tm_idxs)) > 1
            else combined
        )
        # Ribosome.m skips abort records when no peptide or tag was made.
        # Still release every selected ribosome below.
        peptide_idxs = idxs[
            (mrna_pos[idxs - 1] != 0) | (tmrna_pos[idxs - 1] != 0)
        ]
        if peptide_idxs.size:
            old = np.column_stack(
                (
                    bound[peptide_idxs - 1],
                    mrna_pos[peptide_idxs - 1],
                    np.zeros(peptide_idxs.size),
                )
            )
            stalled = states[peptide_idxs - 1] == self.stalledValue
            old[stalled, 2] = tmrna_pos[peptide_idxs[stalled] - 1]
            aborted = np.asarray(poly.abortedPolypeptides)
            if aborted.size == 0:
                poly.abortedPolypeptides = old
            else:
                poly.abortedPolypeptides = np.concatenate(
                    (aborted.reshape((-1, 3)), old), axis=0
                )

        _release_complexes(self, idxs, nRibs)
        _release_tm_rna(self, states, idxs, nTMRNA)
        idx0 = idxs - 1
        states[idx0] = self.notExistValue
        bound[idx0] = self.notExistValue
        mrna_pos[idx0] = self.notExistValue
        tmrna_pos[idx0] = self.notExistValue
        _write_last_column(self.states, states)
        _write_last_column(self.boundMRNAs, bound)
        _write_last_column(self.mRNAPositions, mrna_pos)
        _write_last_column(self.tmRNAPositions, tmrna_pos)
        _assign_by_indices(poly.boundMRNAs, idx0, bound[idx0])
        _assign_by_indices(poly.nascentMonomerLengths, idx0, mrna_pos[idx0])
        _assign_by_indices(poly.proteolysisTagLengths, idx0, tmrna_pos[idx0])

    @property
    def stateOccupancies(self) -> np.ndarray:
        time_points = (
            np.asarray(self.states).shape[2]
            if np.asarray(self.states).ndim > 2
            else 1
        )
        counts = np.concatenate(
            (self.nActive, self.nNotExist, self.nStalled), axis=0
        ).astype(float)
        counts = counts.reshape((3, 1, time_points))
        denominator = np.sum(counts, axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return counts / denominator

    @property
    def nActive(self) -> np.ndarray:
        return _matlab_sum(self.states == self.activeValue)

    @property
    def nNotExist(self) -> np.ndarray:
        return _matlab_sum(self.states == self.notExistValue)

    @property
    def nStalled(self) -> np.ndarray:
        return _matlab_sum(self.states == self.stalledValue)

    @property
    def nMRNAsBound(self) -> np.ndarray:
        lengths = np.asarray(_require(self, "polypeptide").monomerLengths)
        n_monomers = lengths.size
        bound = np.asarray(self.boundMRNAs)
        if bound.ndim == 1:
            bound = bound.reshape((-1, 1, 1))
        elif bound.ndim == 2:
            bound = bound[:, :, None]
        out = np.zeros((n_monomers, 1, bound.shape[2]), dtype=float, order="F")
        for i in range(bound.shape[0]):
            for k in range(bound.shape[2]):
                idx = int(bound[i, 0, k])
                if idx >= 1:
                    if idx > n_monomers:
                        raise IndexError(
                            f"bound mRNA index {idx} exceeds {n_monomers} monomers"
                        )
                    out[idx - 1, 0, k] += 1
        return out


def _require(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None) if obj is not None else None
    if value is None:
        raise RuntimeError(
            f"Ribosome dependency {name!r} has not been assigned"
        )
    return value


def _simulation_value(simulation: Any, name: str) -> Any:
    if isinstance(simulation, Mapping):
        return simulation[name]
    return getattr(simulation, name)


def _simulation_state(simulation: Any, name: str) -> Any:
    if isinstance(simulation, Mapping):
        states = simulation.get("states", {})
        if name in states:
            return states[name]
    getter = getattr(simulation, "state", None)
    if getter is None:
        raise RuntimeError(f"simulation cannot provide state {name!r}")
    return getter(name)


def _last_column(array: Any) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 1:
        return value.copy()
    if value.ndim == 2:
        return value[:, 0].copy()
    return value[:, 0, -1].copy()


def _write_last_column(target: np.ndarray, values: np.ndarray) -> None:
    if target.ndim == 1:
        target[:] = values
    elif target.ndim == 2:
        target[:, 0] = values
    else:
        target[:, 0, -1] = values


def _matlab_sum(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    # MATLAB sum over the first non-singleton dimension.
    axis = next((i for i, n in enumerate(arr.shape) if n != 1), 0)
    result = np.sum(arr, axis=axis, keepdims=True)
    while result.ndim < 2:
        result = result.reshape(result.shape + (1,))
    return result


def _indices(primary: Any, secondary: Any) -> np.ndarray:
    return (
        np.asarray(primary, dtype=int).reshape(-1)[
            np.asarray(secondary, dtype=int).reshape(-1) - 1
        ]
        - 1
    )


def _assign_by_indices(
    array: Any, indices: np.ndarray, values: np.ndarray
) -> None:
    target = np.asarray(array)
    if target.ndim == 1:
        target[indices] = values
    elif target.ndim == 2:
        target[indices, 0] = values
    else:
        target[indices, 0, -1] = values


def _release_complexes(state: Ribosome, idxs: np.ndarray, n_ribs: int) -> None:
    count = idxs.size - n_ribs
    if count <= 0:
        return
    complex_state = _require(state, "complex")
    bound = _indices(
        complex_state.boundIndexs, complex_state.ribosome70SIndexs
    )
    mature = _indices(
        complex_state.matureIndexs,
        np.concatenate(
            (
                np.asarray(complex_state.ribosome30SIndexs).reshape(-1),
                np.asarray(complex_state.ribosome50SIndexs).reshape(-1),
            )
        ),
    )
    _change_counts(
        complex_state.counts,
        bound,
        complex_state.compartment.cytosolIndexs,
        -count,
    )
    _change_counts(
        complex_state.counts,
        mature,
        complex_state.compartment.cytosolIndexs,
        count,
    )


def _release_tm_rna(
    state: Ribosome, states: np.ndarray, idxs: np.ndarray, n_tmrna: int
) -> None:
    count = np.count_nonzero(states[idxs - 1] == state.stalledValue) - n_tmrna
    if count <= 0:
        return
    rna = _require(state, "rna")
    bound = _indices(rna.boundIndexs, rna.matureTMRNAIndexs)
    mature = _indices(rna.matureIndexs, rna.matureTMRNAIndexs)
    _change_counts(rna.counts, bound, rna.compartment.cytosolIndexs, -count)
    _change_counts(rna.counts, mature, rna.compartment.cytosolIndexs, count)


def _change_counts(
    counts: Any, rows: np.ndarray, cols: Any, delta: int
) -> None:
    arr = np.asarray(counts)
    row = np.asarray(rows, dtype=int).reshape(-1)
    col = np.asarray(cols, dtype=int).reshape(-1) - 1
    if arr.ndim == 1:
        arr[row] += delta
    else:
        arr[np.ix_(row, col)] += delta
