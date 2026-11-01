"""RNA polymerase state translated from WholeCell's ``RNAPolymerase``."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np


class RNAPolymerase:
    optionNames = ("verbosity", "seed")
    fixedConstantNames = ("stateExpectations",)
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = (
        "states",
        "positionStrands",
        "transcriptionFactorBindingProbFoldChange",
        "supercoilingBindingProbFoldChange",
    )
    dependentStateNames = (
        "stateOccupancies",
        "nActive",
        "nSpecificallyBound",
        "nNonSpecificallyBound",
        "nFree",
    )
    activelyTranscribingIndex = 1
    specificallyBoundIndex = 2
    nonSpecificallyBoundIndex = 3
    freeIndex = 4
    activelyTranscribingValue = 1
    specificallyBoundValue = -3
    nonSpecificallyBoundValue = -1
    freeValue = -2
    notExistValue = 0
    stateValues = np.array([1, -3, -1, -2])
    dryWeight = 0
    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "stateExpectations",
        "states",
        "positionStrands",
        "transcriptionFactorBindingProbFoldChange",
        "supercoilingBindingProbFoldChange",
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        self.wholeCellModelID, self.name = wholeCellModelID, name
        for key in self._snapshot_names[2:]:
            setattr(self, key, None)
        self.chromosome = self.transcripts = None
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

    def restoreSnapshotRuntime(self):
        for key in (
            "activelyTranscribingIndex",
            "specificallyBoundIndex",
            "nonSpecificallyBoundIndex",
            "freeIndex",
            "activelyTranscribingValue",
            "specificallyBoundValue",
            "nonSpecificallyBoundValue",
            "freeValue",
            "notExistValue",
        ):
            setattr(self, key, int(np.asarray(getattr(self, key)).item()))

    def storeObjectReferences(self, simulation: Any) -> None:
        self.chromosome = simulation.state("Chromosome")
        self.transcripts = simulation.state("Transcript")

    def allocateMemory(self, numTimePoints: int) -> None:
        if (
            not isinstance(numTimePoints, (int, np.integer))
            or numTimePoints < 0
        ):
            raise ValueError("numTimePoints must be a non-negative integer")
        n_tus = np.size(self.chromosome.transcriptionUnitWholeCellModelIDs)
        n = int(numTimePoints)
        self.states = np.zeros((0, 1, n), order="F")
        self.positionStrands = np.zeros((0, 2, n), order="F")
        self.transcriptionFactorBindingProbFoldChange = np.zeros(
            (n_tus, 2, n), order="F"
        )
        self.supercoilingBindingProbFoldChange = np.zeros(
            (n_tus, 2, n), order="F"
        )

    def initialize(self) -> None:
        pass

    def degradeFreePolymerase(self, nPolymerases: int) -> None:
        if nPolymerases < 0:
            raise ValueError("nPolymerases must be non-negative")
        indices = np.flatnonzero(
            np.asarray(self.states).reshape(-1, order="F") == self.freeValue
        )[: int(nPolymerases)]
        flat = np.asarray(self.states).reshape(-1, order="F").copy()
        flat[indices] = self.notExistValue
        self.states = flat.reshape(np.shape(self.states), order="F")

    def releasePolymerase(
        self, posStrnds: Any, proteinIsDegraded: bool
    ) -> None:
        if self.chromosome is None or self.transcripts is None:
            raise RuntimeError(
                "releasePolymerase requires chromosome and transcript references; call storeObjectReferences first"
            )
        c, t = self.chromosome, self.transcripts
        positions = np.asarray(self.positionStrands)
        targets = np.asarray(posStrnds).reshape((-1, 2))
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                "releasePolymerase requires current 2D positionStrands"
            )
        seq_len = int(c.sequenceLen)
        compartments = int(c.nCompartments)
        match = np.zeros(len(positions), dtype=bool)
        for target in targets:
            p = positions.copy()
            p[:, 0] = ((p[:, 0] - target[0]) % seq_len) + target[0]
            match |= np.all(p == target, axis=1)
        ids = np.flatnonzero(match)
        if ids.size == 0:
            return
        states = np.asarray(self.states).reshape(-1)
        transcript_ids = ids[states[ids] > self.activelyTranscribingValue]
        aborted = (
            list(np.asarray(t.abortedTranscripts).reshape((-1, 2)))
            if np.size(t.abortedTranscripts)
            else []
        )
        # MATLAB linear indexing returns scalars from either row or column vectors.
        progress_values = np.asarray(t.boundTranscriptProgress).reshape(
            -1, order="F"
        )
        unit_values = np.asarray(t.boundTranscriptionUnits).reshape(
            -1, order="F"
        )
        for i in transcript_ids:
            progress = progress_values[i]
            if progress > 1:
                aborted.append([unit_values[i], progress - 1])
        t.abortedTranscripts = np.asarray(aborted).reshape((-1, 2))
        states[ids] = (
            self.notExistValue if proteinIsDegraded else self.freeValue
        )
        self.states = states.reshape(np.shape(self.states), order="F")
        self.positionStrands[ids, :] = 0
        t.boundTranscriptionUnits[ids] = t.nullTranscriptValue
        t.boundTranscriptProgress[ids] = t.nullTranscriptValue
        t.boundTranscriptChromosome[ids] = t.nullTranscriptValue

    @property
    def nActive(self) -> Any:
        return self._count(
            np.asarray(self.states) >= self.activelyTranscribingValue
        )

    @property
    def nSpecificallyBound(self) -> Any:
        return self._count(
            np.asarray(self.states) == self.specificallyBoundValue
        )

    @property
    def nNonSpecificallyBound(self) -> Any:
        return self._count(
            np.asarray(self.states) == self.nonSpecificallyBoundValue
        )

    @property
    def nFree(self) -> Any:
        return self._count(np.asarray(self.states) == self.freeValue)

    def _count(self, mask: np.ndarray) -> Any:
        if mask.ndim < 3:
            return np.sum(mask)
        return np.sum(mask, axis=0, keepdims=True)

    @property
    def stateOccupancies(self) -> np.ndarray:
        states = np.asarray(self.states)
        nt = states.shape[2] if states.ndim > 2 else 1
        counts = np.stack(
            [
                np.sum(states >= self.activelyTranscribingValue, axis=0),
                np.sum(states == self.specificallyBoundValue, axis=0),
                np.sum(states == self.nonSpecificallyBoundValue, axis=0),
                np.sum(states == self.freeValue, axis=0),
            ],
            axis=0,
        )
        counts = (
            counts.reshape((4, -1, nt))
            .sum(axis=1, keepdims=True)
            .astype(float)
        )
        denom = counts.sum(axis=0, keepdims=True)
        return np.divide(
            counts, denom, out=np.full_like(counts, np.nan), where=denom != 0
        )
