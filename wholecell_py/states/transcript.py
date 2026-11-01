"""Transcript state translated from WholeCell's ``Transcript``."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np


class Transcript:
    optionNames = ("verbosity", "seed")
    fixedConstantNames = (
        "genomeLength",
        "transcriptionUnitFivePrimeCoordinates",
        "transcriptionUnitDirections",
        "transcriptionUnitSequences",
        "transcriptionUnitLengths",
    )
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = (
        "boundTranscriptionUnits",
        "boundTranscriptProgress",
        "boundTranscriptChromosome",
        "abortedTranscripts",
    )
    dependentStateNames = ("rnaBoundRNAPolymerases",)
    nullTranscriptValue = 0
    newTranscriptValue = 1
    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "genomeLength",
        "transcriptionUnitWholeCellModelIDs",
        "transcriptionUnitFivePrimeCoordinates",
        "transcriptionUnitDirections",
        "transcriptionUnitSequences",
        "transcriptionUnitLengths",
        "nmpMolecularWeights",
        "boundTranscriptionUnits",
        "boundTranscriptProgress",
        "boundTranscriptChromosome",
        "abortedTranscripts",
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
        self.chromosome = None
        restored = dict(snapshot or {})
        restored.update(attributes)
        for key, value in restored.items():
            if key in {
                "totalBaseCounts",
                "abortedSequences",
                "rnaMaxRNAPolymeraseState",
                "rnaBoundRNAPolymerases",
                "dryWeight",
            }:
                continue
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
        self.chromosome = _simulation_state(simulation, "Chromosome")

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        if knowledgeBase is None or simulation is None:
            raise ValueError(
                "knowledgeBase and simulation are required to initialize Transcript constants"
            )
        genome = _get(knowledgeBase, "genome")
        tus = _get(knowledgeBase, "transcriptionUnits")
        self.genomeLength = _get(genome, "sequenceLength")
        self.transcriptionUnitWholeCellModelIDs = _column(
            [_get(t, "wholeCellModelID") for t in tus], object
        )
        self.transcriptionUnitFivePrimeCoordinates = _column(
            [_get(t, "fivePrimeCoordinate") for t in tus]
        )
        self.transcriptionUnitDirections = _column(
            [_get(t, "direction") for t in tus]
        )
        self.transcriptionUnitSequences = _column(
            [_get(t, "sequence") for t in tus], object
        )
        self.transcriptionUnitLengths = _column(
            [_get(t, "sequenceLength") for t in tus]
        )
        metabolite = _simulation_state(simulation, "Metabolite")
        self.nmpMolecularWeights = np.asarray(metabolite.molecularWeights)[
            _indices(metabolite.nmpIndexs)
        ]

    def allocateMemory(self, numTimePoints: int) -> None:
        if (
            not isinstance(numTimePoints, (int, np.integer))
            or numTimePoints < 0
        ):
            raise ValueError("numTimePoints must be a non-negative integer")
        n = int(numTimePoints)
        self.boundTranscriptionUnits = np.zeros((0, 1, n), order="F")
        self.boundTranscriptProgress = np.zeros((0, 1, n), order="F")
        self.boundTranscriptChromosome = np.zeros((0, 1, n), order="F")
        self.abortedTranscripts = np.zeros((0, 2, n), order="F")

    def initialize(self) -> None:
        self.abortedTranscripts = np.zeros((0, 2), order="F")

    @property
    def abortedSequences(self) -> list[str]:
        rows = np.asarray(self.abortedTranscripts).reshape((-1, 2))
        return [
            self._sequence(row[0], row[1]) if row[0] else "" for row in rows
        ]

    @property
    def totalBaseCounts(self) -> np.ndarray:
        counts = np.zeros((1, 4), dtype=int)
        for seq in self._active_sequences() + [
            s for s in self.abortedSequences if s
        ]:
            counts[0] += [seq.count(n) for n in "ACGU"]
        return counts

    @totalBaseCounts.setter
    def totalBaseCounts(self, value):
        # MATLAB declares a stored property with a custom getter. Preserve the
        # exported backing value for inspection; runtime reads recompute it.
        self._exported_totalBaseCounts = np.asarray(value).copy()

    @property
    def dryWeight(self) -> float:
        seqs = self._active_sequences() + [
            s
            for row, s in zip(
                np.asarray(self.abortedTranscripts), self.abortedSequences
            )
            if row[0] != 0
        ]
        seq = "".join(seqs)
        masses = np.asarray(self.nmpMolecularWeights).reshape(-1, order="F")
        mass = np.asarray([seq.count(n) for n in "ACGU"]) @ masses
        return float(
            (mass - (len(seq) - len(seqs)) * (1.0079 + 15.9994))
            / 6.0221417930e23
        )

    @property
    def rnaMaxRNAPolymeraseState(self) -> np.ndarray:
        bound = _time_array(self.boundTranscriptionUnits)
        lengths = _time_array(self.transcriptionUnitLengths)
        result = np.zeros((len(lengths), 1, bound.shape[2]))
        for i in range(bound.shape[0]):
            for k in range(bound.shape[2]):
                idx = int(bound[i, 0, k])
                if idx:
                    result[idx - 1, 0, k] = max(
                        result[idx - 1, 0, k],
                        lengths[i, 0, min(k, lengths.shape[2] - 1)],
                    )
        return result

    @property
    def rnaBoundRNAPolymerases(self) -> np.ndarray:
        bound = _time_array(self.boundTranscriptionUnits)
        n_tus = np.size(self.transcriptionUnitLengths)
        result = np.zeros((n_tus, 1, bound.shape[2]), dtype=int)
        for k in range(bound.shape[2]):
            result[:, 0, k] = np.bincount(
                bound[:, 0, k].astype(int), minlength=n_tus + 1
            )[1 : n_tus + 1]
        return result

    def _active_sequences(self) -> list[str]:
        bound, progress = _time_array(
            self.boundTranscriptionUnits
        ), _time_array(self.boundTranscriptProgress)
        k = bound.shape[2] - 1
        return [
            self._sequence(
                bound[i, 0, k],
                progress[i, 0, min(k, progress.shape[2] - 1)] - 1,
            )
            for i in range(bound.shape[0])
            if progress[i, 0, min(k, progress.shape[2] - 1)]
            > self.nullTranscriptValue
        ]

    def _sequence(self, tu: Any, length: Any) -> str:
        sequences = np.asarray(
            self.transcriptionUnitSequences, dtype=object
        ).reshape(-1, order="F")
        sequence = sequences[int(tu) - 1]
        if isinstance(sequence, np.ndarray):
            sequence = "".join(sequence.ravel(order="F"))
        return str(sequence)[: max(0, int(length))]


def _time_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    return arr[:, :, None] if arr.ndim == 2 else arr


def _column(values: Any, dtype: Any = float) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape((-1, 1), order="F")


def _indices(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=int).reshape(-1, order="F") - 1


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name)


def _simulation_state(simulation: Any, name: str) -> Any:
    if isinstance(simulation, Mapping):
        return simulation.get("states", {}).get(name)
    getter = getattr(simulation, "state", None)
    if getter is None:
        raise RuntimeError(f"simulation cannot provide state {name!r}")
    return getter(name)
