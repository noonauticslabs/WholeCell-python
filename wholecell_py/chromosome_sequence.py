"""Circular, double-stranded chromosome sequence with MATLAB-style coordinates.

Coordinates passed to the explicit methods and ``sequence[row, strand]`` are
one-based. Strand columns 1/3 are the positive strand; 2/4 are its basewise
complement. The negative strand follows the same coordinate direction, as in
the source MATLAB class (complement, without reversing).
"""

from __future__ import annotations

from typing import Any

import numpy as np


_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "GC",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}


def _as_sequence(value: Any) -> str:
    if isinstance(value, str):
        sequence = value
    else:
        array = np.asarray(value)
        if array.dtype.kind in "US":
            if array.ndim > 2 or (array.ndim == 2 and min(array.shape) > 1):
                raise ValueError("positive_strand must be a character vector")
            sequence = "".join(str(x) for x in array.reshape(-1, order="F"))
        else:
            raise TypeError(
                "positive_strand must be a string or character array"
            )
    if any(base not in "ACGT" for base in sequence):
        raise ValueError("positive_strand must contain only A, C, G, and T")
    return sequence


class ChromosomeSequence:
    """Immutable view of the four MATLAB chromosome/strand columns."""

    def __init__(self, positive_strand: Any):
        self.positive_strand = _as_sequence(positive_strand)
        if not self.positive_strand:
            raise ValueError("positive_strand must not be empty")

    @property
    def positive_strand(self) -> str:
        return self._positive_strand

    @positive_strand.setter
    def positive_strand(self, sequence: str) -> None:
        # Keep the historically public, assignable attribute coherent with the
        # lookup arrays and summaries. Strings are immutable, so the derived
        # values remain valid until the attribute is reassigned.
        self._positive_strand = sequence
        self._negative_strand = sequence.translate(_COMPLEMENT)
        self._positive_bases = np.asarray(list(sequence), dtype="U1")
        self._negative_bases = np.asarray(
            list(self._negative_strand), dtype="U1"
        )
        self._positive_base_counts = np.asarray(
            [sequence.count(base) for base in "ACGT"], dtype=np.int64
        )

    def __setstate__(self, state) -> None:
        """Rebuild derived arrays when loading older checkpoint pickles."""
        self.__dict__.update(state)
        sequence = self.__dict__.get("_positive_strand")
        if sequence is not None:
            self.positive_strand = _as_sequence(sequence)

    @property
    def positive_strand_sequence(self) -> str:
        return self.positive_strand

    @property
    def negative_strand(self) -> str:
        return self._negative_strand

    @property
    def negative_strand_sequence(self) -> str:
        return self.negative_strand

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), 4)

    def __len__(self) -> int:
        return len(self.positive_strand)

    def size(self, dimension: int | None = None):
        if dimension is None:
            return self.shape
        if not isinstance(dimension, (int, np.integer)) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        return self.shape[dimension - 1] if dimension <= 2 else 1

    def _positions(self, positions: Any) -> np.ndarray:
        p = np.asarray(positions)
        if p.dtype.kind not in "iu" and (
            p.dtype.kind != "f"
            or not np.all(np.isfinite(p))
            or not np.all(p == np.floor(p))
        ):
            raise IndexError("positions must be finite integers")
        p = p.astype(np.int64)
        return np.mod(p - 1, len(self))

    def _subsequence_bases(
        self, positions: Any, strands: Any = None
    ) -> np.ndarray:
        """Return selected bases as an array, without joining a Python string."""
        if strands is None:
            p = np.asarray(positions)
            if p.size == 0:
                return np.empty((0,), dtype="U1")
            if p.ndim == 2 and p.shape[-1] == 2:
                positions, strands = p[..., 0], p[..., 1]
            else:
                strands = 1
        p = np.asarray(positions)
        s = np.asarray(strands)
        try:
            p, s = np.broadcast_arrays(p, s)
        except ValueError as exc:
            raise ValueError(
                "positions and strands must be broadcast-compatible"
            ) from exc
        idx = self._positions(p)
        negative = np.mod(s, 2) == 0
        return np.where(
            negative, self._negative_bases[idx], self._positive_bases[idx]
        )

    def strand(self, strand: Any = 1) -> str:
        """Return one strand by MATLAB column number (wrapped over 1..4)."""
        col = np.asarray(strand)
        if col.size != 1:
            raise ValueError("strand() accepts one strand number")
        n = int(col.item())
        return self.negative_strand if n % 2 == 0 else self.positive_strand

    def circularposition(self, positions: Any) -> np.ndarray | int:
        """Wrap one-based coordinates into the chromosome; scalar stays scalar."""
        p = np.asarray(positions)
        wrapped = self._positions(p) + 1
        return int(wrapped) if wrapped.ndim == 0 else wrapped

    def subsequence(self, positions: Any, strands: Any = None):
        """Fetch wrapped positions; optional strand values use MATLAB parity."""
        bases = self._subsequence_bases(positions, strands)
        if bases.ndim == 0:
            return str(bases.item())
        if bases.ndim == 1:
            return "".join(bases.tolist())
        return bases

    def basecount(self, strand: int | None = None) -> np.ndarray:
        """Counts in A,C,G,T order, for either or both strand orientations."""
        positive = self._positive_base_counts
        negative = positive[[3, 2, 1, 0]]
        if strand is None:
            return positive + negative
        return (negative if int(strand) % 2 == 0 else positive).copy()

    def get_base_counts(self, strand: int | None = None) -> np.ndarray:
        return self.basecount(strand)

    def getBaseCounts(self, strand=None):
        if strand is None:
            return self.basecount().reshape(4, 1)
        columns = [
            self.basecount(int(s)) for s in np.asarray(strand).ravel(order="F")
        ]
        return (
            np.stack(columns, axis=1)
            if columns
            else np.empty((4, 0), dtype=np.int64)
        )

    def getGCContent(self):
        return (
            self._positive_base_counts[1] + self._positive_base_counts[2]
        ) / len(self)

    def subsequenceBaseCounts(self, positions, strands=None):
        bases = self._subsequence_bases(positions, strands)
        return np.asarray(
            [np.count_nonzero(bases == base) for base in "ACGT"],
            dtype=np.int64,
        ).reshape(4, 1)

    def findSubsequence(self, subsequence, subposition, bothStrands=False):
        """Find circular recognition motifs at a one-based motif position.

        The returned coordinate is the requested ``subposition`` within each
        match, matching ChromosomeSequence.m's use of Bioinformatics ``restrict``.
        IUPAC ambiguity symbols in knowledge-base motifs are supported.
        """
        motif = (
            _as_sequence(subsequence)
            if not isinstance(subsequence, str)
            else subsequence
        )
        if not motif or np.asarray(subposition).size == 0 or not len(self):
            return (
                np.empty((0, 2), dtype=np.int64)
                if bothStrands
                else np.empty(0, dtype=np.int64)
            )
        anchor_values = np.asarray(subposition)
        if anchor_values.size != 1 or anchor_values.dtype.kind not in "iuf":
            raise ValueError("subposition must be a scalar integer")
        anchor = float(anchor_values.item())
        if (
            not np.isfinite(anchor)
            or anchor != np.floor(anchor)
            or anchor < 1
            or anchor > len(motif)
        ):
            raise ValueError("subposition must be within the motif")
        anchor = int(anchor)
        if any(base not in _IUPAC for base in motif):
            raise ValueError("motif contains an unsupported nucleotide code")

        def matches(sequence, query):
            allowed = [_IUPAC[base] for base in query]
            output = []
            for start in range(len(self)):
                if all(
                    sequence[(start + offset) % len(self)] in bases
                    for offset, bases in enumerate(allowed)
                ):
                    output.append((start + anchor - 1) % len(self) + 1)
            return output

        positive = matches(self.positive_strand, motif)
        if not bothStrands:
            return np.asarray(positive, dtype=np.int64)
        negative = matches(self.negative_strand, motif[::-1])
        rows = [(position, 1) for position in positive]
        rows.extend((position, 2) for position in negative)
        return np.asarray(rows, dtype=np.int64).reshape(-1, 2)

    def __getitem__(self, key):
        """Read indexing with one-based MATLAB rows and wrapped strand columns."""
        if isinstance(key, tuple) and len(key) == 2:
            rows, cols = key
            row_values = (
                np.arange(1, len(self) + 1)[rows]
                if isinstance(rows, slice)
                else rows
            )
            col_values = (
                np.arange(1, 5)[cols] if isinstance(cols, slice) else cols
            )
            row_values = np.asarray(row_values)
            col_values = np.asarray(col_values)
            if row_values.ndim > 1 or col_values.ndim > 1:
                raise IndexError("row and column subscripts must be vectors")
            row_values = row_values.reshape(-1)
            col_values = col_values.reshape(-1)
            if not row_values.size or not col_values.size:
                return np.empty((row_values.size, col_values.size), dtype="U1")
            row_idx = self._positions(row_values)
            positive = np.broadcast_to(
                self._positive_bases[row_idx, None],
                (row_idx.size, col_values.size),
            ).copy()
            negative_cols = np.mod(col_values, 2) == 0
            if np.any(negative_cols):
                positive[:, negative_cols] = self._negative_bases[
                    row_idx, None
                ]
            scalar_subscripts = (
                np.ndim(rows) == 0
                and np.ndim(cols) == 0
                and not isinstance(rows, slice)
                and not isinstance(cols, slice)
            )
            if scalar_subscripts:
                return str(positive[0, 0])
            return positive
        if isinstance(key, tuple):
            raise IndexError("ChromosomeSequence supports at most two indices")
        # MATLAB linear indexing traverses the Lx4 array column-major.
        indices = np.asarray(key)
        if indices.dtype.kind == "b":
            raise IndexError("logical indexing is not supported")
        if indices.dtype.kind not in "iu" and (
            indices.dtype.kind != "f"
            or not np.all(np.isfinite(indices))
            or not np.all(indices == np.floor(indices))
        ):
            raise IndexError("linear indices must be finite integers")
        raw = indices.astype(np.int64, copy=False) - 1
        if np.any(raw < 0) or np.any(raw >= 4 * len(self)):
            raise IndexError("linear index outside the 4-column sequence")
        rows = raw % len(self) + 1
        cols = raw // len(self) + 1
        bases = self._subsequence_bases(rows, cols)
        if indices.ndim == 0:
            return str(bases.item())
        if indices.ndim == 1:
            return "".join(bases.tolist())
        return bases.reshape(indices.shape)
