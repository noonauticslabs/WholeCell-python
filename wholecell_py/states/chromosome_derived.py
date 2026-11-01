"""Derived chromosome fields translated from ``Chromosome.m``.

This mixin keeps genomic coverage coordinate sparse. Coordinates and strand IDs
are MATLAB-style one based; chromosome position (dimension one) is circular.
"""

from __future__ import annotations

import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat


class ChromosomeDerived:
    """Calculation methods for polymerization-derived chromosome fields.

    The host supplies chromosome dimensions, gene/TU metadata, and sparse state
    fields. Region accessibility depends on protein-footprint and DNA-damage
    query methods elsewhere in the MATLAB state and is intentionally explicit
    about that dependency.
    """

    def calcUnpolymerizedRegions(self):
        source = self.polymerizedRegions
        p, lengths = _coords_and_lengths(source)
        ntime = int(source.size(3)) if len(source.siz) >= 3 else 1
        outp, outl = [], []
        for strand in range(1, int(self.nCompartments) + 1):
            for time in range(1, ntime + 1):
                intervals = _intervals(
                    p, lengths, strand, time, int(self.sequenceLen)
                )
                cursor = 1
                for start, end in intervals:
                    if cursor < start:
                        outp.append((cursor, strand, time))
                        outl.append(start - cursor)
                    cursor = max(cursor, end + 1)
                if cursor <= self.sequenceLen:
                    outp.append((cursor, strand, time))
                    outl.append(int(self.sequenceLen) - cursor + 1)
        dims = source.siz.copy()
        return CircularSparseMat(
            _coords_for_dims(outp, len(dims)), outl, dims, [1]
        )

    def calcSingleStrandedRegions(self):
        source = self.polymerizedRegions
        p, lengths = _coords_and_lengths(source)
        ntime = int(source.size(3)) if len(source.siz) >= 3 else 1
        outp, outl = [], []
        for strand in range(1, int(self.nCompartments) + 1):
            opposite = strand + 1 if strand % 2 else strand - 1
            for time in range(1, ntime + 1):
                own = _intervals(
                    p, lengths, strand, time, int(self.sequenceLen)
                )
                paired = _intervals(
                    p, lengths, opposite, time, int(self.sequenceLen)
                )
                for start, end in _subtract_intervals(own, paired):
                    outp.append((start, strand, time))
                    outl.append(end - start + 1)
        result = CircularSparseMat(
            _coords_for_dims(outp, len(source.siz)), outl, source.siz, [1]
        )
        return self.mergeAdjacentRegions(result)

    def calcDoubleStrandedRegions(self):
        source = self.polymerizedRegions
        p, lengths = _coords_and_lengths(source)
        ntime = int(source.size(3)) if len(source.siz) >= 3 else 1
        outp, outl = [], []
        for strand in range(1, int(self.nCompartments) + 1):
            opposite = strand + 1 if strand % 2 else strand - 1
            for time in range(1, ntime + 1):
                own = _intervals(
                    p, lengths, strand, time, int(self.sequenceLen)
                )
                paired = _intervals(
                    p, lengths, opposite, time, int(self.sequenceLen)
                )
                for start, end in _intersect_intervals(own, paired):
                    outp.append((start, strand, time))
                    outl.append(end - start + 1)
        result = CircularSparseMat(
            _coords_for_dims(outp, len(source.siz)), outl, source.siz, [1]
        )
        return self.mergeAdjacentRegions(result)

    def calcGeneCopyNumbers(self):
        return np.sum(np.asarray(self.polymerizedGenes), axis=1).reshape(-1, 1)

    def calcPloidy(self):
        _, lengths = _coords_and_lengths(self.polymerizedRegions)
        return float(np.sum(lengths) / (2 * int(self.sequenceLen)))

    def calcPolymerizedGenes(self):
        return self._polymerized_feature_copies(
            self.gene.startCoordinates, self.gene.strands, self.gene.lengths
        )

    def getTranscriptionUnitCopyNumbers(self):
        return np.sum(
            np.asarray(self.polymerizedTranscriptionUnits), axis=1
        ).reshape(-1, 1)

    def calcPolymerizedTranscriptionUnits(self):
        return self._polymerized_feature_copies(
            self.transcriptionUnitStartCoordinates,
            self.transcriptionUnitStrands,
            self.transcriptionUnitLengths,
        )

    def calcCopyNumbers_Accessible(self):
        return np.sum(np.asarray(self.accessibleGenes), axis=1).reshape(-1, 1)

    def calcTranscriptionUnitCopyNumbers_Accessible(self):
        return np.sum(
            np.asarray(self.accessibleTranscriptionUnits), axis=1
        ).reshape(-1, 1)

    def calcAccessibleGenes(self):
        starts = np.asarray(self.gene.startCoordinates).ravel()
        strands = np.asarray(self.gene.strands).ravel()
        lengths = np.asarray(self.gene.lengths).ravel()
        flags = self.isRegionAccessible(
            np.vstack(
                (
                    np.column_stack((starts, strands)),
                    np.column_stack((starts, strands + 2)),
                )
            ),
            np.tile(lengths, 2),
            [],
            [],
            True,
            [],
            False,
            True,
        )[0]
        return np.asarray(flags).reshape(
            -1, int(np.asarray(self.nCompartments).item()) // 2, order="F"
        )

    def calcAccessibleTranscriptionUnits(self):
        starts = np.asarray(self.transcriptionUnitStartCoordinates).ravel()
        strands = np.asarray(self.transcriptionUnitStrands).ravel()
        lengths = np.asarray(self.transcriptionUnitLengths).ravel()
        flags = self.isRegionAccessible(
            np.vstack(
                (
                    np.column_stack((starts, strands)),
                    np.column_stack((starts, strands + 2)),
                )
            ),
            np.tile(lengths, 2),
            [],
            [],
            True,
            [],
            False,
            True,
        )[0]
        return np.asarray(flags).reshape(
            -1, int(np.asarray(self.nCompartments).item()) // 2, order="F"
        )

    def calcLinkingNumbers_minFreeEnergy(self):
        p, lengths = _coords_and_lengths(self.doubleStrandedRegions)
        return CircularSparseMat(
            p,
            lengths / float(self.relaxedBasesPerTurn),
            self.linkingNumbers.siz,
            [1],
        )

    def calcSupercoils(self):
        return self.linkingNumbers - self.linkingNumbers_minFreeEnergy

    def calcSuperhelicalDensity(self):
        lk0 = self.linkingNumbers_minFreeEnergy
        p, deltas = (self.linkingNumbers - lk0).find()
        denominators = lk0.get_coords(p).reshape(-1)
        return CircularSparseMat(
            p, deltas.reshape(-1) / denominators, lk0.siz, [1]
        )

    def calcSupercoiled(self):
        p, lengths = _coords_and_lengths(self.doubleStrandedRegions)
        if not len(p):
            return CircularSparseMat(
                np.empty((0, len(self.linkingNumbers.siz)), int),
                [],
                self.linkingNumbers.siz[:2],
                [1],
            )
        query = _coords_for_dims(p.tolist(), len(self.linkingNumbers.siz))
        lk = self.linkingNumbers.get_coords(query).reshape(-1)
        lk0 = lengths / float(self.relaxedBasesPerTurn)
        sigma = (lk - lk0) / lk0
        flags = (
            np.abs(sigma - self.equilibriumSuperhelicalDensity)
            < self.supercoiledSuperhelicalDensityTolerance
        )
        return CircularSparseMat(
            p[:, :2], flags, self.linkingNumbers.siz[:2], [1]
        )

    def _polymerized_feature_copies(self, starts, strands, lengths):
        starts = np.asarray(starts, dtype=int).reshape(-1)
        strands = np.asarray(strands, dtype=int).reshape(-1)
        lengths = np.asarray(lengths, dtype=int).reshape(-1)
        if not (len(starts) == len(strands) == len(lengths)):
            raise ValueError(
                "feature coordinates, strands, and lengths must align"
            )
        p, covered = _coords_and_lengths(self.polymerizedRegions)
        result = np.zeros((len(starts), 2), dtype=bool)
        for copy in range(2):
            for i, (start, strand, length) in enumerate(
                zip(starts, strands, lengths)
            ):
                result[i, copy] = _is_covered(
                    p,
                    covered,
                    int(start),
                    int(strand + 2 * copy),
                    int(length),
                    self.sequenceLen,
                )
        return result


def _coords_and_lengths(matrix):
    p, vals = matrix.find()
    p = np.asarray(p, dtype=int)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return p, np.asarray(vals).reshape(-1).astype(int)


def _coords_for_dims(rows, ndim):
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0:
        return np.empty((0, ndim), dtype=int)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if ndim == 2:
        return rows[:, :2]
    if rows.shape[1] < ndim:
        rows = np.column_stack(
            (rows, np.ones((len(rows), ndim - rows.shape[1]), dtype=int))
        )
    return rows[:, :ndim]


def _intervals(coords, lengths, strand, time, genome_len):
    pieces = []
    for row, length in zip(coords, lengths):
        if int(row[1]) != strand or (len(row) >= 3 and int(row[2]) != time):
            continue
        if length <= 0:
            continue
        start = int(row[0])
        end = start + int(length) - 1
        if end <= genome_len:
            pieces.append((start, end))
        else:
            pieces.extend(((start, genome_len), (1, end - genome_len)))
    pieces.sort()
    merged = []
    for start, end in pieces:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(source, cuts):
    out = []
    for start, end in source:
        cursor = start
        for a, b in cuts:
            if b < cursor:
                continue
            if a > end:
                break
            if a > cursor:
                out.append((cursor, min(end, a - 1)))
            cursor = max(cursor, b + 1)
            if cursor > end:
                break
        if cursor <= end:
            out.append((cursor, end))
    return out


def _intersect_intervals(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo <= hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _is_covered(coords, lengths, start, strand, length, genome_len):
    if length <= 0:
        return False
    intervals = _intervals(coords, lengths, strand, 1, int(genome_len))
    end = start + length - 1
    query = [(start, min(end, genome_len))]
    if end > genome_len:
        query.append((1, end - genome_len))
    return all(
        any(a <= q0 and b >= q1 for a, b in intervals) for q0, q1 in query
    )
