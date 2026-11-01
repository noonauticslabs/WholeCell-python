"""Sparse chromosome strand views and DNA dry mass translated from Chromosome.m."""

from __future__ import annotations

import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat


HYDROGEN = 1.0079
OXYGEN = 15.9994
AVOGADRO = 6.0221417930e23


class ChromosomeMassView:
    """Mixin for ``getStrandView``, ``mergeAdjacentRegions`` and ``calcDryWeight``.

    Host fields follow the MATLAB Chromosome state. Sparse fields are kept sparse;
    only coordinate rows are traversed. Strand coordinates remain one based.
    """

    _default_strand_fields = (
        "polymerizedRegions",
        "monomerBoundSites",
        "complexBoundSites",
        "gapSites",
        "abasicSites",
        "damagedSugarPhosphates",
        "damagedBases",
        "intrastrandCrossLinks",
        "strandBreaks",
        "hollidayJunctions",
    )

    def getStrandView(self, outputs=None):
        if outputs is None:
            outputs = self._default_strand_fields
        single = isinstance(outputs, str)
        names = [outputs] if single else list(outputs)
        result = [self._strand_view_one(name) for name in names]
        return result[0] if single else tuple(result)

    def _strand_view_one(self, name):
        source = getattr(self, name)
        p, vals = source.find()
        p = _pad_coords(p)
        if name == "polymerizedRegions":
            lengths = vals.reshape(-1).astype(int)
            s1, s2 = map(
                int, np.asarray(self.strandIndexs_negative).reshape(-1)[:2]
            )
            a = [
                (int(row[0]), int(length), tuple(row[2:]))
                for row, length in zip(p, lengths)
                if row[1] == s1
            ]
            b = [
                (int(row[0]), int(length), tuple(row[2:]))
                for row, length in zip(p, lengths)
                if row[1] == s2
            ]
            intersects, merged = [], list(a)
            for start, length, time in b:
                end = start + length - 1
                overlaps = [
                    (x, n, t)
                    for x, n, t in merged
                    if t == time and x <= end and x + n - 1 >= start
                ]
                for x, n, _ in overlaps:
                    lo, hi = max(x, start), min(x + n - 1, end)
                    intersects.append((lo, s2, *time, hi - lo + 1))
                    merged.remove((x, n, time))
                bounds = [(x, n) for x, n, _ in overlaps] + [(start, length)]
                if bounds:
                    merged.append(
                        (
                            min(x for x, _ in bounds),
                            max(x + n - 1 for x, n in bounds)
                            - min(x for x, _ in bounds)
                            + 1,
                            time,
                        )
                    )
            rows = [[x, s1, *t, n] for x, n, t in merged] + intersects
            # Retain any strand/time entries outside the two paired compartments.
            rows.extend(
                [
                    [*row, int(length)]
                    for row, length in zip(p, lengths)
                    if row[1] not in (s1, s2)
                ]
            )
            if rows:
                coords = np.asarray([r[:-1] for r in rows], dtype=int)
                values = np.asarray([r[-1] for r in rows], dtype=int)
            else:
                coords, values = np.empty((0, p.shape[1]), int), np.empty(
                    0, int
                )
            view = CircularSparseMat(
                coords,
                values,
                source.siz,
                getattr(source, "circularDims", [1]),
            )
            return self.mergeAdjacentRegions(view)

        # Other occupancy views reflect sites on the second negative strand into
        # the first, limited to polymerized coverage. This mirrors MATLAB's row
        # traversal, including its historical final-row-only loop bound.
        rp, rv = self.polymerizedRegions.find()
        rp = _pad_coords(rp)
        rlengths = rv.reshape(-1)
        if not len(rp):
            return CircularSparseMat(
                p, vals, source.siz, getattr(source, "circularDims", [1])
            )
        j = (
            len(rp) - 1
        )  # MATLAB source uses `for j = size(...,1)`; preserve oddity.
        row = rp[j]
        end = row[0] + rlengths[j] - 1
        mask = (
            (p[:, 0] >= row[0])
            & (p[:, 0] <= end)
            & (p[:, 1] == self.strandIndexs_negative[0])
            & np.all(p[:, 2:] == row[2:], axis=1)
        )
        p[mask, 1] = self.strandIndexs_negative[1]
        return CircularSparseMat(
            p, vals, source.siz, getattr(source, "circularDims", [1])
        )

    def mergeAdjacentRegions(self, polymerizedRegions):
        p, lengths = polymerizedRegions.find()
        p = _pad_coords(p)
        lengths = lengths.reshape(-1).astype(int)
        if len(lengths) <= 1:
            return polymerizedRegions
        order = np.lexsort(tuple(p[:, i] for i in range(p.shape[1])))
        p, lengths = p[order], lengths[order]
        if np.any(lengths < 0) or any(
            p[i + 1, 0] - p[i, 0] < lengths[i]
            and np.array_equal(p[i + 1, 1:], p[i, 1:])
            for i in range(len(p) - 1)
        ):
            raise ValueError("polymerizedRegions is corrupt")
        outp, outl = [], []
        for row, length in zip(p, lengths):
            if (
                outp
                and row[0] == outp[-1][0] + outl[-1]
                and np.array_equal(row[1:], outp[-1][1:])
            ):
                outl[-1] += int(length)
            else:
                outp.append(row.copy())
                outl.append(int(length))
        return CircularSparseMat(
            np.asarray(outp, dtype=int).reshape(-1, p.shape[1]),
            np.asarray(outl),
            polymerizedRegions.siz,
            getattr(polymerizedRegions, "circularDims", [1]),
        )

    def calcDryWeight(self):
        time_count = int(self.abasicSites.size(3))
        sequence = self.sequence
        length_genome = int(self.sequenceLen)
        counts = np.zeros((4, time_count), dtype=float)
        bonds = np.zeros(time_count, dtype=float)
        p, lengths = self.getStrandView("polymerizedRegions").find()
        p = _pad_coords(p)
        for row, length in zip(p, lengths.reshape(-1).astype(int)):
            t = int(row[2]) - 1 if p.shape[1] > 2 else 0
            strand = int(row[1])
            positions = np.arange(row[0], row[0] + length)
            counts[:, t] += _base_counts(
                sequence,
                positions if length != length_genome else None,
                strand,
            )
            bonds[t] += length - 1
            if length == length_genome or (
                row[0] + length - 1 == length_genome
                and np.any(
                    np.all(
                        p[:, :3] == [1, strand, row[2] if len(row) > 2 else 1],
                        axis=1,
                    )
                )
            ):
                bonds[t] += 1
        weights = np.asarray(
            _field(self.metabolite, "molecularWeights")
        ).reshape(-1, order="F")
        dnmp = _one_index(_field(self.metabolite, "dnmpIndexs"))
        value = weights[dnmp] @ counts - (HYDROGEN + OXYGEN) * bonds
        for k in range(time_count):
            value[k] -= (
                weights[_one_index(_field(self.metabolite, "dr5pIndexs"))][0]
                - 2 * HYDROGEN
            ) * _collapse(self.gapSites, k + 1)
            pos, vals = _at_time(self.abasicSites, k + 1)
            value[k] -= (
                len(vals)
                * weights[_one_index(_field(self.metabolite, "waterIndexs"))][
                    0
                ]
            )
            if len(pos):
                value[k] -= (
                    sequence.subsequenceBaseCounts(pos[:, :2]).ravel()
                    @ weights[dnmp]
                )
            pos, vals = _at_time(self.damagedSugarPhosphates, k + 1)
            value[k] += (
                weights[vals.astype(int) - 1].sum()
                - len(vals)
                * weights[_one_index(_field(self.metabolite, "dr5pIndexs"))][0]
            )
            pos, vals = _at_time(self.damagedBases, k + 1)
            if len(vals):
                value[k] += (
                    weights[vals.astype(int) - 1].sum()
                    - sequence.subsequenceBaseCounts(pos[:, :2]).ravel()
                    @ weights[
                        _one_index(
                            _field(self.metabolite, "unmodifiedBaseIndexs")
                        )
                    ]
                )
            pos, vals = _at_time(self.intrastrandCrossLinks, k + 1)
            if len(vals):
                positions = np.column_stack((pos[:, 0], pos[:, 0] + 1))
                strands = np.repeat(pos[:, 1:2], 2, axis=1)
                value[k] += (
                    weights[vals.astype(int) - 1].sum()
                    - sequence.subsequenceBaseCounts(
                        positions, strands
                    ).ravel()
                    @ weights[dnmp]
                )
            value[k] += (HYDROGEN + OXYGEN) * _collapse(
                self.strandBreaks, k + 1
            )
        shape = (1, 1) if time_count == 1 else (1, 1, time_count)
        return np.asarray(value).reshape(shape) / AVOGADRO


def _pad_coords(coords):
    p = np.asarray(coords, dtype=int)
    if p.size == 0:
        return np.empty(
            (0, max(3, p.shape[1] if p.ndim == 2 else 3)), dtype=int
        )
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.shape[1] < 3:
        p = np.column_stack((p, np.ones((len(p), 3 - p.shape[1]), dtype=int)))
    return p


def _one_index(value):
    return np.asarray(value, dtype=int).reshape(-1, order="F") - 1


def _at_time(mat, time):
    p, v = mat.find()
    p = _pad_coords(p)
    mask = p[:, 2] == time
    return p[mask], np.asarray(v).ravel()[mask]


def _collapse(mat, time):
    p, v = mat.find()
    p = _pad_coords(p)
    return np.asarray(v).ravel()[p[:, 2] == time].sum()


def _sequence_counts(sequence, positions, strands=None):
    pos = np.asarray(positions, dtype=int)
    if pos.size == 0:
        return np.zeros((0, 4), dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 1)
    if pos.shape[1] == 2:
        strings = [
            sequence.subsequence(pos[:, 0], pos[:, 1]),
            sequence.subsequence(pos[:, 1], pos[:, 1]),
        ]
        return np.sum(
            [_count_string(s) for s in strings], axis=0, keepdims=True
        )
    text = sequence.subsequence(
        pos.reshape(-1), 1 if strands is None else strands
    )
    return _count_string(text).reshape(1, 4)


def _base_counts(sequence, positions, strand):
    text = (
        sequence.strand(strand)
        if positions is None
        else sequence.subsequence(positions, strand)
    )
    return _count_string(text)


def _count_string(value):
    return np.asarray([value.count(x) for x in "ACGT"], dtype=float)


def _field(obj, name):
    if isinstance(obj, dict):
        return obj[name]
    fields = getattr(obj, "fields", None)
    if fields is not None and name in fields:
        return fields[name]
    return getattr(obj, name)
