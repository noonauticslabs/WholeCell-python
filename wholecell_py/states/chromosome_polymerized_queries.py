"""Sparse region queries for chromosome synthesis and DNA strandedness."""

from __future__ import annotations

import numpy as np

from .chromosome_derived import _intervals


class ChromosomePolymerizedQueries:
    """Queries over polymerized, single stranded, and double stranded DNA.

    The host supplies the sparse ``polymerizedRegions`` matrix and the derived
    region properties ``singleStrandedRegions``, ``doubleStrandedRegions``,
    and ``unpolymerizedRegions``. Coordinates and strand IDs are MATLAB one based.
    """

    def _query_regions(
        self,
        positionsStrands,
        lengths,
        regions,
        returnExtent,
        checkRegionSupercoiled=False,
        allow_full_strand_fast_path=True,
    ):
        p = np.asarray(positionsStrands, dtype=int)
        if p.size == 0:
            return (
                np.zeros(0, bool),
                np.zeros(0, int),
                p.reshape(0, p.shape[-1] if p.ndim else 2),
                np.zeros(0, int),
            )
        if p.ndim == 1:
            p = p.reshape(1, -1)
        p = p.copy()
        lens = np.asarray(lengths, dtype=int).reshape(-1)
        if len(lens) == 1:
            lens = np.repeat(lens, len(p))
        if len(lens) != len(p) or np.any(lens == 0):
            raise ValueError(
                "lengths must be non-zero and align with query rows"
            )
        coords, values = regions.find()
        coords = np.asarray(coords, dtype=int)
        if coords.size == 0:
            coords = np.empty((0, len(regions.siz)), dtype=int)
        elif coords.ndim == 1:
            coords = coords.reshape(1, -1)
        values = np.asarray(values).reshape(-1).astype(int)
        if checkRegionSupercoiled:
            sup, _ = self.supercoiled.find()
            sup = np.asarray(sup, dtype=int)
            if sup.size == 0:
                sup = np.empty((0, len(regions.siz)), int)
            elif sup.ndim == 1:
                sup = sup.reshape(1, -1)
            valid = {tuple(row) for row in sup.tolist()}
            coords_values = [
                (row, val)
                for row, val in zip(coords, values)
                if tuple(row) in valid
            ]
            coords = np.asarray(
                [x[0] for x in coords_values], dtype=int
            ).reshape(-1, coords.shape[1])
            values = np.asarray([x[1] for x in coords_values], dtype=int)
        extents = np.zeros(len(p), dtype=int)
        genome = int(np.asarray(self.sequenceLen).item())
        if (
            allow_full_strand_fast_path
            and np.all(values == genome)
            and coords.shape[1] == 2
        ):
            extents[np.isin(p[:, 1], coords[:, 1])] = lens[
                np.isin(p[:, 1], coords[:, 1])
            ]
        else:
            # Preserve MATLAB's split ordering and continuation guards, including
            # its asymmetric behavior for negative queries crossing the origin.
            starts = (
                (p[:, 0] + np.minimum(0, lens + 1) - 1) % genome + 1
            ).tolist()
            ends = (np.asarray(starts) + np.abs(lens) - 1).tolist()
            owners = list(range(len(p)))
            width = max(coords.shape[1], p.shape[1]) - 1
            region_keys = np.pad(
                coords[:, 1:],
                ((0, 0), (0, width - coords.shape[1] + 1)),
                constant_values=1,
            )
            keys = np.pad(
                p[:, 1:],
                ((0, 0), (0, width - p.shape[1] + 1)),
                constant_values=1,
            ).tolist()
            crossing = [i for i, end in enumerate(ends) if end > genome]
            for i in crossing:
                starts.append(max(1, starts[i] - genome))
                ends.append(ends[i] - genome)
                ends[i] = genome
                owners.append(i)
                keys.append(keys[i])
            region_starts = coords[:, 0]
            region_ends = region_starts + values - 1
            for i, (start, end, owner, key) in enumerate(
                zip(starts, ends, owners, keys)
            ):
                if lens[owner] > 0:
                    if (
                        i >= len(p)
                        and extents[owner] < genome - starts[owner] + 1
                    ):
                        continue
                    candidates = np.flatnonzero(
                        (start >= region_starts)
                        & (start <= region_ends)
                        & np.all(region_keys == key, axis=1)
                    )
                    if candidates.size:
                        extents[owner] += (
                            min(region_ends[candidates[0]], end) - start + 1
                        )
                else:
                    if i >= len(p) and -extents[owner] < ends[owner]:
                        continue
                    candidates = np.flatnonzero(
                        (end >= region_starts)
                        & (end <= region_ends)
                        & np.all(region_keys == key, axis=1)
                    )
                    if candidates.size:
                        extents[owner] -= (
                            end - max(region_starts[candidates[0]], start) + 1
                        )
        flags = extents == lens
        idxs = np.flatnonzero(flags) + 1
        if returnExtent:
            return (
                np.ones(len(p), dtype=bool),
                np.arange(1, len(p) + 1),
                p,
                extents,
            )
        return flags, idxs, p[flags], extents[flags]

    def isRegionPolymerized(
        self,
        positionsStrands,
        lengths,
        returnExtent=False,
        checkRegionStrandedness=None,
        checkRegionSupercoiled=False,
        polymerizedRegions=None,
    ):
        source = (
            polymerizedRegions
            if polymerizedRegions is not None
            else self.polymerizedRegions
        )
        if checkRegionStrandedness is not None:
            if checkRegionStrandedness == self.dnaStrandedness_dsDNA:
                source = self.doubleStrandedRegions
            elif checkRegionStrandedness == self.dnaStrandedness_ssDNA:
                source = self.singleStrandedRegions
        return self._query_regions(
            positionsStrands,
            lengths,
            source,
            returnExtent,
            (
                checkRegionSupercoiled
                if checkRegionStrandedness
                == getattr(self, "dnaStrandedness_dsDNA", None)
                else False
            ),
        )

    def isRegionNotPolymerized(
        self, positionsStrands, lengths, returnExtent=False
    ):
        return self._query_regions(
            positionsStrands,
            lengths,
            self.unpolymerizedRegions,
            returnExtent,
            allow_full_strand_fast_path=False,
        )

    def isRegionSingleStranded(
        self, positionsStrands, lengths, returnExtent=False
    ):
        return self.isRegionPolymerized(
            positionsStrands, lengths, returnExtent, self.dnaStrandedness_ssDNA
        )

    def isRegionDoubleStranded(
        self,
        positionsStrands,
        lengths,
        returnExtent=False,
        checkRegionSupercoiled=False,
    ):
        return self.isRegionPolymerized(
            positionsStrands,
            lengths,
            returnExtent,
            self.dnaStrandedness_dsDNA,
            checkRegionSupercoiled,
        )


def _split_interval(start, length, genome):
    if length <= 0:
        return []
    end = start + length - 1
    if end <= genome:
        return [(start, end)]
    return [(start, genome), (1, end - genome)]


def _merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _covered(intervals, start, length, genome):
    total = 0
    for a, b in _split_interval(start, length, genome):
        total += sum(max(0, min(b, y) - max(a, x) + 1) for x, y in intervals)
    return total


def _contiguous_extent(intervals, start, signed_length, genome):
    """Accessible prefix from the query anchor, stopping at the first gap."""
    remaining = abs(signed_length)
    total = 0
    forward = signed_length > 0
    while remaining:
        containing = next(
            ((a, b) for a, b in intervals if a <= start <= b), None
        )
        if containing is None:
            break
        a, b = containing
        available = b - start + 1 if forward else start - a + 1
        consumed = min(remaining, available)
        total += consumed
        remaining -= consumed
        if not remaining or (b != genome if forward else a != 1):
            break
        start = 1 if forward else genome
    return total
