"""Chromosome strand-break calculations translated from ``Chromosome.m``."""

from __future__ import annotations

import numpy as np

from ..circular_sparsemat import CircularSparseMat


class ChromosomeBreaks:
    """Derived strand-break views and segment classification."""

    def calcDoubleStrandBreaks(self):
        if int(np.asarray(self.doubleStrandBreakSeparation).item()) != 1:
            raise ValueError(
                "Simulation only valid for doubleStrandBreakSeparation equal 1"
            )
        breaks = self.strandBreaks
        # MATLAB: [value(:,1) value(:,1) value(:,2) value(:,2)]
        coords, vals = breaks.find()
        coords = np.asarray(coords, int)
        occupied = {
            tuple(row)
            for row, value in zip(coords, np.asarray(vals).ravel())
            if value
        }
        selected = [
            row
            for row in coords
            if tuple(
                [row[0], row[1] + 1 if row[1] % 2 else row[1] - 1, *row[2:]]
            )
            in occupied
        ]
        return CircularSparseMat(
            np.asarray(selected, int).reshape((-1, len(breaks.siz))),
            True,
            breaks.siz,
            breaks.circularDims,
        )

    def calcSingleStrandBreaks(self):
        value = self.strandBreaks.copy()
        coords, _ = self.doubleStrandBreaks.find()
        if len(coords):
            value = value.set_coords(coords, 0)
        other = self.getDamagedSites(
            True, False, True, True, False, False, False
        )
        c3, _ = self.shiftCircularSparseMatBond3Prime(other).find()
        c5, _ = self.shiftCircularSparseMatBond5Prime(other).find()
        if len(c3):
            value = value.set_coords(c3, 0)
        if len(c5):
            value = value.set_coords(c5, 0)
        return CircularSparseMat(
            value.subs,
            np.ones(len(value.vals), dtype=bool),
            value.siz,
            getattr(value, "circularDims", [1]),
        )

    def calcStrandBreakClassification(self):
        # Chromosome.m:3435-3540. MATLAB pads the origin-facing end to a
        # whole number of segments, then groups break sites by strand pair,
        # segment, and time. The counts are over fully polymerized duplex
        # segments; all indices below are kept one based until array access.
        genome_length = int(np.asarray(self.sequenceLen).item())
        n_strands = int(np.asarray(self.nCompartments).item())
        segment_length = int(
            np.asarray(
                getattr(self, "strandBreakClassification_segmentLength", 216)
            ).item()
        )
        dsb_separation = int(
            np.asarray(
                getattr(
                    self,
                    "strandBreakClassification_doubleStrandBreakSeparation",
                    10,
                )
            ).item()
        )
        if genome_length < 0 or n_strands < 0 or segment_length <= 0:
            raise ValueError(
                "invalid chromosome dimensions or classification segment length"
            )

        breaks = self.strandBreaks
        n_time = int(breaks.size(3)) if len(breaks.siz) >= 3 else 1
        n_segments = (genome_length + segment_length - 1) // segment_length
        padded_length = n_segments * segment_length
        pad = padded_length - genome_length

        # Count segments where both strands of a chromosome copy contain the
        # complete segment. This reproduces the source's full-segment ranges
        # followed by countUnique(... == 2).
        polymerized_positions, polymerized_lengths = (
            self.polymerizedRegions.find()
        )
        polymerized_positions = np.asarray(polymerized_positions, dtype=int)
        if polymerized_positions.ndim == 1:
            polymerized_positions = polymerized_positions.reshape(1, -1)
        polymerized_lengths = np.asarray(polymerized_lengths).reshape(-1)
        intervals_by_strand: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for row, length in zip(polymerized_positions, polymerized_lengths):
            pos, strand = int(row[0]), int(row[1])
            time = int(row[2]) if len(row) >= 3 else 1
            if length <= 0:
                continue
            # Chromosome.m appends a translated copy only when the interval
            # starts in the padded tail, then clips to padded_length.
            starts = [pos]
            if pos <= pad:
                starts.append(pos + genome_length)
            for start in starts:
                end = min(start + int(length) - 1, padded_length)
                if start <= end:
                    intervals_by_strand.setdefault((strand, time), []).append(
                        (start, end)
                    )

        full_strands = set()
        for (strand, time), intervals in intervals_by_strand.items():
            merged = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            for start, end in merged:
                first_segment = (
                    start - 1 + segment_length - 1
                ) // segment_length + 1
                last_segment = end // segment_length
                for segment in range(first_segment, last_segment + 1):
                    if 1 <= segment <= n_segments:
                        full_strands.add((segment, strand, time))

        duplex_segments = [dict() for _ in range(n_time)]
        for segment, strand, time in full_strands:
            if 1 <= time <= n_time:
                pair = (strand + 1) // 2
                duplex_segments[time - 1].setdefault(
                    (segment, pair), set()
                ).add(strand)
        polymerized_count = np.zeros(n_time, dtype=int)
        for t, segments in enumerate(duplex_segments):
            polymerized_count[t] = sum(
                1
                for (segment, pair), strands in segments.items()
                if {2 * pair - 1, 2 * pair}.issubset(strands)
            )

        break_positions, break_values = breaks.find()
        break_positions = np.asarray(break_positions, dtype=int)
        if break_positions.ndim == 1:
            break_positions = break_positions.reshape(1, -1)
        break_values = np.asarray(break_values).reshape(-1)
        damage_rows = []
        for row, value in zip(break_positions, break_values):
            if not value:
                continue
            pos, strand = int(row[0]), int(row[1])
            time = int(row[2]) if len(row) >= 3 else 1
            candidates = [pos]
            if pos <= pad:
                candidates.append(pos + genome_length)
            for extended_pos in candidates:
                if not 1 <= extended_pos <= padded_length:
                    continue
                segment = (extended_pos - 1) // segment_length + 1
                within = (extended_pos - 1) % segment_length + 1
                pair = (strand + 1) // 2
                damage_rows.append((time, segment, within, strand, pair))

        # MATLAB output rows: NB, SSB, SSB+, 2SSB, DSB, DSB+, DSB++.
        result = np.zeros((7, 1, n_time), dtype=float)
        # find(permute(...)) orders strand fastest, then position, segment,
        # and time. The MATLAB loop consumes contiguous segment/time/strand-pair
        # runs in that order; when multiple chromosome copies have damage at
        # interleaved positions, their rows can split one copy's run. Preserve
        # that source ordering and grouping rather than coalescing by key.
        damage_rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        cursor = 0
        while cursor < len(damage_rows):
            time, segment, _, _, pair = damage_rows[cursor]
            if not (
                1 <= time <= n_time
                and {2 * pair - 1, 2 * pair}.issubset(
                    duplex_segments[time - 1].get((segment, pair), set())
                )
            ):
                # The source tests all remaining damaged regions at once and
                # continues without shrinking subs if any are non-duplex,
                # which can loop forever. Avoid the hang and leave such
                # out-of-invariant states unclassified.
                break
            end = cursor + 1
            while end < len(damage_rows):
                next_time, next_segment, _, _, next_pair = damage_rows[end]
                if (next_time, next_segment, next_pair) != (
                    time,
                    segment,
                    pair,
                ):
                    break
                end += 1
            sites = [(row[2], row[3]) for row in damage_rows[cursor:end]]
            sites.sort(key=lambda item: (item[0], item[1]))
            positions = np.asarray([site[0] for site in sites], dtype=int)
            strands = np.asarray([site[1] for site in sites], dtype=int)
            if len(sites) == 1:
                classification = 2  # SSB
            elif np.all(strands == strands[0]):
                classification = 3  # SSB+
            else:
                dsb_indices = np.flatnonzero(
                    (np.diff(positions) < dsb_separation)
                    & (np.diff(strands) != 0)
                )
                if not len(dsb_indices):
                    classification = 4  # 2SSB
                elif (
                    len(dsb_indices) > 1
                    and positions[dsb_indices[-1]] - positions[dsb_indices[0]]
                    >= dsb_separation
                ):
                    classification = 7  # DSB++
                elif len(dsb_indices) > 1:
                    classification = 6  # DSB+
                else:
                    i = int(dsb_indices[0])
                    adjacent = (
                        i > 0
                        and positions[i] - positions[i - 1] < dsb_separation
                    ) or (
                        i < len(positions) - 2
                        and positions[i + 2] - positions[i + 1]
                        < dsb_separation
                    )
                    classification = 6 if adjacent else 5  # DSB+ / DSB
            result[classification - 1, 0, time - 1] += 1
            cursor = end
        result[0, 0, :] = polymerized_count - np.sum(result[1:, 0, :], axis=0)
        return result
