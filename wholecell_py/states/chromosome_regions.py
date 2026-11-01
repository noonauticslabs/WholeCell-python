"""Circular chromosome region algebra translated from MATLAB ``Chromosome``.

Coordinates are one based, strands are MATLAB compartment IDs, and interval
lengths are signed only for ``splitRegions`` (where sign selects direction).
"""

from __future__ import annotations

import numpy as np


class ChromosomeRegions:
    """Mixin for operations on chromosome ``[position, strand]`` regions.

    The host class supplies ``sequenceLen`` and ``nCompartments``.
    """

    def _region_inputs(self, positions, lengths):
        p = np.asarray(positions, dtype=int).reshape((-1, 2)).copy()
        l = np.asarray(lengths, dtype=int).reshape(-1).copy()
        if len(l) == 1:
            l = np.repeat(l, len(p))
        if len(p) != len(l):
            raise ValueError(
                "positions and lengths must have the same row count"
            )
        return p, l

    def splitOverOriC(self, posStrnds, lens):
        p, l = self._region_inputs(posStrnds, lens)
        ix = np.flatnonzero(
            p[:, 0] + l - 1 > int(np.asarray(self.sequenceLen).item())
        )
        if ix.size:
            extra = np.column_stack((np.ones(ix.size, dtype=int), p[ix, 1]))
            extra_l = (
                p[ix, 0] + l[ix] - int(np.asarray(self.sequenceLen).item()) - 1
            )
            l[ix] = int(np.asarray(self.sequenceLen).item()) - p[ix, 0] + 1
            p = np.vstack((p, extra))
            l = np.r_[l, extra_l]
        return p, l

    def joinSplitOverOriCRegions(self, posStrnds, lens):
        p, l = self._region_inputs(posStrnds, lens)
        ends = p[:, 0] + l - 1
        for strand in np.unique(p[:, 1]):
            a = np.flatnonzero((p[:, 0] == 1) & (p[:, 1] == strand))
            b = np.flatnonzero(
                (ends == int(np.asarray(self.sequenceLen).item()))
                & (p[:, 1] == strand)
            )
            if not len(a) or not len(b) or a[0] == b[0]:
                continue
            ends[b[0]] += ends[a[0]] - p[a[0], 0] + 1
            p = np.delete(p, a[0], axis=0)
            ends = np.delete(ends, a[0])
        return p, ends - p[:, 0] + 1

    def joinSplitRegions(self, posStrnds, lens):
        p, l = self._region_inputs(posStrnds, lens)
        if not len(p):
            return p.reshape((0, 2)), l
        p[:, 0] = (p[:, 0] - 1) % int(np.asarray(self.sequenceLen).item()) + 1
        order = np.lexsort((p[:, 0], p[:, 1]))
        p, l = p[order], l[order]
        starts, ends, strands = p[:, 0].copy(), p[:, 0] + l - 1, p[:, 1]
        keep = np.ones(len(p), dtype=bool)
        for strand in np.unique(strands):
            ids = np.flatnonzero(strands == strand)
            for prev, cur in zip(ids[:-1], ids[1:]):
                if ends[prev] + 1 >= starts[cur]:
                    starts[cur] = starts[prev]
                    ends[cur] = max(ends[prev], ends[cur])
                    keep[prev] = False
            if len(ids) >= 2:
                first = ids[np.flatnonzero(keep[ids])[0]]
                last = ids[-1]
                if ends[last] + 1 >= starts[first] + int(
                    np.asarray(self.sequenceLen).item()
                ):
                    ends[last] = int(np.asarray(self.sequenceLen).item())
                    starts[first] = 1
        return (
            np.column_stack((starts[keep], strands[keep])),
            ends[keep] - starts[keep] + 1,
        )

    def splitRegions(self, rgnPosStrnds, rgnLens, splitLen):
        p, lens = self._region_inputs(rgnPosStrnds, rgnLens)
        if splitLen <= 0:
            raise ValueError("splitLen must be positive")
        out, lengths = [], []
        for (pos, strand), length in zip(p, lens):
            n = abs(int(length)) // splitLen
            for k in range(n):
                out.append(
                    (pos + (1 if length > 0 else -1) * k * splitLen, strand)
                )
                lengths.append(splitLen)
        return np.asarray(out, dtype=int).reshape((-1, 2)), np.asarray(
            lengths, dtype=int
        )

    def intersectRegions(self, posStrndsA, lensA, posStrndsB, lensB):
        pa, la = self.splitOverOriC(posStrndsA, lensA)
        pb, lb = self.splitOverOriC(posStrndsB, lensB)
        pa[:, 0] = (pa[:, 0] - 1) % int(
            np.asarray(self.sequenceLen).item()
        ) + 1
        pb[:, 0] = (pb[:, 0] - 1) % int(
            np.asarray(self.sequenceLen).item()
        ) + 1
        oa, ob = np.lexsort((pa[:, 1], pa[:, 0])), np.lexsort(
            (pb[:, 1], pb[:, 0])
        )
        pa, la, pb, lb = pa[oa], la[oa], pb[ob], lb[ob]
        out, lengths = [], []
        for strand in range(1, int(np.asarray(self.nCompartments).item()) + 1):
            aa = [
                (int(p[0]), int(l)) for p, l in zip(pa, la) if p[1] == strand
            ]
            bb = [
                (int(p[0]), int(l)) for p, l in zip(pb, lb) if p[1] == strand
            ]
            i = j = 0
            while i < len(aa) and j < len(bb):
                a, al = aa[i]
                b, bl = bb[j]
                if a <= b:
                    if a + al > b:
                        out.append((b, strand))
                        lengths.append(min(a + al, b + bl) - b)
                        if a + al < b + bl:
                            i += 1
                        else:
                            j += 1
                    else:
                        i += 1
                else:
                    if b + bl > a:
                        out.append((a, strand))
                        lengths.append(min(b + bl, a + al) - a)
                        if b + bl < a + al:
                            j += 1
                        else:
                            i += 1
                    else:
                        j += 1
        return self.joinSplitOverOriCRegions(
            np.asarray(out, dtype=int).reshape((-1, 2)),
            np.asarray(lengths, dtype=int),
        )

    def excludeRegions(self, incPosStrnds, incLens, excPosStrnds, excLens):
        ip, il = self.joinSplitOverOriCRegions(incPosStrnds, incLens)
        ep, el = self._region_inputs(excPosStrnds, excLens)
        if len(el) == 1 and len(ep) != 1:
            el = np.repeat(el, len(ep))
        ep, el = self.joinSplitRegions(ep, el)
        out = []
        for (start, strand), length in zip(ip, il):
            end = start + length - 1
            cuts = []
            for (x, s), n in zip(ep, el):
                if s != strand:
                    continue
                for shift in (
                    -int(np.asarray(self.sequenceLen).item()),
                    0,
                    int(np.asarray(self.sequenceLen).item()),
                ):
                    a, b = x + shift, x + shift + n - 1
                    if a <= end and b >= start:
                        cuts.append((max(start, a), min(end, b)))
            cuts.sort()
            cursor = int(start)
            for a, b in cuts:
                if a > cursor:
                    out.append((cursor, int(strand), a - 1))
                cursor = max(cursor, b + 1)
            if cursor <= end:
                out.append((cursor, int(strand), end))
        if not out:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)
        p = np.array([[a, s] for a, s, _ in out], dtype=int)
        l = np.array([b - a + 1 for a, _, b in out], dtype=int)
        p[:, 0] = (p[:, 0] - 1) % int(np.asarray(self.sequenceLen).item()) + 1
        p, l = self.joinSplitOverOriCRegions(p, l)
        p[:, 0] = (p[:, 0] - 1) % int(np.asarray(self.sequenceLen).item()) + 1
        order = np.lexsort((p[:, 0], p[:, 1]))
        return p[order], l[order]

    def excludeOverlappingRegions(
        self,
        idxs,
        newIdxs,
        positionsStrands,
        lengths,
        footprint,
        footprint3Prime,
        footprint5Prime,
        isPositionsStrandFootprintCentroid,
        eitherStrand,
    ):
        idxs, new_idxs = np.asarray(idxs, dtype=int).reshape(-1), np.asarray(
            newIdxs, dtype=int
        ).reshape(-1)
        ps, lens = np.asarray(positionsStrands), np.asarray(lengths).reshape(
            -1
        )
        selected = np.r_[idxs, new_idxs]
        tmp = ps[selected - 1].astype(int, copy=True)
        tl = lens[selected - 1]
        starts = tmp[:, 0] + np.minimum(0, tl + 1)
        if isPositionsStrandFootprintCentroid:
            starts[tmp[:, 1] % 2 == 1] -= footprint5Prime
            starts[tmp[:, 1] % 2 == 0] -= footprint3Prime
        ends = starts + np.abs(tl) - 1 + footprint - 1
        strands = (
            np.ceil(tmp[:, 1] / 2).astype(int) if eitherStrand else tmp[:, 1]
        )
        si, ei, st, owner = (
            starts.tolist(),
            ends.tolist(),
            strands.tolist(),
            [0] * len(idxs) + list(range(1, len(new_idxs) + 1)),
        )
        for i in range(len(starts)):
            if starts[i] < 0:
                si.append(
                    int(starts[i] + int(np.asarray(self.sequenceLen).item()))
                )
                ei.append(
                    int(
                        min(
                            int(np.asarray(self.sequenceLen).item()),
                            ends[i] + int(np.asarray(self.sequenceLen).item()),
                        )
                    )
                )
                st.append(int(strands[i]))
                owner.append(owner[i])
        for i in range(len(si)):
            if ei[i] > int(np.asarray(self.sequenceLen).item()):
                si.append(
                    max(1, si[i] - int(np.asarray(self.sequenceLen).item()))
                )
                ei.append(ei[i] - int(np.asarray(self.sequenceLen).item()))
                st.append(st[i])
                owner.append(owner[i])
        keep = np.ones(len(new_idxs), dtype=bool)
        for i in range(len(owner) - 1, 0, -1):
            if owner[i] <= len(idxs):
                continue
            keep[owner[i] - 1] &= not any(
                si[j] <= si[i] <= ei[j] or si[j] <= ei[i] <= ei[j]
                for j in range(i)
                if st[i] == st[j]
            )
        return idxs, new_idxs[keep]
