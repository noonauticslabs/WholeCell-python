"""Chromosome binding, damage, and accessibility queries."""

from __future__ import annotations

import numpy as np


class ChromosomeAccessibility:
    """Sparse chromosome accessibility queries using MATLAB one-based IDs."""

    def getDNAFootprint(self, monomers, complexs):
        def values(name, ids):
            x = np.asarray(getattr(self, name))
            ids = np.asarray(ids, dtype=int)
            return (
                x.reshape(-1, order="F")[ids - 1]
                if ids.size
                else np.empty((0,), dtype=x.dtype)
            )

        m, c = np.asarray(
            [] if monomers is None else monomers, dtype=int
        ).reshape(-1), np.asarray(
            [] if complexs is None else complexs, dtype=int
        ).reshape(
            -1
        )
        mf, cf = values("monomerDNAFootprints", m), values(
            "complexDNAFootprints", c
        )
        footprint = int(
            max([1, *mf.reshape(-1).tolist(), *cf.reshape(-1).tolist()])
        )
        f3, f5 = self.calculateFootprintOverhangs(footprint)
        bind = np.r_[
            values("monomerDNAFootprintBindingStrandedness", m),
            values("complexDNAFootprintBindingStrandedness", c),
        ]
        region = np.r_[
            values("monomerDNAFootprintRegionStrandedness", m),
            values("complexDNAFootprintRegionStrandedness", c),
        ]

        def single(a):
            if not a.size:
                return self.dnaStrandedness_dsDNA
            if np.any(a != a[0]):
                raise ValueError("Footprints must be of a single strandedness")
            return a[0].item()

        return footprint, f3, f5, single(bind), single(region)

    def getReleasableProteins(self, bindingMonomers, bindingComplexs):
        m, c = (
            np.asarray(
                [] if bindingMonomers is None else bindingMonomers, dtype=int
            ).reshape(-1)
            - 1,
            np.asarray(
                [] if bindingComplexs is None else bindingComplexs, dtype=int
            ).reshape(-1)
            - 1,
        )
        catalytic = np.asarray(self.reactionMonomerCatalysisMatrix)[:, m].sum(
            axis=1
        ) + np.asarray(self.reactionComplexCatalysisMatrix)[:, c].sum(axis=1)
        thresholds = np.asarray(self.reactionThresholds).reshape(-1)
        return self._released(
            self.reactionBoundMonomer, catalytic >= thresholds
        ), self._released(self.reactionBoundComplex, catalytic >= thresholds)

    @staticmethod
    def _released(bound, mask):
        a = np.asarray(bound).reshape(-1)[np.asarray(mask, bool)]
        a = np.unique(a[a != 0]).astype(int)
        return np.sort(a) if a.size else np.array([0], dtype=int)

    def isRegionUndamaged(
        self,
        positionsStrands,
        lengths,
        isEitherStrandDamaged=False,
        ignoreDamageFilter=None,
        returnExtent=False,
        isAnyStrandDamaged=False,
    ):
        p = np.asarray(positionsStrands, dtype=int)
        if p.size == 0:
            return (
                np.zeros(0, bool),
                np.zeros(0, int),
                p.reshape(0, 2),
                np.zeros(0, int),
            )
        p = np.atleast_2d(p)
        lens = np.asarray(lengths, dtype=int).ravel()
        if lens.size == 1:
            lens = np.repeat(lens, len(p))
        damaged, values = self.damagedSites.find()
        if not len(values):
            return np.ones(len(p), bool), np.arange(1, len(p) + 1), p, lens
        width = max(p.shape[1], damaged.shape[1]) - 1
        damage_keys = np.pad(
            damaged[:, 1:],
            ((0, 0), (0, width - damaged.shape[1] + 1)),
            constant_values=1,
        )
        keys = np.pad(
            p[:, 1:], ((0, 0), (0, width - p.shape[1] + 1)), constant_values=1
        )
        if isAnyStrandDamaged:
            keys[:, 0] = 1
            damage_keys[:, 0] = 1
        elif isEitherStrandDamaged:
            keys[:, 0] = (keys[:, 0] + 1) // 2
            damage_keys[:, 0] = (damage_keys[:, 0] + 1) // 2
        ign = np.asarray(
            [] if ignoreDamageFilter is None else ignoreDamageFilter, dtype=int
        )
        if ign.size == 0:
            ign = np.empty((len(p), 0), dtype=int)
        else:
            ign = np.atleast_2d(ign)
            if len(ign) == 1:
                ign = np.repeat(ign, len(p), axis=0)
        genome = int(np.asarray(self.sequenceLen).item())
        starts = (
            (p[:, 0] + np.minimum(0, lens + 1) - 1) % genome + 1
        ).tolist()
        ends = (np.asarray(starts) + np.abs(lens) - 1).tolist()
        owners = list(range(len(p)))
        keys = keys.tolist()
        for i in [j for j, end in enumerate(ends) if end > genome]:
            starts.append(max(1, starts[i] - genome))
            ends.append(ends[i] - genome)
            ends[i] = genome
            owners.append(i)
            keys.append(keys[i])
        ext = np.zeros(len(p), dtype=int)
        for i, (start, end, owner, key) in enumerate(
            zip(starts, ends, owners, keys)
        ):
            sites = damaged[
                (damaged[:, 0] >= start)
                & (damaged[:, 0] <= end)
                & np.all(damage_keys == key, axis=1),
                0,
            ]
            if lens[owner] > 0:
                if i >= len(p) and ext[owner] < genome - starts[owner] + 1:
                    continue
                ignored = (
                    ign[owner]
                    if i < len(p)
                    else ign[owner] - (genome - starts[owner] + 1)
                )
                offsets = np.setdiff1d(sites - start + 1, ignored)
                ext[owner] += (
                    offsets.min() - 1 if offsets.size else end - start + 1
                )
            else:
                ignored = (
                    ign[owner] if i < len(p) else ign[owner] + p[owner, 0]
                )
                offsets = np.setdiff1d(
                    end - sites + 1, ignored - lens[owner] - 1
                )
                ext[owner] -= (
                    offsets.min() - 1 if offsets.size else end - start + 1
                )
        flags = np.ones(len(p), dtype=bool) if returnExtent else ext == lens
        return flags, np.flatnonzero(flags) + 1, p[flags], ext[flags]

    def isRegionProteinFree(
        self,
        positionsStrands,
        lengths,
        returnExtent=False,
        releasableMonomerIndexs=None,
        releasableComplexIndexs=None,
        bothStrands=False,
        allStrands=False,
    ):
        from ..sparsemat import SparseMat

        p = np.asarray(positionsStrands, dtype=int).reshape(-1, 2)
        lens = np.asarray(lengths, dtype=int).reshape(-1)
        if not len(p):
            empty = SparseMat(np.empty((0, 2), int), [], [0, 0])
            return (
                np.zeros(0, bool),
                np.zeros(0, int),
                p,
                np.zeros(0, int),
                empty,
                empty.copy(),
            )
        if len(lens) == 1:
            lens = np.repeat(lens, len(p))
        genome = int(np.asarray(self.sequenceLen).item())
        # Preserve MATLAB's region bounds, including its peculiar negative-origin
        # split and the asymmetric coordinate calculation for wrapped queries.
        starts = (p[:, 0] + np.minimum(0, lens + 1) - 1) % genome + 1
        ends = starts + np.abs(lens) - 1
        strands = p[:, 1].copy()
        if bothStrands:
            strands = np.ceil(strands / 2).astype(int)
        if allStrands:
            strands[:] = 1
        region_starts = starts.tolist()
        region_ends = ends.tolist()
        region_strands = strands.tolist()
        region_idxs = (np.arange(len(p)) + 1).tolist()
        for i in np.flatnonzero(ends > genome):
            region_starts.append(max(1, int(starts[i]) - genome))
            region_ends.append(int(ends[i]) - genome)
            region_strands.append(int(strands[i]))
            region_idxs.append(int(i + 1))
            ends[i] = genome
            region_ends[i] = genome
        for i in np.flatnonzero(starts < 0):
            region_starts.append(int(starts[i]) + genome)
            region_ends.append(min(genome, int(ends[i]) + genome))
            region_strands.append(int(strands[i]))
            region_idxs.append(int(i + 1))
            starts[i] = 1
        ext = np.abs(lens).copy()
        out = []
        released_sets = [
            set(
                np.asarray([] if x is None else x, dtype=int)
                .reshape(-1)
                .tolist()
            )
            for x in (releasableMonomerIndexs, releasableComplexIndexs)
        ]
        for which, (mat, footprints, released) in enumerate(
            zip(
                (self.monomerBoundSites, self.complexBoundSites),
                (self.monomerDNAFootprints, self.complexDNAFootprints),
                released_sets,
            )
        ):
            subs, vals = mat.find()
            subs = np.asarray(subs, dtype=int).reshape(-1, 2)
            vals = np.asarray(vals, dtype=int).reshape(-1)
            if not bothStrands:
                binding = np.asarray(
                    getattr(
                        self,
                        (
                            "monomerDNAFootprintBindingStrandedness"
                            if which == 0
                            else "complexDNAFootprintBindingStrandedness"
                        ),
                    )
                ).reshape(-1)[vals - 1]
                isds = binding == int(
                    np.asarray(self.dnaStrandedness_dsDNA).item()
                )
                extra = np.column_stack(
                    (
                        subs[isds, 0],
                        2 * np.ceil(subs[isds, 1] / 2).astype(int) - 1,
                    )
                )
                subs = np.vstack(
                    (
                        subs[~isds],
                        extra,
                        np.column_stack((extra[:, 0], extra[:, 1] + 1)),
                    )
                )
                vals = np.r_[vals[~isds], vals[isds], vals[isds]]
            if allStrands:
                repl = np.asarray(
                    getattr(
                        self,
                        (
                            "monomerIndexs_replisome"
                            if which == 0
                            else "complexIndexs_replisome"
                        ),
                    )
                ).reshape(-1)
                keep = ~np.isin(vals, repl)
                subs, vals = subs[keep], vals[keep]
            fp = np.asarray(footprints).reshape(-1)[vals - 1]
            bstarts = subs[:, 0]
            bends = bstarts + fp - 1
            bstrands = subs[:, 1].copy()
            if bothStrands:
                bstrands = np.ceil(bstrands / 2).astype(int)
            if allStrands:
                bstrands[:] = 1
            wrap = np.flatnonzero(bends > genome)
            bstarts = np.r_[bstarts, np.ones(len(wrap), int)]
            bends = np.r_[bends, bends[wrap] - genome]
            bstrands = np.r_[bstrands, bstrands[wrap]]
            vals = np.r_[vals, vals[wrap]]
            coords = []
            data = []
            for j, (a, b, s, qi) in enumerate(
                zip(region_starts, region_ends, region_strands, region_idxs)
            ):
                mask = (
                    ((bstarts >= a) & (bstarts <= b))
                    | ((bends >= a) & (bends <= b))
                    | ((bstarts <= a) & (bends >= b))
                )
                hit = np.flatnonzero(mask & (bstrands == s))
                if not len(hit):
                    continue
                q = qi - 1
                if lens[q] > 0:
                    pos = bstarts[hit] - a + 1
                    if j >= len(p):
                        pos = pos + (genome - p[q, 0])
                else:
                    pos = b - bends[hit] + 1
                    if j >= len(p):
                        pos = pos + p[q, 0]
                pos = np.maximum(pos, 1).astype(int)
                nr = hit[~np.isin(vals[hit], list(released))]
                if len(nr):
                    ext[q] = min(
                        ext[q], int(np.min(pos[np.isin(hit, nr)])) - 1
                    )
                coords.extend((q + 1, int(x)) for x in pos)
                data.extend(vals[hit].tolist())
            width = max(
                int(max(np.abs(lens), default=0)),
                max((c[1] for c in coords), default=0),
            )
            out.append(
                SparseMat(
                    np.asarray(coords, int).reshape(-1, 2),
                    np.asarray(data, int),
                    [len(p), width],
                )
            )
        signed = np.sign(lens) * ext
        flags = signed == lens
        idx = np.flatnonzero(flags) + 1
        if returnExtent:
            return (
                np.ones(len(p), bool),
                np.arange(1, len(p) + 1),
                p,
                signed,
                *out,
            )
        return flags, idx, p[flags], signed[flags], *out

    def isRegionAccessible(
        self,
        positionsStrands,
        lengths,
        bindingMonomers,
        bindingComplexs,
        isPositionsStrandFootprintCentroid=False,
        ignoreDamageFilter=None,
        returnExtent=False,
        returnOverlappingRegions=True,
        checkRegionSupercoiled=False,
    ):
        p = np.asarray(positionsStrands, dtype=int).reshape(-1, 2)
        lens = np.asarray(lengths, dtype=int).reshape(-1)
        if not len(p):
            return np.zeros(0, bool), np.zeros(0, int), p, np.zeros(0, int)
        if len(lens) == 1:
            lens = np.repeat(lens, len(p))
        if not np.all(lens):
            raise ValueError("Lengths must be non-zero integers")
        fp, f3, f5, bind, region = self.getDNAFootprint(
            bindingMonomers, bindingComplexs
        )
        query = p.copy()
        ign = np.asarray(
            [] if ignoreDamageFilter is None else ignoreDamageFilter, dtype=int
        )
        if ign.size == 0:
            ign = np.zeros((len(p), 0), int)
        elif ign.ndim <= 1:
            ign = ign.reshape(1, -1)
        if len(ign) == 1 and len(p) > 1:
            ign = np.repeat(ign, len(p), axis=0)
        if isPositionsStrandFootprintCentroid:
            for i, (strand, n) in enumerate(zip(p[:, 1], lens)):
                a, b = (f5, f3) if strand % 2 else (f3, f5)
                shift = a if n > 0 else -b
                query[i, 0] -= shift
                ign[i] += shift
        elif np.any(lens < 0):
            query[:, 0] += fp - 1
        qlen = lens + np.sign(lens) * (fp - 1)
        if region == self.dnaStrandedness_dsDNA:
            polymer = self.isRegionDoubleStranded(
                query, qlen, True, checkRegionSupercoiled
            )[3]
        elif region == self.dnaStrandedness_ssDNA:
            polymer = self.isRegionSingleStranded(query, qlen, True)[3]
        elif region == self.dnaStrandedness_xsDNA:
            polymer = qlen.copy()  # Exact source TODO behavior.
        else:
            raise ValueError("Invalid footprint strandedness")
        rm, rc = self.getReleasableProteins(bindingMonomers, bindingComplexs)
        protein = self.isRegionProteinFree(
            query,
            qlen,
            True,
            rm,
            rc,
            bind == self.dnaStrandedness_dsDNA,
            region == self.dnaStrandedness_xsDNA,
        )[3]
        damage = self.isRegionUndamaged(
            query,
            qlen,
            bind == self.dnaStrandedness_dsDNA,
            ign,
            True,
            region == self.dnaStrandedness_xsDNA,
        )[3]
        ext = np.sign(lens) * np.maximum(
            0,
            np.minimum(
                np.minimum(np.abs(protein), np.abs(polymer)), np.abs(damage)
            )
            - (fp - 1),
        )
        if not returnOverlappingRegions:
            candidates = (
                np.flatnonzero(ext != 0 if returnExtent else lens == ext) + 1
            )
            _, selected = self.excludeOverlappingRegions(
                [],
                candidates,
                p,
                ext,
                fp,
                f3,
                f5,
                isPositionsStrandFootprintCentroid,
                bind == self.dnaStrandedness_dsDNA,
            )
            ext[
                np.setdiff1d(
                    np.arange(len(p)),
                    np.asarray(selected, dtype=int).ravel() - 1,
                )
            ] = 0
        flags = np.ones(len(p), bool) if returnExtent else lens == ext
        return flags, np.flatnonzero(flags) + 1, p[flags], ext[flags]

    @staticmethod
    def calculateFootprintOverhangs(footprint):
        f5 = (int(footprint) - 1 + 1) // 2
        return int(footprint) - 1 - f5, f5
