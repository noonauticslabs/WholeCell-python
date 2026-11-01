"""Accessible chromosome region and site sampling."""

from __future__ import annotations

import numpy as np


def _coords_values(matrix):
    coords, values = matrix.find()
    coords = np.asarray(coords, dtype=int)
    if coords.size == 0:
        coords = np.empty((0, matrix.ndims()), dtype=int)
    elif coords.ndim == 1:
        coords = coords.reshape(1, -1)
    return coords, np.asarray(values).reshape(-1)


class ChromosomeSampling:
    """Ports of ``sampleAccessibleRegions``, ``sampleAccessibleSites``, and
    ``getAccessibleRegions`` from ``Chromosome.m``. Host-provided accessibility,
    footprint, release, and exclusion methods remain the source of policy.
    Coordinates returned by these methods are MATLAB one based.
    """

    def sampleAccessibleRegions(
        self,
        nSites,
        weights,
        positionsStrands,
        lengths,
        bindingMonomers,
        bindingComplexs,
        isPositionsStrandFootprintCentroid,
        ignoreDamageFilter,
        returnOverlappingRegions,
        returnExtentAccessible,
        checkRegionSupercoiled=False,
    ):
        positions = np.asarray(positionsStrands, dtype=int).reshape(-1, 2)
        lens = np.asarray(lengths).reshape(-1).copy()
        if lens.size != len(positions):
            raise ValueError(
                "positionsStrands and lengths must have matching rows"
            )
        n_sites = float(nSites)
        if n_sites < 0 or np.isnan(n_sites):
            raise ValueError("nSites must be nonnegative")
        weights = (
            np.ones(len(positions), dtype=float)
            if weights is None or np.asarray(weights).size == 0
            else np.asarray(weights, dtype=float).reshape(-1).copy()
        )
        if len(weights) != len(positions):
            raise ValueError("weights must match positionsStrands")
        footprint, fp3, fp5, binding_strand = self.getDNAFootprint(
            bindingMonomers, bindingComplexs
        )[:4]
        idxs = np.empty(0, dtype=int)
        while np.any(weights) and len(idxs) < n_sites:
            n_more = min(
                max(2 * (n_sites - len(idxs)), 10),
                int(np.count_nonzero(weights)),
            )
            selected = (
                np.asarray(
                    self.randStream.randsample(
                        len(weights), n_more, False, weights
                    ),
                    dtype=int,
                ).reshape(-1)
                - 1
            )
            weights[selected] = 0
            flags, _, _, extents = self.isRegionAccessible(
                positions[selected],
                lens[selected],
                bindingMonomers,
                bindingComplexs,
                isPositionsStrandFootprintCentroid,
                ignoreDamageFilter,
                returnExtentAccessible,
                True,
                checkRegionSupercoiled,
            )
            flags = np.asarray(flags, dtype=bool).reshape(-1)
            new = selected[flags]
            lens[selected[~flags]] = 0
            lens[selected[flags]] = np.asarray(extents).reshape(-1)
            if not returnOverlappingRegions:
                idxs, new = self.excludeOverlappingRegions(
                    idxs + 1,
                    new + 1,
                    positions,
                    lens,
                    footprint,
                    fp3,
                    fp5,
                    isPositionsStrandFootprintCentroid,
                    binding_strand == self.dnaStrandedness_dsDNA,
                )
                idxs, new = (
                    np.asarray(idxs, dtype=int).reshape(-1) - 1,
                    np.asarray(new, dtype=int).reshape(-1) - 1,
                )
            remaining = max(0, n_sites - len(idxs))
            if np.isfinite(remaining):
                new = new[: int(remaining)]
            idxs = np.r_[idxs, new]
        idxs = np.sort(idxs.astype(int))
        tfs = np.zeros(len(positions), dtype=bool)
        tfs[idxs] = True
        return tfs, idxs + 1, positions[idxs], lens[idxs]

    def sampleAccessibleSites(self, prob, nSites, seq):
        dna_length = int(np.asarray(self.sequenceLen).item())
        poly_pos, _ = _coords_values(self.polymerizedRegions)
        n_strands = int(poly_pos[:, 1].max()) if len(poly_pos) else 0
        sequence = "" if seq is None else seq
        seq_len = len(sequence)
        monomer_pos, bound_monomers = _coords_values(self.monomerBoundSites)
        complex_pos, bound_complexes = _coords_values(self.complexBoundSites)
        accessible = int(np.asarray(self.polymerizedRegions.vals).sum())
        if bound_monomers.size:
            accessible -= int(
                np.asarray(self.monomerDNAFootprints)
                .reshape(-1)[bound_monomers.astype(int) - 1]
                .sum()
            )
        if bound_complexes.size:
            accessible -= int(
                np.asarray(self.complexDNAFootprints)
                .reshape(-1)[bound_complexes.astype(int) - 1]
                .sum()
            )
        accessible -= int(self.damagedSites.nnz())
        n_sites = float(nSites)
        if np.isnan(n_sites) or n_sites < 0:
            raise ValueError("nSites must be nonnegative")
        if prob is not None and np.asarray(prob).size:
            gc = sum(base in "GC" for base in sequence)
            content = float(np.asarray(self.sequenceGCContent).item())
            estimate = (
                accessible
                * float(np.asarray(prob).item())
                * (content / 2) ** gc
                * ((1 - content) / 2) ** (seq_len - gc)
            )
            n_sites = min(
                n_sites,
                int(
                    np.asarray(
                        self.randStream.stochasticRound(estimate)
                    ).item()
                ),
            )
        if n_sites <= 0:
            return np.empty((0, 2), dtype=int)
        found = np.empty((0, 2), dtype=int)
        for _ in range(15):
            remaining = n_sites - len(found)
            n_more = (
                int(max(2 * remaining, remaining + 10))
                if np.isfinite(remaining)
                else 2 * dna_length * max(1, n_strands)
            )
            positions = (
                np.ceil(dna_length * self.randStream.rand(n_more, 1))
                .astype(int)
                .reshape(-1)
            )
            strands = (
                np.ceil(n_strands * self.randStream.rand(n_more, 1))
                .astype(int)
                .reshape(-1)
            )
            candidates = np.column_stack((positions, strands))
            if sequence:
                direction = np.where(strands % 2 == 1, 1, -1)
                offsets = np.arange(seq_len)
                subseqs = self.sequence.subsequence(
                    positions[:, None] + direction[:, None] * offsets,
                    strands[:, None],
                )
                subseqs = np.asarray(subseqs)
                if subseqs.ndim == 1:
                    subseqs = subseqs.reshape(-1, 1)
                query = np.asarray(list(sequence))
                if query.ndim == 1:
                    matches = np.all(subseqs == query[None, :], axis=1)
                else:
                    matches = np.any(
                        np.all(
                            subseqs[:, None, :] == query[None, :, :], axis=2
                        ),
                        axis=1,
                    )
                candidates = candidates[matches]
                query_len = seq_len
            else:
                query_len = 1
            _, ids, _, _ = self.isRegionAccessible(
                candidates, query_len, [], [], True, None, False, False
            )
            ids = np.asarray(ids, dtype=int).reshape(-1) - 1
            found = np.unique(np.vstack((found, candidates[ids])), axis=0)
            if len(found) >= n_sites:
                found, _ = self.randStream.randomlySelectNRows(
                    found, int(n_sites)
                )
                break
        if n_sites > 1 and len(found):
            found = found[np.lexsort((found[:, 1], found[:, 0]))]
        return found

    def getAccessibleRegions(
        self, monomerIdx, complexIdx, checkRegionSupercoiled=False
    ):
        footprint, _, _, binding_strand, region_strand = self.getDNAFootprint(
            monomerIdx, complexIdx
        )
        releasable_m, releasable_c = self.getReleasableProteins(
            monomerIdx, complexIdx
        )
        releasable_m = np.asarray(releasable_m, dtype=int).reshape(-1)
        releasable_c = np.asarray(releasable_c, dtype=int).reshape(-1)
        mon_pos, mons = _coords_values(self.monomerBoundSites)
        cpx_pos, cpxs = _coords_values(self.complexBoundSites)
        mon_fp = np.asarray(
            self.monomerDNAFootprintBindingStrandedness
        ).reshape(-1)
        cpx_fp = np.asarray(
            self.complexDNAFootprintBindingStrandedness
        ).reshape(-1)
        excluded_pos, excluded_len = [], []

        def add_bound(coords, ids, allowed, binding, footprints):
            for coordinate, value in zip(coords, ids.astype(int)):
                if value in allowed or binding[value - 1] not in (
                    self.dnaStrandedness_ssDNA,
                    self.dnaStrandedness_dsDNA,
                ):
                    continue
                strands = [int(coordinate[1])]
                if binding[value - 1] == self.dnaStrandedness_dsDNA:
                    base = 2 * int(np.ceil(coordinate[1] / 2))
                    strands = [base - 1, base]
                for strand in strands:
                    excluded_pos.append((int(coordinate[0]), strand))
                    excluded_len.append(int(footprints[value - 1]))

        add_bound(
            mon_pos,
            mons,
            set(releasable_m.tolist()),
            mon_fp,
            self.monomerDNAFootprints,
        )
        add_bound(
            cpx_pos,
            cpxs,
            set(releasable_c.tolist()),
            cpx_fp,
            self.complexDNAFootprints,
        )
        dmg_pos, _ = _coords_values(self.damagedSites)
        excluded_pos.extend(map(tuple, dmg_pos.tolist()))
        excluded_len.extend([1] * len(dmg_pos))
        excluded_pos = np.asarray(excluded_pos, dtype=int).reshape(-1, 2)
        excluded_len = np.asarray(excluded_len, dtype=int)
        if binding_strand == self.dnaStrandedness_dsDNA and len(excluded_pos):
            excluded_pos[:, 1] = (excluded_pos[:, 1] + 1) // 2
        if (
            region_strand == self.dnaStrandedness_ssDNA
            and binding_strand == self.dnaStrandedness_ssDNA
        ):
            polymer_pos, polymer_len = _coords_values(
                self.singleStrandedRegions
            )
        elif region_strand == self.dnaStrandedness_dsDNA:
            polymer_pos, polymer_len = _coords_values(
                self.doubleStrandedRegions
            )
            if binding_strand == self.dnaStrandedness_dsDNA:
                keep = polymer_pos[:, 1] % 2 == 1
                polymer_pos, polymer_len = (
                    polymer_pos[keep].copy(),
                    polymer_len[keep],
                )
                polymer_pos[:, 1] = (polymer_pos[:, 1] + 1) // 2
            if checkRegionSupercoiled:
                super_pos, _ = _coords_values(self.supercoiled)
                supers = {tuple(row) for row in super_pos}
                if binding_strand == self.dnaStrandedness_dsDNA:
                    keep = np.asarray(
                        [(p[0], 2 * p[1]) in supers for p in polymer_pos]
                    )
                else:
                    keep = np.asarray(
                        [tuple(p) in supers for p in polymer_pos]
                    )
                polymer_pos, polymer_len = polymer_pos[keep], polymer_len[keep]
        else:
            raise ValueError(f"unsupported strandedness: {region_strand}")
        regions, lengths = self.excludeRegions(
            polymer_pos, polymer_len, excluded_pos, excluded_len
        )
        regions = np.asarray(regions, dtype=int).reshape(-1, 2)
        lengths = np.asarray(lengths, dtype=int).reshape(-1)
        if binding_strand == self.dnaStrandedness_dsDNA:
            regions[:, 1] = 2 * regions[:, 1] - 1
        keep = lengths >= int(footprint)
        return regions[keep], lengths[keep]
