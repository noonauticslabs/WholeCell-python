"""Creation and identity changes for chromosome-bound proteins."""

import numpy as np


class ChromosomeBindingCreation:
    """Mixin implementing the corresponding Chromosome.m binding methods.

    Region sampling, exclusion, and release policy are supplied by the host.
    """

    def setSiteProteinBound(
        self,
        positionsStrands,
        maxBindings,
        weights,
        binding_monomerIndexs,
        binding_complexIndexs,
        mainEffects_monomerIndexs,
        mainEffects_complexIndexs,
        isBindingStable,
        isPositionsStrandFootprintCentroid,
        lengths,
        isBindingProcessive,
        ignoreDamageFilter,
        checkRegionSupercoiled=False,
    ):
        monomers = np.asarray(binding_monomerIndexs).reshape(-1)
        complexes = np.asarray(binding_complexIndexs).reshape(-1)
        positions = np.asarray(positionsStrands, dtype=int).reshape((-1, 2))
        lengths = np.asarray(lengths).reshape(-1)
        if isBindingStable and monomers.size + complexes.size > 1:
            raise ValueError("Can only bind 1 protein at a time")
        if (
            isBindingStable
            and not isBindingProcessive
            and np.any(lengths != 1)
        ):
            raise ValueError(
                "If any(lengths~=1) then if binding is stable it must be processive"
            )
        if lengths.size == 1:
            lengths = np.repeat(lengths, len(positions))

        tfs, idxs, positions, lengths = self.sampleAccessibleRegions(
            maxBindings,
            weights,
            positions,
            lengths,
            monomers,
            complexes,
            isPositionsStrandFootprintCentroid,
            ignoreDamageFilter,
            not isBindingStable,
            isBindingProcessive,
            checkRegionSupercoiled,
        )
        tfs = np.asarray(tfs)
        idxs = np.asarray(idxs)
        positions = np.asarray(positions, dtype=int).reshape((-1, 2))
        lengths = np.asarray(lengths).reshape(-1)
        footprint, footprint3, footprint5, binding_strand, region_strand = (
            self.getDNAFootprint(monomers, complexes)
        )
        footprint = np.asarray(footprint).reshape(-1)
        fp = int(footprint[0]) if footprint.size else 0

        release_positions = positions.copy()
        release_positions[:, 0] += np.minimum(0, lengths + 1).astype(int)
        if isPositionsStrandFootprintCentroid:
            odd = release_positions[:, 1] % 2 == 1
            release_positions[odd, 0] -= int(
                np.asarray(footprint5).reshape(-1)[0]
            )
            release_positions[~odd, 0] -= int(
                np.asarray(footprint3).reshape(-1)[0]
            )

        region_ds = (
            int(np.asarray(region_strand).reshape(-1)[0])
            == self.dnaStrandedness_xsDNA
        )
        bind_ds = (
            int(np.asarray(binding_strand).reshape(-1)[0])
            == self.dnaStrandedness_dsDNA
        )
        if region_ds:
            mpos, mvals = self.monomerBoundSites.find()
            cpos, cvals = self.complexBoundSites.find()
            releasable_m, releasable_c = self.getReleasableProteins(
                monomers, complexes
            )
            keepm = ~np.isin(
                mvals.reshape(-1), np.asarray(releasable_m).reshape(-1)
            )
            keepc = ~np.isin(
                cvals.reshape(-1), np.asarray(releasable_c).reshape(-1)
            )
            mpos, mvals = mpos[keepm], mvals[keepm]
            cpos, cvals = cpos[keepc], cvals[keepc]
            four = np.vstack(
                [
                    np.column_stack(
                        (release_positions[:, 0], np.full(len(positions), s))
                    )
                    for s in (1, 2, 3, 4)
                ]
            )
            lens = np.tile(fp + np.abs(lengths) - 1, 4)
            occupied_pos = np.vstack((mpos, cpos))
            occupied_fp = np.r_[
                np.asarray(self.monomerDNAFootprints)[
                    mvals.astype(int).reshape(-1) - 1
                ],
                np.asarray(self.complexDNAFootprints)[
                    cvals.astype(int).reshape(-1) - 1
                ],
            ]
            release_positions, release_lengths = self.excludeRegions(
                four, lens, occupied_pos, occupied_fp
            )
        else:
            release_lengths = fp + np.abs(lengths) - 1

        released_m, released_c, side_effects = self.setRegionProteinUnbound(
            release_positions,
            release_lengths,
            mainEffects_monomerIndexs,
            mainEffects_complexIndexs,
            region_ds,
            bind_ds,
            False,
            False,
        )

        if isBindingStable:
            bind_positions = positions.copy()
            if isBindingProcessive:
                bind_positions[:, 0] += (lengths - np.sign(lengths)).astype(
                    int
                )
            if isPositionsStrandFootprintCentroid:
                odd = bind_positions[:, 1] % 2 == 1
                bind_positions[odd, 0] -= int(
                    np.asarray(footprint5).reshape(-1)[0]
                )
                bind_positions[~odd, 0] -= int(
                    np.asarray(footprint3).reshape(-1)[0]
                )
            if monomers.size:
                self.monomerBoundSites[bind_positions] = monomers[0]
                released_m = np.asarray(released_m).copy()
                released_m[
                    np.asarray(mainEffects_monomerIndexs).reshape(-1)
                    == monomers[0]
                ] -= len(idxs)
            else:
                self.complexBoundSites[bind_positions] = complexes[0]
                released_c = np.asarray(released_c).copy()
                released_c[
                    np.asarray(mainEffects_complexIndexs).reshape(-1)
                    == complexes[0]
                ] -= len(idxs)
        return (
            released_m,
            released_c,
            side_effects,
            tfs,
            idxs,
            positions,
            lengths,
        )

    def modifyBoundProtein(
        self,
        positionsStrands,
        newMonomers,
        newComplexs,
        mainEffects_monomerIndexs,
        mainEffects_complexIndexs,
    ):
        pos = np.asarray(positionsStrands, dtype=int).reshape((-1, 2))
        newm = np.asarray(newMonomers).reshape(-1)
        newc = np.asarray(newComplexs).reshape(-1)
        if np.any((newm != 0) & (newc != 0)):
            raise ValueError("Only 1 protein can be bound at each site")
        oldm = np.asarray(self.monomerBoundSites[pos]).reshape(-1)
        oldc = np.asarray(self.complexBoundSites[pos]).reshape(-1)
        oldfp = np.zeros(len(pos), dtype=float)
        newfp = np.zeros(len(pos), dtype=float)
        nz = oldm != 0
        oldfp[nz] = np.asarray(self.monomerDNAFootprints).reshape(-1)[
            oldm[nz].astype(int) - 1
        ]
        nz = oldc != 0
        oldfp[nz] = np.asarray(self.complexDNAFootprints).reshape(-1)[
            oldc[nz].astype(int) - 1
        ]
        nz = newm != 0
        newfp[nz] = np.asarray(self.monomerDNAFootprints).reshape(-1)[
            newm[nz].astype(int) - 1
        ]
        nz = newc != 0
        newfp[nz] = np.asarray(self.complexDNAFootprints).reshape(-1)[
            newc[nz].astype(int) - 1
        ]
        # Intentionally enforce the source's stated safety invariant for all sites.
        # MATLAB's zeros(n)/any/if bug bypasses this check for n > 1.
        if np.any(newfp > oldfp):
            raise ValueError(
                "New protein footprints cannot be larger that old ones"
            )
        nz = newm != 0
        self.monomerBoundSites[pos[nz]] = newm[nz]
        nz = newc != 0
        self.complexBoundSites[pos[nz]] = newc[nz]

        def tally(old, new, main):
            result = np.zeros((np.asarray(main).size, 1))
            main = np.asarray(main).reshape(-1)
            for values, sign in ((old[old != 0], 1), (new[new != 0], -1)):
                mapped = np.asarray(
                    [
                        (
                            np.flatnonzero(main == x)[0]
                            if np.any(main == x)
                            else -1
                        )
                        for x in values
                    ],
                    dtype=int,
                )
                if np.any(mapped < 0):
                    raise ValueError(
                        "All binding/releasing proteins must be mainEffects"
                    )
                if mapped.size:
                    ids, counts = np.unique(mapped, return_counts=True)
                    result[ids, 0] += sign * counts
            return result

        released_m = tally(oldm, newm, mainEffects_monomerIndexs)
        released_c = tally(oldc, newc, mainEffects_complexIndexs)
        return released_m, released_c, np.empty((0, 1), dtype=object)
