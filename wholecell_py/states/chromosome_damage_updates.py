"""Chromosome damage mutations translated from ``Chromosome.m``."""

from __future__ import annotations

import numpy as np

from ..circular_sparsemat import CircularSparseMat
from ..side_effect import SimulationStateSideEffect


class ChromosomeDamageUpdates:
    """Damage-site mutation methods (coordinates are MATLAB one based)."""

    def setSiteDamaged(
        self,
        damageType,
        damageSubType,
        probDamage,
        maxDamages,
        vulnerableMotif,
        vulnerableMotifType,
    ):
        # MATLAB returns an empty column of side effects even though this
        # operation has no effects on other states.
        def text(value):
            while (
                isinstance(value, np.ndarray)
                and value.dtype == object
                and value.size == 1
            ):
                value = value.item()
            return (
                "".join(np.asarray(value).ravel(order="F"))
                if isinstance(value, np.ndarray) and value.dtype.kind in "US"
                else value
            )

        damageType, vulnerableMotifType, vulnerableMotif = map(
            text, (damageType, vulnerableMotifType, vulnerableMotif)
        )
        effects = np.empty((0, 1), dtype=object)
        positions = np.empty((0, 2), dtype=int)
        probability = float(np.asarray(probDamage).item())
        maximum = float(np.asarray(maxDamages).item())
        if probability == 0 or maximum == 0:
            return positions, effects

        if isinstance(vulnerableMotif, str):
            positions = np.asarray(
                self.sampleAccessibleSites(
                    probability, maximum, vulnerableMotif
                ),
                dtype=int,
            ).reshape(-1, 2)
        else:
            target = getattr(self, vulnerableMotifType)
            if target.nnz() == 0:
                return positions, effects
            damage_view = self.damagedSites_nonRedundant
            # The source performs elementwise equality against both sparse
            # matrices, then find() on the resulting logical matrix.
            motif_pos, motif_vals = target.find()
            damage_pos, damage_vals = damage_view.find()
            motif_pos = np.asarray(motif_pos, dtype=int).reshape(
                -1, len(target.siz)
            )
            motif_vals = np.asarray(motif_vals).reshape(-1)
            damage_pos = np.asarray(damage_pos, dtype=int).reshape(
                -1, len(target.siz)
            )
            damage_vals = np.asarray(damage_vals).reshape(-1)
            motif_value = np.asarray(vulnerableMotif).item()
            motif = {
                tuple(p)
                for p, v in zip(motif_pos, motif_vals)
                if v == motif_value
            }
            damage = {
                tuple(p)
                for p, v in zip(damage_pos, damage_vals)
                if v == motif_value
            }
            if motif_value == 0:
                # Equality to zero includes implicit sparse entries in MATLAB.
                eligible = np.ones(tuple(target.siz), dtype=bool)
                for coords, vals in (
                    (motif_pos, motif_vals),
                    (damage_pos, damage_vals),
                ):
                    nonzero = coords[vals != 0]
                    if len(nonzero):
                        eligible[tuple((nonzero - 1).T)] = False
                linear = np.flatnonzero(eligible.ravel(order="F"))
                positions = (
                    np.column_stack(
                        np.unravel_index(linear, eligible.shape, order="F")
                    )
                    + 1
                )
            else:
                common = motif.intersection(damage)
                positions = np.asarray(
                    sorted(common, key=lambda row: tuple(reversed(row))),
                    dtype=int,
                ).reshape(-1, 2)
            if positions.size == 0:
                return positions, effects
            n = int(
                np.asarray(
                    self.randStream.stochasticRound(
                        len(positions) * probability
                    )
                ).item()
            )
            maximum = min(maximum, n)
            if maximum == 0:
                return np.empty((0, 2), dtype=int), effects
            positions, _ = self.randStream.randomlySelectNRows(
                positions, int(maximum)
            )
            positions = np.asarray(positions, dtype=int).reshape(-1, 2)

        if positions.size == 0:
            return np.empty((0, 2), dtype=int), effects
        matrix = getattr(self, damageType)
        coords = positions
        if len(matrix.siz) > 2:
            coords = np.column_stack(
                (positions, np.ones(len(positions), dtype=int))
            )
        values = np.full(len(coords), damageSubType)
        setattr(self, damageType, matrix.set_coords(coords, values))
        return positions, effects
