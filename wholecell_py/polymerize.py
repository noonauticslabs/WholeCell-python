"""Source greedy polymerization allocator, with explicit independent RNG input."""

import numpy as np
from .randstream import RandStream


def polymerize(
    sequences,
    baseAmounts,
    bases,
    basePadValue,
    energy=0,
    energyCostPerBase=0,
    randStream=None,
):
    """Return progress, remaining bases, base costs, remaining energy, energy cost.

    Equivalent allocation algorithm to sim/util/polymerize.m. Random tie
    choices use the caller's stream; no MATLAB state or trajectory is replayed.
    """
    sequences = np.asarray(sequences).copy()
    if sequences.ndim != 2:
        raise ValueError("sequences must be a row-per-polymer matrix")
    original_shape = np.asarray(baseAmounts).shape
    amounts = np.asarray(baseAmounts, dtype=float).ravel(order="F").copy()
    bases = np.asarray(list(bases) if isinstance(bases, str) else bases).ravel(
        order="F"
    )
    energy = float(np.asarray(energy).item())
    cost = float(np.asarray(energyCostPerBase).item())
    rng = randStream if randStream is not None else RandStream()
    progress = np.zeros(sequences.shape[0], dtype=int)
    costs = np.zeros(bases.size)
    spent = 0.0
    active = np.arange(sequences.shape[0])
    lengths = np.max(
        np.where(
            sequences != basePadValue, np.arange(sequences.shape[1]) + 1, 0
        ),
        axis=1,
        initial=0,
    )
    if not (np.any(amounts < 0) or np.any(np.isnan(amounts))):
        while sequences.size and energy >= cost:
            keep = progress[active] < lengths[active]
            active, sequences = active[keep], sequences[keep]
            if not sequences.size:
                break
            counts = np.array(
                [np.sum(sequences == base, axis=0) for base in bases]
            )
            cumulative = np.cumsum(
                np.column_stack((np.zeros(bases.size), counts)), axis=1
            )
            limits = np.array(
                [
                    np.flatnonzero(row <= amount)[-1]
                    for row, amount in zip(cumulative, amounts)
                ]
            )
            # MATLAB sum reduces the first nonsingleton dimension. For a
            # single base this is the sequence-position axis, a source edge
            # case that can ignore the energy bound (retain, do not repair).
            energy_counts = (
                cumulative.sum() if bases.size == 1 else cumulative.sum(axis=0)
            )
            permitted = np.flatnonzero(
                np.atleast_1d(energy_counts) * cost <= energy
            )
            energy_limit = permitted[-1] if permitted.size else np.inf
            elongation = int(min(limits.min(), energy_limit))
            usage = cumulative[:, elongation]
            limiting = (
                np.flatnonzero(limits == elongation)
                if elongation + 1 < cumulative.shape[1]
                else np.empty(0, int)
            )
            progress[active] = np.minimum(
                progress[active] + elongation, lengths[active]
            )
            amounts -= usage
            costs += usage
            energy -= cost * usage.sum()
            spent += cost * usage.sum()
            sequences = sequences[:, elongation:]
            if not sequences.size:
                break
            if limiting.size:
                keep = np.ones(len(active), bool)
                for i in limiting:
                    indices = np.flatnonzero(sequences[:, 0] == bases[i])
                    order = (
                        np.asarray(
                            rng.randperm(len(indices)), dtype=int
                        ).ravel()
                        - 1
                    )
                    keep[indices[order[int(amounts[i]) :]]] = False
                active, sequences = active[keep], sequences[keep]
            elif elongation == 0:
                n = int(np.fix(energy / cost))
                selected = (
                    np.asarray(rng.randperm(len(active)), dtype=int).ravel()[
                        :n
                    ]
                    - 1
                )
                progress[active[selected]] += 1
                usage = np.array(
                    [np.sum(sequences[selected, 0] == base) for base in bases]
                )
                amounts -= usage
                costs += usage
                energy -= cost * n
                spent += cost * n
                break
    return (
        progress[:, None],
        amounts.reshape(original_shape, order="F"),
        costs[:, None],
        energy,
        spent,
    )
