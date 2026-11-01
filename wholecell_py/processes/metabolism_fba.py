"""Source-order construction and pruning of Metabolism.m's FBA network."""

import numpy as np


def _column(values):
    return np.asarray(values).reshape(-1, 1)


def _blocks(matrix):
    nonzero = matrix != 0
    rows = np.zeros(matrix.shape[0], dtype=int)
    cols = np.zeros(matrix.shape[1], dtype=int)
    er, ec = ~nonzero.any(axis=1), ~nonzero.any(axis=0)
    block = int(er.any() or ec.any())
    rows[er], cols[ec] = 1, 1
    for row in range(len(rows)):
        if rows[row]:
            continue
        block += 1
        queue = [(0, row)]
        while queue:
            axis, i = queue.pop()
            assignments = rows if axis == 0 else cols
            if assignments[i]:
                continue
            assignments[i] = block
            neighbors = np.flatnonzero(
                nonzero[i] if axis == 0 else nonzero[:, i]
            )
            queue.extend((1 - axis, int(j)) for j in neighbors)
    return rows, cols


def _null_active(matrix):
    """Nonzero rows of MATLAB null(A,'r'), using RREF pivoting/tolerance."""
    a = np.array(matrix, dtype=float, copy=True)
    m, n = a.shape
    tolerance = (
        max(m, n)
        * np.finfo(float).eps
        * (np.linalg.norm(a, np.inf) if m and n else 0)
    )
    row, pivots = 0, []
    for col in range(n):
        if row == m:
            break
        pivot = row + int(np.argmax(np.abs(a[row:, col])))
        if abs(a[pivot, col]) <= tolerance:
            a[row:, col] = 0
            continue
        a[[row, pivot], col:] = a[[pivot, row], col:]
        a[row, col:] /= a[row, col]
        for other in range(m):
            if other != row:
                a[other, col:] -= a[other, col] * a[row, col:]
        pivots.append(col)
        row += 1
    free = np.setdiff1d(np.arange(n), pivots)
    active = np.zeros(n, dtype=bool)
    active[free] = True
    if pivots and free.size:
        active[pivots] = np.any(a[: len(pivots), free] != 0, axis=1)
    return active


def formulate_fba(p, production=None, energy=None, simplify=True):
    def ids(names):
        return np.asarray(p.substrateIndexs(names), dtype=int).ravel() - 1

    cyt = int(np.asarray(p.compartmentIndexs_cytosol).item()) - 1
    ext = int(np.asarray(p.compartmentIndexs_extracellular).item()) - 1
    stoich = np.asarray(p.reactionStoichiometryMatrix)
    ns, nr, nc = stoich.shape
    if production is not None and np.size(production):
        production = np.array(production, copy=True)
        production[np.abs(production) < 1e-4] = 0
        p.metabolismProduction = np.zeros(
            (ns, p.substrateMetaboliteGlobalIndexs.shape[1])
        )
        p.metabolismProduction[
            np.asarray(p.substrateMetaboliteLocalIndexs).ravel() - 1
        ] = production[
            p.substrateMetaboliteGlobalIndexs - 1,
            p.substrateMetaboliteCompartmentIndexs - 1,
        ]
    if energy is not None and np.size(energy):
        p.unaccountedEnergyConsumption = energy
    original = np.asarray(p.metabolismProduction)
    if np.any(np.sum(original != 0, axis=1) > 1):
        raise ValueError(
            "metabolism production spans multiple compartments for one substrate"
        )
    new, recycle = np.maximum(0, original), np.minimum(0, original)
    ntp, ndp, nmp = (
        ids(["ATP", "CTP", "GTP", "UTP"]),
        ids(["ADP", "CDP", "GDP", "UDP"]),
        ids(["AMP", "CMP", "GMP", "UMP"]),
    )
    recycled_nucleotides = recycle[ndp] + recycle[nmp]
    recycle[ntp] -= recycled_nucleotides
    new[ntp] += recycled_nucleotides
    water = ids(["H2O"])
    recycle[water] -= recycled_nucleotides.sum(axis=0)
    new[water] += recycled_nucleotides.sum(axis=0)
    hydrolysis = ids(["ATP", "ADP", "PI", "H2O", "H"])
    energy_vector = float(
        np.asarray(p.unaccountedEnergyConsumption).item()
    ) * np.array([-1, 1, 1, -1, 1])
    recycle[hydrolysis, cyt] += energy_vector
    new[hydrolysis, cyt] -= energy_vector
    met = ids(["MET", "H", "H2O"])
    formyl_met = recycle[ids(["FMET"])[0], cyt] * np.array([1, 1, -1])
    new[met, cyt] += formyl_met
    recycle[met, cyt] -= formyl_met
    folate = ids(["FTHF10", "FOR", "THF"])
    total = new[folate, cyt] + recycle[folate, cyt]
    signs = np.array([1, -1, -1])
    new[folate, cyt] = np.maximum(
        0, total - signs * max(0, np.min(total / signs))
    )
    recycle[folate, cyt] = total - new[folate, cyt]
    p.metabolismNewProduction, p.metabolismRecyclingProduction = new, recycle
    nz = np.flatnonzero(recycle.ravel(order="F") != 0)
    i, j = np.unravel_index(nz, recycle.shape, order="F")
    ni, total_sub = len(nz), ns + 8
    total_rxn = nr + ns + ni + 2
    conversion = np.arange(nr)
    external = nr + np.arange(ns)
    internal = nr + ns + np.arange(ni)
    limited = internal[recycle[i, j] < 0]
    unlimited = internal[recycle[i, j] > 0]
    biomass_production, biomass_exchange = total_rxn - 2, total_rxn - 1
    constraints = np.arange(ns, ns + 7) + cyt * total_sub
    biomass_row = ns + 7 + cyt * total_sub
    matrix = np.zeros((total_sub, total_rxn, nc))
    matrix[:ns, :nr, :] = stoich
    matrix[np.arange(ns), external, ext] = 1
    matrix[i, internal, j] = -1
    # Source ismember selects the first occurrence among recycling rows.
    location = {}
    for index, substrate in enumerate(i):
        location.setdefault(int(substrate), index)
    groups = [list(v) for v in zip(ntp, ndp, nmp)] + [
        list(ids(["FTHF10", "THF"])),
        list(ids(["FTHF10", "FOR", "FMET"])),
        list(ids(["MET", "FMET"])),
    ]
    for row, members in enumerate(groups):
        columns = [
            internal[location[int(member)]]
            for member in members
            if int(member) in location
        ]
        matrix[ns + row, columns, cyt] = 1
    matrix[:ns, biomass_production, :] = -new
    matrix[ns + 7, biomass_production, cyt] = matrix[
        ns + 7, biomass_exchange, cyt
    ] = 1
    full = matrix.copy()
    full[:ns, biomass_production, :] = -original
    matrix = matrix.transpose(0, 2, 1).reshape(
        total_sub * nc, total_rxn, order="F"
    )
    full = full.transpose(0, 2, 1).reshape(
        total_sub * nc, total_rxn, order="F"
    )
    catalysis = np.zeros((total_rxn, np.size(p.enzymeWholeCellModelIDs)))
    catalysis[:nr] = p.reactionCatalysisMatrix
    bounds = np.tile([-np.inf, np.inf], (total_rxn, 1))
    p.reactionBounds = np.where(
        np.asarray(p.reactionBounds) < 0,
        -np.inf,
        np.where(np.asarray(p.reactionBounds) > 0, np.inf, p.reactionBounds),
    )
    bounds[:nr] = p.reactionBounds
    ex_bounds = np.asarray(p.substrateExternalExchangeBounds)
    for column in range(2):
        valid = ~np.isnan(ex_bounds[:, column])
        bounds[external[valid], column] = ex_bounds[valid, column]
    bounds[limited, 1], bounds[unlimited, 0] = 0, 0
    bounds[biomass_production] = [0, np.inf]
    bounds[biomass_exchange] = [-np.inf, 0]
    enzyme_bounds = np.full((total_rxn, 2), np.nan)
    enzyme_bounds[:nr] = p.enzymeBounds
    objective = np.zeros(total_rxn)
    objective[biomass_production] = 1e3
    if limited.size:
        objective[limited] = np.divide(1.0, np.minimum(0, new).sum())
    rows, cols = np.arange(total_sub * nc), np.arange(total_rxn)
    removed_rows, removed_cols = np.zeros((len(rows), 3)), np.zeros(
        (len(cols), 3)
    )
    iteration = 0

    def prune(keep_rows, keep_cols, inner, reason):
        nonlocal rows, cols
        removed_rows[rows[~keep_rows]] = [iteration, inner, reason]
        removed_cols[cols[~keep_cols]] = [iteration, inner, reason]
        rows, cols = rows[keep_rows], cols[keep_cols]

    if simplify:
        while True:
            iteration += 1
            row_blocks, col_blocks = _blocks(full[np.ix_(rows, cols)])
            if np.all(row_blocks == 1):
                break
            block = row_blocks[rows == biomass_row]
            if block.size != 1:
                raise ValueError("source FBA pruning removed biomass row")
            prune(row_blocks == block.item(), col_blocks == block.item(), 0, 1)
            flags, inner = [True] * 3, 0
            while any(flags):
                inner += 1
                flags = [True] * 3
                while flags[0]:
                    current = full[np.ix_(rows, cols)]
                    leaves = np.count_nonzero(current, axis=1) == 1
                    affected = np.any(current[leaves] != 0, axis=0)
                    prune(~leaves, ~affected, inner, 2)
                    flags[0] = bool(np.any(leaves))
                current = full[np.ix_(rows, cols)]
                with np.errstate(invalid="ignore"):
                    lower, upper = (
                        current * bounds[cols, 0],
                        current * bounds[cols, 1],
                    )
                reversible = (
                    np.any(lower < 0, axis=1) | np.any(upper < 0, axis=1)
                ) & (np.any(lower > 0, axis=1) | np.any(upper > 0, axis=1))
                keep_cols = ~np.any(current[~reversible] != 0, axis=0)
                prune(reversible, keep_cols, inner, 3)
                flags[1] = not np.all(reversible)
                current = full[np.ix_(rows, cols)]
                active = _null_active(current)
                keep_rows = np.any(current[:, active] != 0, axis=1)
                prune(keep_rows, active, inner, 4)
                flags[2] = not np.all(keep_rows)
    rows = np.union1d(rows, constraints)
    removed_rows[constraints] = 0
    substrate_rows = np.concatenate(
        [np.arange(ns) + c * total_sub for c in range(nc)]
    )
    sub, comp = rows % total_sub, rows // total_sub
    normal = sub < ns
    p.fbaSubstrateIndexs_substrates = _column(np.flatnonzero(normal) + 1)
    p.fbaSubstrateIndexs_metaboliteInternalExchangeConstraints = _column(
        np.flatnonzero(np.isin(rows, constraints)) + 1
    )
    p.substrateIndexs_fba = _column(sub[normal] + comp[normal] * ns + 1)
    p.substrateIndexs_externalExchangedMetabolites = _column(
        cols[np.isin(cols, external)] - nr + 1
    )
    p.substrateIndexs_internalExchangedMetabolites = _column(nz + 1)
    p.substrateIndexs_internalExchangedLimitedMetabolites = _column(
        np.flatnonzero(recycle.ravel(order="F") < 0) + 1
    )
    p.fbaSubstrateIndexs_biomass = _column(
        np.flatnonzero(rows == biomass_row) + 1
    )
    if not np.all(np.isin(limited, cols)):
        raise ValueError(
            "source FBA pruning removed limited exchange reactions"
        )
    p.reactionIndexs_fba = _column(
        np.flatnonzero(np.isin(conversion, cols)) + 1
    )
    for name, indices in (
        ("metabolicConversion", conversion),
        ("metaboliteExternalExchange", external),
        ("metaboliteInternalExchange", internal),
        ("metaboliteInternalLimitedExchange", limited),
        ("metaboliteInternalUnlimitedExchange", unlimited),
        ("biomassProduction", [biomass_production]),
        ("biomassExchange", [biomass_exchange]),
    ):
        setattr(
            p,
            "fbaReactionIndexs_" + name,
            _column(np.flatnonzero(np.isin(cols, indices)) + 1),
        )
    p.fbaReactionStoichiometryMatrix = matrix[np.ix_(rows, cols)]
    p.fbaReactionCatalysisMatrix = catalysis[cols]
    p.fbaRightHandSide = np.zeros((len(rows), 1))
    p.fbaReactionBounds, p.fbaEnzymeBounds, p.fbaObjective = (
        bounds[cols],
        enzyme_bounds[cols],
        _column(objective[cols]),
    )
    return removed_rows[substrate_rows], removed_cols[conversion]
