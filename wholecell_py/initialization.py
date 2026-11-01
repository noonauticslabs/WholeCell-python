"""Fresh cell sampling from fitted constants, from Simulation/initializeState.m.

This does not fit constants or build a model from a raw knowledge base. A
snapshot supplies the fitted object graph; its dynamic state is reallocated.
"""

import hashlib
import warnings
from pathlib import Path

import numpy as np

from .molecule_count_state import MoleculeCountState
from .runner import _text, load_model
from .side_effect import apply_side_effects
from .simulation import _conditions, _one_based

AVOGADRO = MoleculeCountState.nAvogadro


def _vec(value):
    return np.asarray(value).ravel(order="F")


def _idx(value):
    return _one_based(value, label="initialization index").ravel(order="F") - 1


def _scalar(value):
    return float(np.asarray(value).item())


def _dense(value):
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def _runtime_dimensions(obj):
    # MATLAB omits trailing singleton dimensions. Allocation methods also
    # support time series; the live single-cell runtime uses two dimensions.
    for key, value in vars(obj).copy().items():
        if isinstance(value, np.ndarray) and value.ndim > 2:
            shape = list(value.shape)
            while len(shape) > 2 and shape[-1] == 1:
                shape.pop()
            if tuple(shape) != value.shape:
                setattr(obj, key, value.reshape(shape, order="F"))


def _sample_proteins(sim, rna_weight, protein_weight):
    r, pm, mass, time = (
        sim.state(n) for n in ("Rna", "ProteinMonomer", "Mass", "Time")
    )
    rng = sim.randStream
    mature = _idx(pm.matureIndexs)
    gene_mrna = _idx(sim.gene.mRNAIndexs)
    composition = _dense(r.matureRNAGeneComposition)
    cycle = _scalar(time.cellCycleLength)
    decay = _vec(pm.decayRates)[mature]
    weights = _vec(pm.molecularWeights)
    fraction = _scalar(mass.initialFractionAAsInMonomers)
    if _text(sim.macromoleculeStateInitialization) == "expected":
        expression = np.zeros(weights.size)
        expression[mature] = (
            composition[gene_mrna] @ _vec(r.expression)[_idx(r.matureIndexs)]
        ) / (np.log(2) / cycle + decay)
        expression /= expression.sum()
        counts = rng.stochasticRound(
            fraction
            * protein_weight
            * AVOGADRO
            * expression
            / (weights @ expression)
        )
        pm.counts[np.arange(weights.size), _idx(pm.compartments)] = counts
        return
    n = _scalar(pm.macromoleculeStateInitializationVariation)
    mrna = _idx(r.matureMRNAIndexs)
    mature_mrna = _idx(r.matureIndexs)[mrna]
    expression = _vec(r.expression)[mature_mrna].copy()
    expression /= expression.sum()
    if rna_weight > 0:
        rna_decay = _vec(r.decayRates)[mature_mrna]
        production = expression * rna_decay
        production /= production.sum()
        counts = (
            expression
            * rna_weight
            * _scalar(r.weightFractionMRNA)
            * AVOGADRO
            / (expression @ _vec(r.molecularWeights)[mature_mrna])
        )
        total = (
            (counts.sum() * np.log(2) / cycle + rna_decay @ counts)
            * cycle
            / np.log(2)
        )
        sampled = _vec(rng.mnrnd(int(np.floor(n * total + 0.5)), production))
        expression = np.zeros_like(expression)
        for i, count in enumerate(sampled):
            if count > 0:
                expression[i] = count * np.mean(
                    rng.random(
                        "exponential", 1 / rna_decay[i], [int(count), 1]
                    )
                )
    if protein_weight <= 0:
        return
    production = composition[np.ix_(gene_mrna, mrna)] @ expression
    production /= production.sum()
    expression = production / (np.log(2) / cycle + decay)
    expression /= expression.sum()
    counts = (
        protein_weight * AVOGADRO * expression / (expression @ weights[mature])
    )
    total = counts @ (1 + cycle / np.log(2) * decay)
    sampled = _vec(rng.mnrnd(int(np.floor(n * total + 0.5)), expression))
    sampled = rng.stochasticRound(
        n
        * sampled
        * protein_weight
        * fraction
        * AVOGADRO
        / (sampled @ weights[mature])
    )
    cumulative = np.cumsum(sampled).astype(np.int64)
    order = rng.randperm(int(cumulative[-1]))
    selected = np.searchsorted(
        cumulative, order[: int(cumulative[-1] // 2)], side="left"
    )
    selected_counts = np.bincount(selected, minlength=mature.size)
    pm.counts[mature, _idx(pm.compartments)[mature]] = rng.stochasticRound(
        2 / n * selected_counts
    )


def _expected_complexes(sim):
    """Source mass-action allocation over connected subunit networks."""
    from scipy.sparse import bmat, csr_matrix
    from scipy.sparse.csgraph import connected_components

    r, pm, pc = (
        sim.state(n) for n in ("Rna", "ProteinMonomer", "ProteinComplex")
    )
    composition = _dense(pc.proteinComplexComposition)
    matrix = composition.sum(axis=2)
    genes = sim.gene
    subunits = np.zeros(matrix.shape[0])
    subunits[_idx(genes.mRNAIndexs)] = pm.counts[_idx(pm.matureIndexs)].sum(
        axis=1
    )
    for gene_field, rna_field in (
        ("rRNAIndexs", "matureRRNAIndexs"),
        ("sRNAIndexs", "matureSRNAIndexs"),
        ("tRNAIndexs", "matureTRNAIndexs"),
    ):
        subunits[_idx(getattr(genes, gene_field))] = r.counts[
            _idx(getattr(r, rna_field))
        ].sum(axis=1)
    processes = list(np.asarray(sim.processes, dtype=object).ravel(order="F"))
    formation_index = next(
        i + 1
        for i, p in enumerate(processes)
        if _text(p.wholeCellModelID) == "Process_MacromolecularComplexation"
    )
    excluded = (
        _vec(pc.formationProcesses)[_idx(pc.matureIndexs)] != formation_index
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        complexes = np.floor(np.nanmin(subunits[:, None] / matrix, axis=0))
    complexes[excluded] = 0
    edges = csr_matrix(matrix != 0)
    _, labels = connected_components(
        bmat([[None, edges], [edges.T, None]], format="csr")
    )
    for label in np.unique(labels):
        rows = np.flatnonzero(labels[: matrix.shape[0]] == label)
        cols = np.flatnonzero(labels[matrix.shape[0] :] == label)
        if cols.size <= 1:
            continue
        block = matrix[np.ix_(rows, cols)]
        available = subunits[rows].copy()
        allocated = np.zeros(cols.size)
        for iteration in range(100000):
            with np.errstate(divide="ignore", invalid="ignore"):
                rates = np.prod(
                    (available / available.mean())[:, None] ** block, axis=0
                )
                capacity = np.nanmin(available[:, None] / block, axis=0)
            rates[excluded[cols]] = 0
            if not np.any(rates) or np.max(capacity) < 1e-3:
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                allocated += rates * np.nanmin(available / (block @ rates))
            available = subunits[rows] - block @ allocated
        else:
            raise RuntimeError("Initial complex allocation did not converge")
        complexes[cols] = np.floor(allocated)
    if not np.all(np.isfinite(complexes)):
        raise ValueError("Nonfinite initial complex allocation")
    mature = _idx(pc.matureIndexs)
    pc.counts[mature, _idx(pc.compartments)[mature]] = complexes
    rna_rows = np.setdiff1d(
        np.arange(len(_idx(r.matureIndexs))), _idx(r.matureMRNAIndexs)
    )
    gene_rows = np.setdiff1d(
        np.arange(matrix.shape[0]), _idx(genes.mRNAIndexs)
    )
    r.counts[_idx(r.matureIndexs)[rna_rows]] -= np.einsum(
        "ijk,j->ik", composition[gene_rows], complexes
    )
    pm.counts[_idx(pm.matureIndexs)] -= np.einsum(
        "ijk,j->ik", composition[_idx(genes.mRNAIndexs)], complexes
    )


def _initialize_once(sim):
    mode = _text(sim.macromoleculeStateInitialization)
    if mode not in ("expected", "multinomial"):
        raise ValueError(f"Unsupported macromolecule initialization: {mode}")
    states = list(np.asarray(sim.states, dtype=object).ravel(order="F"))
    processes = list(np.asarray(sim.processes, dtype=object).ravel(order="F"))
    for state in states:
        state.allocateMemory(1)
        _runtime_dimensions(state)
    for process in processes:
        process.allocateMemoryForState(1)
        _runtime_dimensions(process)
    t, s, m, c, r, pm, mass, geom = (
        sim.state(n)
        for n in (
            "Time",
            "Stimulus",
            "Metabolite",
            "Chromosome",
            "Rna",
            "ProteinMonomer",
            "Mass",
            "Geometry",
        )
    )
    t.initialize()
    _runtime_dimensions(t)
    rng = sim.randStream
    total_weight = _scalar(
        rng.random("norm", 1, _scalar(mass.cellInitialMassVariation))
    )
    biomass = total_weight * np.asarray(m.biomassComposition)
    rna_weight = (
        total_weight
        * _scalar(mass.cellInitialDryWeight)
        * _scalar(mass.dryWeightFractionRNA)
    )
    protein_weight = (
        total_weight
        * _scalar(mass.cellInitialDryWeight)
        * _scalar(mass.dryWeightFractionProtein)
    )
    cytosol = int(_idx(sim.compartment.cytosolIndexs)[0])
    m.counts = np.maximum(0, rng.stochasticRound(biomass)).astype(float)
    c.initialize()
    _runtime_dimensions(c)
    chromosome_composition = np.r_[_vec(c.sequence.getBaseCounts()), 0].astype(
        float
    )
    repair = sim.process("DNARepair")
    methylated = 2 * np.asarray(repair.RM_MunI_RecognitionSites).shape[0]
    chromosome_composition[4] = methylated
    chromosome_composition[0] -= methylated
    m.counts[
        np.r_[_idx(m.dnmpIndexs), _idx(m.getIndexs("m6dAMP"))], cytosol
    ] -= chromosome_composition
    expression = _vec(r.expression)
    weights = _vec(r.molecularWeights) / AVOGADRO
    target = _scalar(mass.initialFractionNTPsInRNAs) * rna_weight
    if mode == "expected":
        counts = rng.stochasticRound(
            target * expression / (weights @ expression)
        )
    else:
        draws = (
            rng.randsample(
                expression.size,
                int(np.ceil(2 * rna_weight / (weights @ expression))),
                True,
                expression,
            )
            - 1
        )
        crossing = np.flatnonzero(np.cumsum(weights[draws]) > target)
        if not crossing.size:
            raise RuntimeError(
                "RNA sample did not reach source initialization mass cutoff"
            )
        counts = rng.stochasticRound(
            np.bincount(draws[: crossing[0]], minlength=expression.size)
        )
    r.counts[:, cytosol] = counts
    m.counts -= _dense(r.baseCounts).T @ r.counts
    _sample_proteins(sim, rna_weight, protein_weight)
    m.counts[:, cytosol] -= _dense(pm.baseCounts).T @ pm.counts.sum(axis=1)
    if mode == "expected":
        _expected_complexes(sim)
    volume = (
        total_weight
        * _scalar(mass.cellInitialDryWeight)
        / (1 - _scalar(mass.fractionWetWeight))
        / _scalar(geom.density)
    )
    for indices, concentration in (
        (m.ntpIndexs, m.meanNTPConcentration),
        (m.ndpIndexs, m.meanNDPConcentration),
    ):
        ids = _idx(indices)
        m.counts[ids[[0, 2]], cytosol] = np.floor(
            1e-3 * AVOGADRO * volume * _scalar(concentration) + 0.5
        )
        m.counts[ids[[1, 3]], cytosol] = 0
    m.counts[_idx(m.nmpIndexs), cytosol] = np.floor(
        1e-3 * AVOGADRO * volume * _scalar(m.meanNMPConcentration) + 0.5
    )
    ndp = m.counts[_idx(m.ndpIndexs), cytosol].sum()
    nmp = m.counts[_idx(m.nmpIndexs), cytosol].sum()
    for indices, value in (
        (m.phosphateIndexs, ndp),
        (m.diphosphateIndexs, nmp),
        (m.hydrogenIndexs, ndp + nmp),
    ):
        m.counts[_idx(indices), cytosol] = value
    for name in ("Mass", "Geometry", "FtsZRing", "Transcript", "Polypeptide"):
        state = sim.state(name)
        state.initialize()
        _runtime_dimensions(state)

    def conditions():
        m.counts = _conditions(m.counts, m.setCounts, _scalar(t.values))
        s.values = _conditions(s.values, s.setValues, _scalar(t.values))

    def clamp_metabolites():
        if np.any(m.counts < np.minimum(0, -biomass) * 0.15):
            warnings.warn(
                "Initialization metabolites below source tolerance before clamping",
                RuntimeWarning,
            )
        m.counts = np.maximum(0, m.counts)

    conditions()
    clamp_metabolites()
    order = _idx(sim.processInitOrderIndexs)
    if sorted(order.tolist()) != list(range(len(processes))):
        raise ValueError(
            "Initialization order must contain every process once"
        )
    for i in order:
        process = processes[i]
        process.simulationStateSideEffects = []
        process.copyFromState()
        try:
            process.initializeState()
        except Exception as error:
            raise RuntimeError(
                f"Initialization failed in {_text(process.wholeCellModelID)}"
            ) from error
        process.copyToState()
        apply_side_effects(process.simulationStateSideEffects, sim)
    for name, field in (
        ("Polypeptide", "abortedPolypeptides"),
        ("Transcript", "abortedTranscripts"),
    ):
        state = sim.state(name)
        setattr(state, field, getattr(state, field)[:0])
    clamp_metabolites()
    mass.calcMass()
    geom.calculateVolume()
    conditions()
    for process in processes:
        process.copyFromState()


def initialize_model(template, *, seed=1, max_attempts=100, mode=None):
    """Sample a new t=0 cell; return only after the original growth filter passes.

    Each rejection reloads fitted constants and uses a newly sampled seed.
    Implementation errors propagate immediately. Rejection exhaustion raises;
    it never silently returns an unacceptable cell. RNGs remain independent
    NumPy streams, as in the trajectory runner, not MATLAB bitstream replay.
    """
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    template = Path(template)
    attempts = []
    attempt_seed = int(seed)
    for attempt in range(max_attempts):
        graph = load_model(template, seed=attempt_seed)
        sim = graph.simulation
        sim.seed = attempt_seed
        if mode is not None:
            sim.macromoleculeStateInitialization = mode
        _initialize_once(sim)
        reaction = sim.state("MetabolicReaction")
        growth = _scalar(reaction.growth)
        mean = _scalar(reaction.meanInitialGrowthRate)
        width = _scalar(reaction.initialGrowthFilterWidth)
        if (
            not np.isfinite([growth, mean, width]).all()
            or mean <= 0
            or width < 0
        ):
            raise ValueError(
                "Invalid initialization growth or growth-filter constants"
            )
        bias = abs(growth - mean) / mean
        attempts.append(
            dict(seed=attempt_seed, growth=growth, relative_error=bias)
        )
        if bias <= width:
            for name, state in graph.states.items():
                counts = getattr(state, "counts", None)
                if counts is not None and (
                    not np.isfinite(counts).all() or np.any(counts < -1e-9)
                ):
                    raise ValueError(f"Invalid initialized counts in {name}")
            graph.initialization_provenance = dict(
                kind="fresh cell from MATLAB-fitted constants",
                root_seed=int(seed),
                template_sha256=hashlib.sha256(
                    template.read_bytes()
                ).hexdigest(),
                mode=_text(sim.macromoleculeStateInitialization),
                attempts=attempts,
                validation="experimental; no MATLAB initialization ensemble validation",
            )
            return graph
        attempt_seed = int(sim.randStream.randi(2**32).item()) - 1
    raise RuntimeError(
        f"Initialization growth filter rejected all {max_attempts} attempts: {attempts}"
    )
