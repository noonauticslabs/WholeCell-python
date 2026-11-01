from types import SimpleNamespace

import numpy as np

from wholecell_py.simulation import Simulation


class FixedRandom:
    def __init__(self, orders):
        self.orders = list(orders)

    def randperm(self, n):
        result = self.orders.pop(0)
        assert len(result) == n
        return np.asarray(result)


class ResourceProcess:
    def __init__(self, name, state, demand, use, log):
        self.name = name
        self.state = state
        self.demand = np.asarray(demand)
        self.use = use
        self.log = log
        self.substrateMetaboliteGlobalCompartmentIndexs = np.array([1])
        self.substrateMetaboliteLocalIndexs = np.array([1])
        self.substrates = np.zeros((1, 1), dtype=int)
        self.simulationStateSideEffects = []

    def copyFromState(self):
        self.log.append(
            (
                self.name,
                "copy",
                int(np.asarray(self.state.Metabolite.counts).ravel()[0]),
            )
        )

    def calcResourceRequirements_Current(self):
        return self.demand.copy()

    def evolveState(self):
        self.log.append((self.name, "evolve", int(self.substrates[0, 0])))
        self.substrates[0, 0] -= min(self.use, self.substrates[0, 0])

    def copyToState(self):
        self.log.append(
            (
                self.name,
                "write",
                int(np.asarray(self.state.Metabolite.counts).ravel()[0]),
            )
        )


def make_sim(
    processes,
    order,
    *,
    counts=None,
    stim_conditions=None,
    media_conditions=None,
):
    states = {
        "Time": SimpleNamespace(values=np.array([0.0])),
        "Metabolite": SimpleNamespace(
            counts=np.array(counts if counts is not None else [10]),
            setCounts=np.asarray(
                (
                    media_conditions
                    if media_conditions is not None
                    else np.empty((0, 6))
                ),
                dtype=float,
            ),
        ),
        "Stimulus": SimpleNamespace(
            values=np.array([0.0]),
            setValues=np.asarray(
                (
                    stim_conditions
                    if stim_conditions is not None
                    else np.empty((0, 6))
                ),
                dtype=float,
            ),
        ),
    }
    shared = SimpleNamespace(**states)
    for p in processes:
        p.state = shared
    return Simulation(states, processes, FixedRandom(order), 1.0)


def test_demand_shares_are_computed_before_runs_and_integer_allocations_are_fair():
    log = []
    a = ResourceProcess("A", None, [[1]], 2, log)
    b = ResourceProcess("B", None, [[1]], 1, log)
    sim = make_sim([a, b], [[1, 2]], counts=[10])
    original_counts = sim.states["metabolite"].counts
    req, alloc, use = sim.evolve_state()
    np.testing.assert_array_equal(req, [[1, 1]])
    np.testing.assert_array_equal(alloc, [[5, 5]])
    np.testing.assert_array_equal(use, [[2, 1]])
    np.testing.assert_array_equal(sim.states["metabolite"].counts, [7])
    # Scheduler writeback emulates MATLAB value semantics for aliased arrays.
    np.testing.assert_array_equal(original_counts, [10])
    assert (
        "B",
        "copy",
        8,
    ) in log  # second process sees first process writeback


def test_tRNA_must_precede_translation_and_conditions_apply_inclusive_time_window():
    log = []
    trna = ResourceProcess("tRNAAminoacylation", None, [[0]], 0, log)
    translation = ResourceProcess("Translation", None, [[0]], 0, log)
    sim = make_sim(
        [trna, translation],
        [[2, 1], [1, 2]],
        counts=[3],
        stim_conditions=[[1, 1, 8, 1, 1, 1]],
        media_conditions=[[1, 1, 4, 1, 1, 1]],
    )
    sim.evolve_state()
    events = [entry[0] for entry in log if entry[1] == "evolve"]
    assert events == ["tRNAAminoacylation", "Translation"]
    np.testing.assert_array_equal(sim.states["stimulus"].values, [8])
    np.testing.assert_array_equal(sim.states["metabolite"].counts, [4])


def test_column_major_condition_indices_and_substrate_gather_scatter():
    # Condition position 2 addresses the first-column/second-row MATLAB element.
    log = []
    p = ResourceProcess("P", None, [[1], [2]], 1, log)
    p.substrateMetaboliteGlobalCompartmentIndexs = np.array([2, 3])
    p.substrateMetaboliteLocalIndexs = np.array([1, 2])
    p.substrates = np.zeros((2, 1), dtype=int)
    states = {
        "Time": SimpleNamespace(values=np.array([0.0])),
        "Metabolite": SimpleNamespace(
            counts=np.array([[5, 0], [5, 0]]),
            setCounts=np.array([[1, 1, 9, 0, 1, 2]]),
        ),
        "Stimulus": SimpleNamespace(
            values=np.zeros(1), setValues=np.empty((0, 6))
        ),
    }
    p.state = SimpleNamespace(**states)
    sim = Simulation(states, [p], FixedRandom([[1]]), 1.0)
    sim.evolve_state()
    assert states["Metabolite"].counts[1, 0] == 9


def test_nonallocated_molecule_updates_and_side_effects_remain_visible():
    class Effect:
        def updateSimulationState(self, simulation):
            simulation.states["metabolite"].counts[1] += 3

    class Writer(ResourceProcess):
        def evolveState(self):
            super().evolveState()
            self.simulationStateSideEffects = Effect()

        def copyToState(self):
            self.state.Metabolite.counts[1] += 7

    class Reader(ResourceProcess):
        def evolveState(self):
            assert self.state.Metabolite.counts[1] == 12
            super().evolveState()

    writer = Writer("writer", None, [[1]], 1, [])
    reader = Reader("reader", None, [[1]], 1, [])
    sim = make_sim([writer, reader], [[1, 2]], counts=[10, 2])
    sim.evolve_state()
    np.testing.assert_array_equal(sim.states["metabolite"].counts, [8, 12])


def test_matlab_double_index_maps_are_accepted_without_rounding():
    process = ResourceProcess("P", None, [[1]], 1, [])
    process.substrateMetaboliteLocalIndexs = np.array([[1.0]])
    process.substrateMetaboliteGlobalCompartmentIndexs = np.array([[1.0]])
    simulation = make_sim([process], [[1]], counts=[5])
    simulation.evolve_state()
    np.testing.assert_array_equal(simulation.states["metabolite"].counts, [4])
