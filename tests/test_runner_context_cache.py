from types import SimpleNamespace

import numpy as np

from wholecell_py.runner import SimulationContext


class FixedOrder:
    def randperm(self, n):
        return np.arange(1, n + 1)


class EmptyProcess:
    def __init__(self, name):
        self.wholeCellModelID = name
        self.substrateMetaboliteLocalIndexs = np.empty((0, 1), dtype=int)
        self.substrateMetaboliteGlobalCompartmentIndexs = np.empty(
            (0, 1), dtype=int
        )
        self.substrates = np.empty((0, 1))
        self.simulationStateSideEffects = []

    def copyFromState(self):
        pass

    def calcResourceRequirements_Current(self):
        return self.substrates

    def evolveState(self):
        pass

    def copyToState(self):
        pass


def make_context():
    states = np.array(
        [
            SimpleNamespace(
                wholeCellModelID="State_Time", values=np.array([0.0])
            ),
            SimpleNamespace(
                wholeCellModelID="State_Metabolite",
                counts=np.array([[1.0]]),
                setCounts=np.empty((0, 6)),
            ),
            SimpleNamespace(
                wholeCellModelID="State_Stimulus",
                values=np.array([0.0]),
                setValues=np.empty((0, 6)),
            ),
        ],
        dtype=object,
    )
    processes = np.array(
        [EmptyProcess("Process_A"), EmptyProcess("Process_B")], dtype=object
    )
    context = SimulationContext()
    context.states = states
    context.processes = processes
    context.randStream = FixedOrder()
    context.stepSizeSec = np.array([1.0])
    context.processIndex_tRNAAminoacylation = np.array([1])
    context.processIndex_translation = np.array([2])
    return context


def test_coordinator_reuses_stable_runtime_objects_and_rebuilds_on_replacement():
    context = make_context()
    context.evolveState()
    coordinator = context._evolution_coordinator
    context.evolveState()
    assert context._evolution_coordinator is coordinator
    assert context.states[0].values[0] == 2

    replacement = EmptyProcess("Process_B_reloaded")
    context.processes[1] = replacement
    context.evolveState()
    assert context._evolution_coordinator is not coordinator
    assert context._evolution_coordinator.processes[1] is replacement

    rebuilt = context._evolution_coordinator
    context.stepSizeSec = np.array([2.0])
    context.evolveState()
    assert context._evolution_coordinator is not rebuilt
    assert context.states[0].values[0] == 5
