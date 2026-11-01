import numpy as np

from wholecell_py.processes.ftsz_polymerization import FtsZPolymerization


def test_initialize_state_approaches_enzyme_steady_state_and_restores_substrates():
    process = FtsZPolymerization()
    process.enzymes = np.array([0.0])
    process.substrates = np.array([[4.0], [7.0]])
    calls = []

    def evolve():
        calls.append(1)
        process.substrates[:] = 1e6
        process.enzymes[:] = min(len(calls), 1)

    process.evolveState = evolve
    process.initializeState()

    assert len(calls) == 2
    np.testing.assert_array_equal(process.enzymes, [1.0])
    np.testing.assert_array_equal(process.substrates, [[4.0], [7.0]])
