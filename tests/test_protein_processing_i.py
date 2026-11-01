import numpy as np
from pathlib import Path

from wholecell_py.processes.protein_processing_i import ProteinProcessingI
from wholecell_py.randstream import RandStream
from wholecell_py.snapshot import load_snapshot


class Stream:
    def stochasticRound(self, values):
        return np.floor(np.asarray(values) + 0.5).astype(int)

    def mnrnd(self, n, p):
        p = np.asarray(p, dtype=float)
        if p.sum() == 0:
            return np.zeros_like(p, dtype=int)
        # Deterministic multinomial stand-in for this contract test.
        return np.floor(int(n) * p / p.sum()).astype(int)


def process():
    return ProteinProcessingI(
        deformylaseSpecificRate=38,
        methionineAminoPeptidaseSpecificRate=6,
        nascentMonomerNTerminalMethionineCleavages=np.array([True, False]),
        unprocessedMonomerWholeCellModelIDs=["a", "b"],
        processedMonomerWholeCellModelIDs=["a", "b"],
        substrates=np.array([10.0, 0.0, 0.0, 0.0]),
        enzymes=np.array([1.0, 1.0]),
        unprocessedMonomers=np.array([[3.0], [4.0]]),
        processedMonomers=np.zeros((2, 1)),
        randStream=Stream(),
        stepSizeSec=1,
    )


def test_resource_demand_uses_water_limit_and_cleavable_species():
    p = process()
    demand = p.calcResourceRequirements_Current()
    assert demand[0] == 10


def test_evolve_processes_cleavage_and_deformylation_and_balances_metabolites():
    p = process()
    p.evolveState()
    np.testing.assert_array_equal(p.unprocessedMonomers, [[0], [0]])
    np.testing.assert_array_equal(p.processedMonomers, [[3], [4]])
    assert p.substrates[0] == 0
    assert (
        p.substrates[3] == 7 and p.substrates[1] == 7 and p.substrates[2] == 3
    )


def test_snapshot_roundtrip_preserves_singleton_time_axis():
    snapshot = Path(__file__).parents[1] / "reference" / "snapshot_t0.json"
    g = load_snapshot(snapshot)
    p = ProteinProcessingI(**g.processes["ProteinProcessingI"].fields)
    p.randStream = RandStream(seed=1)
    p.copyFromState()
    assert p.unprocessedMonomers.ndim == 3
    assert p.unprocessedMonomers.shape[1:] == (1, 1)
    p.calcResourceRequirements_Current()
    p.evolveState()
    p.copyToState()
    assert np.asarray(p.monomer.counts).ndim == 3
