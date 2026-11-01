import numpy as np
from types import SimpleNamespace

from wholecell_py.processes.protein_processing_ii import ProteinProcessingII
from wholecell_py.randstream import RandStream


def configured():
    p = ProteinProcessingII(
        substrates=np.array([10.0, 0.0, 10.0, 0.0, 0.0]),
        enzymes=np.array([1.0, 1.0]),
        unprocessedMonomers=np.array([2.0, 1.0, 4.0]),
        processedMonomers=np.zeros(3),
        signalSequenceMonomers=np.zeros(3),
        lipoproteinMonomerIndexs=np.array([1]),
        secretedMonomerIndexs=np.array([2]),
        unprocessedMonomerIndexs=np.array([3]),
        lipoproteinSignalPeptidaseSpecificRate=10.0,
        lipoproteinDiacylglycerylTransferaseSpecificRate=10.0,
        randStream=RandStream(seed=3),
        stepSizeSec=1.0,
    )
    return p


def test_lifecycle_resource_requirements():
    p = configured()
    state = SimpleNamespace(
        monomerProductions=np.array([1.0, 2.0, 3.0]),
        monomerProductions0=np.array([1.0, 2.0, 3.0]),
    )
    bm, by, lo, hi = p.calcResourceRequirements_LifeCycle(None, state)
    np.testing.assert_array_equal(bm, [3, 0, 1, 0, 0])
    np.testing.assert_array_equal(by, [0, 0, 0, 1, 1])
    np.testing.assert_allclose(lo, [0.6, 0.2])
    assert np.all(np.isinf(hi))


def test_evolve_processes_and_passes_through_unaffected_protein():
    p = configured()
    p.evolveState()
    # Enzyme and substrate capacities suffice to process all three targets.
    np.testing.assert_array_equal(p.unprocessedMonomers, [0, 0, 0])
    np.testing.assert_array_equal(p.processedMonomers, [2, 1, 4])
    np.testing.assert_array_equal(p.signalSequenceMonomers, [2, 1, 0])
    np.testing.assert_array_equal(p.substrates, [7, 0, 8, 2, 2])


def test_evolve_lipoprotein_transfer_with_column_shaped_snapshot_state():
    p = configured()
    # Snapshot state arrays retain their singleton compartment dimension.
    p.unprocessedMonomers = p.unprocessedMonomers.reshape(-1, 1)
    p.processedMonomers = p.processedMonomers.reshape(-1, 1)
    p.signalSequenceMonomers = p.signalSequenceMonomers.reshape(-1, 1)
    p.evolveState()
    np.testing.assert_array_equal(p.unprocessedMonomers.ravel(), [0, 0, 0])
    np.testing.assert_array_equal(p.processedMonomers.ravel(), [2, 1, 4])
    np.testing.assert_array_equal(p.signalSequenceMonomers.ravel(), [2, 1, 0])
    np.testing.assert_array_equal(p.substrates, [7, 0, 8, 2, 2])


def test_current_requirements_respect_enzyme_rates_and_substrate_rows():
    p = configured()
    np.testing.assert_array_equal(
        p.calcResourceRequirements_Current(), [3, 0, 2, 0, 0]
    )


def test_column_constant_transferase_cap_limits_lipid_processing():
    p = configured()
    p.lipoproteinDiacylglycerylTransferaseSpecificRate = np.array([[1.0]])
    p.stepSizeSec = np.array([[1.0]])
    p.evolveState()
    np.testing.assert_array_equal(p.unprocessedMonomers, [1, 0, 0])
    np.testing.assert_array_equal(p.processedMonomers, [1, 1, 4])
    np.testing.assert_array_equal(p.substrates, [8, 0, 9, 1, 1])


def test_column_constant_zero_peptidase_blocks_signal_processing():
    p = configured()
    p.lipoproteinSignalPeptidaseSpecificRate = np.array([[0.0]])
    p.stepSizeSec = np.array([[1.0]])
    p.evolveState()
    np.testing.assert_array_equal(p.unprocessedMonomers, [2, 1, 0])
    np.testing.assert_array_equal(p.processedMonomers, [0, 0, 4])
    np.testing.assert_array_equal(p.substrates, [10, 0, 10, 0, 0])
