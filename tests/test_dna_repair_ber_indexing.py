from types import SimpleNamespace

import numpy as np

from wholecell_py.processes.dna_repair import DNARepair
from wholecell_py.randstream import RandStream


class BindingChromosome:
    """Small binding-state fixture returning MATLAB-style candidate row indices."""

    def __init__(self):
        self.damagedBases = np.zeros((6, 2), dtype=int)
        self.abasicSites = np.zeros((6, 2), dtype=int)
        self.abasicSites[2, 0] = 1
        self.strandBreaks3 = np.zeros((6, 2), dtype=int)
        self.strandBreaks5 = np.zeros((6, 2), dtype=int)
        self.damagedSugarPhosphates = np.ones((6, 2), dtype=int)
        self.strandBreaks = np.zeros((6, 2), dtype=int)
        self.bound_position = None

    def setSiteProteinBound(self, positions, maximum, *args):
        self.bound_position = np.asarray(positions, dtype=int).copy()
        # bindProteinToChromosome's idxs are one based: the only candidate is row 1.
        return (
            np.zeros((1, 1)),
            np.zeros((0, 1)),
            np.empty((0, 1), object),
            np.array([True]),
            np.array([1]),
            self.bound_position.copy(),
            np.array([1]),
        )

    @staticmethod
    def shiftPositionsStrandsBond3Prime(positions):
        shifted = np.asarray(positions, dtype=int).copy()
        shifted[:, 0] += 1
        return shifted


def test_ber_single_abasic_site_uses_one_based_binding_index_as_row_one():
    p = DNARepair()
    chromosome = BindingChromosome()
    p.chromosome = chromosome
    p.enzymeMonomerLocalIndexs = np.array([1])
    p.enzymeMonomerGlobalIndexs = np.array([10])
    p.enzymeComplexLocalIndexs = np.array([], dtype=int)
    p.enzymeComplexGlobalIndexs = np.array([], dtype=int)
    p.enzymes = np.array([1.0])
    p.boundEnzymes = np.array([0.0])
    p.simulationStateSideEffects = np.empty((0, 1), object)
    p.enzymeIndexs_BER_baseExcision = np.array([], dtype=int)
    p.reactionIndexs_BER_baseexcision = np.array([], dtype=int)
    p.reactionIndexs_BER_aplyase = np.array([1])
    p.reactionIndexs_BER_apendonuclease = np.array([2])
    p.reactionIndexs_BER_drpesterase = np.array([3])
    p.reactionIndexs_BER_dr5plyase = np.array([4])
    p.enzymeIndexs_apurinicEndonuclease = np.array([1])
    p.enzymeIndexs_polymerase = np.array([1])
    p.reactionCatalysisMatrix = np.array([[1], [1], [0], [0]])
    p.reactionSmallMoleculeStoichiometryMatrix = np.array(
        [[-1, -2, 0, 0], [1, 2, 0, 0]], dtype=float
    )
    p.substrates = np.array([[1.0], [0.0]])
    p.enzymeBounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=float)
    p.stepSizeSec = 1
    p.randStream = RandStream(seed=1)
    # Ensure the first AP lyase reaction is considered, then the alternative.
    p.randStream.randperm = lambda n: np.array([1, 2])

    p.evolveState_BER()

    np.testing.assert_array_equal(chromosome.bound_position, [[3, 1]])
    assert chromosome.damagedSugarPhosphates[2, 0] == 0
    assert chromosome.strandBreaks[3, 0] == 1
    np.testing.assert_array_equal(p.substrates, [[0.0], [1.0]])


def test_matlab_candidate_indices_select_last_dsb_without_overrun():
    rows = np.array([[10, 1], [20, 1]])
    np.testing.assert_array_equal(DNARepair._matlab_rows(rows, [2]), [[20, 1]])


def test_dense_find_coordinates_keep_matlab_column_major_order():
    mask = np.array([[False, True], [True, False]])
    np.testing.assert_array_equal(
        DNARepair._matlab_find_coordinates(mask), [[2, 1], [1, 2]]
    )


def test_odd_compartment_find_coordinates_use_compressed_chromosome_copy_ids():
    np.testing.assert_array_equal(
        DNARepair._chromosome_copy_coordinates([[10, 1], [20, 3]]),
        [[10, 1], [20, 2]],
    )


def test_sparse_subscripts_sort_by_matlab_column_major_linear_index():
    coords = np.array([[1, 2], [2, 1]])
    np.testing.assert_array_equal(
        DNARepair._sort_matlab_subscripts(coords, 3), [[2, 1], [1, 2]]
    )
