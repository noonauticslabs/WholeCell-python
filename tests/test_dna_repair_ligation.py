from types import SimpleNamespace
import numpy as np
from wholecell_py.processes.dna_repair import DNARepair
from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.randstream import RandStream


def test_ligation_uses_resource_limited_column_stoichiometry():
    p = DNARepair()
    breaks = CircularSparseMat([[1, 1], [2, 1], [3, 1]], [1, 1, 1], [10, 4], 1)
    p.chromosome = SimpleNamespace(
        strandBreaks=breaks, singleStrandBreaks=breaks
    )
    p.reactionIndexs_ligation = 1
    p.enzymeIndexs_ligase = 1
    p.reactionSmallMoleculeStoichiometryMatrix = np.array(
        [[-1.0], [1.0], [0.0]]
    )
    p.substrates = np.array([[2.0], [0.0], [0.0]])
    p.enzymes = np.array([[10.0]])
    p.enzymeBounds = np.array([[0.0, 1.0]])
    p.stepSizeSec = 1
    p.randStream = RandStream(seed=1)
    p.bindProteinToChromosome = lambda positions, enzyme, n, *args: (
        None,
        None,
        positions[:n],
        n,
        None,
    )
    p.evolveState_Ligate()
    np.testing.assert_array_equal(p.substrates, [[0.0], [2.0], [0.0]])
    assert p.chromosome.strandBreaks.nnz() == 1
