import numpy as np
import pytest

from wholecell_py.states.rna import Rna
from wholecell_py.states.protein_monomer import ProteinMonomer
from wholecell_py.states.protein_complex import ProteinComplex


@pytest.mark.parametrize("state_type", [Rna, ProteinMonomer, ProteinComplex])
def test_get_indexes_preserves_matrix_element_positions(state_type):
    # MATLAB ismember(A,B) returns Locb with A's shape and element positions.
    state = state_type(
        wholeCellModelIDs=np.array(["C", "A", "D", "B"]),
        matureIndexs=np.array([2, 4, 1, 3]),
    )
    query = np.array([["A", "B", "missing"], ["D", "C", "A"]])
    np.testing.assert_array_equal(
        state.getIndexs(query), [[1, 2, 0], [4, 3, 1]]
    )
