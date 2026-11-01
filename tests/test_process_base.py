import numpy as np
from types import SimpleNamespace

from wholecell_py.process_base import Process


def test_process_dry_weight_includes_attached_state_weights():
    p = Process(
        states=[SimpleNamespace(dryWeight=3.0), SimpleNamespace(dryWeight=7.0)]
    )
    p.getDryWeight = lambda: np.array([[2.0]])
    np.testing.assert_array_equal(p.dryWeight, [[12.0]])
    p.states = None
    np.testing.assert_array_equal(p.dryWeight, [[2.0]])


def test_copy_helpers_keep_matlab_one_based_f_order_and_copy():
    global_state = np.arange(1, 3 * 2 * 2 + 1).reshape((3, 2, 2), order="F")
    got = Process().copyFromStateHelper(
        global_state, np.array([[1, 3], [2, 1]]), np.array([[2, 1], [1, 2]])
    )
    np.testing.assert_array_equal(got[:, :, 0], [[4, 3], [2, 4]])
    assert not np.shares_memory(got, global_state)


def test_copy_to_helper_writes_mapping_without_aliasing_local_state():
    state = {"counts": np.zeros((3, 2, 2))}
    p = Process()
    p.substrates = np.arange(1, 9).reshape((2, 2, 2), order="F")
    p.copyToStateHelper(
        state, "counts", "substrates", [[1, 2]], [[1, 2]], [[1]]
    )
    expected = np.zeros((3, 2, 2))
    expected[0, 0, :] = [1, 5]
    expected[1, 1, :] = [3, 7]
    np.testing.assert_array_equal(state["counts"], expected)
    assert not np.shares_memory(state["counts"], p.substrates)


def test_index_conversion_uses_snapshot_one_based_indices():
    p = Process(
        substrateMetaboliteLocalIndexs=[2, 4],
        substrateMetaboliteGlobalIndexs=[8, 9],
    )
    np.testing.assert_array_equal(p.substrateToMetabolite([4, 2, 99]), [9, 8])


def test_multitime_gather_reads_live_mature_and_bound_maps():
    p = Process()
    p.rna = SimpleNamespace(
        matureIndexs=np.array([1.0]), boundIndexs=np.array([2.0])
    )
    p.enzymeWholeCellModelIDs = ["enzyme"]
    p.enzymeBoundMonomerGlobalCompartmentIndexs = np.array([[1.0]])
    for kind in ("Stimulus", "Metabolite", "RNA", "Monomer", "Complex"):
        setattr(p, "enzyme" + kind + "LocalIndexs", np.empty((0, 1)))
        setattr(p, "enzyme" + kind + "GlobalIndexs", np.empty((0, 1)))
        setattr(p, "enzyme" + kind + "CompartmentIndexs", np.empty((0, 1)))
        setattr(
            p, "enzyme" + kind + "GlobalCompartmentIndexs", np.empty((0, 1))
        )
    p.enzymeRNALocalIndexs = np.array([[1.0]])
    p.enzymeRNAGlobalIndexs = np.array([[1.0]])
    p.enzymeRNACompartmentIndexs = np.array([[1.0]])
    p.enzymeRNAGlobalCompartmentIndexs = np.array([[1.0]])
    p.enzymeBoundRNAGlobalCompartmentIndexs = np.array([[2.0]])

    values = np.arange(1.0, 13.0).reshape((3, 2, 2), order="F")
    empty = np.empty((3, 2, 2))
    arrays = (empty, empty, values, empty, empty)
    gathered, bound = p._copy_type("enzyme", arrays, bound=True)
    np.testing.assert_array_equal(gathered[0, 0], values[0, 0])
    np.testing.assert_array_equal(bound[0, 0], values[1, 0])

    p.rna.matureIndexs[:] = 3
    p.rna.boundIndexs[:] = 1
    gathered, bound = p._copy_type("enzyme", arrays, bound=True)
    np.testing.assert_array_equal(gathered[0, 0], values[2, 0])
    np.testing.assert_array_equal(bound[0, 0], values[0, 0])
