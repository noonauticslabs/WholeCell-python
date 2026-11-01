import numpy as np
import pytest

from wholecell_py.states.host import Host


def test_allocate_memory_uses_matlab_state_shapes_and_bool_dtype():
    host = Host("State_Host", "Host")
    host.allocateMemory(4)
    host.initialize()
    assert host.isBacteriumAdherent.shape == (1, 1, 4)
    assert host.isBacteriumAdherent.dtype == np.bool_
    assert host.isTLRActivated.shape == (3, 1, 4)
    assert host.isTLRActivated.dtype == np.bool_
    np.testing.assert_array_equal(host.isBacteriumAdherent, False)
    np.testing.assert_array_equal(host.isTLRActivated, False)


def test_snapshot_and_keyword_attributes_are_restored_with_array_copy():
    tlr = np.ones((3, 1, 2), dtype=bool)
    host = Host(
        snapshot={"isTLRActivated": tlr, "wholeCellModelID": "State_Host"},
        isNFkBActivated=True,
    )
    assert host.isTLRActivated is not tlr
    np.testing.assert_array_equal(host.isTLRActivated, tlr)
    assert host.wholeCellModelID == "State_Host"
    assert host.isNFkBActivated is True
    assert host.stateNames == (
        "isBacteriumAdherent",
        "isTLRActivated",
        "isNFkBActivated",
        "isInflammatoryResponseActivated",
    )


def test_allocate_memory_validates_timepoint_count():
    host = Host()
    with pytest.raises(TypeError, match="integer"):
        host.allocateMemory(1.5)
    with pytest.raises(ValueError, match="non-negative"):
        host.allocateMemory(-1)
