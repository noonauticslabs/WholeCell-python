import numpy as np

from wholecell_py.processes.host_interaction import HostInteraction
from wholecell_py.states.host import Host


def make_process(enzymes):
    host = Host(
        tlrIDs=("TLR1", "TLR2", "TLR6"), isTLRActivated=np.zeros((3, 1), bool)
    )
    process = HostInteraction(
        enzymes=np.asarray(enzymes, dtype=bool).reshape(15, 1), host=host
    )
    return process, host


def test_resource_requirements_match_source_dimensions():
    p = HostInteraction(substrates=np.ones((2, 3)))
    bm, by, lo, hi = p.calcResourceRequirements_LifeCycle()
    assert bm.shape == by.shape == (0,)
    assert lo.shape == hi.shape == (15,)
    assert np.all(lo == 0) and np.all(np.isinf(hi))
    np.testing.assert_array_equal(
        p.calcResourceRequirements_Current(), np.zeros((2, 3))
    )


def test_host_activation_and_inflammatory_rules():
    enzymes = np.zeros(15, bool)
    enzymes[[2, 3, 5, 6, 9, 10, 11, 12]] = (
        True  # required organelle/core/adhesins
    )
    enzymes[[1, 13]] = True  # TLR1/2 ligands
    p, h = make_process(enzymes)
    p.evolveState()
    assert h.isBacteriumAdherent
    np.testing.assert_array_equal(
        h.isTLRActivated.ravel(), [True, True, False]
    )
    assert h.isNFkBActivated and h.isInflammatoryResponseActivated


def test_nonadherent_cell_does_not_activate_tlrs_or_adhesion_response():
    enzymes = np.zeros(15, bool)
    enzymes[[1, 8, 13]] = True
    p, h = make_process(enzymes)
    p.initializeState()
    assert not h.isBacteriumAdherent
    np.testing.assert_array_equal(
        h.isTLRActivated.ravel(), [False, False, False]
    )
    assert not h.isNFkBActivated and not h.isInflammatoryResponseActivated


def test_antigen_response_requires_adherence_when_nfkb_is_inactive():
    enzymes = np.zeros(15, bool)
    enzymes[[2, 3, 5, 6, 9, 10, 11, 12, 0]] = True
    p, h = make_process(enzymes)
    p.evolveState()
    assert h.isBacteriumAdherent
    assert h.isInflammatoryResponseActivated
