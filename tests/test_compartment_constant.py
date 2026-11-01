from pathlib import Path

import numpy as np

from wholecell_py.compartment import Compartment
from wholecell_py.snapshot import load_snapshot


SNAPSHOT = (
    Path(__file__).resolve().parents[1] / "reference" / "snapshot_t0.json"
)
MATLAB_CLASS = "edu.stanford.covert.cell.sim.constant.Compartment"


def test_compartment_constants_load_from_snapshot_and_lookup_ids():
    graph = load_snapshot(SNAPSHOT, class_registry={MATLAB_CLASS: Compartment})
    compartment = graph.constants["compartment"]

    assert isinstance(compartment, Compartment)
    assert compartment.count.item() == 6
    ids = [
        (
            value.item()
            if isinstance(value, np.ndarray) and value.size == 1
            else "".join(value.ravel())
        )
        for value in compartment.wholeCellModelIDs[:, 0]
    ]
    assert ids == ["c", "d", "e", "m", "tc", "tm"]
    np.testing.assert_array_equal(
        compartment.cellularIndexs[:, 0], [1, 2, 4, 5, 6]
    )
    np.testing.assert_array_equal(
        compartment.getIndexs(["c", "tm", "missing"]), [1, 6, 0]
    )
    assert compartment.getIndexs("tc") == 5
