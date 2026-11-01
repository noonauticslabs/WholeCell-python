from pathlib import Path

import numpy as np
import pytest

from wholecell_py.gene import Gene
from wholecell_py.snapshot import load_snapshot


SNAPSHOT = Path(__file__).parents[1] / "reference" / "snapshot_t0.json"


def test_snapshot_constants_and_known_id_lookup_match_export():
    graph = load_snapshot(SNAPSHOT)
    source = graph.constants["gene"].fields
    gene = Gene(snapshot=source)

    assert gene.wholeCellModelIDs.shape == (525, 1)
    known = gene.wholeCellModelIDs[[0, 147, 524], 0]
    np.testing.assert_array_equal(gene.getIndexs(known), [1, 148, 525])
    assert gene.getIndexs("not-a-gene") == 0
    assert gene.getIndexsByPosition(int(gene.startCoordinates[0, 0])) == 1
    assert gene.getIndexsByPosition(-1) is None
    with pytest.raises(ValueError):
        gene.lengths[0, 0] = 0


def test_constructor_kwargs_override_snapshot_and_preserve_shapes():
    gene = Gene(
        snapshot={"wholeCellModelIDs": np.array([["old"]]), "lengths": [[4]]},
        wholeCellModelIDs=np.array([["new"], ["next"]]),
    )
    np.testing.assert_array_equal(gene.getIndexs(["next", "missing"]), [2, 0])
    assert gene.lengths.shape == (1, 1)
