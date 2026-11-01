import json
from pathlib import Path

import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.chromosome import Chromosome

ROOT = Path(__file__).resolve().parents[1]


def test_all_region_query_outputs_match_matlab_for_intact_and_partial_chromosomes():
    graph = load_snapshot(
        ROOT / "reference/snapshot_t0.json", runtime_values=True
    )
    chromosome = Chromosome(snapshot=graph.states["Chromosome"].fields)
    cases = json.loads((ROOT / "reference/region_queries.json").read_text())[
        "cases"
    ]

    def unpack(value):
        return np.asarray(value["data"]).reshape(value["shape"], order="F")

    for case in cases:
        chromosome.polymerizedRegions = CircularSparseMat(
            unpack(case["polymerizedSubs"]),
            unpack(case["polymerizedLengths"]),
            [int(np.asarray(chromosome.sequenceLen).item()), 4],
            1,
        )
        result = getattr(chromosome, case["method"])(
            unpack(case["positions"]), unpack(case["lengths"]), case["extent"]
        )
        for actual, field in zip(
            result, ("flags", "indices", "resultPositions", "resultLengths")
        ):
            expected = unpack(case[field])
            np.testing.assert_array_equal(
                np.asarray(actual).ravel(order="F"),
                expected.ravel(order="F"),
                err_msg=f"scenario={case['scenario']} method={case['method']} extent={case['extent']} field={field}",
            )


def test_chromosome_mass_recomputes_from_sparse_damage_and_live_sequence():
    graph = load_snapshot(
        ROOT / "reference/snapshot_t0.json", runtime_values=True
    )
    chromosome = Chromosome(snapshot=graph.states["Chromosome"].fields)
    expected = json.loads(
        (ROOT / "reference/state_weights_t0.json").read_text()
    )["Chromosome"]["data"]
    np.testing.assert_allclose(
        chromosome.calcDryWeight(), expected, rtol=1e-14, atol=0
    )
    previous = chromosome.dryWeight.copy()
    chromosome.damagedBases = CircularSparseMat(
        [], [], chromosome.damagedBases.siz, 1
    )
    assert not np.array_equal(chromosome.dryWeight, previous)


def test_occupied_and_damaged_region_queries_match_matlab():
    graph = load_snapshot(
        ROOT / "reference/snapshot_t0.json", runtime_values=True
    )
    chromosome = Chromosome(snapshot=graph.states["Chromosome"].fields)
    reference = json.loads(
        (ROOT / "reference/accessibility_queries.json").read_text()
    )

    def unpack(value):
        return np.asarray(value["data"]).reshape(value["shape"], order="F")

    for field, record in reference["inputState"].items():
        setattr(
            chromosome,
            field,
            CircularSparseMat(
                unpack(record["subs"]),
                unpack(record["vals"]),
                [int(np.asarray(chromosome.sequenceLen).item()), 4],
                1,
            ),
        )
    positions, lengths = unpack(reference["positions"]), unpack(
        reference["lengths"]
    )
    for case in reference["cases"]:
        if case["method"] == "isRegionProteinFree":
            result = chromosome.isRegionProteinFree(
                positions,
                lengths,
                case["extent"],
                [0],
                [0],
                case["both"],
                case["any"],
            )
        else:
            result = chromosome.isRegionUndamaged(
                positions,
                lengths,
                case["both"],
                [],
                case["extent"],
                case["any"],
            )
        for actual, field in zip(
            result,
            (
                "flags",
                "indices",
                "positions",
                "lengths",
                "monomers",
                "complexs",
            ),
        ):
            if hasattr(actual, "full"):
                actual = actual.full()
            np.testing.assert_array_equal(
                np.asarray(actual).ravel(order="F"),
                unpack(case[field]).ravel(order="F"),
                err_msg=f"{case['method']} extent={case['extent']} both={case['both']} any={case['any']} field={field}",
            )


def test_all_exported_chromosome_derived_views_match_runtime_restored_matlab_states():
    from wholecell_py.runner import load_model
    from wholecell_py.sparsemat import SparseMat
    from wholecell_py.states.chromosome import _DERIVED

    for name in ("snapshot_t0.json", "snapshot_seed22_t4000.json"):
        graph = load_model(ROOT / "reference" / name, seed=1)
        state = graph.states["Chromosome"]
        for field in _DERIVED:
            expected = getattr(state, "_exported_" + field, None)
            if expected is None:
                continue
            actual = getattr(state, field)
            if isinstance(expected, SparseMat):
                assert isinstance(actual, SparseMat), field
                np.testing.assert_array_equal(
                    actual.siz, expected.siz, err_msg=field
                )
                np.testing.assert_array_equal(
                    actual.subs, expected.subs, err_msg=field
                )
                np.testing.assert_array_equal(
                    actual.vals, expected.vals, err_msg=field
                )
            else:
                np.testing.assert_array_equal(
                    np.asarray(actual).reshape(-1),
                    np.asarray(expected).reshape(-1),
                    err_msg=f"{name}: {field}",
                )
