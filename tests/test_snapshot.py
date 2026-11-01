import json

import numpy as np
import pytest

from wholecell_py.snapshot import (
    ChromosomeSequenceData,
    DataObject,
    MatlabFunctionExpression,
    RNGData,
    ValueObjectData,
    load_snapshot,
)
from wholecell_py.sparsemat import SparseMat


def write_snapshot(tmp_path, objects, *, complete=True, errors=None):
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "complete": complete,
                "errors": errors or [],
                "objects": objects,
            }
        )
    )
    return path


def test_arrays_restore_dtype_shape_fortran_order_and_nonfinite(tmp_path):
    fields = {
        "matrix": {
            "kind": "array",
            "dtype": "int16",
            "shape": [2, 2],
            "data": [1, 2, 3, 4],
        },
        "special": {
            "kind": "array",
            "dtype": "double",
            "shape": [3, 1],
            "data": [0, 0, 0],
            "nonfinite_indices": [1, 2, 3],
            "nonfinite_values": ["NaN", "Inf", "-Inf"],
        },
        "chars": {"kind": "char", "shape": [1, 3], "data": "abc"},
    }
    graph = load_snapshot(
        write_snapshot(
            tmp_path, [{"id": "simulation", "class": "Sim", "fields": fields}]
        )
    )
    assert graph.simulation.matrix.dtype == np.int16
    np.testing.assert_array_equal(graph.simulation.matrix, [[1, 3], [2, 4]])
    assert np.isnan(graph.simulation.special[0, 0])
    assert np.isposinf(graph.simulation.special[1, 0])
    assert np.isneginf(graph.simulation.special[2, 0])
    assert graph.simulation.chars.shape == (1, 3)


def test_multicolumn_cells_structs_and_value_objects_keep_column_order(
    tmp_path,
):
    fields = {
        "cells": {"kind": "cell", "shape": [2, 2], "data": [1, 2, 3, 4]},
        "structs": {
            "kind": "struct",
            "shape": [2, 2],
            "data": [{"n": n} for n in range(4)],
        },
        "values": {
            "kind": "value_objects",
            "class": "Value",
            "shape": [2, 2],
            "data": [{"n": n} for n in range(4)],
        },
    }
    graph = load_snapshot(
        write_snapshot(
            tmp_path, [{"id": "simulation", "class": "Sim", "fields": fields}]
        )
    )
    np.testing.assert_array_equal(graph.simulation.cells, [[1, 3], [2, 4]])
    assert graph.simulation.structs[0, 1]["n"] == 2
    assert graph.simulation.values[1, 0].n == 1


def test_graph_identity_cycles_maps_and_tagged_data(tmp_path):
    refs = {"kind": "reference", "id": "state:CellMass"}
    fields = {
        "self": {"kind": "reference", "id": "simulation"},
        "state": refs,
        "cell": {"kind": "cell", "shape": [1, 2], "data": [refs, None]},
        "struct": {
            "kind": "struct",
            "shape": [1, 1],
            "data": [{"owner": refs}],
        },
        "sparse": {
            "kind": "sparse",
            "class": "MATLAB.sparse",
            "shape": [3, 3],
            "subs": {
                "kind": "array",
                "dtype": "int64",
                "shape": [1, 2],
                "data": [2, 3],
            },
            "vals": {
                "kind": "array",
                "dtype": "double",
                "shape": [1, 1],
                "data": [5],
            },
            "circularDimensions": {
                "kind": "array",
                "dtype": "int64",
                "shape": [1, 1],
                "data": [2],
            },
        },
        "sequence": {
            "kind": "chromosome_sequence",
            "positive_strand": {"kind": "char", "shape": [1, 2], "data": "AC"},
        },
        "rng": {"kind": "rng", "type": "mt", "seed": 42, "state": [1, 2]},
        "fn": {"kind": "matlab_function", "expression": "@(x)x+1"},
    }
    records = [
        {"id": "simulation", "class": "Sim", "fields": fields},
        {"id": "constant:compartment", "class": "Comp", "fields": {}},
        {"id": "constant:gene", "class": "Gene", "fields": {}},
        {
            "id": "state:CellMass",
            "class": "Mass",
            "fields": {"back": {"kind": "reference", "id": "simulation"}},
        },
        {"id": "process:Metabolism", "class": "Met", "fields": {"mass": refs}},
    ]
    graph = load_snapshot(write_snapshot(tmp_path, records))
    assert graph.simulation.self is graph.simulation
    state = graph.states["CellMass"]
    assert graph.simulation.state is state and state.back is graph.simulation
    assert graph.processes["Metabolism"].mass is state
    assert (
        graph.constants["compartment"] is graph.objects["constant:compartment"]
    )
    assert graph.simulation.cell[0, 0] is state
    assert graph.simulation.struct[0, 0]["owner"] is state
    assert isinstance(graph.simulation.sparse, SparseMat)
    assert graph.simulation.sparse.circular_dimensions[0, 0] == 2
    assert graph.simulation.sequence == ChromosomeSequenceData(
        graph.simulation.sequence.positive_strand
    )
    assert isinstance(graph.simulation.rng, RNGData)
    assert isinstance(graph.simulation.fn, MatlabFunctionExpression)


def test_incomplete_and_strict_runtime_errors_are_explicit(tmp_path):
    path = write_snapshot(
        tmp_path,
        [{"id": "simulation", "class": "Sim", "fields": {}}],
        complete=False,
        errors=[{"object": "simulation", "field": "x", "error": "bad"}],
    )
    with pytest.raises(ValueError, match="incomplete"):
        load_snapshot(path)
    assert isinstance(
        load_snapshot(path, allow_incomplete=True).simulation, DataObject
    )
    with pytest.raises(ValueError, match="no runtime class"):
        load_snapshot(
            write_snapshot(
                tmp_path, [{"id": "simulation", "class": "Sim", "fields": {}}]
            ),
            strict_runtime=True,
        )


def test_value_objects_keep_shape_and_unresolved_matlab_class(tmp_path):
    value = {
        "kind": "value_objects",
        "class": "pkg.SideEffect",
        "shape": [1, 2],
        "data": [
            {
                "state": {
                    "kind": "array",
                    "dtype": "uint64",
                    "shape": [1, 1],
                    "data": [2**63 + 5],
                }
            },
            {
                "state": {
                    "kind": "array",
                    "dtype": "uint32",
                    "shape": [1, 1],
                    "data": [7],
                }
            },
        ],
    }
    graph = load_snapshot(
        write_snapshot(
            tmp_path,
            [
                {
                    "id": "simulation",
                    "class": "Sim",
                    "fields": {"effects": value},
                }
            ],
        )
    )
    effects = graph.simulation.effects
    assert effects.shape == (1, 2)
    assert isinstance(effects[0, 0], ValueObjectData)
    assert effects[0, 0].class_name == "pkg.SideEffect"
    assert effects[0, 0].state.dtype == np.uint64
    assert effects[0, 0].state[0, 0] == 2**63 + 5
    with pytest.raises(ValueError, match="no runtime class"):
        load_snapshot(
            write_snapshot(
                tmp_path,
                [
                    {
                        "id": "simulation",
                        "class": "Sim",
                        "fields": {"effects": value},
                    }
                ],
            ),
            strict_runtime=True,
        )
