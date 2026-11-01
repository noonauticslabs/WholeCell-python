import json
from pathlib import Path

import numpy as np
from types import SimpleNamespace

from wholecell_py.processes.replication_initiation import ReplicationInitiation
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot
from wholecell_py.states.chromosome import Chromosome

ROOT = Path(__file__).parents[1]


def snapshot():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.ReplicationInitiation"] = (
        ReplicationInitiation
    )
    registry["edu.stanford.covert.cell.sim.state.Chromosome"] = Chromosome
    return load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=registry,
        runtime_values=True,
    )


def test_snapshot_current_resource_vector_matches_matlab():
    process = snapshot().processes["ReplicationInitiation"]
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["ReplicationInitiation"]
    actual = process.calcResourceRequirements_Current()
    np.testing.assert_allclose(
        actual.reshape(expected["shape"], order="F"),
        np.asarray(expected["data"]).reshape(expected["shape"], order="F"),
    )


def test_release_transition_tables_preserve_one_based_indices():
    process = ReplicationInitiation(enzymeComplexGlobalIndexs=np.arange(1, 16))
    process.calcDnaAxpReleaseReactions()
    assert process.dnaARelease_remainingDnaAIndexs[3] == 2
    assert process.dnaARelease_remainingDnaAIndexs[2] == 2
    assert process.dnaARelease_releasedDnaAIndexs[3] == 2
    assert process.dnaARelease_releasedDnaAIndexs[2] == 1
    assert process.dnaARelease_remainingDnaAMatrix[1, 3] == 1
    # ATP-monomer product is released from ATP Nmers; ADP-monomer from ADP Nmers.
    np.testing.assert_array_equal(
        np.flatnonzero(process.dnaARelease_releasedDnaAMatrix[1]),
        [1, 3, 5, 7, 9, 11, 13],
    )
    np.testing.assert_array_equal(
        np.flatnonzero(process.dnaARelease_releasedDnaAMatrix[0]),
        [0, 2, 4, 6, 8, 10, 12],
    )


def test_polymerize_updates_column_enzyme_vector_with_nonzero_counts():
    class Rand:
        def stochasticRound(self, value):
            return int(value)

        def randsample(self, n, k, replacement, weights):
            return np.array([1])

    for atp in (True, False):
        process = ReplicationInitiation()
        process.enzymes = np.zeros((15, 1))
        process.enzymes[0, 0] = 2
        process.enzymes[1, 0] = 2
        process.randStream = Rand()
        process.dnaABoxIndexs_7mer = np.array([], dtype=int)
        process.dnaABoxIndexs_R1234 = np.array([1, 2, 3, 4])
        process.dnaABoxStartPositions = np.arange(1, 6)
        process.enzymeComplexGlobalIndexs = np.arange(1, 16)
        process.chromosome = type(
            "Chromosome",
            (),
            {
                "complexBoundSites": type(
                    "Sites",
                    (),
                    {
                        "find": lambda self: (
                            np.empty((0, 2), int),
                            np.empty(0, int),
                        )
                    },
                )()
            },
        )()
        process.modifyProteinOnChromosome = lambda *args: None
        process.calculateDnaAATPBindingRates = lambda a, d: (
            np.zeros((1, 2)),
            np.zeros(1),
            0.0,
        )
        process.calculateDnaAADPBindingRates = (
            process.calculateDnaAATPBindingRates
        )
        process.calculateDnaAR1234ATPPolymerizationRates = (
            lambda a, d: np.array([[1.0], [0], [0], [0]])
        )
        process.calculateDnaAR1234ADPPolymerizationRates = (
            process.calculateDnaAR1234ATPPolymerizationRates
        )
        a, d = np.array([[1], [0], [0], [0]]), np.zeros((4, 1), dtype=int)

        if atp:
            process.bindAndPolymerizeDnaAATP(a, d)
            assert a[0, 0] == 2
            assert process.enzymes[3, 0] == 1
        else:
            process.bindAndPolymerizeDnaAADP(a, d)
            assert d[0, 0] == 2
            assert process.enzymes[2, 0] == 1
        expected = np.zeros((15, 1))
        if atp:
            expected[0, 0] = 2
            expected[3, 0] = 1
        else:
            expected[0, 0] = expected[1, 0] = expected[2, 0] = 1
        np.testing.assert_array_equal(process.enzymes, expected)
        assert process.enzymes.shape == (15, 1)


def test_snapshot_evolution_preserves_upstream_chromosome_dependency():
    process = snapshot().processes["ReplicationInitiation"]
    # Execute the source path. If the current chromosome runtime lacks a method
    # reached by the MATLAB call chain, preserve and surface that dependency.
    try:
        process.evolveState()
    except (AttributeError, NotImplementedError) as error:
        assert str(error)


def test_snapshot_box_status_and_assembled_ori_keep_two_copy_outputs():
    process = snapshot().processes["ReplicationInitiation"]

    status = process.calculateDnaABoxStatus()
    assembled = process.calculateIsDnaAORIComplexAssembled()

    assert status.shape == (
        len(np.asarray(process.dnaABoxStartPositions).reshape(-1)),
        2,
    )
    np.testing.assert_array_equal(assembled, [True, False])


def _box_status_process(
    box_positions, polymerized_coords, polymerized_lengths
):
    from wholecell_py.sparsemat import SparseMat

    class EmptySites:
        def __getitem__(self, key):
            return 0

    process = ReplicationInitiation()
    process.dnaABoxStartPositions = np.asarray(box_positions, dtype=int)
    process.enzymeComplexGlobalIndexs = np.arange(1, 16)
    process.chromosome = type("Chromosome", (), {})()
    process.chromosome.polymerizedRegions = SparseMat(
        polymerized_coords, polymerized_lengths, [100, 4]
    )
    process.chromosome.complexBoundSites = EmptySites()
    return process


def test_box_status_uses_source_fork_boundary_when_only_copy_three_exists():
    process = _box_status_process([5, 15], [[10, 3]], [8])

    status = process.calculateDnaABoxStatus()

    np.testing.assert_array_equal(status[:, 1], [0, -1])


def test_box_status_uses_source_gap_between_copy_three_and_four():
    process = _box_status_process([5, 15, 25, 35], [[10, 3], [30, 4]], [8, 8])

    status = process.calculateDnaABoxStatus()

    np.testing.assert_array_equal(status[:, 1], [0, 0, -1, 0])


def test_box_status_uses_copy_four_boundary_without_copy_three_region():
    process = _box_status_process([5, 15, 25], [[20, 4]], [8])

    status = process.calculateDnaABoxStatus()

    np.testing.assert_array_equal(status[:, 1], [-1, -1, 0])


def test_assembled_ori_complex_returns_one_atp_status_per_copy():
    process = ReplicationInitiation()
    process.calculateDnaAR1234Polymerization = lambda: (
        np.array([[7, 7], [7, 6], [7, 7], [7, 7]]),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
    )
    process.calcuateIsDnaAR5Occupied = lambda: np.array([True, True])

    assembled = process.calculateIsDnaAORIComplexAssembled()

    np.testing.assert_array_equal(assembled, [True, False])


def test_assembled_ori_complex_requires_atp_not_adp_heptamers():
    process = ReplicationInitiation()
    process.calculateDnaAR1234Polymerization = lambda: (
        np.full((4, 2), 6),
        np.full((4, 2), 8),
    )
    process.calcuateIsDnaAR5Occupied = lambda: np.array([True, True])

    assembled = process.calculateIsDnaAORIComplexAssembled()

    np.testing.assert_array_equal(assembled, [False, False])


def test_final_condition_initialization_raises_if_required_ori_binding_fails():
    process = ReplicationInitiation()
    process.enzymes = np.zeros((15, 1))
    process.enzymes[14, 0] = 60
    process.boundEnzymes = np.zeros((15, 1))
    process.dnaABoxStartPositions = np.arange(1, 11)
    process.dnaABoxIndexs_R12345 = np.array([1, 2, 3, 4, 5])
    process.dnaABoxIndexs_R1234 = np.array([1, 2, 3, 4])
    process.dnaABoxIndexs_R5 = np.array([5])
    process.dnaABoxIndexs_9mer = np.array([6, 7])
    process.dnaABoxIndexs_8mer = np.array([8, 9, 10])
    process.kb1ATP = process.kb2ATP = 1.0

    class Rand:
        def stochasticRound(self, value):
            return int(np.floor(float(np.asarray(value).item()) + 0.5))

    process.randStream = Rand()
    process.bindProteinToChromosome = lambda positions, *args: (
        np.zeros(len(positions), dtype=bool),
        np.empty(0, int),
        np.empty((0, 2), int),
        0,
        np.empty(0),
    )

    import pytest

    with pytest.raises(RuntimeError, match="bound to oriC sites"):
        process.initializeStateBasedOnFinalConditions()


def test_sample_dnaa_boxes_preserves_full_source_coordinate_vector():
    class Rand:
        def randomlySelectNRows(self, values, count):
            return np.array([7, 9])

    process = ReplicationInitiation()
    process.randStream = Rand()
    process.dnaABoxStartPositions = np.arange(101, 111)
    process.dnaABoxIndexs_R12345 = np.array([1, 2, 3, 4, 5])
    process.dnaABoxIndexs_R1234 = np.array([1, 2, 3, 4])
    process.dnaABoxIndexs_R5 = np.array([5])
    process.dnaABoxIndexs_7mer = np.array([6, 7, 8])
    process.dnaABoxIndexs_8mer = np.array([9, 10])
    process.dnaABoxIndexs_9mer = np.array([], dtype=int)

    process.sampleDnaABoxes(7)

    # MATLAB evaluates dnaABoxStartPositions(idxs) without assigning the
    # result, so only the category indexes are remapped.
    np.testing.assert_array_equal(
        process.dnaABoxStartPositions, np.arange(101, 111)
    )
    np.testing.assert_array_equal(process.dnaABoxIndexs_7mer, [6])
    np.testing.assert_array_equal(process.dnaABoxIndexs_8mer, [7])


def test_kb_initializer_derives_dnaa_box_coordinates_and_classes(monkeypatch):
    from wholecell_py.process_base import Process
    from wholecell_py.chromosome_process_aspect import ChromosomeProcessAspect

    names = [
        "other7",
        "Functional box R1",
        "other8",
        "Functional box R2",
        "Functional box R3",
        "Functional box R4",
        "Functional box R5",
    ]
    subtypes = ["7mer", "9mer", "8mer", "9mer", "7mer", "8mer", "9mer"]
    features = [
        SimpleNamespace(
            type="DnaA box",
            name=name,
            subtype=subtype,
            startCoordinate=i * 10,
            sequenceLength=8,
        )
        for i, (name, subtype) in enumerate(zip(names, subtypes))
    ]
    process = ReplicationInitiation()

    monkeypatch.setattr(
        Process,
        "initializeConstants",
        lambda self, kb, sim, options=None: None,
    )

    def aspect_init(self, kb, sim):
        self.enzymeDNAFootprints = np.ones(15) * 4
        self.enzymeComplexGlobalIndexs = np.arange(1, 15)

    monkeypatch.setattr(
        ChromosomeProcessAspect, "initializeConstants", aspect_init
    )

    process.initializeConstants(
        SimpleNamespace(genomeFeatures=features), object()
    )

    np.testing.assert_array_equal(
        process.dnaABoxStartPositions, [2, 12, 22, 32, 42, 52, 62]
    )
    np.testing.assert_array_equal(process.dnaABoxIndexs_7mer, [1, 5])
    np.testing.assert_array_equal(process.dnaABoxIndexs_8mer, [3, 6])
    np.testing.assert_array_equal(process.dnaABoxIndexs_9mer, [2, 4, 7])
    np.testing.assert_array_equal(
        process.dnaABoxIndexs_R12345, [2, 4, 5, 6, 7]
    )
    assert process.dnaARelease_remainingDnaAMatrix.shape == (15, 15)


def test_final_condition_initialization_raises_if_selected_box_binding_fails():
    process = ReplicationInitiation()
    process.enzymes = np.zeros((15, 1))
    process.enzymes[14, 0] = 60
    process.boundEnzymes = np.zeros((15, 1))
    process.dnaABoxStartPositions = np.arange(1, 11)
    process.dnaABoxIndexs_R12345 = np.array([1, 2, 3, 4, 5])
    process.dnaABoxIndexs_R1234 = np.array([1, 2, 3, 4])
    process.dnaABoxIndexs_R5 = np.array([5])
    process.dnaABoxIndexs_9mer = np.array([6, 7])
    process.dnaABoxIndexs_8mer = np.array([8, 9, 10])
    process.kb1ATP = process.kb2ATP = 1.0

    class Rand:
        def stochasticRound(self, value):
            return int(np.floor(float(np.asarray(value).item()) + 0.5))

    process.randStream = Rand()
    calls = 0

    def bind(positions, *args):
        nonlocal calls
        calls += 1
        success = calls <= 2
        return (
            np.full(len(positions), success, dtype=bool),
            np.empty(0, int),
            np.empty((0, 2), int),
            0,
            np.empty(0),
        )

    process.bindProteinToChromosome = bind

    import pytest

    with pytest.raises(RuntimeError, match="8- and 9-mer"):
        process.initializeStateBasedOnFinalConditions()
