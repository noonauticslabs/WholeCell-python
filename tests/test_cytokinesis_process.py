import json
from pathlib import Path

import numpy as np
from types import SimpleNamespace as NS

from wholecell_py.randstream import RandStream
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot
from wholecell_py.processes.cytokinesis import Cytokinesis


def test_initialize_constants_derives_chromosome_footprints():
    process = Cytokinesis()
    process.enzymeWholeCellModelIDs = ["monomer", "complex"]
    process.enzymeMonomerLocalIndexs = [1]
    process.enzymeMonomerGlobalIndexs = [2]
    process.enzymeComplexLocalIndexs = [2]
    process.enzymeComplexGlobalIndexs = [1]
    process.chromosome = NS(
        monomerDNAFootprints=np.array([5, 8]),
        complexDNAFootprints=np.array([12]),
        calculateFootprintOverhangs=lambda x: (x / 2, x / 2),
    )
    process.initializeConstants()
    np.testing.assert_array_equal(process.enzymeDNAFootprints, [8, 12])
    np.testing.assert_array_equal(process.enzymeDNAFootprints5Prime, [4, 6])


ROOT = Path(__file__).resolve().parents[1]


def snapshot_registry():
    registry = implemented_registry()
    # The process registration is owned by the workspace's runtime registry;
    # keep this focused fixture usable while that mapping is being integrated.
    registry["edu.stanford.covert.cell.sim.process.Cytokinesis"] = Cytokinesis
    return registry


def test_reference_snapshot_current_requirements_and_copy_roundtrip():
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=snapshot_registry(),
    )
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )
    process = graph.processes["Cytokinesis"]
    process.randStream = RandStream(seed=1)
    process.copyFromState()
    np.testing.assert_array_equal(
        process.calcResourceRequirements_Current().ravel(order="F"),
        expected["Cytokinesis"]["data"],
    )

    # The snapshot's local substrate/enzyme values round-trip to linked states.
    before_substrates = process.substrates.copy()
    before_enzymes = process.enzymes.copy()
    before_bound = process.boundEnzymes.copy()
    process.copyToState()
    process.copyFromState()
    np.testing.assert_array_equal(process.substrates, before_substrates)
    np.testing.assert_array_equal(process.enzymes, before_enzymes)
    np.testing.assert_array_equal(process.boundEnzymes, before_bound)


def test_snapshot_evolve_respects_chromosome_guard_and_uses_runtime_stream():
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=snapshot_registry(),
    )
    process = graph.processes["Cytokinesis"]
    process.copyFromState()

    class NoEvents:
        def rand(self):
            return 1.0

    process.randStream = NoEvents()
    substrates = process.substrates.copy()
    ring = [
        process.ftsZRing.numEdgesOneStraight,
        process.ftsZRing.numEdgesTwoStraight,
        process.ftsZRing.numEdgesTwoBent,
        process.ftsZRing.numResidualBent,
    ]
    process.evolveState()
    # At t0 the chromosome is not segregated, so the process returns untouched.
    if not process.chromosome.segregated:
        np.testing.assert_array_equal(process.substrates, substrates)
        np.testing.assert_array_equal(
            [
                process.ftsZRing.numEdgesOneStraight,
                process.ftsZRing.numEdgesTwoStraight,
                process.ftsZRing.numEdgesTwoBent,
                process.ftsZRing.numResidualBent,
            ],
            ring,
        )


def test_matlab_scalar_array_late_ring_dissociation_and_hydrolysis():
    """Late-ring reactions preserve MATLAB stoichiometry with column arrays."""
    from types import SimpleNamespace

    class Events:
        def rand(self):
            return 0.5

    def fixture():
        p = Cytokinesis()
        p.chromosome = SimpleNamespace(segregated=True)
        p.geometry = SimpleNamespace(pinched=True)
        p.ftsZRing = SimpleNamespace(
            numFtsZSubunitsPerFilament=np.array([[9.0]]),
            numEdges=1,
            numEdgesOneStraight=0,
            numEdgesTwoStraight=0,
            numEdgesTwoBent=0,
            numResidualBent=0,
        )
        p.randStream = Events()
        p.rateFilamentBindingMembrane = 0
        p.rateFilamentDissociation = 1
        p.rateFtsZGtpHydrolysis = 1
        p.enzymes = np.zeros((4, 1))
        p.boundEnzymes = np.zeros((4, 1))
        p.substrates = np.array([[0.0], [100.0], [0.0]])
        return p

    p = fixture()
    p.ftsZRing.numResidualBent = 1
    p.boundEnzymes[1] = 1
    p.evolveState()
    assert p.enzymes[2, 0] == 9
    assert p.boundEnzymes[1, 0] == 0
    assert p.ftsZRing.numResidualBent == 0

    p = fixture()
    p.ftsZRing.numEdgesTwoBent = 1
    p.boundEnzymes[1] = 2
    p.evolveState()
    assert p.enzymes[2, 0] == 9
    assert p.boundEnzymes[1, 0] == 1
    assert p.ftsZRing.numResidualBent == 1

    p = fixture()
    p.rateFilamentDissociation = 0
    p.ftsZRing.numEdgesTwoStraight = 1
    p.boundEnzymes[0] = 2
    p.evolveState()
    np.testing.assert_array_equal(p.substrates[:, 0], [18, 82, 18])
    np.testing.assert_array_equal(p.boundEnzymes[:, 0], [0, 2, 0, 0])
    assert p.ftsZRing.numEdgesTwoBent == 1
