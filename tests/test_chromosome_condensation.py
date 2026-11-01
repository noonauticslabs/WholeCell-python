import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from wholecell_py.processes.chromosome_condensation import (
    ChromosomeCondensation,
)
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot
from wholecell_py.randstream import RandStream
from wholecell_py.states.chromosome import Chromosome

ROOT = Path(__file__).parents[1]


def snapshot():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.ChromosomeCondensation"] = (
        ChromosomeCondensation
    )
    registry["edu.stanford.covert.cell.sim.state.Chromosome"] = Chromosome
    return load_snapshot(
        ROOT / "reference" / "snapshot_t0.json",
        class_registry=registry,
        runtime_values=True,
        rng_factory=lambda _: RandStream(seed=1),
    )


def test_snapshot_current_resource_vector():
    graph = snapshot()
    process = graph.processes["ChromosomeCondensation"]
    assert isinstance(process, ChromosomeCondensation)
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["ChromosomeCondensation"]
    actual = process.calcResourceRequirements_Current()
    np.testing.assert_allclose(
        actual.reshape(expected["shape"], order="F"),
        np.asarray(expected["data"]).reshape(expected["shape"], order="F"),
    )


def test_evolution_uses_chromosome_aspect_binding_runtime():
    graph = snapshot()
    process = graph.processes["ChromosomeCondensation"]
    substrates = process.substrates.copy()
    enzymes = process.enzymes.copy()
    free_adp = enzymes[int(process.enzymeIndexs_SMC_ADP) - 1].copy()
    process.evolveState()
    assert (
        process.substrates[int(process.substrateIndexs_atp) - 1].item()
        <= substrates[int(process.substrateIndexs_atp) - 1].item()
    )
    assert np.all(process.enzymes >= 0)


def test_kb_initialization_builds_process_mappings_before_chromosome_footprints():
    process = ChromosomeCondensation()
    process.compartment = NS(
        count=2, extracellularIndexs=2, cytosolIndexs=1, membraneIndexs=2
    )
    process.stimulus = NS(wholeCellModelIDs=[], names=[])
    process.metabolite = NS(
        wholeCellModelIDs=["ATP", "ADP", "PI", "H2O", "H"],
        names=["ATP", "ADP", "Pi", "Water", "Hydrogen"],
        molecularWeights=[507, 427, 95, 18, 1],
        hydrophobicIndexs=[],
    )
    for name in ("rna", "monomer"):
        setattr(
            process,
            name,
            NS(
                wholeCellModelIDs=[],
                names=[],
                matureIndexs=[],
                boundIndexs=[],
                molecularWeights=[],
                compartments=[],
            ),
        )
    smc_ids = ["MG_213_214_298_6MER", "MG_213_214_298_6MER_ADP"]
    process.complex = NS(
        wholeCellModelIDs=smc_ids,
        names=smc_ids,
        matureIndexs=[1, 2],
        boundIndexs=[1, 2],
        molecularWeights=[100, 100],
        compartments=[1, 1],
    )
    process.gene = NS(mRNAIndexs=[])
    process.chromosome = NS(
        complexDNAFootprints=np.array([630, 630]),
        monomerDNAFootprints=np.empty(0),
        calculateFootprintOverhangs=lambda footprints: (
            footprints,
            footprints,
        ),
    )
    process.initializeConstants(
        NS(
            processes=[
                NS(
                    wholeCellModelID=process.wholeCellModelID,
                    parameters=[
                        NS(name="smcSepNt", index="", defaultValue=7000),
                        NS(
                            name="smcSepProbCenter",
                            index="",
                            defaultValue=3500,
                        ),
                    ],
                )
            ]
        ),
        NS(stepSizeSec=1, verbosity=0, seed=4),
    )
    assert process.smcSepNt == 7000
    assert process.smcSepProbCenter == 3500
    np.testing.assert_array_equal(
        np.asarray(process.enzymeWholeCellModelIDs).ravel(), smc_ids
    )
    np.testing.assert_array_equal(process.enzymeDNAFootprints, [630, 630])
