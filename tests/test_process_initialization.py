from types import SimpleNamespace as NS
import numpy as np
import pytest
from wholecell_py.process_base import Process


def process():
    p = Process("P")
    p.compartment = NS(
        count=3, cytosolIndexs=1, membraneIndexs=2, extracellularIndexs=3
    )
    p.stimulus = NS(wholeCellModelIDs=["light"], names=["Light"])
    p.metabolite = NS(
        wholeCellModelIDs=["water", "lipid"],
        names=["Water", "Lipid"],
        molecularWeights=[18, 90],
        hydrophobicIndexs=[2],
    )
    for name, ident in [("rna", "R"), ("monomer", "M"), ("complex", "C")]:
        setattr(
            p,
            name,
            NS(
                wholeCellModelIDs=["nascent", ident, "bound"],
                names=["N", ident, "B"],
                matureIndexs=[2],
                boundIndexs=[3],
                molecularWeights=[10, 20, 30],
                compartments=[1, 2, 1],
            ),
        )
    p.stimuliWholeCellModelIDs = ["light"]
    p.substrateWholeCellModelIDs = ["lipid", "R", "water"]
    p.enzymeWholeCellModelIDs = ["M", "C"]
    return p


def test_mapping_preserves_mature_relative_and_global_linear_indices():
    p = process()
    v = p.initializeConstantsHelper(["C", "water", "R", "M", "light", "lipid"])
    assert len(v) == 26
    np.testing.assert_array_equal(v[24].ravel(), [1, 1, 1, 1, 1, 2])
    np.testing.assert_array_equal(v[25].ravel(), [20, 18, 20, 20, 0, 90])
    np.testing.assert_array_equal(
        v[8].ravel(), [1, 4]
    )  # metabolite water cytosol, lipid membrane
    np.testing.assert_array_equal(
        v[12].ravel(), [2]
    )  # mature RNA absolute row 2
    np.testing.assert_array_equal(
        v[16].ravel(), [5]
    )  # mature monomer row 2, compartment 2
    np.testing.assert_array_equal(
        v[22].ravel(), [6]
    )  # bound monomer row 3, compartment 2


def test_retained_compartments_and_empty_types():
    v = process().initializeConstantsHelper(["M"], True)
    np.testing.assert_array_equal(v[16], [[2, 5, 8]])
    np.testing.assert_array_equal(v[22], [[3, 6, 9]])
    assert v[4].shape == (0, 3)
    with pytest.raises(ValueError, match="invalidIDs"):
        process().initializeConstantsHelper(["unknown"])


def test_initialize_parameter_growth_options_and_override():
    p = process()
    p.someIndex = [2, 4]
    kb = NS(
        processes=[
            NS(
                wholeCellModelID="P",
                parameters=[
                    NS(name="rate", index="someIndex", defaultValue=[7, 9]),
                    NS(name="scalar", index="", defaultValue=5),
                ],
            )
        ]
    )
    sim = NS(stepSizeSec=2, verbosity=3, seed=41)
    p.initializeConstants(kb, sim, {"retainEnzymeCompartments": True})
    np.testing.assert_array_equal(p.rate, [[0], [7], [0], [9]])
    assert (p.scalar, p.stepSizeSec, p.seed, p.verbosity, p.initialized) == (
        5,
        2,
        41,
        3,
        True,
    )
    np.testing.assert_array_equal(
        p.enzymeMonomerGlobalCompartmentIndexs, [[2, 5, 8]]
    )
    p.initializeConstants_overrideSubstrates(["water"])
    np.testing.assert_array_equal(
        p.substrateMetaboliteGlobalCompartmentIndexs, [[1]]
    )
    np.testing.assert_array_equal(p.substrateMolecularWeights, [[18]])


def test_sparse_state_gather_preserves_time_and_column_order():
    from wholecell_py.sparsemat import SparseMat

    s = SparseMat([[2, 1, 1], [1, 2, 2]], [[8], [9]], [2, 2, 2])
    g = np.array([[2, 1]])
    c = np.array([[1, 2]])
    actual = Process._copyFromStateHelper(s, g, c)
    np.testing.assert_array_equal(actual, [[[8, 0], [0, 9]]])


def test_simulation_lookup_full_short_and_missing():
    from wholecell_py.runner import SimulationContext

    ctx = SimulationContext()
    obj = NS(wholeCellModelID="State_Rna")
    ctx.states = [obj]
    assert ctx.state("Rna") is obj
    assert ctx.state("State_Rna") is obj
    assert ctx.state("absent").shape == (0, 0)


def test_copy_to_state_helper_accepts_singleton_time_omitted():
    p = Process()
    p.substrates = np.array([[7, 8]])
    state = NS(counts=np.zeros((2, 2)))
    p.copyToStateHelper(
        state, "counts", "substrates", [[2, 1]], [[1, 2]], [[1]]
    )
    np.testing.assert_array_equal(state.counts[:, :, 0], [[0, 8], [7, 0]])


def test_compartment_source_initialization():
    from wholecell_py.compartment import Compartment

    kb = dict(
        numCompartments=2,
        compartments=[
            dict(wholeCellModelID="c", name="Cytosol"),
            dict(wholeCellModelID="e", name="External"),
        ],
    )
    for prefix in (
        "cytosol",
        "chromosome",
        "membrane",
        "terminalOrganelleCytosol",
        "terminalOrganelleMembrane",
        "extracellular",
        "cellular",
    ):
        kb[prefix + "CompartmentIndexs"] = (
            [2] if prefix == "extracellular" else [1]
        )
    c = Compartment()
    c.initializeConstants(kb)
    assert c.count == 2
    assert c.getIndexs("e") == 2
    np.testing.assert_array_equal(c.names, [["Cytosol"], ["External"]])


def test_complete_multitime_copy_with_column_shaped_mature_bound_indices():
    p = process()
    for name in ("rna", "monomer", "complex"):
        state = getattr(p, name)
        state.matureIndexs = np.array([[2]])
        state.boundIndexs = np.array([[3]])
    for name in ("stimulus", "metabolite", "rna", "monomer", "complex"):
        state = getattr(p, name)
        n = len(state.wholeCellModelIDs)
        setattr(
            state,
            "values" if name == "stimulus" else "counts",
            np.arange(n * 3 * 2).reshape(n, 3, 2).astype(float),
        )
    p.initializeConstants(
        NS(processes=[NS(wholeCellModelID="P", parameters=[])]),
        NS(stepSizeSec=1, verbosity=0, seed=1),
    )
    p.copyFromState()
    np.testing.assert_array_equal(p.enzymes[0, 0], p.monomer.counts[1, 1, :])
    np.testing.assert_array_equal(
        p.boundEnzymes[0, 0], p.monomer.counts[2, 1, :]
    )
    p.enzymes[0, 0, :] = [101, 102]
    p.boundEnzymes[0, 0, :] = [201, 202]
    p.copyToState()
    np.testing.assert_array_equal(p.monomer.counts[1, 1, :], [101, 102])
    np.testing.assert_array_equal(p.monomer.counts[2, 1, :], [201, 202])
