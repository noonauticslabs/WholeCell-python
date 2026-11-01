import numpy as np

from wholecell_py.processes.chromosome_segregation import ChromosomeSegregation
from wholecell_py.process_base import Process
from wholecell_py.chromosome_process_aspect import ChromosomeProcessAspect
from wholecell_py.sparsemat import SparseMat


class State:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class Simulation:
    def __init__(self, chromosome):
        self.compartment = object()
        self.gene = object()
        self._states = {
            "Geometry": State(),
            "Stimulus": State(),
            "Metabolite": State(),
            "Rna": State(),
            "ProteinMonomer": State(),
            "ProteinComplex": State(),
            "Chromosome": chromosome,
        }

    def state(self, name):
        return self._states[name]


def replicated_chromosome(*, segregated=False):
    # The sparse dimensions represent a large genome with only a few occupied
    # positions; reductions must not turn it into a dense array.
    return State(
        segregated=segregated,
        polymerizedRegions=SparseMat(
            [[1, 1], [10_000_000, 2]], [10, 10], [10_000_000, 2]
        ),
        supercoiled=SparseMat([[1, 1], [1, 2]], [1, 1], [1, 2]),
        nCompartments=2,
        sequenceLen=10,
    )


def test_process_stores_chromosome_state_and_resource_demand():
    chromosome = replicated_chromosome()
    process = ChromosomeSegregation(gtpCost=3)
    process.storeObjectReferences(Simulation(chromosome))
    process.substrates = np.zeros(5)
    process.enzymes = np.ones(5)

    assert process.chromosome is chromosome
    assert chromosome in process.states
    np.testing.assert_array_equal(
        process.calcResourceRequirements_Current(), [3, 0, 0, 3, 0]
    )


def test_initializer_composes_source_process_and_chromosome_aspect(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        Process,
        "initializeConstants",
        lambda self, kb, sim, options=None: calls.append(("process", options)),
    )
    monkeypatch.setattr(
        ChromosomeProcessAspect,
        "initializeConstants",
        lambda self, kb, sim: calls.append(("chromosome", None)),
    )
    process = ChromosomeSegregation()

    process.initializeConstants(
        object(), object(), {"retainEnzymeCompartments": True}
    )

    assert calls == [
        ("process", {"retainEnzymeCompartments": True}),
        ("chromosome", None),
    ]


def test_evolution_mutates_linked_chromosome_and_substrates_only_when_guards_pass():
    chromosome = replicated_chromosome()
    process = ChromosomeSegregation(gtpCost=3)
    process.storeObjectReferences(Simulation(chromosome))
    process.substrates = np.array([4, 0, 0, 5, 0], dtype=float)
    process.enzymes = np.ones(5)

    process.evolveState()

    assert chromosome.segregated is True
    np.testing.assert_array_equal(process.substrates, [1, 3, 3, 2, 3])


def test_evolution_obeys_each_guard_and_lifecycle_requirements():
    chromosome = replicated_chromosome()
    process = ChromosomeSegregation(gtpCost=2)
    process.chromosome = chromosome
    process.substrates = np.array([2, 0, 0, 2, 0], dtype=float)
    process.enzymes = np.array([1, 1, 1, 1, 0])
    assert not np.any(process.calcResourceRequirements_Current())
    process.evolveState()
    assert not chromosome.segregated
    np.testing.assert_array_equal(process.substrates, [2, 0, 0, 2, 0])

    biomass, byproducts, min_enz, max_enz = (
        process.calcResourceRequirements_LifeCycle()
    )
    np.testing.assert_array_equal(biomass, [2, 0, 0, 2, 0])
    np.testing.assert_array_equal(byproducts, [0, 2, 2, 0, 2])
    np.testing.assert_array_equal(min_enz, [2] * 5)
    assert np.all(np.isinf(max_enz))


def test_snapshot_column_constants_segregate_without_broadcast_failure():
    process = ChromosomeSegregation(gtpCost=np.array([[3.0]]))
    process.chromosome = replicated_chromosome()
    process.substrates = np.array([[4.0], [0.0], [0.0], [5.0], [0.0]])
    process.enzymes = np.ones((5, 1))
    process.substrateIndexs_gtp = np.array([[1]])
    alias = process.substrates
    process.evolveState()
    assert process.chromosome.segregated
    assert alias is process.substrates
    np.testing.assert_array_equal(alias[:, 0], [1, 3, 3, 2, 3])
    process.evolveState()
    np.testing.assert_array_equal(alias[:, 0], [1, 3, 3, 2, 3])


def test_segregation_single_subscript_only_updates_first_time_plane():
    process = ChromosomeSegregation(gtpCost=np.array([[3.0]]))
    process.chromosome = replicated_chromosome()
    process.substrates = np.column_stack(
        ([4.0, 0.0, 0.0, 5.0, 0.0], [99.0] * 5)
    )
    process.enzymes = np.ones((5, 1))
    process.evolveState()
    np.testing.assert_array_equal(process.substrates[:, 0], [1, 3, 3, 2, 3])
    np.testing.assert_array_equal(process.substrates[:, 1], [99] * 5)
