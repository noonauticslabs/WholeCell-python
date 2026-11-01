import numpy as np

from wholecell_py.states.mass import Mass


class _State:
    def __init__(self, dry, wet=None):
        self._dry = np.asarray(dry, dtype=float)
        self._wet = None if wet is None else np.asarray(wet, dtype=float)

    @property
    def dryWeight(self):
        return self._dry

    @property
    def wetWeight(self):
        return self._wet


class _Compartment:
    cytosolIndexs = np.array([1])
    cellularIndexs = np.array([1, 2])
    extracellularIndexs = np.array([3])


def test_calc_mass_uses_live_weights_and_1_based_compartments():
    shape = (1, 3, 2)
    water = np.array([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]])
    met_dry = np.array([[[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]])
    dna_dry = np.array([[[1.0, 2.0]]])  # compact cytosolic-only state
    rna_dry = np.full(shape, 0.1)
    monomer_dry = np.full(shape, 0.2)
    complex_dry = np.full(shape, 0.3)
    met = _State(met_dry, water)
    dna, rna = _State(dna_dry), _State(rna_dry)
    monomer, complex_ = _State(monomer_dry), _State(complex_dry)
    mass = Mass()
    mass.compartment = _Compartment()
    mass.states = [met, dna, rna, monomer, complex_]
    mass.metabolite, mass.chromosome = met, dna
    mass.rna, mass.monomer, mass.complex = rna, monomer, complex_

    mass.initialize()

    expected_total = water + met_dry + rna_dry + monomer_dry + complex_dry
    expected_total[:, 0, :] += dna_dry[:, 0, :]
    np.testing.assert_allclose(mass.total, expected_total)
    np.testing.assert_allclose(mass.waterWt, water)
    np.testing.assert_allclose(mass.metaboliteWt, met_dry)
    np.testing.assert_allclose(mass.dnaWt[:, 0, :], dna_dry[:, 0, :])
    np.testing.assert_allclose(mass.proteinWt, monomer_dry + complex_dry)
    np.testing.assert_allclose(mass.cell[:, 2, :], 0)
    np.testing.assert_allclose(mass.media[:, :2, :], 0)
    np.testing.assert_allclose(
        mass.cellDry[:, :2, :], expected_total[:, :2, :] - water[:, :2, :]
    )
    assert mass.total.shape == shape


def test_snapshot_arrays_are_copied_and_constants_survive():
    original = np.array([1.0, 2.0])
    mass = Mass(
        snapshot={
            "cellInitialDryWeight": original,
            "dryWeightFractionDNA": np.array([0.2]),
            "total": np.ones((1, 1, 1)),
        }
    )
    assert mass.cellInitialDryWeight is not original
    assert "cellInitialDryWeight" in mass.fixedConstantNames
    assert "dryWeightFractionDNA" in mass.fittedConstantNames
    assert "total" in mass.dependentStateNames
