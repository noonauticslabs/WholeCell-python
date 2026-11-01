import warnings

import numpy as np

from wholecell_py.states.geometry import Geometry


class MassStub:
    def __init__(self):
        self.cell = np.array([[[2.0, 3.0], [4.0, 5.0]]])
        self.cellInitialDryWeight = 2.0
        self.fractionWetWeight = 0.5
        self.initialBiomassConcentration = 4.0


def test_snapshot_constructor_and_matlab_shaped_allocation():
    state = Geometry(
        "id", "Geometry", density=1000, snapshot={"volume": np.ones((1, 1, 2))}
    )
    state.allocateMemory(3)
    assert state.width.shape == state.pinchedDiameter.shape == (1, 1, 3)
    assert state.density == 1000


def test_volume_initialize_and_live_mass_reference():
    state = Geometry(density=2.0)
    mass = MassStub()

    class Simulation:
        def state(self, key):
            assert key == "Mass"
            return mass

    state.storeObjectReferences(Simulation())
    state.calculateVolume()
    np.testing.assert_allclose(state.volume, [[[3.0, 4.0]]])
    state.initialize()
    assert state.mass is mass
    np.testing.assert_allclose(state.pinchedDiameter, state.width)
    np.testing.assert_allclose(
        state.width, Geometry.calculateWidth(state.volume)
    )
    assert state.chamberVolume == 1.0


def test_geometry_and_dependent_getters():
    width = np.array([[[2e-7, 2e-7]]])
    pinch = np.array([[[2e-7, 1e-7]]])
    volume = np.array([[[1e-17, 1.5e-17]]])
    state = Geometry(width=width, pinchedDiameter=pinch, volume=volume)
    cy, area, length = Geometry.calculateGeometry(width, pinch, volume)
    assert cy.shape == area.shape == length.shape == (1, 1, 2)
    np.testing.assert_allclose(state.cylindricalLength, cy)
    np.testing.assert_allclose(state.surfaceArea, area)
    np.testing.assert_allclose(state.totalLength, length)
    np.testing.assert_allclose(state.pinchedCircumference, pinch * np.pi)
    np.testing.assert_array_equal(state.pinched, False)


def test_divided_geometry_uses_source_sentinel_values():
    state = Geometry(width=2e-7, pinchedDiameter=0, volume=1e-17)
    assert state.width == -1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert (
            state.cylindricalLength
            == state.surfaceArea
            == state.totalLength
            == -1
        )
    assert caught
    assert state.pinched
