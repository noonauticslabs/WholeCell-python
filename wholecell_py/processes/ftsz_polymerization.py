"""FtsZ filament assembly process translated from WholeCell."""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get

_N_AVOGADRO = 6.0221417930e23


class FtsZPolymerization(Process):
    optionNames__ = ()
    fixedConstantNames__ = (
        "maxPolymerLength",
        "activationFwd",
        "activationRev",
        "exchangeFwd",
        "exchangeRev",
        "nucleationFwd",
        "nucleationRev",
        "elongationFwd",
        "elongationRev",
    )
    fittedConstantNames__ = ()
    localStateNames__ = ()
    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = ("GDP", "GTP", "PI", "H2O", "H")
    (
        substrateIndexs_gdp,
        substrateIndexs_gtp,
        substrateIndexs_phosphate,
        substrateIndexs_water,
        substrateIndexs_hydrogen,
    ) = (1, 2, 3, 4, 5)
    enzymeWholeCellModelIDs = (
        "MG_224_MONOMER",
        "MG_224_MONOMER_GDP",
        "MG_224_MONOMER_GTP",
        "MG_224_2MER_GTP",
        "MG_224_3MER_GTP",
        "MG_224_4MER_GTP",
        "MG_224_5MER_GTP",
        "MG_224_6MER_GTP",
        "MG_224_7MER_GTP",
        "MG_224_8MER_GTP",
        "MG_224_9MER_GTP",
    )
    (
        enzymeIndexs_FtsZ,
        enzymeIndexs_FtsZ_GDP,
        enzymeIndexs_FtsZ_GTP,
        enzymeIndexs_FtsZ_dimer,
        enzymeIndexs_FtsZ_9mer,
    ) = (1, 2, 3, 4, 11)
    enzymeIndexs_FtsZ_activated = np.arange(3, 12).reshape(-1, 1)

    def __init__(
        self,
        wholeCellModelID="Process_FtsZPolymerization",
        name="FtsZPolymerization",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.maxPolymerLength = getattr(self, "maxPolymerLength", 9)
        for k, v in dict(
            activationFwd=1.1,
            activationRev=0.01,
            exchangeFwd=1e4,
            exchangeRev=5e3,
            nucleationFwd=4.2e6,
            nucleationRev=40.0,
            elongationFwd=5.1e6,
            elongationRev=2.9,
        ).items():
            setattr(self, k, getattr(self, k, v))

    def initializeConstants(self, *args, **kwargs):
        super().initializeConstants(*args, **kwargs)
        self.maxPolymerLength = len(self.enzymeIndexs_FtsZ_activated)

    def calcResourceRequirements_LifeCycle(self, constants, states):
        from .cytokinesis import Cytokinesis

        ring = _get(_get(constants, "states"), "FtsZRing")
        _, gtp_cost = Cytokinesis.calcRequiredPinchingCycles(
            _get(self.geometry, "width"),
            _get(ring, "numFtsZSubunitsPerFilament"),
            _get(ring, "filamentLengthInNm"),
        )
        complex_productions = np.asarray(
            _get(states, "complexProductions")
        ).reshape(-1)
        activated = (
            np.asarray(self.enzymeIndexs_FtsZ_activated, dtype=int).ravel() - 1
        )
        global_indices = (
            np.asarray(self.enzymeGlobalIndexs, dtype=int).ravel() - 1
        )
        gtp_cost = (
            gtp_cost
            + np.arange(1, 10) @ complex_productions[global_indices[activated]]
        )
        gdp_index = int(np.asarray(self.enzymeIndexs_FtsZ_GDP).item()) - 1
        gdp_global = global_indices[gdp_index]
        gdp_9mer = (
            np.asarray(
                self.complex.getIndexs("MG_224_9MER_GDP"), dtype=int
            ).ravel()
            - 1
        )
        gdp_cost = (
            gtp_cost
            - complex_productions[gdp_global]
            - 9 * complex_productions[gdp_9mer[0]]
        )
        biomass = np.zeros((len(self.substrateWholeCellModelIDs), 1))
        byproducts = np.zeros_like(biomass)
        biomass[int(self.substrateIndexs_gtp) - 1, 0] = gtp_cost
        byproducts[int(self.substrateIndexs_gdp) - 1, 0] = gdp_cost
        minimum = np.zeros((len(self.enzymeWholeCellModelIDs), 1))
        maximum = np.full_like(minimum, np.inf)
        return biomass, byproducts, minimum, maximum

    def initializeState(self):
        max_steps = 50
        tol = max(1, 0.002 * np.asarray(self.enzymes).sum())
        substrates = np.asarray(self.substrates).copy()
        self.substrates[...] = 1e6
        enzymes = np.asarray(self.enzymes).copy()
        for _ in range(max_steps):
            self.evolveState()
            current = np.asarray(self.enzymes)
            if np.linalg.norm(enzymes - current, ord=np.inf) < tol:
                break
            enzymes = current.copy()
        self.substrates = substrates

    def calcResourceRequirements_Current(self):
        r = np.zeros_like(np.asarray(self.substrates), dtype=float)
        r[1] = (
            np.asarray(self.enzymes).reshape(-1)[0]
            + np.asarray(self.enzymes).reshape(-1)[1]
        )
        return r

    @staticmethod
    def _diff(y, p):
        a, ar, ex, er, nu, nur, el, elr, gdp, gtp = p
        d = np.zeros_like(y)
        d[0] = -a * y[0] + ar * y[2]
        d[1] = -ex * y[1] * gtp + er * y[2] * gdp
        d[2] = (
            a * y[0]
            - ar * y[2]
            + ex * y[1] * gtp
            - er * y[2] * gdp
            - 2 * nu * y[2] ** 2
            + 2 * nur * y[3]
            - el * y[2] * y[3:-1].sum()
            + elr * y[4:].sum()
        )
        d[3] = nu * y[2] ** 2 - nur * y[3] - el * y[2] * y[3] + elr * y[4]
        for i in range(4, len(y) - 1):
            d[i] = (
                el * y[2] * y[i - 1]
                - elr * y[i]
                - el * y[2] * y[i]
                + elr * y[i + 1]
            )
        d[-1] = el * y[2] * y[-2] - elr * y[-1]
        return d

    @staticmethod
    def _jac(y, p):
        a, ar, ex, er, nu, nur, el, elr, gdp, gtp = p
        n = len(y)
        J = np.zeros((n, n))
        J[0, 0], J[0, 2] = -a, ar
        J[1, 1], J[1, 2] = -ex * gtp, er * gdp
        J[2, 0], J[2, 1], J[2, 2] = (
            a,
            ex * gtp,
            -ar - er * gdp - 4 * nu * y[2] - el * y[3:-1].sum(),
        )
        J[2, 3:] += elr
        J[2, 3:-1] -= el * y[2]
        J[2, 3] += 2 * nur
        J[3, 2], J[3, 3], J[3, 4] = (
            2 * nu * y[2] - el * y[3],
            -nur - el * y[2],
            elr,
        )
        for i in range(4, n - 1):
            J[i, 2] = el * (y[i - 1] - y[i])
            J[i, i - 1] = el * y[2]
            J[i, i] = -elr - el * y[2]
            J[i, i + 1] = elr
        J[-1, 2] = el * y[-2]
        J[-1, -2] = el * y[2]
        J[-1, -1] = -elr
        return J

    def integrateODEs(self, y0):
        # WholeCell's adaptive, linearly implicit modified ode23s integrator.
        y = np.asarray(y0, dtype=float).reshape(-1)
        yout = [y.copy()]
        tout = [0.0]
        rate_constants = [
            float(np.asarray(value).item())
            for value in (
                self.activationFwd,
                self.activationRev,
                self.exchangeFwd,
                self.exchangeRev,
                self.nucleationFwd,
                self.nucleationRev,
                self.elongationFwd,
                self.elongationRev,
            )
        ]
        substrate_concentrations = self.moleculesToConcentration(
            self.substrates
        ).reshape(-1)
        p = np.r_[rate_constants, substrate_concentrations[[0, 1]]]
        tf = float(np.asarray(self.stepSizeSec).item())
        hmax = 0.1 * tf
        threshold = 0.1 / _N_AVOGADRO / float(self.geometry.volume)
        rtol = 0.01
        pow_ = 1 / 3
        d = 1 / (2 + np.sqrt(2))
        e32 = 6 + np.sqrt(2)
        t = 0.0
        f0 = self._diff(y, p)
        J = self._jac(y, p)
        wt = np.maximum(abs(y), threshold)
        rh = 1.25 * np.sqrt(0.5 * np.max(abs(J @ f0 / wt))) / rtol**pow_
        ah = 1 / rh
        first = False
        while t < tf:
            hmin = 16 * np.finfo(float).eps * t
            ah = min(hmax, max(hmin, ah))
            done = 1.1 * ah >= abs(tf - t)
            h = tf - t if done else ah
            if first:
                J = self._jac(y, p)
            while True:
                M = np.eye(len(y)) - h * d * J
                k1 = np.linalg.solve(M, f0)
                f1 = self._diff(y + 0.5 * h * k1, p)
                k2 = np.linalg.solve(M, f1 - k1) + k1
                tn = tf if done else t + h
                h = tn - t
                yn = y + h * k2
                f2 = self._diff(yn, p)
                k3 = np.linalg.solve(M, f2 - e32 * (k2 - f1) - 2 * (k1 - f0))
                err = (
                    ah
                    / 6
                    * np.max(
                        abs(
                            (k1 - 2 * k2 + k3)
                            / np.maximum(
                                np.maximum(abs(y), abs(yn)), threshold
                            )
                        )
                    )
                )
                if err <= rtol:
                    break
                if ah <= hmin:
                    return np.asarray(tout), np.array(yout).T
                ah = max(hmin, ah * max(0.1, 0.8 * (rtol / err) ** pow_))
                h = ah
                done = False
            y = yn
            t = tn
            tout.append(t)
            yout.append(y.copy())
            f0 = f2
            first = True
            if done:
                break
            temp = 1.25 * (err / rtol) ** pow_
            ah = ah / temp if temp > 0.2 else 5 * ah
        return np.asarray(tout), np.array(yout).T

    def discretizeEnzymes(self, concentrations):
        molecule_counts = self.concentrationToMolecules(
            concentrations
        ).reshape(-1, 1)
        e = (
            np.asarray(self.randStream.stochasticRound(molecule_counts))
            .reshape(-1)
            .astype(int)
        )
        max_length = int(np.asarray(self.maxPolymerLength).item())
        weights = np.concatenate(([1, 1], np.arange(1, max_length + 1)))
        while True:
            delta = int(weights @ (e - np.asarray(self.enzymes).reshape(-1)))
            if delta == 0:
                return e
            idx = (
                int(
                    np.ceil(
                        len(weights)
                        * float(np.asarray(self.randStream.rand()).item())
                    )
                )
                - 1
            )
            if delta > 0 and e[idx] > 0:
                e[idx] -= 1
                if idx > 1:
                    e[idx - 1] += 1
            elif delta < 0:
                if idx > 1 and e[idx - 1] > 0:
                    e[idx - 1] -= 1
                    e[idx] += 1
                elif idx <= 1:
                    e[idx] += 1

    def applySubstrateLimits(self, enzymes, substrates):
        e = np.asarray(enzymes).copy()
        s = np.asarray(substrates).reshape(-1).copy()
        old = np.asarray(self.enzymes).reshape(-1)
        max_length = int(np.asarray(self.maxPolymerLength).item())
        ngtp = np.concatenate(([0, 0], np.arange(1, max_length + 1)))
        ngdp = np.concatenate(([0, 1], np.zeros(max_length)))
        while ngtp @ (e - old) > s[1]:
            i = np.flatnonzero(e[2:])[0] + 2
            e[i] -= 1
            e[0] += i - 1
        while ngdp @ (e - old) > s[0] + min(s[3], s[1] - ngtp @ (e - old)):
            e[1] -= 1
            e[0] += 1
        s[1] -= ngtp @ (e - old)
        s[0] -= ngdp @ (e - old)
        s += np.array([1, -1, 1, -1, 1]) * max(0, -s[0])
        return e.reshape(np.asarray(self.enzymes).shape), s.reshape(
            np.asarray(substrates).shape
        )

    def evolveState(self):
        if not np.any(self.enzymes):
            return
        _, ys = self.integrateODEs(self.moleculesToConcentration(self.enzymes))
        valid = np.flatnonzero(np.all(ys >= 0, axis=0))
        enz = self.discretizeEnzymes(ys[:, valid[-1]])
        self.enzymes, self.substrates = self.applySubstrateLimits(
            enz, self.substrates
        )

    def moleculesToConcentration(self, count):
        return np.asarray(count) / (_N_AVOGADRO * float(self.geometry.volume))

    def concentrationToMolecules(self, conc):
        return np.asarray(conc) * (_N_AVOGADRO * float(self.geometry.volume))
