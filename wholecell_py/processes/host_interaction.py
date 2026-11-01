"""Host interaction Boolean rules ported from MATLAB.

Source: WholeCell/src/+edu/+stanford/+covert/+cell/+sim/+process/HostInteraction.m
All exported enzyme indices remain MATLAB one-based indices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..process_base import Process, _get


class HostInteraction(Process):
    enzymeWholeCellModelIDs = (
        "MG_075_MONOMER",
        "MG_149_MONOMER",
        "MG_191_MONOMER",
        "MG_192_MONOMER",
        "MG_200_MONOMER",
        "MG_217_MONOMER",
        "MG_218_MONOMER",
        "MG_288_MONOMER",
        "MG_309_MONOMER",
        "MG_312_MONOMER",
        "MG_317_MONOMER",
        "MG_318_MONOMER",
        "MG_386_MONOMER",
        "MG_412_MONOMER",
        "MG_410_411_412_PENTAMER",
    )
    enzymeIndexs_terminalOrganelle = np.array([3, 4, 6, 7, 10, 11, 12, 13])
    enzymeIndexs_terminalOrganelleCore = np.array([7, 10, 11])
    enzymeIndexs_terminalOrganelleLiprotein = np.array([3, 4, 5, 12, 13])
    enzymeIndexs_liprotein = np.array([1, 2, 3, 4, 5, 9, 12, 13, 14, 15])
    enzymeIndexs_adhesin = np.array([3, 4, 6, 12])
    enzymeIndexs_antigen = np.array([1, 2, 9, 14, 15])
    enzymeIndexs_tlrLigand = np.array([2, 9, 14, 15])
    enzymeIndexs_tlr12Ligand = np.array([2, 14, 15])
    enzymeIndexs_tlr26Ligand = np.array([9])
    stimuliWholeCellModelIDs: tuple[str, ...] = ()
    substrateWholeCellModelIDs: tuple[str, ...] = ()

    def __init__(
        self,
        wholeCellModelID="Process_HostInteraction",
        name="Host Interaction",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)

    def storeObjectReferences(self, simulation: Any) -> None:
        super().storeObjectReferences(simulation)
        statefn = getattr(simulation, "state", None)
        if callable(statefn):
            self.host = statefn("Host")
        else:
            states = _get(simulation, "states")
            self.host = next(
                v for k, v in states.items() if str(k).lower() == "host"
            )
        self.states = [*(self.states or []), self.host]

    def calcResourceRequirements_LifeCycle(self, *args, **kwargs):
        return (
            np.zeros(len(self.substrateWholeCellModelIDs)),
            np.zeros(len(self.substrateWholeCellModelIDs)),
            np.zeros(len(self.enzymeWholeCellModelIDs)),
            np.full(len(self.enzymeWholeCellModelIDs), np.inf),
        )

    def calcResourceRequirements_Current(self):
        return np.zeros_like(np.asarray(self.substrates))

    def initializeState(self):
        self.evolveState()

    @staticmethod
    def _onebased(values, indices):
        array = np.asarray(values)
        flat = array.ravel(order="F")
        return flat[np.asarray(indices, dtype=int).ravel() - 1]

    def evolveState(self):
        h = self.host
        enzymes = np.asarray(self.enzymes)
        if enzymes.ndim == 1:
            enzymes = enzymes.reshape((-1, 1))
        # MATLAB all/any across the selected matrix are scalar reductions.
        h.isBacteriumAdherent = bool(
            np.all(
                self._onebased(enzymes, self.enzymeIndexs_terminalOrganelle)
            )
            and np.all(self._onebased(enzymes, self.enzymeIndexs_adhesin))
        )
        adherent = bool(np.asarray(h.isBacteriumAdherent).ravel(order="F")[0])
        tlr = np.asarray(h.isTLRActivated, dtype=bool).copy()
        if tlr.ndim == 0:
            tlr = tlr.reshape((1, 1))
        tlr_flat = tlr.ravel(order="F")
        ligand12 = bool(
            np.any(self._onebased(enzymes, self.enzymeIndexs_tlr12Ligand))
        )
        ligand26 = bool(
            np.any(self._onebased(enzymes, self.enzymeIndexs_tlr26Ligand))
        )
        tlr_flat[int(h.tlrIndexs_1) - 1] = adherent and ligand12
        tlr_flat[int(h.tlrIndexs_2) - 1] = adherent and (ligand12 or ligand26)
        tlr_flat[int(h.tlrIndexs_6) - 1] = adherent and ligand26
        h.isTLRActivated = tlr_flat.reshape(tlr.shape, order="F")
        tlr1, tlr2, tlr6 = (
            bool(tlr_flat[int(i) - 1])
            for i in (h.tlrIndexs_1, h.tlrIndexs_2, h.tlrIndexs_6)
        )
        h.isNFkBActivated = (tlr2 and tlr1) or (tlr2 and tlr6)
        antigen = bool(
            np.any(self._onebased(enzymes, self.enzymeIndexs_antigen))
        )
        h.isInflammatoryResponseActivated = bool(
            h.isNFkBActivated or (adherent and antigen)
        )
