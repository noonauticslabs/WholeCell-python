"""Runtime counterpart of MATLAB's compartment constants."""

from __future__ import annotations

from typing import Any

import numpy as np


class Compartment:
    """Compartment IDs, names, and MATLAB-compatible one-based indices."""

    def __init__(self, **snapshot_fields: Any) -> None:
        self.count = None
        self.wholeCellModelIDs = None
        self.names = None
        self.cytosolIndexs = None
        self.chromosomeIndexs = None
        self.membraneIndexs = None
        self.terminalOrganelleCytosolIndexs = None
        self.terminalOrganelleMembraneIndexs = None
        self.extracellularIndexs = None
        self.cellularIndexs = None
        for name, value in snapshot_fields.items():
            setattr(self, name, value)

    def initializeConstants(self, knowledgeBase, simulation=None):
        """Copy compartment constants from the KB as in Compartment.m."""

        def field(obj, name):
            return obj[name] if isinstance(obj, dict) else getattr(obj, name)

        self.count = field(knowledgeBase, "numCompartments")
        records = field(knowledgeBase, "compartments")
        self.wholeCellModelIDs = np.asarray(
            [field(r, "wholeCellModelID") for r in records], dtype=object
        ).reshape(-1, 1)
        self.names = np.asarray(
            [field(r, "name") for r in records], dtype=object
        ).reshape(-1, 1)
        for prefix in (
            "cytosol",
            "chromosome",
            "membrane",
            "terminalOrganelleCytosol",
            "terminalOrganelleMembrane",
            "extracellular",
            "cellular",
        ):
            setattr(
                self,
                prefix + "Indexs",
                np.array(
                    field(knowledgeBase, prefix + "CompartmentIndexs"),
                    copy=True,
                ),
            )

    def getIndexs(self, wholeCellModelIDs: Any) -> Any:
        """Return MATLAB-style one-based locations, using zero for unknown IDs."""
        ids = self.wholeCellModelIDs
        if ids is None:
            raise RuntimeError(
                "compartment constants have not been initialized"
            )
        ids = np.asarray(ids, dtype=object)
        # Snapshot character arrays are MATLAB char matrices; cells hold these
        # directly while an individual char row is represented as U1 elements.
        known = [_as_id(value) for value in ids.ravel(order="F")]
        lookup = {value: index for index, value in enumerate(known, start=1)}
        if isinstance(wholeCellModelIDs, str):
            return lookup.get(wholeCellModelIDs, 0)
        query = np.asarray(wholeCellModelIDs, dtype=object)
        result = np.asarray(
            [lookup.get(_as_id(value), 0) for value in query.ravel(order="F")],
            dtype=int,
        )
        return result.reshape(query.shape, order="F")


def _as_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "US":
            return "".join(str(part) for part in value.ravel(order="F"))
        if value.size == 1:
            return _as_id(value.item())
    return str(value)
