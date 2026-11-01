"""N-dimensional sparse matrices with circular coordinate dimensions.

Coordinates and dimension numbers use MATLAB's one-based convention. Storage
remains coordinate sparse; wrapping is applied before delegating to SparseMat.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from .sparsemat import SparseMat, _integer_coordinates


class CircularSparseMat(SparseMat):
    """SparseMat with selected dimensions that wrap around their extent."""

    def __init__(self, *args: Any):
        circular_dims: Any = ()
        if len(args) == 0:
            super().__init__()
        elif len(args) == 1:
            src = args[0]
            circular_dims = (
                src.circularDims if isinstance(src, CircularSparseMat) else ()
            )
            super().__init__(src)
        elif len(args) == 2:
            mat, circular_dims = args
            super().__init__(mat)
        elif len(args) == 3:
            subs, vals, siz = args
            super().__init__(subs, vals, siz)
        elif len(args) == 4:
            subs, vals, siz, circular_dims = args
            shape = _integer_coordinates(siz, "siz").reshape(-1)
            if shape.size == 1:
                shape = np.r_[shape, 1]
            coords = _integer_coordinates(subs, "subs").copy()
            if coords.size:
                if coords.ndim == 1:
                    coords = coords.reshape(1, -1)
                dims0 = _dims(circular_dims)
                for dim in dims0:
                    d = dim - 1
                    if d < 0 or d >= len(shape):
                        raise ValueError(
                            "circularDims must be within 1..ndims"
                        )
                    if shape[d] <= 0:
                        raise ValueError(
                            "cannot wrap coordinates in an empty dimension"
                        )
                    coords[:, d] = (coords[:, d] - 1) % shape[d] + 1
            super().__init__(coords, vals, siz)
        else:
            raise TypeError(
                "CircularSparseMat accepts zero, one, two, three, or four arguments"
            )

        cd = _dims(circular_dims)
        if len(set(cd)) != len(cd) or any(
            d < 1 or d > len(self.siz) for d in cd
        ):
            raise ValueError(
                "circularDims must be distinct positive integers in 1..ndims(this)"
            )
        self.circularDims = np.asarray(
            cd if len(cd) == 1 else sorted(cd), dtype=np.int64
        )

    def copy(self):
        return CircularSparseMat(self, self.circularDims)

    def isDimCircular(self, dim: int) -> bool:
        return bool(np.any(self.circularDims == int(dim)))

    def isequal(self, other: Any) -> bool:
        return (
            isinstance(other, SparseMat)
            and np.array_equal(self.siz, other.siz)
            and np.array_equal(self.subs, other.subs)
            and np.array_equal(self.vals, other.vals)
            and np.array_equal(
                self.circularDims,
                getattr(other, "circularDims", np.array([], dtype=int)),
            )
        )

    def get_coords(self, coords):
        return super().get_coords(self._wrap_coords(coords))

    def reshape(self, *dims):
        # CircularSparseMat.m delegates reshape to SparseMat with oldDims=zeros,
        # so a reshape deliberately clears circular metadata.
        return self._from_sparse(super().reshape(*dims), ())

    def permute(self, dims):
        order = np.asarray(dims, dtype=np.int64).reshape(-1)
        result = super().permute(order)
        mapped = [
            axis + 1
            for axis, original_axis in enumerate(order)
            if original_axis in self.circularDims
        ]
        return self._from_sparse(result, mapped)

    def squeeze(self):
        old_dims = np.flatnonzero(self.siz != 1) + 1
        result = super().squeeze()
        if old_dims.size < 2:
            old_dims = np.r_[old_dims, 0]
        mapped = [
            axis + 1
            for axis, original_axis in enumerate(old_dims)
            if original_axis in self.circularDims
        ]
        return self._from_sparse(result, mapped)

    def sum(self, dim=1):
        if dim > self.ndims():
            return self.copy()
        result = super().sum(dim)
        # WholeCell SparseMat.sum passes every remaining original axis to the
        # CircularSparseMat constructor, even if it was not circular before.
        remaining_dims = np.delete(np.arange(1, self.ndims() + 1), dim - 1)
        return self._from_sparse(result, remaining_dims)

    def set_coords(self, coords, values):
        return CircularSparseMat._from_sparse(
            super().set_coords(self._wrap_coords(coords), values),
            self.circularDims,
        )

    def __getitem__(self, key):
        # A single N-column coordinate array follows SparseMat's coordinate API.
        if isinstance(key, (list, np.ndarray)):
            a = np.asarray(key)
            if a.ndim <= 2 and (a.ndim == 1 or a.shape[-1] == self.ndims()):
                return self.get_coords(a)
        if not isinstance(key, tuple):
            key = (key,)
        selectors = list(key)
        if len(selectors) > self.ndims():
            raise IndexError("too many subscripts")
        if len(selectors) == 1 and self.ndims() > 1:
            # MATLAB linear indexing; circular dimensions do not apply.
            return SparseMat.get_coords(
                self,
                self.ind2sub(np.asarray(selectors[0]).reshape(-1), self.siz),
            )
        selectors += [slice(None)] * (self.ndims() - len(selectors))
        axes = [self._selector(s, i) for i, s in enumerate(selectors)]
        coords = np.asarray(list(product(*axes)), dtype=np.int64).reshape(
            -1, self.ndims()
        )
        return self.get_coords(coords)

    def __setitem__(self, key, values):
        if isinstance(key, (list, np.ndarray)):
            a = np.asarray(key)
            if a.ndim <= 2 and (a.ndim == 1 or a.shape[-1] == self.ndims()):
                coords = self._wrap_coords(a)
                updated = super().set_coords(coords, values)
                self._adopt(updated)
                return
        selectors = list(key if isinstance(key, tuple) else (key,))
        if len(selectors) == 1 and self.ndims() > 1:
            coords = self.ind2sub(
                np.asarray(selectors[0]).reshape(-1), self.siz
            )
        else:
            if len(selectors) > self.ndims():
                raise IndexError("too many subscripts")
            selectors += [slice(None)] * (self.ndims() - len(selectors))
            axes = [self._selector(s, i) for i, s in enumerate(selectors)]
            coords = np.asarray(list(product(*axes)), dtype=np.int64).reshape(
                -1, self.ndims()
            )
        self._adopt(super().set_coords(self._wrap_coords(coords), values))

    def _selector(self, selector, axis):
        if isinstance(selector, slice):
            start = 1 if selector.start is None else int(selector.start)
            stop = (
                int(self.siz[axis])
                if selector.stop is None
                else int(selector.stop)
            )
            step = 1 if selector.step is None else int(selector.step)
            vals = np.arange(
                start, stop + (1 if step > 0 else -1), step, dtype=np.int64
            )
        else:
            vals = _integer_coordinates(selector, "subscript").reshape(-1)
        if axis + 1 in self.circularDims and len(vals):
            vals = (vals - 1) % self.siz[axis] + 1
        elif np.any(vals < 1) or np.any(vals > self.siz[axis]):
            raise IndexError("subscript out of range")
        return vals.tolist()

    def _wrap_coords(self, coords):
        c = _integer_coordinates(coords, "coordinates").copy()
        if c.size == 0:
            return c.reshape(0, self.ndims())
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if c.shape[1] != self.ndims():
            raise ValueError("coordinate width must equal ndims")
        for dim in self.circularDims:
            d = int(dim) - 1
            if self.siz[d] <= 0:
                raise IndexError("cannot wrap indices in an empty dimension")
            c[:, d] = (c[:, d] - 1) % self.siz[d] + 1
        return c

    def _adopt(self, other):
        self.siz, self.subs, self.vals = other.siz, other.subs, other.vals

    @classmethod
    def _from_sparse(cls, sparse, dims):
        return cls(sparse.subs, sparse.vals, sparse.siz, dims)

    def normalize(self, *args, **kwargs):
        old_dims = args[3] if len(args) > 3 else kwargs.get("oldDims")
        if old_dims is None:
            old_dims = np.arange(1, self.ndims() + 1, dtype=np.int64)
        old_dims = np.asarray(old_dims, dtype=np.int64).reshape(-1)
        result = super().normalize(*args, **kwargs)
        result.circularDims = np.asarray(
            [
                i + 1
                for i, original in enumerate(old_dims)
                if original in self.circularDims
            ],
            dtype=np.int64,
        )
        result.circularDims = result.circularDims[
            result.circularDims <= result.ndims()
        ]
        return result

    def tprod(self, *args, **kwargs):
        if len(args) != 3:
            raise TypeError("tprod expects other, Aid, and Bid")
        other, aid, bid = args
        other_matrix = (
            other if isinstance(other, SparseMat) else SparseMat(other)
        )
        result = super().tprod(other_matrix, aid, bid)
        aid = np.asarray(
            [aid] if np.isscalar(aid) else aid, dtype=np.int64
        ).reshape(-1)
        bid = np.asarray(
            [bid] if np.isscalar(bid) else bid, dtype=np.int64
        ).reshape(-1)
        aod = np.setdiff1d(np.arange(1, self.ndims() + 1), aid)
        bod = np.setdiff1d(np.arange(1, other_matrix.ndims() + 1), bid)
        circular_dims = [
            int(np.where(aod == dim)[0][0] + 1)
            for dim in self.circularDims
            if dim in aod
        ]
        other_dims = getattr(
            other_matrix, "circularDims", np.empty(0, dtype=np.int64)
        )
        circular_dims.extend(
            int(np.where(bod == dim)[0][0] + len(aod) + 1)
            for dim in other_dims
            if dim in bod
        )
        return CircularSparseMat._from_sparse(result, circular_dims)

    def __repr__(self):
        return f"CircularSparseMat(siz={self.siz.tolist()}, nnz={self.nnz()}, circularDims={self.circularDims.tolist()})"


def _dims(value):
    a = _integer_coordinates(value, "circularDims").reshape(-1)
    return [int(x) for x in a]
