"""Coordinate sparse arrays with the indexing rules used by WholeCell's SparseMat.

Coordinates and linear indices exposed by this class are ONE-BASED. Linear
index conversion follows MATLAB/Fortran order. Instances behave as value
objects: mutating methods return an independent SparseMat.
"""

from __future__ import annotations

from typing import Any, Iterable
import numpy as np


def _coordinate_grid(siz):
    """Yield all one-based output coordinates in Fortran axis order."""
    shape = tuple(int(x) for x in np.asarray(siz).reshape(-1))
    if not shape:
        yield ()
        return
    if any(x == 0 for x in shape):
        return
    for coord in np.ndindex(shape):
        yield tuple(x + 1 for x in coord)


def _stored_at_free(mat, inner_dims, free_dims, inner):
    return {
        tuple(int(x) for x in row[free_dims])
        for row in mat.subs
        if tuple(row[inner_dims]) == inner
    }


def _integer_coordinates(value, name):
    array = np.asarray(value)
    if array.dtype.kind in "iu":
        return array.astype(np.int64, copy=False)
    if array.dtype.kind == "b":
        return array.astype(np.int64, copy=False)
    if (
        array.dtype.kind != "f"
        or not np.isfinite(array).all()
        or np.any(array != np.floor(array))
    ):
        raise ValueError(f"{name} must contain finite integers")
    return array.astype(np.int64, copy=False)


class SparseMat:
    """N-dimensional sparse value array stored as ``subs``, ``vals``, ``siz``."""

    def __init__(self, *args: Any):
        if not args:
            self.siz = np.array([0, 2], dtype=np.int64)
            self.subs = np.empty((0, 2), dtype=np.int64)
            self.vals = np.empty((0, 1), dtype=float)
        elif len(args) == 1:
            a = args[0]
            if isinstance(a, SparseMat):
                self.siz, self.subs, self.vals = (
                    a.siz.copy(),
                    a.subs.copy(),
                    a.vals.copy(),
                )
            elif isinstance(a, dict):
                self.__init__(a["subs"], a["vals"], a["siz"])
                return
            else:
                dense = np.asarray(a)
                if dense.ndim == 0:
                    dense = dense.reshape(1, 1)
                if dense.ndim == 1:
                    dense = dense.reshape(-1, 1)
                self.siz = np.asarray(dense.shape, dtype=np.int64)
                self.subs = np.argwhere(dense != 0).astype(np.int64) + 1
                self.vals = (
                    dense[tuple((self.subs - 1).T)].reshape(-1, 1)
                    if len(self.subs)
                    else np.empty((0, 1), dtype=dense.dtype)
                )
                self._normalize()
        elif len(args) == 3:
            subs, vals, siz = args
            self.siz = _integer_coordinates(siz, "siz").reshape(-1).copy()
            if np.any(self.siz < 0):
                raise ValueError("invalid sparse dimensions or coordinates")
            if self.siz.size == 1:
                self.siz = np.r_[self.siz, 1]
            self.subs = _integer_coordinates(subs, "subs").copy()
            if self.subs.size == 0:
                self.subs = np.empty((0, self.siz.size), dtype=np.int64)
            if self.subs.ndim == 1:
                self.subs = self.subs.reshape(1, -1)
            values = np.asarray(vals)
            if values.ndim > 2:
                raise ValueError("vals must be a vector or scalar")
            self.vals = values.reshape(-1, 1).copy()
            if self.vals.size == 1 and len(self.subs) != 1:
                self.vals = np.repeat(self.vals, len(self.subs), axis=0)
            if self.subs.shape[1] > len(self.siz) and np.all(
                self.subs[:, len(self.siz) :] == 1
            ):
                self.subs = self.subs[:, : len(self.siz)]
            if self.subs.shape[1] != len(self.siz) or len(self.vals) != len(
                self.subs
            ):
                raise ValueError("subscript/value dimensions mismatch")
            if (
                np.any(self.siz < 0)
                or np.any(self.subs < 1)
                or (len(self.subs) and np.any(self.subs > self.siz))
            ):
                raise ValueError("invalid sparse dimensions or coordinates")
            self._deduplicate_last()
            self._normalize()
        else:
            raise TypeError("SparseMat accepts zero, one, or three arguments")

    def _deduplicate_last(self):
        last = {}
        for i, row in enumerate(self.subs):
            last[tuple(row)] = i
        ix = sorted(last.values())
        self.subs, self.vals = self.subs[ix], self.vals[ix]

    def _normalize(self):
        self.vals = self.vals.reshape(-1, 1)
        keep = self.vals[:, 0] != 0
        self.subs, self.vals = self.subs[keep], self.vals[keep]
        # MATLAB sort_subs orders by column-major linear index.
        order = np.lexsort(
            tuple(self.subs[:, j] for j in range(self.subs.shape[1]))
        )
        self.subs, self.vals = self.subs[order], self.vals[order]
        while len(self.siz) > 2 and self.siz[-1] == 1:
            if len(self.subs) and np.any(self.subs[:, -1] != 1):
                break
            self.siz, self.subs = self.siz[:-1], self.subs[:, :-1]
        if len(self.siz) == 1:
            self.siz = np.r_[self.siz, 1]

    def normalize(
        self,
        reSize: bool = True,
        reFind: bool = True,
        reSort: bool = True,
        oldDims: Any = None,
    ):
        """Return a normalized copy, following WholeCell ``SparseMat.normalize``.

        ``oldDims`` is carried for subclasses that track dimension metadata;
        base SparseMat only normalizes the coordinate representation.
        """
        del oldDims
        siz = np.asarray(self.siz, dtype=np.int64).reshape(-1).copy()
        subs = np.asarray(self.subs, dtype=np.int64).copy()
        vals = np.asarray(self.vals).reshape(-1, 1).copy()
        if reSize:
            if siz.size == 0:
                siz = np.array([0, 0], dtype=np.int64)
                if subs.size or vals.size:
                    raise ValueError("index exceeds matrix dimensions")
                subs = np.empty((0, 2), dtype=np.int64)
                vals = np.empty((0, 1), dtype=vals.dtype)
            elif siz.size == 1:
                siz = np.r_[siz, 1]
                subs = np.column_stack(
                    (subs, np.ones(len(subs), dtype=np.int64))
                )
            # MATLAB SparseMat retains at least two dimensions and removes
            # trailing singleton dimensions only when requested.
            if siz.size > 2 and siz[-1] == 1:
                keep_dims = max(
                    2,
                    (
                        int(np.flatnonzero(siz != 1)[-1] + 1)
                        if np.any(siz != 1)
                        else 2
                    ),
                )
                if subs.size and np.any(subs[:, keep_dims:] != 1):
                    raise ValueError("index exceeds matrix dimensions")
                siz, subs = siz[:keep_dims], subs[:, :keep_dims]
        if reFind and vals.size:
            keep = vals[:, 0] != 0
            subs, vals = subs[keep], vals[keep]
        if reSort and len(subs):
            order = np.lexsort(tuple(subs[:, j] for j in range(subs.shape[1])))
            subs, vals = subs[order], vals[order]
        result = self.__class__.__new__(self.__class__)
        result.siz, result.subs, result.vals = siz, subs, vals
        return result

    def copy(self):
        return SparseMat(self)

    @property
    def inds(self):
        return self.sub2ind(self.subs)

    def size(self, k=None):
        return (
            self.siz.copy()
            if k is None
            else (int(self.siz[k - 1]) if 1 <= k <= len(self.siz) else 1)
        )

    def ndims(self):
        return len(self.siz)

    def numel(self):
        return int(np.prod(self.siz, dtype=np.int64))

    def nnz(self):
        return len(self.vals)

    def length(self):
        return int(max(self.siz, default=0))

    def isempty(self):
        return not bool(np.all(self.siz))

    def full(self):
        out = np.zeros(tuple(self.siz), dtype=self.vals.dtype)
        if len(self.subs):
            out[tuple((self.subs - 1).T)] = self.vals[:, 0]
        return out

    def find(self):
        return self.subs.copy(), self.vals.copy()

    def get_coords(self, coords):
        c = _integer_coordinates(coords, "coordinates")
        if c.size == 0:
            return np.zeros((0, 1), dtype=self.vals.dtype)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if c.shape[1] != len(self.siz):
            raise ValueError("coordinate width must equal ndims")
        if np.any(c < 1) or np.any(c > self.siz):
            raise IndexError("subscript out of range")
        # SparseMat normalizes entries into MATLAB column-major order. Use
        # those integer linear indices for the common lookup path; constructing
        # Python tuple keys and a dict for every read was particularly costly
        # for repeated chromosome queries. The arrays are intentionally public
        # and mutable, so detect callers that have broken the normalized order
        # and retain the historical last-entry-wins behavior for that case.
        if (
            self.subs.ndim != 2
            or self.subs.shape[1] != len(self.siz)
            or len(self.vals) != len(self.subs)
            or (
                len(self.subs)
                and np.any((self.subs < 1) | (self.subs > self.siz))
            )
        ):
            lookup = {tuple(s): v for s, v in zip(self.subs, self.vals[:, 0])}
            return np.asarray(
                [lookup.get(tuple(row), 0) for row in c]
            ).reshape(-1, 1)
        strides = np.r_[1, np.cumprod(self.siz[:-1], dtype=np.int64)]
        keys = 1 + (self.subs - 1) @ strides
        if len(keys) > 1 and (np.any(keys[1:] <= keys[:-1])):
            lookup = {tuple(s): v for s, v in zip(self.subs, self.vals[:, 0])}
            return np.asarray(
                [lookup.get(tuple(row), 0) for row in c]
            ).reshape(-1, 1)

        result_indices = np.full(len(c), -1, dtype=np.int64)
        if len(keys):
            query = 1 + (c - 1) @ strides
            positions = np.searchsorted(keys, query)
            found = positions < len(keys)
            found[found] &= keys[positions[found]] == query[found]
            result_indices[found] = positions[found]
        matched = result_indices >= 0
        # Preserve NumPy's historical list-inference behavior: a fully
        # matched read keeps the values dtype, a mixed hit/miss read promotes
        # with Python's integer zero, and an all-missing read is int64.
        if not np.any(matched):
            result_dtype = np.dtype(np.int64)
        elif np.all(matched):
            result_dtype = self.vals.dtype
        else:
            result_dtype = np.result_type(self.vals.dtype, np.int64)
        result = np.zeros(len(c), dtype=result_dtype)
        result[matched] = self.vals[result_indices[matched], 0]
        return result.reshape(-1, 1)

    def set_coords(self, coords, values):
        c = _integer_coordinates(coords, "coordinates")
        v = np.asarray(values).reshape(-1)
        if c.size == 0:
            return self.copy()
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if (
            c.shape[1] != len(self.siz)
            or np.any(c < 1)
            or np.any(c > self.siz)
        ):
            raise IndexError("invalid coordinates")
        if len(v) == 1:
            v = np.repeat(v, len(c))
        if len(v) != len(c):
            raise ValueError("coordinate/value count mismatch")
        d = {tuple(s): x for s, x in zip(self.subs, self.vals[:, 0])}
        for s, x in zip(c, v):
            if x == 0:
                d.pop(tuple(s), None)
            else:
                d[tuple(s)] = x
        keys = sorted(d)
        return SparseMat(
            np.asarray(keys, dtype=np.int64).reshape(-1, len(self.siz)),
            np.asarray([d[k] for k in keys]),
            self.siz,
        )

    def take(self, *ranges):
        """Gather inclusive one-based coordinate ranges (one per dimension)."""
        if len(ranges) == 1:
            return self.get_coords(ranges[0])
        grids = np.meshgrid(
            *[np.asarray(r, dtype=np.int64).reshape(-1) for r in ranges],
            indexing="ij",
        )
        c = np.stack([g.ravel(order="F") for g in grids], axis=1)
        return self.get_coords(c)

    def assign(self, coords, values):
        return self.set_coords(coords, values)

    def sub2ind(self, subs, siz=None):
        s = np.asarray(subs, dtype=np.int64)
        z = self.siz if siz is None else np.asarray(siz, dtype=np.int64)
        if s.size == 0:
            return np.empty((0, 1), dtype=np.int64)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        if s.shape[1] != len(z) or np.any(s < 1) or np.any(s > z):
            raise IndexError("invalid subscripts")
        return (1 + (s - 1) @ np.r_[1, np.cumprod(z[:-1])]).reshape(-1, 1)

    def ind2sub(self, inds, siz=None):
        z = self.siz if siz is None else np.asarray(siz, dtype=np.int64)
        ix = np.asarray(inds, dtype=np.int64).reshape(-1) - 1
        if np.any(ix < 0) or np.any(ix >= np.prod(z)):
            raise IndexError("linear index out of range")
        return np.column_stack(
            [ix // int(np.prod(z[:i])) % z[i] + 1 for i in range(len(z))]
        ).astype(np.int64)

    def reshape(self, *dims):
        if len(dims) == 1 and not np.isscalar(dims[0]):
            dims = tuple(np.asarray(dims[0]).reshape(-1))
        dims = np.asarray(dims, dtype=np.int64)
        if np.count_nonzero(dims == -1) > 1:
            raise ValueError("one inferred dimension allowed")
        if np.any(dims == -1):
            dims[dims == -1] = self.numel() // int(np.prod(dims[dims != -1]))
        if np.any(dims < 0) or int(np.prod(dims)) != self.numel():
            raise ValueError("element count must not change")
        return SparseMat(self.ind2sub(self.inds, dims), self.vals, dims)

    def permute(self, dims):
        p = np.asarray(dims, dtype=int) - 1
        if sorted(p.tolist()) != list(range(len(self.siz))):
            raise ValueError("invalid permutation")
        return SparseMat(self.subs[:, p], self.vals, self.siz[p])

    def tprod(self, other, Aid, Bid):
        """Tensor contraction along paired, one-based dimensions.

        The coordinate join handles ordinary finite entries sparsely. As in
        WholeCell's implementation, NaNs propagate across every output value
        touched by their contracted coordinate, and Inf multiplied by an
        implicit zero contributes NaN.
        """
        b = other if isinstance(other, SparseMat) else SparseMat(other)
        a = self
        aid = np.asarray(
            [Aid] if np.isscalar(Aid) else Aid, dtype=np.int64
        ).reshape(-1)
        bid = np.asarray(
            [Bid] if np.isscalar(Bid) else Bid, dtype=np.int64
        ).reshape(-1)
        if (
            aid.size != bid.size
            or np.any(aid < 1)
            or np.any(aid > a.ndims())
            or np.any(bid < 1)
            or np.any(bid > b.ndims())
            or len(np.unique(aid)) != len(aid)
            or len(np.unique(bid)) != len(bid)
        ):
            raise ValueError(
                "inner dimensions must be unique valid dimensions of equal length"
            )
        aid0, bid0 = aid - 1, bid - 1
        if not np.array_equal(a.siz[aid0], b.siz[bid0]):
            raise ValueError("paired inner dimensions must have equal sizes")
        aod = np.setdiff1d(np.arange(a.ndims()), aid0)
        bod = np.setdiff1d(np.arange(b.ndims()), bid0)
        out_siz = np.r_[a.siz[aod], b.siz[bod]]
        if not out_siz.size:
            out_siz = np.array([1, 1], dtype=np.int64)
        elif out_siz.size == 1:
            out_siz = np.r_[out_siz, 1]

        av = a.vals[:, 0]
        bv = b.vals[:, 0]
        a_nan, b_nan = np.isnan(av), np.isnan(bv)
        a_inf, b_inf = np.isinf(av), np.isinf(bv)
        out = {}

        def add(coord, value):
            key = tuple(int(x) for x in coord)
            if key in out:
                out[key] = out[key] + value
            else:
                out[key] = value

        # Multiply stored nonzero pairs whose contracted coordinates agree.
        b_by_inner = {}
        for i, row in enumerate(b.subs):
            b_by_inner.setdefault(tuple(row[bid0]), []).append(i)
        for i, ar in enumerate(a.subs):
            for j in b_by_inner.get(tuple(ar[aid0]), ()):
                add(np.r_[ar[aod], b.subs[j, bod]], av[i] * bv[j])

        # A stored NaN times any value (including implicit zero) is NaN.
        for i in np.flatnonzero(a_nan):
            for coord in _coordinate_grid(b.siz[bod]):
                add(np.r_[a.subs[i, aod], coord], np.nan)
        for j in np.flatnonzero(b_nan):
            for coord in _coordinate_grid(a.siz[aod]):
                add(np.r_[coord, b.subs[j, bod]], np.nan)

        # Infinite stored values times implicit zeros are NaN. Explicit
        # nonzero products were already included above.
        if np.any(a_inf):
            for i in np.flatnonzero(a_inf):
                inner = tuple(a.subs[i, aid0])
                for coord in _coordinate_grid(b.siz[bod]):
                    if tuple(coord) not in _stored_at_free(
                        b, bid0, bod, inner
                    ):
                        add(np.r_[a.subs[i, aod], coord], np.nan)
        if np.any(b_inf):
            for j in np.flatnonzero(b_inf):
                inner = tuple(b.subs[j, bid0])
                for coord in _coordinate_grid(a.siz[aod]):
                    if tuple(coord) not in _stored_at_free(
                        a, aid0, aod, inner
                    ):
                        add(np.r_[coord, b.subs[j, bod]], np.nan)

        coords, values = [], []
        for coord, value in out.items():
            if value != 0 or np.isnan(value):
                coords.append(coord)
                values.append(value)
        subs = np.asarray(coords, dtype=np.int64).reshape(-1, len(out_siz))
        result = SparseMat(subs, np.asarray(values), out_siz)
        # The MATLAB method returns the circular subclass when the right
        # operand is circular and the left operand is plain SparseMat.
        if other.__class__.__name__ == "CircularSparseMat":
            right_free = np.setdiff1d(np.arange(1, b.ndims() + 1), bid)
            circular = [
                int(np.where(right_free == dim)[0][0] + len(aod) + 1)
                for dim in other.circularDims
                if dim in right_free
            ]
            from .circular_sparsemat import CircularSparseMat

            return CircularSparseMat._from_sparse(result, circular)
        return result

    def transpose(self):
        if self.ndims() > 2:
            raise ValueError("transpose requires 2-D array")
        return self.permute([2, 1])

    T = property(transpose)

    def squeeze(self):
        dims = np.flatnonzero(self.siz != 1)
        subs = self.subs[:, dims]
        siz = self.siz[dims]
        if len(dims) < 2:
            padding = 2 - len(dims)
            subs = np.column_stack(
                (subs, np.ones((len(subs), padding), dtype=np.int64))
            )
            siz = np.r_[siz, np.ones(padding, dtype=np.int64)]
        return SparseMat(subs, self.vals, siz)

    def sum(self, dim=1):
        d = dim - 1
        if d < 0:
            raise ValueError("invalid dimension")
        if d >= len(self.siz):
            return self.copy()
        shape = self.siz.copy()
        shape[d] = 1
        if not len(self.subs):
            return SparseMat(np.empty((0, len(shape)), int), [], shape)
        ss = self.subs.copy()
        ss[:, d] = 1
        keys, inv = np.unique(ss, axis=0, return_inverse=True)
        # MATLAB accumarray(..., @sum) counts logical true values numerically;
        # NumPy's in-place add into a bool array instead performs logical OR.
        dtype = (
            np.float64
            if self.vals.dtype == np.dtype(bool)
            else self.vals.dtype
        )
        vals = np.zeros(len(keys), dtype=dtype)
        np.add.at(vals, inv, self.vals[:, 0].astype(dtype, copy=False))
        return SparseMat(keys, vals, shape)

    def collapse(self, dim=None):
        if dim is None:
            return np.sum(self.vals)
        dims = np.asarray([dim] if np.isscalar(dim) else dim, dtype=int)
        if np.all(dims < 0):
            dims = np.setdiff1d(np.arange(1, self.ndims() + 1), -dims)
        elif np.any(dims < 0):
            raise ValueError("cannot mix positive and negative dimensions")
        result = self
        for d in dims:
            result = result.sum(int(d))
        # MATLAB squeeze/double result; this materializes only the reduced output.
        return np.squeeze(result.full()).astype(float)

    def __repr__(self):
        return f"SparseMat(siz={self.siz.tolist()}, nnz={self.nnz()})"

    def __getitem__(self, coords):
        return self.get_coords(coords)

    def __setitem__(self, coords, values):
        result = self.set_coords(coords, values)
        self.subs, self.vals, self.siz = result.subs, result.vals, result.siz

    def _binary(self, other, op):
        if not isinstance(other, SparseMat):
            scalar = np.asarray(other)
            if scalar.size != 1:
                other = SparseMat(scalar)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    fill = op(0, scalar.item())
                if fill != 0:
                    # MATLAB explicitly materializes implicit entries when
                    # they become true/nonzero/NaN (eq zero, scalar addition,
                    # multiplication by Inf, and division by zero).
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return SparseMat(op(self.full(), scalar.item()))
                return SparseMat(
                    self.subs, op(self.vals[:, 0], scalar.item()), self.siz
                )
        if not np.array_equal(self.siz, other.siz):
            if other.numel() == 1:
                other = SparseMat(
                    np.full(tuple(self.siz), other.full().item())
                )
            elif self.numel() == 1:
                return other._binary(self.full().item(), lambda a, b: op(b, a))
            else:
                raise ValueError("sparse array sizes must agree")
        with np.errstate(divide="ignore", invalid="ignore"):
            implicit = op(0, 0)
            if implicit != 0:
                return SparseMat(op(self.full(), other.full()))
        keys, inverse = np.unique(
            np.vstack((self.subs, other.subs)), axis=0, return_inverse=True
        )
        a = np.zeros(len(keys), dtype=self.vals.dtype)
        b = np.zeros(len(keys), dtype=other.vals.dtype)
        a[inverse[: self.nnz()]] = self.vals[:, 0]
        b[inverse[self.nnz() :]] = other.vals[:, 0]
        return SparseMat(keys, np.asarray(op(a, b)).reshape(-1), self.siz)

    def __add__(self, o):
        return self._binary(o, np.add)

    def __radd__(self, o):
        return self + o

    def __sub__(self, o):
        return self._binary(o, np.subtract)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        return self._binary(o, np.multiply)

    def __rmul__(self, o):
        return self * o

    def __truediv__(self, o):
        return self._binary(o, np.divide)

    def __neg__(self):
        return SparseMat(self.subs, -self.vals, self.siz)

    def __pos__(self):
        return self.copy()

    def __eq__(self, o):
        return self._binary(o, np.equal)

    def __ne__(self, o):
        return self._binary(o, np.not_equal)

    def __lt__(self, o):
        return self._binary(o, np.less)

    def __le__(self, o):
        return self._binary(o, np.less_equal)

    def __gt__(self, o):
        return self._binary(o, np.greater)

    def __ge__(self, o):
        return self._binary(o, np.greater_equal)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise NotImplementedError(f"SparseMat.{name} is not ported yet")
