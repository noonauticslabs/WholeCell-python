"""NumPy-backed port of the simulation-facing MATLAB ``RandStream`` methods.

Random draws are statistically equivalent, not bit-for-bit compatible with MATLAB.
Indices returned by MATLAB-facing methods remain one-based.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


class RandStream:
    def __init__(
        self,
        type: str = "mt19937ar",
        *args: Any,
        seed: int | None = None,
        **kwargs: Any,
    ):
        # Accept MATLAB-style ('Seed', value) as well as Python seed=.
        if len(args) % 2:
            raise TypeError(
                "MATLAB constructor options must be name/value pairs"
            )
        opts = {
            str(args[i]).lower(): args[i + 1] for i in range(0, len(args), 2)
        }
        opts.update({str(k).lower(): v for k, v in kwargs.items()})
        self.type = type
        self.seed = seed if seed is not None else opts.get("seed", 0)
        self._rng = np.random.default_rng(self.seed)
        self._initial_state = deepcopy(self._rng.bit_generator.state)

    @staticmethod
    def _shape(args: tuple[Any, ...]) -> tuple[int, ...]:
        if not args:
            return (1, 1)
        if len(args) == 1 and np.ndim(args[0]) > 0:
            shape = tuple(int(x) for x in np.asarray(args[0]).reshape(-1))
        elif len(args) == 1:
            n = int(args[0])
            shape = (n, n)
        else:
            shape = tuple(int(x) for x in args)
        if any(x < 0 for x in shape):
            raise ValueError("dimensions must be nonnegative")
        return shape

    def reset(self, *args: Any) -> None:
        seed = int(args[0]) if args else self.seed
        self._rng = np.random.default_rng(seed)
        if not args:
            self._rng.bit_generator.state = deepcopy(self._initial_state)

    def save(self) -> dict[str, Any]:
        return deepcopy(self._rng.bit_generator.state)

    def restore(self, state: dict[str, Any]) -> None:
        self._rng.bit_generator.state = deepcopy(state)

    @property
    def state(self) -> dict[str, Any]:
        return self.save()

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self.restore(value)

    def rand(self, *dims: Any) -> np.ndarray:
        return self._rng.random(self._shape(dims))

    def randn(self, *dims: Any) -> np.ndarray:
        return self._rng.standard_normal(self._shape(dims))

    def randi(self, imax: int, *dims: Any) -> np.ndarray:
        imax = int(imax)
        if imax < 1:
            raise ValueError("upper bound must be positive")
        shape = self._shape(dims)
        if imax == 1:
            return np.ones(shape, dtype=np.int64)
        return self._rng.integers(1, imax + 1, size=shape, dtype=np.int64)

    def randperm(self, n: int, k: int | None = None) -> np.ndarray:
        n = int(n)
        k = n if k is None else int(k)
        if n < 0 or k < 0 or k > n:
            raise ValueError("randperm requires 0 <= k <= n")
        return self._rng.permutation(n)[:k].astype(np.int64) + 1

    def binornd(self, n: Any, p: Any, *dims: Any) -> np.ndarray:
        n, p = np.broadcast_arrays(n, p)
        return self._rng.binomial(
            n, p, size=self._shape(dims) if dims else None
        )

    def poissrnd(self, lam: Any, *dims: Any) -> np.ndarray:
        return self._rng.poisson(lam, size=self._shape(dims) if dims else None)

    def mnrnd(self, n: Any, p: Any, m: int | None = None) -> np.ndarray:
        trials = np.asarray(n)
        probabilities = np.asarray(p, dtype=float)
        if trials.ndim > 2 or (trials.ndim == 2 and min(trials.shape) != 1):
            raise ValueError("n must be a scalar or vector")
        if (
            not np.all(np.isfinite(trials))
            or np.any(trials < 0)
            or np.any(trials != np.floor(trials))
        ):
            raise ValueError("n must contain nonnegative integers")
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        elif probabilities.ndim != 2:
            raise ValueError(
                "p must be a vector or matrix of row distributions"
            )
        if probabilities.shape[1] == 1 and probabilities.shape[0] > 1:
            probabilities = probabilities.T
        if m is not None:
            if np.ndim(m) != 0 or not np.isfinite(m) or int(m) != m or m < 0:
                raise ValueError("m must be a nonnegative integer")
            if trials.size != 1 or probabilities.shape[0] != 1:
                raise ValueError(
                    "with m, n must be scalar and p must be a vector"
                )
            trials = np.repeat(trials.reshape(-1), int(m))
            probabilities = np.repeat(probabilities, int(m), axis=0)
        elif trials.size == 1 and probabilities.shape[0] > 1:
            trials = np.repeat(trials.reshape(-1), probabilities.shape[0])
        if probabilities.shape[0] == 1 and trials.size > 1:
            probabilities = np.repeat(probabilities, trials.size, axis=0)
        if probabilities.shape[0] != trials.size:
            raise ValueError("n and p must have matching distribution rows")
        if np.any(probabilities < 0) or not np.isfinite(probabilities).all():
            raise ValueError("probabilities must be finite and nonnegative")
        # MATLAB mnrnd preserves rows whose probabilities do not sum to one
        # as NaN; silently renormalizing changes the specified distribution.
        out = np.full(probabilities.shape, np.nan, dtype=float)
        for i, (count, probs) in enumerate(
            zip(trials.reshape(-1), probabilities)
        ):
            if np.isclose(probs.sum(), 1.0, rtol=1e-7, atol=1e-12):
                out[i] = self._rng.multinomial(int(count), probs / probs.sum())
        return out

    def random(self, name: str, *args: Any) -> np.ndarray:
        """MATLAB Statistics Toolbox distribution facade used by the model."""
        kind = str(name).lower()
        if kind in ("poiss", "poisson"):
            if not args:
                raise TypeError("poisson random requires a mean")
            lam, rest = args[0], args[1:]
            value = self.poissrnd(lam, *rest)
            return (
                np.asarray(value).reshape((1, 1))
                if np.ndim(value) == 0
                else value
            )
        if kind == "binomial":
            if len(args) < 2:
                raise TypeError("binomial random requires n and p")
            value = self.binornd(args[0], args[1], *args[2:])
            return (
                np.asarray(value).reshape((1, 1))
                if np.ndim(value) == 0
                else value
            )
        if kind in ("norm", "normal"):
            if len(args) < 2:
                raise TypeError(
                    "normal random requires mean and standard deviation"
                )
            shape = self._shape(tuple(args[2:])) if len(args) > 2 else (1, 1)
            return self._rng.normal(args[0], args[1], size=shape)
        if kind == "exponential":
            if not args:
                raise TypeError("exponential random requires a mean")
            scale, rest = args[0], args[1:]
            return self._rng.exponential(
                scale, size=self._shape(tuple(rest)) if rest else (1, 1)
            )
        raise NotImplementedError(f"unsupported random distribution {name!r}")

    def randsample(
        self, n: int, k: int, replacement: bool = False, w: Any = None
    ) -> np.ndarray:
        n, k = int(n), int(k)
        if n < 0 or k < 0 or (not replacement and k > n):
            raise ValueError("invalid sample size")
        if w is None:
            return (
                self._rng.choice(n, size=k, replace=replacement).astype(
                    np.int64
                )
                + 1
            )
        # MATLAB arguments have value semantics: the weighted-without-
        # replacement path may zero weights internally, but must not mutate the
        # caller's input array. np.asarray can return the original float array.
        weights = np.asarray(w, dtype=float).reshape(-1).copy()
        if (
            weights.size != n
            or np.any(weights < 0)
            or not np.isfinite(weights).all()
        ):
            raise ValueError(
                "weights must be finite, nonnegative, and length n"
            )
        if not replacement and k > 1:
            if np.all(weights == weights[0]):
                return self.randsample(n, k, False)
            # Preserve MATLAB wrapper behavior: max-float weights are forced in,
            # then remaining draws are weighted with replacement and deduplicated.
            chosen = (
                np.flatnonzero(weights >= np.finfo(float).max).astype(np.int64)
                + 1
            )
            if chosen.size > k:
                return chosen[self._rng.permutation(chosen.size)[:k]]
            if chosen.size == k:
                return chosen
            weights[chosen - 1] = 0
            have = np.zeros(n, dtype=bool)
            have[chosen - 1] = True
            out = list(chosen)
            while len(out) < k and np.any(weights):
                draw = self.randsample(n, k - len(out), True, weights)
                for idx in draw:
                    if not have[idx - 1]:
                        out.append(int(idx))
                        have[idx - 1] = True
                weights[draw - 1] = 0
            return np.asarray(out, dtype=np.int64)
        if k == 1:
            replacement = True
        if weights.sum() == 0:
            return np.empty((0,), dtype=np.int64)
        return (
            self._rng.choice(
                n, size=k, replace=replacement, p=weights / weights.sum()
            ).astype(np.int64)
            + 1
        )

    def randCounts(self, counts: Any, N: int) -> np.ndarray:
        counts = np.asarray(counts)
        if counts.ndim > 2 or (counts.ndim == 2 and min(counts.shape) != 1):
            raise ValueError("counts must be a vector")
        if np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError("counts must be nonnegative integers")
        N = int(N)
        total = int(counts.sum())
        if N < 0 or N > total:
            raise ValueError("N must be between zero and sum(counts)")
        if N == total:
            return counts.copy()
        positive = True
        if N > total / 2:
            positive = False
            N = total - N
        cumulative = np.cumsum(counts, dtype=np.int64)
        selected = np.zeros(counts.shape, dtype=np.int64)
        for _ in range(N):
            idx = int(
                np.searchsorted(
                    cumulative,
                    self.randi(int(cumulative[-1])).item(),
                    side="left",
                )
            )
            selected.flat[idx] += 1
            cumulative[idx:] -= 1
        return selected if positive else counts - selected

    def stochasticRound(self, value: Any) -> np.ndarray:
        value = np.asarray(value)
        with np.errstate(invalid="ignore"):
            up = self._rng.random(value.shape) < np.mod(value, 1)
        rounded = np.where(up, np.ceil(value), np.floor(value))
        # MATLAB keeps Inf/NaN as doubles; casting these to int64 produces
        # spurious negative limits in enzyme-limited reaction allocation.
        if not np.all(np.isfinite(rounded)):
            return rounded
        return rounded.astype(np.int64)

    def randomlySelectNRows(
        self, mat: Any, nRndRows: int
    ) -> tuple[np.ndarray, np.ndarray]:
        mat = np.asarray(mat)
        if mat.ndim == 1:
            mat = mat.reshape((-1, 1))
        k = min(mat.shape[0], int(nRndRows))
        idx = np.sort(self.randsample(mat.shape[0], k, False))
        return mat[idx - 1, :], idx

    def randomlySelectRows(
        self, mat: Any, prob: float
    ) -> tuple[np.ndarray, np.ndarray]:
        mat = np.asarray(mat)
        n = int(self.stochasticRound(np.asarray(prob * mat.shape[0])).item())
        return self.randomlySelectNRows(mat, n)
