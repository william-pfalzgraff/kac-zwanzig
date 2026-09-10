"""Estimators of the velocity autocorrelation function and its derivative from sampled trajectories.

Ensemble average over ``n_traj`` independent canonical initial conditions::

    R(t)  = (1/n) sum_n v_n(t) v_n(0)                 raw VACF
    C(t)  = R(t)/R(0)            ("sample" normalization, default; C(0) = 1 exactly)
    C(t)  = R(t)/(kT/M)          ("exact" normalization)

Derivative estimators (all exact per trajectory, no finite differences)::

    acc:  (1/n) sum_n dv_n/dt(t) v_n(0)               <v'(t) v(0)>
    sym:  average of acc and  -(1/n) sum_n v_n(t) dv_n/dt(0)   (stationarity partner)

Collapsed form.  Substituting ``v_n(t) = sum_k [alpha_kn cos + beta_kn sin]`` gives
``R(t) = sum_k [abar_k cos(nu_k t) + bbar_k sin(nu_k t)]`` with
``abar_k = (1/n) sum_n alpha_kn v_n(0)``, so the ``n_t x n_traj`` trajectory array is
never formed; cost ``O(N (n_traj + n_t))``.

Groups ("blocks").  The trajectories are split into ``n_blocks`` independent groups and a
complete raw estimate is kept per group.  The group spread gives standard errors
(:meth:`VACFEstimate.stderr`), and resampling groups with replacement gives a bootstrap
(:meth:`VACFEstimate.bootstrap`) whose replicate curves keep the time correlation of the
noise and handle the ratio normalization exactly.  This is *not* the block averaging used
on single long MD time series: the groups here are independent by construction.

Noise statistics (Gaussian process, first-order propagation through the ratio)::

    exact  normalization:  Var[C(t)] = (1 + C^2)/n,   Cov = (C(t-t') + C(t)C(t'))/n
    sample normalization:  Var[C(t)] = (1 - C^2)/n,   Cov = (C(t-t') - C(t)C(t'))/n
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Optional

import numpy as np

from .bath import cos_sum, sin_sum
from .propagate import acceleration, velocity
from .sample import default_chunk, iter_sample_modes
from .system import System


def _normalize(R, Rdot_acc, Rdot_sym, R0, normalization, kT_over_M):
    if normalization == "sample":
        s = R0
    elif normalization == "exact":
        s = kT_over_M
    else:
        raise ValueError("normalization must be 'sample' or 'exact'")
    return R / s, Rdot_acc / s, Rdot_sym / s


@dataclass
class BootstrapResult:
    """Replicate curves from resampling groups of trajectories with replacement.

    Attributes
    ----------
    C, Cdot_acc, Cdot_sym : ndarray, shape (n_boot, n_t)
        One normalized curve per replicate, in the estimate's normalization.
    t : ndarray
    """
    t: np.ndarray
    C: np.ndarray
    Cdot_acc: np.ndarray
    Cdot_sym: np.ndarray

    def _pick(self, which):
        return {"C": self.C, "acc": self.Cdot_acc, "sym": self.Cdot_sym}[which]

    def std(self, which: str = "C") -> np.ndarray:
        """Bootstrap standard deviation at each time (a standard error for the pooled estimate)."""
        return self._pick(which).std(axis=0, ddof=1)

    def band(self, which: str = "C", level: float = 0.95):
        """Pointwise percentile band ``(lower, upper)`` at the given confidence level."""
        a = 100 * (1 - level) / 2
        x = self._pick(which)
        return np.percentile(x, a, axis=0), np.percentile(x, 100 - a, axis=0)

    def covariance(self, which: str = "C") -> np.ndarray:
        """Bootstrap covariance matrix across times, shape ``(n_t, n_t)``."""
        return np.cov(self._pick(which).T)


@dataclass
class VACFEstimate:
    """A sampled VACF with derivative estimates, per-group raw sums, standard errors and a bootstrap.

    Attributes
    ----------
    t : ndarray
    C, Cdot_acc, Cdot_sym : ndarray
        Pooled normalized estimates: the VACF, the ``<v'(t) v(0)>`` derivative estimator and
        its symmetrized version.
    n_traj : int
    normalization : {"sample", "exact"}
    R0 : float
        Sampled ``<v(0)^2>``.
    block_sizes : ndarray, shape (n_blocks,)
        Trajectories in each group.
    blocks_R, blocks_Racc, blocks_Rsym : ndarray, shape (n_blocks, n_t)
        Raw (unnormalized) per-group estimates.
    blocks_R0 : ndarray, shape (n_blocks,)
        Raw per-group ``<v(0)^2>``.
    kT_over_M : float
    """
    t: np.ndarray
    C: np.ndarray
    Cdot_acc: np.ndarray
    Cdot_sym: np.ndarray
    n_traj: int
    normalization: str
    R0: float
    block_sizes: np.ndarray
    blocks_R: np.ndarray
    blocks_Racc: np.ndarray
    blocks_Rsym: np.ndarray
    blocks_R0: np.ndarray
    kT_over_M: float

    @property
    def n_blocks(self) -> int:
        return self.block_sizes.size

    def _normalized_blocks(self):
        out = [_normalize(self.blocks_R[b], self.blocks_Racc[b], self.blocks_Rsym[b], self.blocks_R0[b],
                          self.normalization, self.kT_over_M) for b in range(self.n_blocks)]
        return (np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out]))

    @property
    def blocks_C(self) -> np.ndarray:
        """Per-group normalized VACF estimates, shape ``(n_blocks, n_t)``."""
        return self._normalized_blocks()[0]

    @property
    def blocks_Cdot_acc(self) -> np.ndarray:
        return self._normalized_blocks()[1]

    @property
    def blocks_Cdot_sym(self) -> np.ndarray:
        return self._normalized_blocks()[2]

    def stderr(self, which: str = "C") -> np.ndarray:
        """Standard error of the pooled estimate from the spread of the group estimates."""
        b = {"C": 0, "acc": 1, "sym": 2}[which]
        return self._normalized_blocks()[b].std(axis=0, ddof=1) / np.sqrt(self.n_blocks)

    def bootstrap(self, n_boot: int = 1000, seed: Optional[int] = None,
                  rng: Optional[np.random.Generator] = None) -> BootstrapResult:
        """Resample the groups of trajectories with replacement and re-form the pooled estimates.

        Each replicate draws ``n_blocks`` groups with replacement, pools their raw sums
        (weighted by group size) and normalizes, so the ratio ``R(t)/R(0)`` is handled
        exactly and the replicate curves keep the time correlation of the noise.  Use the
        replicates to put error bars on anything computed *from* ``C`` and ``dC/dt``, e.g.
        a memory kernel recovered by a solver.

        Parameters
        ----------
        n_boot : int
            Number of replicates.
        seed, rng : optional
            Random state.

        Returns
        -------
        BootstrapResult
            ``.C``, ``.Cdot_acc``, ``.Cdot_sym`` of shape ``(n_boot, n_t)``; ``.std()``, ``.band()``.

        Notes
        -----
        With ``B`` groups the bootstrap distribution is built from ``B`` distinct curves, so
        use ``n_blocks >= 50`` (the default) when you intend to bootstrap.
        """
        if self.n_blocks < 2:
            raise ValueError("bootstrap needs at least two groups of trajectories")
        rng = np.random.default_rng(seed) if rng is None else rng
        B = self.n_blocks
        counts = rng.multinomial(B, np.full(B, 1.0 / B), size=n_boot).astype(float)   # (n_boot, B)
        W = counts * self.block_sizes[None, :]
        tot = W.sum(axis=1, keepdims=True)
        R = (W @ self.blocks_R) / tot
        Racc = (W @ self.blocks_Racc) / tot
        Rsym = (W @ self.blocks_Rsym) / tot
        R0 = (W @ self.blocks_R0) / tot[:, 0]
        s = R0[:, None] if self.normalization == "sample" else self.kT_over_M
        return BootstrapResult(self.t, R / s, Racc / s, Rsym / s)


def vacf_ensemble(system: System, t, n_traj: int, rng: np.random.Generator,
                  normalization: str = "sample", n_blocks: int = 50,
                  chunk: Optional[int] = None, dtype=np.float64) -> VACFEstimate:
    """Ensemble VACF and derivative estimates from ``n_traj`` fresh canonical samples (collapsed form).

    Parameters
    ----------
    system : System
    t : array_like
        Output times.
    n_traj : int
        Number of independent trajectories.
    rng : numpy.random.Generator
    normalization : {"sample", "exact"}
    n_blocks : int
        Number of independent groups the trajectories are split into (for standard errors
        and the bootstrap).  Cheap: each group adds one evaluation of a cosine series.
    chunk : int, optional
        Trajectories per memory chunk.

    Returns
    -------
    VACFEstimate
    """
    t = np.asarray(t, dtype=float).ravel()
    nu = system.nu
    n_modes = nu.size
    if chunk is None:
        chunk = default_chunk(system)
    n_blocks = max(1, min(n_blocks, n_traj))
    sizes = np.full(n_blocks, n_traj // n_blocks)
    sizes[: n_traj - sizes.sum()] += 1

    abar = np.zeros((n_blocks, n_modes))
    bbar = np.zeros((n_blocks, n_modes))
    gbar = np.zeros((n_blocks, n_modes))
    dbar = np.zeros((n_blocks, n_modes))
    for b, nb in enumerate(sizes):
        for alpha, beta in iter_sample_modes(system, int(nb), rng, chunk, dtype):
            v0 = alpha.sum(axis=0)                     # v_n(0)
            vd0 = (nu[:, None] * beta).sum(axis=0)      # dv_n/dt(0)
            abar[b] += alpha @ v0
            bbar[b] += beta @ v0
            gbar[b] += alpha @ vd0
            dbar[b] += beta @ vd0
        abar[b] /= nb
        bbar[b] /= nb
        gbar[b] /= nb
        dbar[b] /= nb

    def evaluate(ab, bb, gb, db):
        R = cos_sum(ab, nu, t) + sin_sum(bb, nu, t)
        Racc = -sin_sum(nu * ab, nu, t) + cos_sum(nu * bb, nu, t)
        Rsym = 0.5 * (Racc - (cos_sum(gb, nu, t) + sin_sum(db, nu, t)))
        return R, Racc, Rsym, float(ab.sum())

    kT_over_M = system.kT / system.M
    raw = [evaluate(abar[b], bbar[b], gbar[b], dbar[b]) for b in range(n_blocks)]
    blocks_R = np.array([r[0] for r in raw])
    blocks_Racc = np.array([r[1] for r in raw])
    blocks_Rsym = np.array([r[2] for r in raw])
    blocks_R0 = np.array([r[3] for r in raw])
    w = sizes / sizes.sum()
    R, Racc, Rsym, R0 = w @ blocks_R, w @ blocks_Racc, w @ blocks_Rsym, float(w @ blocks_R0)
    C, Cacc, Csym = _normalize(R, Racc, Rsym, R0, normalization, kT_over_M)
    return VACFEstimate(t, C, Cacc, Csym, n_traj, normalization, R0, sizes.astype(float),
                        blocks_R, blocks_Racc, blocks_Rsym, blocks_R0, kT_over_M)


def vacf_direct(system: System, t, alpha: np.ndarray, beta: np.ndarray, normalization: str = "sample"):
    """Reference estimator that forms the trajectories explicitly.  Returns ``(C, Cdot_acc, Cdot_sym)``."""
    t = np.asarray(t, dtype=float).ravel()
    v = velocity(system, t, alpha, beta)
    a = acceleration(system, t, alpha, beta)
    v0 = alpha.sum(axis=0)
    vd0 = (system.nu[:, None] * beta).sum(axis=0)
    R = (v * v0[:, None]).mean(axis=0)
    Racc = (a * v0[:, None]).mean(axis=0)
    Rsym = 0.5 * (Racc - (v * vd0[:, None]).mean(axis=0))
    R0 = float((v0 ** 2).mean())
    return _normalize(R, Racc, Rsym, R0, normalization, system.kT / system.M)


# --- time-origin averaging (demonstration only: the finite bath is integrable) --------

def vacf_time_average(v: np.ndarray, n_lags: int) -> np.ndarray:
    """Raw time-origin-averaged autocorrelation of ONE uniformly sampled trajectory.

    ``R[m] = mean_i v[i+m] v[i]`` for ``m < n_lags``, by FFT.  Normalize with ``R[0]`` or
    ``kT/M`` as for the ensemble estimator.  For the Kac-Zwanzig bath this estimator does
    *not* converge to the canonical VACF as the trajectory grows: each mode's energy is
    conserved, so ``R -> (kT/M) sum_k a_k^2 E_k cos(nu_k t)`` with ``E_k`` fixed
    exponential variates, leaving a floor ``Var = (kT/M)^2 sum_k a_k^4 cos^2(nu_k t)``.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(v, nfft)
    R = np.fft.irfft(F * np.conj(F), nfft)[:n_lags]
    return R / (n - np.arange(n_lags))


def time_average_floor(system: System, t) -> np.ndarray:
    """Predicted standard deviation of the infinitely long time average (exact normalization)."""
    nu, a2 = system.spectral_measure()
    return np.sqrt(cos_sum(a2 ** 2, 2 * nu, t) / 2 + (a2 ** 2).sum() / 2)   # sum a^4 cos^2 = sum a^4 (1 + cos 2 nu t)/2


# --- noise statistics --------------------------------------------------------------

def predicted_variance(C_exact: np.ndarray, n_traj: int, normalization: str = "sample") -> np.ndarray:
    """Variance of the normalized VACF estimator at each time, from the exact normalized ``C``."""
    C = np.asarray(C_exact, dtype=float)
    if normalization == "sample":
        return np.maximum(1.0 - C ** 2, 0.0) / n_traj      # clip: C(0) = 1 + round-off would give a tiny negative
    if normalization == "exact":
        return (1.0 + C ** 2) / n_traj
    raise ValueError("normalization must be 'sample' or 'exact'")


def predicted_covariance(C_exact: np.ndarray, n_traj: int, normalization: str = "sample") -> np.ndarray:
    """Covariance matrix of the estimator on a *uniform* time grid starting at ``t = 0``."""
    C = np.asarray(C_exact, dtype=float)
    n = C.size
    idx = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    lag = C[idx]
    outer = np.outer(C, C)
    if normalization == "sample":
        return (lag - outer) / n_traj
    if normalization == "exact":
        return (lag + outer) / n_traj
    raise ValueError("normalization must be 'sample' or 'exact'")


def predicted_variance_derivative(Cdot_exact: np.ndarray, K0_over_M: float, n_traj: int) -> np.ndarray:
    """Variance of the ``acc`` derivative estimator with exact normalization: ``(K0/M + Cdot^2)/n``."""
    return (K0_over_M + np.asarray(Cdot_exact, dtype=float) ** 2) / n_traj


# --- finite differences ------------------------------------------------------------

def fd_weights(offsets, deriv: int = 1) -> np.ndarray:
    """Finite-difference weights for ``d^deriv/dt^deriv`` on integer ``offsets`` (unit spacing)."""
    s = np.asarray(offsets, dtype=float)
    n = s.size
    A = np.vander(s, n, increasing=True).T   # A[m, i] = s_i^m
    rhs = np.zeros(n)
    rhs[deriv] = factorial(deriv)
    return np.linalg.solve(A, rhs)


def finite_difference(C, dt: float, order: int = 2, even: bool = True) -> np.ndarray:
    """``dC/dt`` on a uniform grid by central differences of the given (even) order.

    With ``even=True`` the origin is handled by the symmetry ``C(-t) = C(t)``, so the
    same central stencil applies at ``t = 0`` (and gives exactly zero there).  The last
    ``order/2`` points use one-sided stencils of the same order.
    """
    C = np.asarray(C, dtype=float).ravel()
    if order % 2 or order < 2:
        raise ValueError("order must be a positive even integer")
    h = order // 2
    n = C.size
    if n < order + 1:
        raise ValueError("too few points for the requested order")
    w = fd_weights(np.arange(-h, h + 1))
    if even:
        ext = np.concatenate([C[h:0:-1], C])
        left = np.convolve(ext, w[::-1], mode="valid")     # values for indices 0 .. n-1-h
        out = np.empty(n)
        out[: n - h] = left
        out[0] = 0.0                                        # exact by symmetry
    else:
        out = np.empty(n)
        out[h: n - h] = np.convolve(C, w[::-1], mode="valid")
        for i in range(h):
            wi = fd_weights(np.arange(-i, order + 1 - i))
            out[i] = wi @ C[: order + 1]
    for i in range(n - h, n):
        back = n - 1 - i
        wi = fd_weights(np.arange(-(order - back), back + 1))
        out[i] = wi @ C[n - order - 1:]
    return out / dt
