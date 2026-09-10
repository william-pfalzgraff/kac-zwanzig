"""Harmonic baths for the Kac-Zwanzig model: spectral densities, frequency grids, kernels.

Overview
--------
A bath is two arrays: frequencies ``omega_j`` (ascending) and *kernel weights*
``k_j = c_j**2 / (m_j * omega_j**2) >= 0``.  Bath masses are set to one without loss of
generality (they only rescale the bath coordinates).  Everything the particle ever feels
is the memory kernel

    K_N(t) = sum_j k_j cos(omega_j t),                                (exact for the finite bath)

whose continuum counterpart is the cosine transform of the spectral density,

    K_inf(t) = (2/pi) int_0^inf J(w)/w cos(w t) dw,      J(w) = (pi/2) sum_j k_j w_j delta(w - w_j)

(Caldeira-Leggett convention: strictly Ohmic ``J = eta w`` gives friction exactly ``eta``).

Three ways to build a bath:

* presets :func:`ohmic` and :func:`debye` (closed-form grids and continuum kernels),
* any callable ``J(omega)`` through :meth:`Bath.from_spectral_density` (numerical quantiles),
* arbitrary arrays through :class:`Bath` itself.

Grids
-----
``"uniform"``
    midpoints ``w_j = (j - 1/2) dw`` on ``[0, omega_max]``, weights ``k_j = (2/pi) J(w_j)/w_j dw``.
    A midpoint rule for the cosine transform.  By Poisson summation the error is a sum of
    smooth, alternating-sign images of the kernel, ``K_N - K_inf = sum_{m != 0} (-1)^m
    K_inf(t - 2 pi m/dw)``, about ``2 K_inf(2 pi/dw)`` in the window ``t < pi/dw``.
    Second order in ``dw`` when ``J/w`` has a kink at ``w = 0`` (Ohmic), spectral when it is
    analytic (Debye).  Use midpoints: the endpoint grid ``w_j = j dw`` carries a constant
    bias ``dw * J'(0)/pi``.
``"invcdf"``
    quantiles of the density ``rho(w) ~ J(w)/w`` with equal weights ``k_j = K(0)/N``.
    No truncation and ``K(0)`` exact, but the highest few modes have scrambled phases and
    leave a high-frequency ripple in ``K_N`` that decays only like ``N**-1`` (Ohmic) or
    ``N**-3/4`` (Debye).  This is the grid of the "Discrete Oscillators" notebook.

See ``notes/frequency_grids.pdf`` for derivations and measurements.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_T_CHUNK = 512


def cos_sum(weights: np.ndarray, freqs: np.ndarray, t, chunk: int = _T_CHUNK) -> np.ndarray:
    """Evaluate ``sum_j weights_j cos(freqs_j t)`` for every ``t`` (any shape), chunked over ``t``."""
    t = np.asarray(t, dtype=float)
    flat = t.ravel()
    out = np.empty(flat.size)
    for s in range(0, flat.size, chunk):
        out[s:s + chunk] = np.cos(np.multiply.outer(flat[s:s + chunk], freqs)) @ weights
    return out.reshape(t.shape)


def sin_sum(weights: np.ndarray, freqs: np.ndarray, t, chunk: int = _T_CHUNK) -> np.ndarray:
    """Evaluate ``sum_j weights_j sin(freqs_j t)`` for every ``t`` (any shape), chunked over ``t``."""
    t = np.asarray(t, dtype=float)
    flat = t.ravel()
    out = np.empty(flat.size)
    for s in range(0, flat.size, chunk):
        out[s:s + chunk] = np.sin(np.multiply.outer(flat[s:s + chunk], freqs)) @ weights
    return out.reshape(t.shape)


def _uniform_grid(N: int, omega_max: float, midpoint: bool = True):
    dw = omega_max / N
    j = np.arange(1, N + 1, dtype=float)
    omega = (j - 0.5) * dw if midpoint else j * dw
    return omega, dw


@dataclass(frozen=True)
class Bath:
    """A discrete harmonic bath.

    Parameters
    ----------
    omega : array_like, shape (N,)
        Oscillator frequencies, positive and distinct (sorted on construction).
    k : array_like, shape (N,)
        Kernel weights ``k_j = c_j**2/(m_j omega_j**2) >= 0``.  ``K_N(t) = sum_j k_j cos(omega_j t)``.
    name : str, optional
        Label used by :meth:`describe`.
    params : dict, optional
        Parameters the bath was built from (recorded for provenance only).
    kernel_continuum : callable, optional
        ``K_inf(t)`` of the underlying continuum, if known.
    spectral_density_continuum : callable, optional
        ``J(omega)`` of the underlying continuum, if known.

    Attributes
    ----------
    N : int
        Number of oscillators.
    K0 : float
        ``K_N(0) = sum_j k_j``, the total counter-term stiffness.

    Examples
    --------
    >>> import numpy as np
>>> import kac_zwanzig as kz
    >>> b = kz.Bath(omega=[0.5, 1.0, 2.0], k=[0.2, 0.3, 0.1])
    >>> float(b.kernel(0.0)), b.N
    (0.6, 3)
    """

    omega: np.ndarray
    k: np.ndarray
    name: str = "custom"
    params: dict = field(default_factory=dict)
    kernel_continuum: Optional[Callable] = None
    spectral_density_continuum: Optional[Callable] = None

    def __post_init__(self):
        omega = np.array(self.omega, dtype=float, copy=True).ravel()
        k = np.array(self.k, dtype=float, copy=True).ravel()
        if omega.shape != k.shape:
            raise ValueError("omega and k must have the same shape")
        if np.any(omega <= 0):
            raise ValueError("all bath frequencies must be positive")
        if np.any(k < 0):
            raise ValueError("kernel weights must be non-negative")
        order = np.argsort(omega, kind="stable")
        omega, k = omega[order], k[order]
        if np.any(np.diff(omega) <= 0):
            raise ValueError("bath frequencies must be distinct")
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "k", k)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_spectral_density(cls, J: Callable, N: int = 2000, grid: str = "uniform",
                              omega_max: Optional[float] = None, name: str = "custom",
                              n_quad: int = 200_000, params: Optional[dict] = None) -> Bath:
        """Discretize an arbitrary continuum spectral density ``J(omega)``.

        Parameters
        ----------
        J : callable
            Vectorized ``J(omega)`` for ``omega > 0``.  Must vanish at least linearly at
            ``omega -> 0`` so that ``J(omega)/omega`` is integrable there.
        N : int
            Number of oscillators.
        grid : {"uniform", "invcdf"}
            See the module docstring.
        omega_max : float
            Upper limit of the uniform grid, or of the numerical cumulative integral used to
            place the ``"invcdf"`` nodes and to evaluate :meth:`kernel_continuum`.  Choose it
            where ``J(omega)/omega`` is negligible.
        n_quad : int
            Points of the linear quadrature grid on ``(0, omega_max]`` used for the
            cumulative integral and for the numerical continuum kernel.

        Returns
        -------
        Bath
            With :meth:`kernel_continuum` evaluated by quadrature on the same grid (accurate
            while ``omega_max * t / n_quad`` stays small, i.e. for ``t`` up to a few hundred
            in units of ``1/omega_max`` times ``n_quad``).

        Notes
        -----
        The presets :func:`ohmic` and :func:`debye` use closed-form grids and kernels and
        should be preferred when they apply.  This constructor reproduces them to quadrature
        accuracy; it exists so that any ``J`` (super-Ohmic, structured, tabulated) can be
        used with the same downstream machinery.

        Examples
        --------
        >>> import numpy as np
>>> import kac_zwanzig as kz
        >>> J = lambda w: 0.5 * w**3 * np.exp(-w)          # super-Ohmic
        >>> b = kz.Bath.from_spectral_density(J, N=500, omega_max=30)
        >>> round(float(b.K0), 4)                          # (2/pi) * 0.5 * int w^2 e^{-w} dw = 2/pi
        0.6366
        """
        if omega_max is None:
            raise ValueError("omega_max is required (the uniform grid's cutoff, or the range of the numerical CDF)")
        if grid not in ("uniform", "invcdf"):
            raise ValueError(f"unknown grid {grid!r}; use 'uniform' or 'invcdf'")

        def f(w):
            w = np.asarray(w, dtype=float)
            return (2.0 / np.pi) * np.asarray(J(w), dtype=float) / w

        wq = np.linspace(0.0, omega_max, n_quad + 1)[1:]
        fq = f(wq)
        dq = wq[1] - wq[0]
        # cumulative integral from 0, taking f flat on the first sliver (0, wq[0])
        F = np.empty(n_quad + 1)
        F[0] = 0.0
        F[1:] = fq[0] * wq[0] + np.concatenate([[0.0], np.cumsum(0.5 * (fq[1:] + fq[:-1]) * dq)])
        wgrid = np.concatenate([[0.0], wq])
        K0 = float(F[-1])

        if grid == "uniform":
            omega, dw = _uniform_grid(N, omega_max, True)
            k = f(omega) * dw
        else:
            u = (np.arange(1, N + 1, dtype=float) - 0.5) / N
            omega = np.interp(u * K0, F, wgrid)
            k = np.full(N, K0 / N)

        wts = np.full(n_quad, dq)          # trapezoid on [wq[0], omega_max] ...
        wts[0] = 0.5 * dq + wq[0]          # ... plus the first sliver (0, wq[0]) with f taken flat
        wts[-1] = 0.5 * dq
        fw = fq * wts

        def K_inf(t, _chunk=32):
            t = np.asarray(t, dtype=float)
            flat = t.ravel()
            out = np.empty(flat.size)
            for s in range(0, flat.size, _chunk):
                out[s:s + _chunk] = np.cos(np.multiply.outer(flat[s:s + _chunk], wq)) @ fw
            return out.reshape(t.shape)

        p = dict(params or {}, N=N, grid=grid, omega_max=omega_max, n_quad=n_quad)
        return cls(omega, k, name, p, K_inf, lambda w: np.asarray(J(w), dtype=float))

    # ------------------------------------------------------------------ basic quantities
    @property
    def N(self) -> int:
        return self.omega.size

    @property
    def K0(self) -> float:
        """``K_N(0) = sum_j k_j``: total counter-term stiffness; ``2 x`` the reorganization energy."""
        return float(self.k.sum())

    @property
    def omega_max(self) -> float:
        return float(self.omega[-1])

    def kernel(self, t) -> np.ndarray:
        """Exact finite-bath memory kernel ``K_N(t) = sum_j k_j cos(omega_j t)``.

        Parameters
        ----------
        t : array_like
            Times, any shape.

        Returns
        -------
        ndarray
            Same shape as ``t``.  This is the exact friction kernel of the finite bath, the
            one multiplying ``v`` in the equation of motion (units of mass/time^2); the
            memory kernel per unit mass is ``System.memory_kernel(t) = (K_N(t) + M Omega^2)/M``.
        """
        return cos_sum(self.k, self.omega, t)

    def kernel_dot(self, t) -> np.ndarray:
        """``dK_N/dt = -sum_j k_j omega_j sin(omega_j t)``."""
        return -sin_sum(self.k * self.omega, self.omega, t)

    def kernel_integral(self, t) -> np.ndarray:
        """``int_0^t K_N(s) ds = sum_j (k_j/omega_j) sin(omega_j t)`` (the running friction)."""
        return sin_sum(self.k / self.omega, self.omega, t)

    def J(self, omega) -> np.ndarray:
        """Continuum spectral density ``J(omega)`` the bath was built from (raises if unknown)."""
        if self.spectral_density_continuum is None:
            raise ValueError("this bath has no continuum spectral density attached")
        return self.spectral_density_continuum(np.asarray(omega, dtype=float))

    def spectral_density_histogram(self, bins=60) -> tuple[np.ndarray, np.ndarray]:
        """Histogram estimate of ``J(omega)`` from the discrete bath, for plotting.

        The discrete ``J`` is a sum of deltas, ``(pi/2) sum_j k_j omega_j delta(omega - omega_j)``;
        binning that weight and dividing by the bin width gives a density comparable to the
        continuum ``J``.

        Parameters
        ----------
        bins : int or array_like
            Number of equal bins on ``[0, omega_max]`` or explicit bin edges.

        Returns
        -------
        centers, values : ndarray
            Bin centers and the binned spectral density.
        """
        edges = np.linspace(0.0, self.omega_max, int(bins) + 1) if np.isscalar(bins) else np.asarray(bins, dtype=float)
        hist, _ = np.histogram(self.omega, bins=edges, weights=0.5 * np.pi * self.k * self.omega)
        return 0.5 * (edges[1:] + edges[:-1]), hist / np.diff(edges)

    def describe(self) -> str:
        p = ", ".join(f"{a}={b}" for a, b in self.params.items())
        return (f"{self.name} bath, N={self.N}, K0={self.K0:.6g}, "
                f"omega in [{self.omega[0]:.3g}, {self.omega_max:.3g}]  ({p})")


# ---------------------------------------------------------------------- shared helper

def bath_from_spectral_density(J: Callable, K0: float, quantile: Callable, N: int, grid: str,
                               omega_max: Optional[float], midpoint: bool = True):
    """Build ``(omega, k)`` for a continuum ``J`` with a *known* closed-form quantile function.

    Used by the presets.  ``quantile(u)`` returns the frequency below which the fraction
    ``u`` of the density ``rho(w) ~ J(w)/w`` lies (either tail convention works, the
    midpoints are symmetric).  For arbitrary ``J`` use :meth:`Bath.from_spectral_density`.
    """
    if grid == "uniform":
        if omega_max is None:
            raise ValueError("omega_max is required for the uniform grid")
        omega, dw = _uniform_grid(N, omega_max, midpoint)
        k = (2.0 / np.pi) * J(omega) / omega * dw
    elif grid == "invcdf":
        u = (np.arange(1, N + 1, dtype=float) - 0.5) / N
        omega = np.sort(quantile(u))
        k = np.full(N, K0 / N)
    else:
        raise ValueError(f"unknown grid {grid!r}; use 'uniform' or 'invcdf'")
    return omega, k


# ---------------------------------------------------------------------- presets

def ohmic(eta: float = 1.0, omega_c: float = 1.0, N: int = 2000, grid: str = "uniform",
          omega_max: Optional[float] = None, midpoint: bool = True) -> Bath:
    """Ohmic bath with exponential cutoff, ``J(omega) = eta * omega * exp(-omega/omega_c)``.

    Parameters
    ----------
    eta : float
        Friction coefficient: ``int_0^inf K_inf dt = eta``.
    omega_c : float
        Cutoff frequency; the kernel decays on the time scale ``1/omega_c``.
    N : int
        Number of oscillators.
    grid : {"uniform", "invcdf"}
        ``"uniform"`` (recommended) or the notebook's inverse-CDF grid
        ``omega_j = -omega_c ln((j - 1/2)/N)`` with equal weights.
    omega_max : float, optional
        Cutoff of the uniform grid; default ``25 omega_c`` (``e^-25 ~ 1e-11``).
    midpoint : bool
        Midpoint (default) or endpoint placement on the uniform grid.  Keep ``True``.

    Returns
    -------
    Bath
        With ``kernel_continuum(t) = (2 eta omega_c/pi) / (1 + (omega_c t)**2)`` and
        ``K0 = 2 eta omega_c/pi`` (up to the midpoint-rule error ``~dw**2/24``).

    Notes
    -----
    For the free particle the normalized VACF depends on ``eta``, ``M`` and ``omega_c`` only
    through ``eta/(M omega_c)``; ``omega_c`` otherwise just sets the unit of time.  In units
    ``M = omega_c = 1``: ``eta = 1`` gives a single shallow negative dip (MD-like),
    ``eta = 5`` a clearly caged oscillation, ``eta = 20`` many oscillations.

    Examples
    --------
    >>> import kac_zwanzig as kz
    >>> b = kz.ohmic(eta=1.0, N=2000)
    >>> round(b.K0, 4), b.N, round(b.omega_max, 3)
    (0.6366, 2000, 24.994)
    """
    if omega_max is None and grid == "uniform":
        omega_max = 25.0 * omega_c
    K0 = 2.0 * eta * omega_c / np.pi

    def J(w):
        w = np.asarray(w, dtype=float)
        return eta * w * np.exp(-w / omega_c)

    def quantile(u):
        return -omega_c * np.log(u)

    def K_inf(t):
        t = np.asarray(t, dtype=float)
        return K0 / (1.0 + (omega_c * t) ** 2)

    omega, k = bath_from_spectral_density(J, K0, quantile, N, grid, omega_max, midpoint)
    params = dict(eta=eta, omega_c=omega_c, N=N, grid=grid, omega_max=omega_max, midpoint=midpoint)
    return Bath(omega, k, "ohmic", params, K_inf, J)


def debye(lmbda: float = 1.0, omega_c: float = 1.0, N: int = 2000, grid: str = "invcdf",
          omega_max: Optional[float] = None, midpoint: bool = True) -> Bath:
    """Debye bath, ``J(omega) = 2 lmbda omega_c omega / (omega_c**2 + omega**2)``.

    Parameters
    ----------
    lmbda : float
        Reorganization energy ``(1/pi) int J/omega domega``; ``K0 = 2 lmbda``.
    omega_c : float
        Kernel decay rate: ``K_inf(t) = 2 lmbda exp(-omega_c t)``.
    N : int
        Number of oscillators.
    grid : {"invcdf", "uniform"}
        ``"invcdf"`` (default) is the notebook's tan grid ``omega_c tan(pi (j-1/2)/(2N))``:
        no truncation and ``K0`` exact, but a kernel ripple of a few ``1e-3`` at ``N = 4000``.
        ``"uniform"`` with a large ``omega_max`` is much smoother for ``t > 1/omega_max`` but
        misses ``(2/pi) K0 arctan(omega_c/omega_max)`` of ``K0`` at the cusp.
    omega_max : float, optional
        Cutoff of the uniform grid; default ``100 omega_c``.

    Returns
    -------
    Bath

    Notes
    -----
    The continuum VACF of a free particle in this bath is closed-form, see
    :func:`debye_vacf_continuum`; the dimensionless control parameter is ``lmbda/(M omega_c**2)``
    with critical damping at ``1/8``.
    """
    if omega_max is None and grid == "uniform":
        omega_max = 100.0 * omega_c
    K0 = 2.0 * lmbda

    def J(w):
        w = np.asarray(w, dtype=float)
        return 2.0 * lmbda * omega_c * w / (omega_c ** 2 + w ** 2)

    def quantile(u):
        return omega_c * np.tan(0.5 * np.pi * u)

    def K_inf(t):
        t = np.asarray(t, dtype=float)
        return K0 * np.exp(-omega_c * t)

    omega, k = bath_from_spectral_density(J, K0, quantile, N, grid, omega_max, midpoint)
    params = dict(lmbda=lmbda, omega_c=omega_c, N=N, grid=grid, omega_max=omega_max, midpoint=midpoint)
    return Bath(omega, k, "debye", params, K_inf, J)


def debye_vacf_continuum(t, lmbda: float = 1.0, omega_c: float = 1.0, M: float = 1.0) -> np.ndarray:
    """Closed-form continuum velocity autocorrelation of a free particle in a Debye bath.

    From ``C~(s) = 1/(s + K~(s)/M)`` with ``K~(s) = 2 lmbda/(s + omega_c)``:
    ``C~(s) = (s + omega_c)/(s**2 + omega_c s + 2 lmbda/M)``, two exponentials with
    ``s_pm = (-omega_c pm sqrt(omega_c**2 - 8 lmbda/M))/2`` (complex when underdamped).

    Parameters
    ----------
    t : array_like
    lmbda, omega_c, M : float

    Returns
    -------
    ndarray
        Normalized ``C(t)``, ``C(0) = 1``.
    """
    t = np.asarray(t, dtype=float)
    disc = omega_c ** 2 - 8.0 * lmbda / M
    if abs(disc) < 1e-14 * omega_c ** 2:  # critically damped double root
        return np.exp(-0.5 * omega_c * t) * (1.0 + 0.5 * omega_c * t)
    r = np.sqrt(disc + 0j)
    sp, sm = (-omega_c + r) / 2, (-omega_c - r) / 2
    C = ((sp + omega_c) * np.exp(sp * t) - (sm + omega_c) * np.exp(sm * t)) / (sp - sm)
    return C.real
