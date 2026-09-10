"""Exact propagation of sampled initial conditions: velocity, acceleration, position and random force.

    v(t)  = sum_k [ alpha_k cos(nu_k t) + beta_k sin(nu_k t) ]
    a(t)  = sum_k nu_k [ -alpha_k sin(nu_k t) + beta_k cos(nu_k t) ]
    q(t)  = q(0) + sum_{nu_k > 0} [ alpha_k sin(nu_k t) + beta_k (1 - cos(nu_k t)) ] / nu_k  +  alpha_0 t
    F(t)  = sum_j sqrt(k_j) omega_j [ y_j(0) cos(omega_j t) + p_j(0)/omega_j sin(omega_j t) ],   y_j = x_j - g_j q

The random force ``F`` is the one in the exact generalized Langevin equation

    M dv/dt = -M Omega^2 q - int_0^t K_N(t - s) v(s) ds + F(t),

and satisfies ``<F(t) F(0)> = kT K_N(t)`` and ``<F(t) v(0)> = 0`` over the canonical ensemble.
It needs the bath's own initial coordinates, hence a :class:`~kac_zwanzig.sample.ModeState` and the
eigenvector matrix (``(N+1)^2`` memory).  Two matrix products per time chunk; any output
grid; no time-step error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .sample import ModeState
from .system import System

_T_CHUNK = 1024


@dataclass
class Trajectory:
    """A batch of exact trajectories on a common time grid.

    Attributes
    ----------
    t : ndarray, shape (n_t,)
    v, a : ndarray, shape (n_traj, n_t)
        Particle velocity and acceleration ``dv/dt``.
    q : ndarray or None, shape (n_traj, n_t)
        Particle position; ``q(0) = 0`` for the free particle, the sampled ``q(0)`` for a
        trapped particle with a random force; ``None`` if not requested.
    alpha, beta : ndarray, shape (N+1, n_traj)
        Mode amplitudes; re-evaluate on another grid with :func:`velocity` etc.
    F : ndarray or None, shape (n_traj, n_t)
        Random force of the GLE, if requested.
    state : ModeState or None
        Full mode state, if the random force was requested.
    """
    t: np.ndarray
    v: np.ndarray
    a: np.ndarray
    q: Optional[np.ndarray]
    alpha: np.ndarray
    beta: np.ndarray
    F: Optional[np.ndarray] = None
    state: Optional[ModeState] = None

    @property
    def n_traj(self) -> int:
        return self.v.shape[0]


def velocity(system: System, t, alpha: np.ndarray, beta: np.ndarray, chunk_t: int = _T_CHUNK) -> np.ndarray:
    """Particle velocity for every trajectory.

    Parameters
    ----------
    system : System
    t : array_like, shape (n_t,)
    alpha, beta : ndarray, shape (N+1, n_traj)
        Mode amplitudes from :func:`kac_zwanzig.sample_modes` or a :class:`~kac_zwanzig.sample.ModeState`.

    Returns
    -------
    ndarray, shape (n_traj, n_t)
    """
    t = np.asarray(t, dtype=float).ravel()
    nu = system.nu
    out = np.empty((alpha.shape[1], t.size))
    for s in range(0, t.size, chunk_t):
        ts = t[s:s + chunk_t]
        arg = np.multiply.outer(ts, nu)
        out[:, s:s + chunk_t] = (np.cos(arg) @ alpha + np.sin(arg) @ beta).T
    return out


def acceleration(system: System, t, alpha: np.ndarray, beta: np.ndarray, chunk_t: int = _T_CHUNK) -> np.ndarray:
    """Particle acceleration ``dv/dt`` for every trajectory; same signature and shape as :func:`velocity`."""
    t = np.asarray(t, dtype=float).ravel()
    nu = system.nu
    na = nu[:, None] * alpha
    nb = nu[:, None] * beta
    out = np.empty((alpha.shape[1], t.size))
    for s in range(0, t.size, chunk_t):
        ts = t[s:s + chunk_t]
        arg = np.multiply.outer(ts, nu)
        out[:, s:s + chunk_t] = (np.cos(arg) @ nb - np.sin(arg) @ na).T
    return out


def position(system: System, t, alpha: np.ndarray, beta: np.ndarray, q0=0.0,
             chunk_t: int = _T_CHUNK) -> np.ndarray:
    """Particle position ``q(t) = q0 + int_0^t v``, for every trajectory.

    ``q0`` may be a scalar or an array of shape ``(n_traj,)``.  The zero mode (free
    particle) contributes a uniform drift ``alpha_0 t``.  Same shape as :func:`velocity`.
    """
    t = np.asarray(t, dtype=float).ravel()
    nu = system.nu
    pos = nu > 0
    inv = 1.0 / nu[pos]
    ap = alpha[pos] * inv[:, None]
    bp = beta[pos] * inv[:, None]
    drift = alpha[~pos].sum(axis=0)                     # (n_traj,)
    out = np.empty((alpha.shape[1], t.size))
    for s in range(0, t.size, chunk_t):
        ts = t[s:s + chunk_t]
        arg = np.multiply.outer(ts, nu[pos])
        out[:, s:s + chunk_t] = (np.sin(arg) @ ap + (1.0 - np.cos(arg)) @ bp).T + drift[:, None] * ts[None, :]
    return out + np.asarray(q0, dtype=float).reshape(-1, 1) if np.ndim(q0) else out + float(q0)


def initial_position(system: System, state: ModeState) -> np.ndarray:
    """Sampled ``q(0)`` for each trajectory of a mode state, shape ``(n_traj,)``."""
    U = system.eigenvectors()
    return (U[0] @ state.Q) / np.sqrt(system.M)


def random_force(system: System, t, state: ModeState, chunk_t: int = _T_CHUNK) -> np.ndarray:
    """Exact random force ``F(t)`` of the GLE for every trajectory of a mode state.

    ``M dv/dt = -M Omega^2 q - int_0^t K_N(t - s) v(s) ds + F(t)``, with
    ``F(t) = sum_j sqrt(k_j) omega_j [y_j(0) cos(omega_j t) + p_j(0)/omega_j sin(omega_j t)]`` and
    ``y_j = x_j - g_j q`` the bath coordinates relative to the particle.  Force units
    (divide by ``M`` for the per-unit-mass form that goes with :meth:`System.memory_kernel`).

    Parameters
    ----------
    system : System
    t : array_like, shape (n_t,)
    state : ModeState
        From :func:`kac_zwanzig.sample_state` (the bath coordinates are reconstructed from it with
        the eigenvector matrix, ``(N+1)^2`` doubles, computed once and cached).

    Returns
    -------
    ndarray, shape (n_traj, n_t)
    """
    t = np.asarray(t, dtype=float).ravel()
    U = system.eigenvectors()
    b = system.bath
    x = U[1:] @ state.Q                                  # bath coordinates (masses 1), (N, n_traj)
    pb = U[1:] @ state.P                                 # bath momenta
    q0 = (U[0] @ state.Q) / np.sqrt(system.M)
    g = np.sqrt(b.k) / b.omega
    y = x - g[:, None] * q0[None, :]
    A = (np.sqrt(b.k) * b.omega)[:, None] * y            # cos coefficients
    Bs = np.sqrt(b.k)[:, None] * pb                      # sin coefficients
    out = np.empty((state.n_traj, t.size))
    for s in range(0, t.size, chunk_t):
        ts = t[s:s + chunk_t]
        arg = np.multiply.outer(ts, b.omega)
        out[:, s:s + chunk_t] = (np.cos(arg) @ A + np.sin(arg) @ Bs).T
    return out


def gle_residual(system: System, traj: Trajectory) -> np.ndarray:
    """``M dv/dt + M Omega^2 q + int_0^t K_N(t - s) v(s) ds - F(t)`` on the trajectory's uniform grid.

    Zero up to the trapezoid-rule error of the convolution (``O(dt^2)``).  A direct check
    that the velocity, acceleration, position, kernel and random force of a trajectory
    are mutually consistent.  Requires ``traj.F`` (and ``traj.q`` if ``Omega > 0``).
    """
    if traj.F is None:
        raise ValueError("trajectory has no random force; use trajectory(..., random_force=True)")
    t = traj.t
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt, rtol=1e-8, atol=1e-12):
        raise ValueError("gle_residual needs a uniform time grid")
    K = system.bath.kernel(t)
    conv = np.empty_like(traj.v)
    for n in range(traj.n_traj):
        full = np.convolve(traj.v[n], K)[: t.size] * dt
        conv[n] = full - 0.5 * dt * (K * traj.v[n, 0] + K[0] * traj.v[n])
    res = system.M * traj.a + conv - traj.F
    if system.Omega > 0:
        if traj.q is None:
            raise ValueError("trapped particle: positions are needed for the GLE residual")
        res = res + system.M * system.Omega ** 2 * traj.q
    return res
