"""Canonical initial conditions.

In the canonical ensemble the normal modes are independent Gaussians,
``P_k ~ N(0, kT)`` and ``Q_k ~ N(0, kT/nu_k^2)``, so the particle-velocity amplitudes
``alpha_k = a_k P_k/sqrt(M)`` and ``beta_k = -a_k nu_k Q_k/sqrt(M)`` are independent
``N(0, a_k^2 kT/M)``.  No equilibration, no thermostat.

Two samplers:

* :func:`sample_modes` draws ``alpha, beta`` directly.  All that velocity, acceleration,
  position and the correlation-function estimators need.
* :func:`sample_state` draws the full mode state ``(P, Q)`` and derives ``alpha, beta``
  from it.  Needed when the bath's own coordinates are required, i.e. for the random
  force (:func:`kac_zwanzig.random_force`).

The free particle's zero mode is a rigid translation; its ``Q_0`` is set to zero (nothing
observable depends on it) and its ``P_0`` is drawn like any other momentum.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .system import System


@dataclass
class ModeState:
    """Canonical initial conditions in (mass-weighted) normal-mode coordinates.

    Attributes
    ----------
    P, Q : ndarray, shape (N+1, n_traj)
        Mode momenta and coordinates at ``t = 0``; ``Q[0] = 0`` for the free particle.
    alpha, beta : ndarray, shape (N+1, n_traj)
        The particle-velocity amplitudes derived from them.
    """
    P: np.ndarray
    Q: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray

    @property
    def n_traj(self) -> int:
        return self.P.shape[1]


def sample_modes(system: System, n_traj: int, rng: np.random.Generator,
                 dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_traj`` canonical initial conditions as amplitudes.

    Returns ``alpha, beta`` of shape ``(N+1, n_traj)``.
    """
    sigma = system.amplitude_sigma().astype(dtype)[:, None]
    alpha = sigma * rng.standard_normal((sigma.size, n_traj), dtype=dtype)
    beta = sigma * rng.standard_normal((sigma.size, n_traj), dtype=dtype)
    beta[system.nu == 0.0] = 0.0
    return alpha, beta


def sample_state(system: System, n_traj: int, rng: np.random.Generator, dtype=np.float64) -> ModeState:
    """Draw ``n_traj`` canonical initial conditions as a full mode state ``(P, Q)`` plus ``alpha, beta``."""
    nu = system.nu
    a = system.a
    sk = np.sqrt(system.kT)
    P = sk * rng.standard_normal((nu.size, n_traj), dtype=dtype)
    Q = np.zeros((nu.size, n_traj), dtype=dtype)
    pos = nu > 0
    Q[pos] = (sk / nu[pos])[:, None] * rng.standard_normal((int(pos.sum()), n_traj), dtype=dtype)
    alpha = (a / np.sqrt(system.M))[:, None] * P
    beta = -(a * nu / np.sqrt(system.M))[:, None] * Q
    return ModeState(P, Q, alpha, beta)


def iter_sample_modes(system: System, n_traj: int, rng: np.random.Generator,
                      chunk: int, dtype=np.float64) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(alpha, beta)`` chunks whose trajectory counts add up to ``n_traj``."""
    done = 0
    while done < n_traj:
        n = min(chunk, n_traj - done)
        yield sample_modes(system, n, rng, dtype)
        done += n


def default_chunk(system: System, budget_bytes: int = 256 * 2 ** 20) -> int:
    """Trajectories per chunk so that ``alpha`` and ``beta`` together stay under ``budget_bytes``."""
    per_traj = 2 * 8 * (system.bath.N + 1)
    return max(1, budget_bytes // per_traj)
