"""kac_zwanzig: an exactly solvable Kac-Zwanzig bath.

A particle linearly coupled to ``N`` harmonic oscillators is a quadratic system, so its
dynamics, its memory kernel and its velocity autocorrelation function are all available
in closed form (up to one symmetric eigensolve).  This package generates:

* spectral densities and their discretizations      -> :func:`ohmic`, :func:`debye`, :meth:`Bath.from_spectral_density`
* the exact finite-bath kernel and its continuum    -> :meth:`Bath.kernel`, :meth:`Bath.kernel_continuum`
* the exact correlation function and derivative     -> :meth:`System.vacf`, :meth:`System.vacf_dot`
* the exact memory kernel of the velocity GLE       -> :meth:`System.memory_kernel`
* exact trajectories (velocity, acceleration, position, random force) from canonical
  initial conditions                                -> :meth:`System.trajectory`
* sampled correlation functions from ``n`` trajectories, with derivative estimators,
  standard errors and a bootstrap                   -> :meth:`System.sampled_vacf`
* a bundle of ``C``, ``dC/dt`` and the exact kernel for testing any kernel-extraction method
                                                    -> :meth:`System.inversion_data`

The package depends only on numpy and knows nothing about any particular solver.

Quick start
-----------
>>> import numpy as np
>>> import kac_zwanzig as kz
>>> bath = kz.ohmic(eta=1.0, omega_c=1.0, N=2000)        # J = eta w exp(-w/omega_c), uniform grid
>>> s = kz.System(bath, M=1.0, kT=1.0)                    # free particle
>>> t = np.arange(0, 20, 0.05)
>>> C, Cdot, Gamma = s.vacf(t), s.vacf_dot(t), s.memory_kernel(t)
>>> traj = s.trajectory(t, n_traj=1, seed=0, random_force=True)   # traj.v, traj.a, traj.q, traj.F
>>> est = s.sampled_vacf(t, n_traj=10_000, seed=1)        # est.C ~ C +- sqrt((1 - C^2)/10_000)
>>> boot = est.bootstrap(1000)                            # replicate curves, boot.std(), boot.band()
>>> data = s.inversion_data(t, est, derivative="acc")     # data.C, data.Cdot, data.memory_kernel

Conventions
-----------
Units are the user's; the presets are natural with ``M = kT = omega_c = 1``.  The
correlation function is normalized, ``C(0) = 1``.  The memory kernel ``Gamma`` is defined by

    dC/dt = -int_0^t Gamma(t - s) C(s) ds,        Gamma(t) = (K_N(t) + M Omega^2)/M,

so ``Gamma(0) > 0``; a solver that writes the equation with the opposite sign wants ``-Gamma``.

See ``README.md`` for the API table, ``docs/theory.md`` for the derivations and
``docs/frequency_grids.pdf`` for the frequency-grid analysis.
"""
from .bath import Bath, bath_from_spectral_density, cos_sum, debye, debye_vacf_continuum, ohmic, sin_sum
from .estimate import (
                   BootstrapResult,
                   VACFEstimate,
                   fd_weights,
                   finite_difference,
                   predicted_covariance,
                   predicted_variance,
                   predicted_variance_derivative,
                   time_average_floor,
                   vacf_direct,
                   vacf_ensemble,
                   vacf_time_average,
)
from .propagate import Trajectory, acceleration, gle_residual, initial_position, position, random_force, velocity
from .sample import ModeState, default_chunk, iter_sample_modes, sample_modes, sample_state
from .system import InversionData, System

__version__ = "0.4.0"

sampled_vacf = vacf_ensemble
"""Alias: ``kac_zwanzig.sampled_vacf(system, t, n_traj, rng)`` is :func:`vacf_ensemble`."""

__all__ = [
    # baths
    "Bath", "ohmic", "debye", "debye_vacf_continuum", "bath_from_spectral_density", "cos_sum", "sin_sum",
    # particle + bath
    "System", "InversionData",
    # sampling and propagation
    "sample_modes", "sample_state", "ModeState", "iter_sample_modes", "default_chunk",
    "velocity", "acceleration", "position", "initial_position", "random_force", "gle_residual", "Trajectory",
    # estimators
    "VACFEstimate", "BootstrapResult", "vacf_ensemble", "sampled_vacf", "vacf_direct",
    "vacf_time_average", "time_average_floor",
    "finite_difference", "fd_weights", "predicted_variance", "predicted_covariance", "predicted_variance_derivative",
    "__version__",
]
