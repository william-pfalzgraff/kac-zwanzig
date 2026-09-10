"""Particle + bath: normal modes, the particle's spectral measure, exact correlation functions.

Model
-----
    H = p^2/(2M) + (1/2) M Omega^2 q^2 + sum_j [ p_j^2/2 + (omega_j^2/2) (x_j - g_j q)^2 ],
    g_j = sqrt(k_j)/omega_j

(``Omega = 0``, the free particle, by default).  The mass-weighted Hessian is an arrowhead
matrix,

    D_00 = (K0 + M Omega^2)/M,   D_0j = D_j0 = -sqrt(k_j/M) omega_j,   D_jj = omega_j^2,

whose eigenvalues ``nu_k^2`` and squared particle components ``a_k^2 = U_0k^2`` (the
particle's *spectral measure*) give everything in closed form:

    C(t)   = sum_k a_k^2 cos(nu_k t)                      normalized velocity autocorrelation
    dC/dt  = -sum_k a_k^2 nu_k sin(nu_k t)
    v(t)   = sum_k [alpha_k cos(nu_k t) + beta_k sin(nu_k t)],   alpha_k, beta_k ~ N(0, a_k^2 kT/M)

and the exact generalized Langevin equation for the velocity autocorrelation,

    dC/dt = -int_0^t Gamma(t - s) C(s) ds,        Gamma(t) = (K_N(t) + M Omega^2)/M,

a Volterra equation of the first kind for the *memory kernel* ``Gamma`` when ``C`` and
``dC/dt`` are known.  ``Gamma`` is the bath kernel per unit mass (the Mori memory function
of ``v``); the equation is linear in ``C``, so it holds equally for the unnormalized
``<v(t)v(0)>`` and does not involve ``kT``.  Solvers differ in sign conventions; this
package always reports the physical ``Gamma`` above (``Gamma(0) = (K0 + M Omega^2)/M > 0``).

For the free particle there is exactly one zero mode (translation), pinned to ``nu_0 = 0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bath import Bath, cos_sum, sin_sum


@dataclass
class InversionData:
    """Everything a first-kind Volterra solver needs to recover the memory kernel, plus the answer.

    The equation is ``dC/dt = -int_0^t Gamma(t - s) C(s) ds`` on the uniform grid ``t``.

    Attributes
    ----------
    t : ndarray
        Uniform time grid starting at 0.
    dt : float
    C : ndarray
        Normalized velocity autocorrelation (exact or sampled): the *kernel* of the
        first-kind equation.
    Cdot : ndarray
        ``dC/dt`` (exact, estimated, or finite-differenced): the right-hand side.
    memory_kernel : ndarray
        The exact ``Gamma(t)`` a solver should return.
    memory_kernel_0 : float
        ``Gamma(0) = (K0 + M Omega^2)/M``, useful as a starting value for solvers that need one.
    source : str
        ``"exact"`` or a description of the sampled estimate and derivative used.
    """
    t: np.ndarray
    dt: float
    C: np.ndarray
    Cdot: np.ndarray
    memory_kernel: np.ndarray
    memory_kernel_0: float
    source: str = "exact"

    def residual(self, Gamma: np.ndarray) -> float:
        """Max absolute error of a candidate ``Gamma`` against the exact memory kernel (common length)."""
        Gamma = np.asarray(Gamma, dtype=float).ravel()
        n = min(Gamma.size, self.memory_kernel.size)
        return float(np.abs(Gamma[:n] - self.memory_kernel[:n]).max())


class System:
    """A particle of mass ``M`` coupled to a :class:`~kac_zwanzig.bath.Bath` at temperature ``kT``.

    Parameters
    ----------
    bath : Bath
    M : float
        Particle mass.
    kT : float
        Temperature.  Only sets the velocity scale: normalized correlation functions do
        not depend on it.
    Omega : float
        Optional harmonic trap frequency.  ``0`` (default) is the free particle.  A trap
        adds the constant ``Omega^2`` to the memory kernel of the velocity GLE.

    Notes
    -----
    The spectral measure is computed by a dense symmetric eigensolve on first use and
    cached (``N = 2000`` takes under a second, ``N = 5000`` about 15 s).

    Examples
    --------
    >>> import numpy as np
>>> import kac_zwanzig as kz
    >>> s = kz.System(kz.ohmic(eta=1.0, N=2000))
    >>> t = np.arange(0, 20, 0.05)
    >>> C, Cdot, Gamma = s.vacf(t), s.vacf_dot(t), s.memory_kernel(t)
    >>> traj = s.trajectory(t, n_traj=3, seed=0)          # traj.v, traj.a, traj.q: (3, len(t))
    >>> est = s.sampled_vacf(t, n_traj=10_000, seed=1)    # est.C, est.Cdot_acc, est.stderr()
    >>> data = s.inversion_data(t, est)                   # data.C, data.Cdot, data.memory_kernel
    """

    def __init__(self, bath: Bath, M: float = 1.0, kT: float = 1.0, Omega: float = 0.0):
        if M <= 0 or kT <= 0 or Omega < 0:
            raise ValueError("need M > 0, kT > 0, Omega >= 0")
        self.bath = bath
        self.M = float(M)
        self.kT = float(kT)
        self.Omega = float(Omega)
        self._nu = None
        self._a = None
        self._U = None

    # ------------------------------------------------------------------ normal modes
    def hessian(self) -> np.ndarray:
        """Mass-weighted Hessian (arrowhead), shape ``(N+1, N+1)``; index 0 is the particle."""
        b = self.bath
        D = np.diag(np.concatenate([[(b.K0 + self.M * self.Omega ** 2) / self.M], b.omega ** 2]))
        off = -np.sqrt(b.k / self.M) * b.omega
        D[0, 1:] = off
        D[1:, 0] = off
        return D

    def spectral_measure(self, force: bool = False):
        """Mode frequencies and particle weights.

        Returns
        -------
        nu : ndarray, shape (N+1,)
            Normal-mode frequencies, ascending; ``nu[0] == 0`` for the free particle.
        a2 : ndarray, shape (N+1,)
            Squared particle components of the normalized eigenvectors, ``sum(a2) == 1``.
            ``C(t) = sum_k a2_k cos(nu_k t)``.
        """
        if self._nu is None or force:
            self._solve(keep_vectors=False)
        return self._nu, self._a ** 2

    def _solve(self, keep_vectors: bool):
        lam, U = np.linalg.eigh(self.hessian())
        if self.Omega == 0.0:
            if abs(lam[0]) > 1e-9 * abs(lam).max():
                raise RuntimeError(f"expected a zero mode, got lambda_0 = {lam[0]:.3e}")
            lam[0] = 0.0
        if lam.min() < -1e-9 * abs(lam).max():
            raise RuntimeError("Hessian is not positive semi-definite")
        self._nu = np.sqrt(np.clip(lam, 0.0, None))
        self._a = U[0].copy()
        self._U = U if keep_vectors else None

    def eigenvectors(self) -> np.ndarray:
        """Full eigenvector matrix ``U`` (``(N+1, N+1)``; columns are modes, row 0 is the particle).

        Needed only to reconstruct bath coordinates (random force).  Computed on first use
        and cached; costs ``(N+1)^2`` doubles.
        """
        if self._U is None:
            self._solve(keep_vectors=True)
        return self._U

    @property
    def nu(self) -> np.ndarray:
        return self.spectral_measure()[0]

    @property
    def a2(self) -> np.ndarray:
        return self.spectral_measure()[1]

    @property
    def a(self) -> np.ndarray:
        """Signed particle components ``U_0k`` (first row of the eigenvector matrix)."""
        self.spectral_measure()
        return self._a

    def zero_mode_weight_formula(self) -> float:
        """Closed form ``a_0^2 = 1/(1 + sum_j k_j/omega_j^2 / M)`` (free particle only)."""
        b = self.bath
        return 1.0 / (1.0 + float((b.k / b.omega ** 2).sum()) / self.M)

    def amplitude_sigma(self) -> np.ndarray:
        """Standard deviation of the mode amplitudes ``alpha_k, beta_k``: ``sqrt(a_k^2 kT/M)``."""
        return np.sqrt(self.a2 * self.kT / self.M)

    # ------------------------------------------------------------------ exact correlation functions
    def vacf(self, t) -> np.ndarray:
        """Exact normalized velocity autocorrelation ``C(t) = <v(t)v(0)>/<v^2> = sum_k a_k^2 cos(nu_k t)``.

        Parameters
        ----------
        t : array_like
            Times, any shape (need not be uniform).

        Returns
        -------
        ndarray
            ``C(t)``, with ``C(0) = 1`` to round-off.
        """
        nu, a2 = self.spectral_measure()
        return cos_sum(a2, nu, t)

    def vacf_dot(self, t) -> np.ndarray:
        """Exact ``dC/dt = -sum_k a_k^2 nu_k sin(nu_k t)``."""
        nu, a2 = self.spectral_measure()
        return -sin_sum(a2 * nu, nu, t)

    def vacf_ddot(self, t) -> np.ndarray:
        """Exact ``d^2C/dt^2``; ``C''(0) = -Gamma(0) = -(K0 + M Omega^2)/M``."""
        nu, a2 = self.spectral_measure()
        return -cos_sum(a2 * nu ** 2, nu, t)

    def vacf_integral(self, t) -> np.ndarray:
        """``int_0^t C(s) ds`` (the zero mode contributes ``a_0^2 t``)."""
        nu, a2 = self.spectral_measure()
        pos = nu > 0
        out = sin_sum(a2[pos] / nu[pos], nu[pos], t)
        if not pos.all():
            out = out + a2[~pos].sum() * np.asarray(t, dtype=float)
        return out

    # ------------------------------------------------------------------ the memory kernel
    def memory_kernel(self, t) -> np.ndarray:
        """Exact memory kernel ``Gamma(t) = (K_N(t) + M Omega^2)/M`` of the velocity GLE, per unit mass.

        Defined by ``dC/dt = -int_0^t Gamma(t - s) C(s) ds``; this is what a kernel-extraction
        method fed ``C`` and ``dC/dt`` should recover (the Mori memory function of ``v``).  It
        differs from ``bath.kernel`` only by the factor ``1/M`` (and the trap's ``Omega^2``):
        the equation is linear in ``C``, so the normalization of ``C`` and the temperature
        drop out.  Units of 1/time^2; ``Gamma(0) > 0``.  Solvers that write the equation
        with the opposite sign want ``-Gamma``.
        """
        return (self.bath.kernel(t) + self.M * self.Omega ** 2) / self.M

    @property
    def memory_kernel_0(self) -> float:
        """``Gamma(0) = (K0 + M Omega^2)/M``."""
        return (self.bath.K0 + self.M * self.Omega ** 2) / self.M

    def memory_kernel_continuum(self, t):
        """Continuum-limit memory kernel, for orientation only (``None`` if the bath has no continuum)."""
        if self.bath.kernel_continuum is None:
            return None
        return (self.bath.kernel_continuum(t) + self.M * self.Omega ** 2) / self.M

    def inversion_data(self, t, estimate=None, derivative: str = "acc") -> InversionData:
        """Package ``C`` and ``dC/dt`` on a uniform grid together with the exact memory kernel.

        Parameters
        ----------
        t : array_like
            Uniform grid starting at 0.
        estimate : VACFEstimate, optional
            A sampled estimate on the same grid (from :meth:`sampled_vacf`).  ``None`` uses
            the exact ``C`` and ``dC/dt`` (the noise-free problem).
        derivative : {"acc", "sym", "fd2", "fd4", "fd6", "exact"}
            Which ``dC/dt`` to use with a sampled estimate: the acceleration cross-correlation
            ``<v'(t) v(0)>`` (default), its symmetrized version, central finite differences of
            the sampled ``C`` of the given order (even extension at ``t = 0``), or the exact
            derivative (to isolate the effect of noise in ``C`` alone).

        Returns
        -------
        InversionData
            ``data.C``, ``data.Cdot``, ``data.dt``, and the exact ``data.memory_kernel`` to
            compare a solver's output against (``data.residual(Gamma)``).
        """
        from .estimate import finite_difference
        t = np.asarray(t, dtype=float).ravel()
        if t.size < 2 or abs(t[0]) > 1e-12:
            raise ValueError("t must start at 0 and have at least two points")
        dt = float(t[1] - t[0])
        if not np.allclose(np.diff(t), dt, rtol=1e-8, atol=1e-12):
            raise ValueError("t must be uniform")
        if estimate is None:
            C, Cdot, source = self.vacf(t), self.vacf_dot(t), "exact"
        else:
            if estimate.t.shape != t.shape or not np.allclose(estimate.t, t):
                raise ValueError("estimate was computed on a different time grid")
            C = estimate.C
            if derivative == "acc":
                Cdot = estimate.Cdot_acc
            elif derivative == "sym":
                Cdot = estimate.Cdot_sym
            elif derivative == "exact":
                Cdot = self.vacf_dot(t)
            elif derivative.startswith("fd"):
                Cdot = finite_difference(C, dt, order=int(derivative[2:]))
            else:
                raise ValueError(f"unknown derivative {derivative!r}")
            source = (f"sampled ({estimate.n_traj} trajectories, {estimate.normalization} normalization), "
                      f"derivative={derivative}")
        return InversionData(t, dt, np.asarray(C), np.asarray(Cdot), self.memory_kernel(t),
                             self.memory_kernel_0, source)

    # ------------------------------------------------------------------ sampling conveniences
    def trajectory(self, t, n_traj: int = 1, seed: Optional[int] = None,
                   rng: Optional[np.random.Generator] = None, positions: bool = True,
                   random_force: bool = False):
        """Exact trajectories from fresh canonical initial conditions.

        Parameters
        ----------
        t : array_like
            Output times (any grid; there is no time step).
        n_traj : int
            Number of independent trajectories.
        seed, rng : optional
            Random state; pass ``rng`` to continue a stream, or ``seed`` for reproducibility.
        positions : bool
            Also integrate the particle position ``q(t)``.
        random_force : bool
            Also compute the random force ``F(t)`` of the GLE (needs the eigenvector matrix,
            ``(N+1)^2`` doubles, cached after the first call).  Positions start at
            ``q(0) = 0`` for the free particle (its absolute position is a symmetry); a
            trapped particle with ``random_force=True`` starts from its sampled ``q(0)``,
            which the GLE's trap term needs.

        Returns
        -------
        Trajectory
            Fields ``t``, ``v``, ``a`` (= ``dv/dt``), ``q`` (or ``None``), ``F`` (or ``None``),
            each ``(n_traj, len(t))``, plus the mode amplitudes ``alpha, beta`` so the same
            trajectories can be re-evaluated on another grid with :func:`kac_zwanzig.velocity`.
            :func:`kac_zwanzig.gle_residual` checks ``M dv/dt + M Omega^2 q + int K v = F`` on it.
        """
        from .propagate import Trajectory, acceleration, initial_position, position, velocity
        from .propagate import random_force as _random_force
        from .sample import sample_modes, sample_state
        rng = np.random.default_rng(seed) if rng is None else rng
        t = np.asarray(t, dtype=float).ravel()
        if random_force:
            state = sample_state(self, int(n_traj), rng)
            alpha, beta = state.alpha, state.beta
            q0 = initial_position(self, state) if self.Omega > 0 else 0.0
            F = _random_force(self, t, state)
        else:
            state, F, q0 = None, None, 0.0
            alpha, beta = sample_modes(self, int(n_traj), rng)
        v = velocity(self, t, alpha, beta)
        a = acceleration(self, t, alpha, beta)
        q = position(self, t, alpha, beta, q0=q0) if positions else None
        return Trajectory(t=t, v=v, a=a, q=q, alpha=alpha, beta=beta, F=F, state=state)

    def sampled_vacf(self, t, n_traj: int, seed: Optional[int] = None,
                     rng: Optional[np.random.Generator] = None, normalization: str = "sample",
                     n_blocks: int = 50, chunk: Optional[int] = None):
        """Sampled VACF and derivative estimates from ``n_traj`` independent trajectories.

        This is what "simulation data" means in the test bed: an ensemble average over
        independent canonical initial conditions (time-origin averaging of one trajectory
        does not converge for this integrable bath, see :func:`kac_zwanzig.vacf_time_average`).

        Parameters
        ----------
        t : array_like
            Output times.
        n_traj : int
            Number of trajectories.  Noise on ``C`` is ``sqrt((1 - C^2)/n_traj)`` with the
            default normalization.
        seed, rng : optional
            Random state.
        normalization : {"sample", "exact"}
            Divide by the sampled ``<v(0)^2>`` (default; ``C(0) == 1`` exactly, the
            simulation-like choice) or by the exact ``kT/M``.
        n_blocks : int
            Number of independent groups the trajectories are split into.  The group
            spread gives ``est.stderr()``; resampling groups gives ``est.bootstrap()``.
            Keep it at 50 or more if you intend to bootstrap; the cost is negligible.
        chunk : int, optional
            Trajectories per chunk (memory control; default about 256 MB).

        Returns
        -------
        VACFEstimate
            ``est.C``, ``est.Cdot_acc`` (``<v'(t)v(0)>``), ``est.Cdot_sym``, ``est.stderr()``,
            ``est.bootstrap()`` and the per-group raw sums.  Cost is ``O(N (n_traj + len(t)))``;
            ``10^6`` trajectories at ``N = 2000`` take about 30 s.
        """
        from .estimate import vacf_ensemble
        rng = np.random.default_rng(seed) if rng is None else rng
        return vacf_ensemble(self, t, int(n_traj), rng, normalization=normalization,
                             n_blocks=n_blocks, chunk=chunk)

    def describe(self) -> str:
        return (f"System(M={self.M}, kT={self.kT}, Omega={self.Omega}) on {self.bath.describe()}; "
                f"Gamma(0)={self.memory_kernel_0:.6g}, "
                f"cage frequency sqrt(Gamma(0))={np.sqrt(self.memory_kernel_0):.4g}")
