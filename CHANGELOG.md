# Changelog

## 0.4.0 — 2026-09-10

First public release. Renamed from the internal package `kz` to `kac_zwanzig`
(distribution `kac-zwanzig`); the API is otherwise that of the internal 0.3.0.

- Baths: `ohmic`, `debye` presets with closed-form continuum kernels; `Bath.from_spectral_density`
  for any spectral density; midpoint-uniform and inverse-CDF frequency grids.
- Exact finite-bath quantities: kernel, memory kernel, velocity autocorrelation function and its
  derivatives, from one symmetric eigensolve of the arrowhead Hessian.
- Exact trajectories: velocity, acceleration, position and the random force of the generalized
  Langevin equation, from canonical initial conditions; `gle_residual` verifies the equation on a trajectory.
- Sampled correlation functions from independent trajectories in a collapsed form that never
  stores the trajectories; acceleration-based and finite-difference derivative estimators;
  standard errors from groups of trajectories; a group bootstrap that preserves the time
  correlation of the noise; analytic noise formulas for both normalizations.
- `System.inversion_data` bundles `C`, `dC/dt` and the exact memory kernel for any
  kernel-extraction method.
- Examples: quickstart (script and executed notebook), frequency-grid comparison, sampling-noise
  and derivative study, orientation plots, and a deliberately simple trapezoid inversion.
- Documentation: introductory README with numbered equations, `docs/theory.md`, and the
  frequency-grid notes `docs/frequency_grids.pdf`.
