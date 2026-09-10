"""A deliberately simple memory-kernel solver, to show how the package's inversion data is used.

The generalized Langevin equation for the normalized velocity autocorrelation function,

    dC/dt = -int_0^t Gamma(t - s) C(s) ds,

is a Volterra integral equation of the first kind for the memory kernel Gamma.  On a uniform
grid the trapezoid rule turns it into a triangular linear system that can be solved one
point at a time:

    Cdot_i = -dt [ Gamma_i C_0 / 2 + sum_{j=1}^{i-1} Gamma_{i-j} C_j + Gamma_0 C_i / 2 ],

so Gamma_i follows from Gamma_0 ... Gamma_{i-1}.  It needs the starting value
Gamma_0 = -C''(0).  On exact data the error is O(dt^2).  On sampled data it amplifies the
noise, as every first-kind inversion does, which is the effect kac_zwanzig exists to study.
Use a real solver (product-integration collocation, regularized methods) for serious work.

Run as a script to see both regimes and write examples/output/naive_inversion.png.
"""
import pathlib

import numpy as np


def trapezoid_inversion(C, Cdot, dt, Gamma0):
    """Solve ``dC/dt = -int_0^t Gamma(t-s) C(s) ds`` for ``Gamma`` on a uniform grid by the trapezoid rule.

    Parameters
    ----------
    C, Cdot : array_like, shape (n,)
        The correlation function and its derivative at ``t = 0, dt, 2 dt, ...``.
    dt : float
    Gamma0 : float
        ``Gamma(0) = -C''(0)``, known exactly for the model or estimated from the data.

    Returns
    -------
    ndarray, shape (n,)
    """
    C = np.asarray(C, dtype=float)
    Cdot = np.asarray(Cdot, dtype=float)
    n = C.size
    G = np.empty(n)
    G[0] = Gamma0
    for i in range(1, n):
        history = np.dot(G[i - 1:0:-1], C[1:i]) if i > 1 else 0.0        # sum_{j=1}^{i-1} Gamma_{i-j} C_j
        G[i] = (-Cdot[i] / dt - history - 0.5 * G[0] * C[i]) / (0.5 * C[0])
    return G


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import kac_zwanzig as kz

    s = kz.System(kz.ohmic(eta=1.0, N=2000))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

    # exact data: second-order convergence
    for dt, col in ((0.2, "C0"), (0.1, "C1"), (0.05, "C2")):
        t = np.arange(0, 12 + 1e-9, dt)
        data = s.inversion_data(t)
        G = trapezoid_inversion(data.C, data.Cdot, data.dt, data.memory_kernel_0)
        print(f"exact data, dt = {dt:<5}: max |Gamma - exact| = {data.residual(G):.2e}")
        ax[0].plot(t, G - data.memory_kernel, color=col, label=f"dt = {dt}")
    ax[0].set_title("exact C, dC/dt: error of the trapezoid inversion")
    ax[0].set_xlabel("t")
    ax[0].legend()

    # sampled data: noise amplification, with bootstrap error bars
    t = np.arange(0, 12 + 1e-9, 0.1)
    n = 100_000
    est = s.sampled_vacf(t, n_traj=n, seed=1)
    data = s.inversion_data(t, est, derivative="acc")
    G = trapezoid_inversion(data.C, data.Cdot, data.dt, data.memory_kernel_0)
    boot = est.bootstrap(300, seed=2)
    reps = np.array([trapezoid_inversion(boot.C[b], boot.Cdot_acc[b], data.dt, data.memory_kernel_0)
                     for b in range(boot.C.shape[0])])
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    print(f"sampled data ({n} trajectories, dt = 0.1): max |Gamma - exact| = {data.residual(G):.2e}, "
          f"noise in C was {np.sqrt(kz.predicted_variance(s.vacf(t), n)).max():.1e}")
    ax[1].plot(t, data.memory_kernel, "k", label="exact Γ")
    ax[1].fill_between(t, lo, hi, alpha=0.3, label="bootstrap 95% band")
    ax[1].plot(t, G, lw=0.9, label=f"recovered from {n} sampled trajectories")
    ax[1].set_title("sampled data: the inversion amplifies the noise")
    ax[1].set_xlabel("t")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "naive_inversion.png", dpi=130)
    print("wrote", out / "naive_inversion.png")
