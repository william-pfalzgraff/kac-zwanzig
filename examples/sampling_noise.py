"""Milestone 2 figures: the sampling layer (eta = 5, uniform grid, N = 2000).

  figures/m2_sampled_vacf.png       sampled VACF vs exact, pulls, RMS error vs number of trajectories
  figures/m2_derivative.png         derivative estimators: acceleration cross-correlation vs finite differences
  figures/m2_time_average_trap.png  why time-origin averaging of one trajectory does not converge here

Writes PNG figures to examples/output/ (a few minutes; the 10^6-trajectory estimate and
the 10^6-point trajectory dominate).
"""
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import kac_zwanzig as kz

OUT = pathlib.Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

ETA = 5.0
SYS = kz.System(kz.ohmic(eta=ETA, N=2000), M=1.0, kT=1.0)
DT = 0.025
T = np.arange(0, 20 + DT / 2, DT)
C_EX, CD_EX = SYS.vacf(T), SYS.vacf_dot(T)
KT_OVER_M = SYS.kT / SYS.M


def estimates(n_list, seed=10):
    """One ensemble estimate per n (exact normalization); the sample normalization is derived from R0."""
    out = {}
    for n in n_list:
        t0 = time.perf_counter()
        est = kz.vacf_ensemble(SYS, T, int(n), np.random.default_rng(seed + int(np.log10(n))),
                               normalization="exact", n_blocks=10)
        f = KT_OVER_M / est.R0
        out[int(n)] = dict(exact=est.C, sample=est.C * f, acc_exact=est.Cdot_acc, acc_sample=est.Cdot_acc * f,
                           sym_sample=est.Cdot_sym * f, est=est)
        print(f"  n_traj = {n:.0e}: {time.perf_counter() - t0:.1f} s")
    return out


def fig_sampled_vacf(E):
    fig, ax = plt.subplots(3, 1, figsize=(6.5, 9.4), gridspec_kw={"hspace": 0.32})
    # A: two noise levels against the exact curve, with predicted 2-sigma bands
    ax[0].plot(T, C_EX, color=INK, lw=1.2, label="exact")
    for n, col in zip((1000, 100000), SERIES):
        sig = np.sqrt(kz.predicted_variance(C_EX, n, "sample"))
        ax[0].fill_between(T, C_EX - 2 * sig, C_EX + 2 * sig, color=col, alpha=0.18, lw=0)
        ax[0].plot(T, E[n]["sample"], color=col, lw=0.9, label=f"sampled, {n:.0e} trajectories (band: predicted ±2σ)")
    ax[0].set_ylabel("C(t)")
    ax[0].set_title(f"Sampled VACF, η = {ETA:g}, sample normalization", loc="left")
    ax[0].legend(loc="upper right")
    ax[0].set_xlim(0, 12)
    # B: pulls
    n = 10000
    for norm, col in (("sample", SERIES[0]), ("exact", SERIES[1])):
        sig = np.sqrt(kz.predicted_variance(C_EX, n, norm))
        ok = sig > 1e-6
        ax[1].plot(T[ok], (E[n][norm] - C_EX)[ok] / sig[ok], color=col, lw=0.8, label=f"{norm} normalization")
    for y in (-2, -1, 1, 2):
        ax[1].axhline(y, color=INK2, lw=0.5, ls=":" if abs(y) == 1 else "--")
    ax[1].set_ylabel("(Ĉ − C) / σ_pred")
    ax[1].set_title(f"Pulls against the predicted standard deviation, {n:.0e} trajectories", loc="left")
    ax[1].legend(loc="upper right")
    ax[1].set_ylim(-4, 4)
    # C: RMS error vs n
    ns = sorted(E)
    for norm, col, mk in (("sample", SERIES[0], "o"), ("exact", SERIES[1], "s")):
        rms = [np.sqrt(((E[n][norm] - C_EX) ** 2).mean()) for n in ns]
        pred = [np.sqrt(kz.predicted_variance(C_EX, n, norm).mean()) for n in ns]
        ax[2].loglog(ns, rms, mk, color=col, ms=6, label=f"{norm} normalization, measured")
        ax[2].loglog(ns, pred, color=col, lw=1.0, ls="--", label=f"{norm} normalization, predicted")
    ax[2].set_xlabel("number of trajectories")
    ax[2].set_ylabel("RMS(Ĉ − C) over t ≤ 20")
    ax[2].set_title("Noise scales as 1/√(trajectories); sample normalization is quieter", loc="left")
    ax[2].legend(loc="lower left")
    fig.savefig(OUT / "m2_sampled_vacf.png")
    plt.close(fig)


def fd_on(Chat, step):
    """Finite differences of a sampled C sub-sampled to spacing step*DT, returned on that coarse grid."""
    return {o: kz.finite_difference(Chat[::step], DT * step, order=o) for o in (2, 4, 6)}


def fig_derivative(E):
    fig, ax = plt.subplots(3, 1, figsize=(6.5, 9.6), gridspec_kw={"hspace": 0.34})
    for row, n in enumerate((10000, 1000000)):
        Ch = E[n]["sample"]
        a = ax[row]
        for step, o, col in ((2, 2, SERIES[0]), (2, 6, SERIES[1]), (8, 2, SERIES[2]), (8, 6, SERIES[3])):
            err = fd_on(Ch, step)[o] - CD_EX[::step]
            a.plot(T[::step], err, color=col, lw=0.8, label=f"finite difference, order {o}, Δt = {DT * step:g}")
        a.plot(T, E[n]["acc_sample"] - CD_EX, color=INK, lw=0.9, label="⟨v̇(t)v(0)⟩ estimator (no differencing)")
        a.set_ylabel("error in dC/dt")
        a.set_title(f"{n:.0e} trajectories", loc="left")
        a.set_xlim(0, 12)
        lim = 3 * np.sqrt((0.5 * KT_OVER_M * SYS.bath.K0 / SYS.M + 1) / n) * 2
        a.set_ylim(-max(lim, 0.02), max(lim, 0.02))
        if row == 0:
            a.legend(loc="upper right", ncol=1)
    # C: RMS error vs dt
    steps = [1, 2, 4, 8, 16, 32]
    dts = [DT * s for s in steps]
    for o, col in zip((2, 4, 6), SERIES):
        trunc = [np.sqrt(((kz.finite_difference(C_EX[::s], DT * s, order=o) - CD_EX[::s]) ** 2).mean()) for s in steps]
        ax[2].loglog(dts, trunc, color=col, lw=1.0, ls="--", label=f"order {o}, noise-free C (truncation only)")
    for o, col in ((2, SERIES[0]), (6, SERIES[2])):
        for n, mk, mfc in ((10000, "o", col), (1000000, "s", "white")):
            noisy = [np.sqrt(((fd_on(E[n]["sample"], s)[o] - CD_EX[::s]) ** 2).mean()) for s in steps]
            ax[2].loglog(dts, noisy, mk, color=col, ms=5.5, mfc=mfc, label=f"order {o} on sampled C, {n:.0e} traj.")
    for n, ls in ((10000, "-"), (1000000, "-.")):
        rms = np.sqrt(((E[n]["acc_sample"] - CD_EX) ** 2).mean())
        ax[2].axhline(rms, color=INK, lw=0.9, ls=ls, label=f"⟨v̇(t)v(0)⟩ estimator, {n:.0e} traj.")
    ax[2].set_xlabel("Δt of the sampled C(t)")
    ax[2].set_ylabel("RMS error in dC/dt")
    ax[2].set_ylim(1e-5, 1)
    ax[2].set_title("Differencing error vs sampling noise (RMS over t ≤ 20)", loc="left")
    ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8, ncol=2)
    fig.savefig(OUT / "m2_derivative.png")
    plt.close(fig)


def fig_time_average():
    rng = np.random.default_rng(77)
    alpha, beta = kz.sample_modes(SYS, 1, rng)
    dt = 0.05
    n_lags = int(20 / dt) + 1
    t = np.arange(n_lags) * dt
    C = SYS.vacf(t)
    floor = kz.time_average_floor(SYS, t)
    fig, ax = plt.subplots(3, 1, figsize=(6.5, 9.4), gridspec_kw={"hspace": 0.34})
    lengths = [500, 5000, 50000]
    rms = []
    for L, col in zip(lengths, SERIES):
        t0 = time.perf_counter()
        tt = np.arange(int(L / dt)) * dt
        v = kz.velocity(SYS, tt, alpha, beta)[0]
        R = kz.vacf_time_average(v, n_lags) / KT_OVER_M
        print(f"  time average over T = {L}: {time.perf_counter() - t0:.1f} s")
        if L == lengths[-1]:
            ax[0].plot(t, R, color=col, lw=0.9, label=f"time average, one trajectory of length T = {L}")
        ax[1].plot(t, R - C, color=col, lw=0.8, label=f"T = {L}")
        rms.append(np.sqrt(((R - C) ** 2).mean()))
    ax[0].plot(t, C, color=INK, lw=1.2, label="exact canonical C(t)")
    ax[0].set_ylabel("C(t)")
    ax[0].set_title("One long trajectory, time-origin averaging (exact normalization)", loc="left")
    ax[0].legend(loc="upper right")
    ax[1].plot(t, floor, color=INK, lw=1.0, ls="--", label="predicted floor  √(Σ a_k⁴ cos² ν_k t)")
    ax[1].plot(t, -floor, color=INK, lw=1.0, ls="--")
    ax[1].set_ylabel("time average − exact")
    ax[1].set_title("The error stops shrinking: mode energies are frozen (integrable bath)", loc="left")
    ax[1].legend(loc="upper right", fontsize=8.5)
    n_eff = 1.0 / (SYS.a2 ** 2).sum()
    ax[2].loglog(lengths, rms, "o-", color=SERIES[0], label="RMS error of the time average")
    ax[2].axhline(np.sqrt((floor ** 2).mean()), color=INK, ls="--", lw=1.0, label="predicted floor (RMS)")
    ens = []
    for L in lengths:
        e = kz.vacf_ensemble(SYS, t, int(L / dt) // 400 + 1, np.random.default_rng(L),
                             normalization="exact", n_blocks=1)
        ens.append(np.sqrt(((e.C - C) ** 2).mean()))
    ax[2].loglog(lengths, ens, "s-", color=SERIES[1], label="ensemble average, one trajectory per 20 time units")
    ax[2].set_xlabel("trajectory length T  (units of 1/ω_c)")
    ax[2].set_ylabel("RMS error over t ≤ 20")
    ax[2].set_title(f"Effective number of modes 1/Σa_k⁴ = {n_eff:.0f}", loc="left")
    ax[2].legend(loc="lower left", fontsize=8.5)
    fig.savefig(OUT / "m2_time_average_trap.png")
    plt.close(fig)


if __name__ == "__main__":
    print("ensemble estimates:")
    E = estimates([1000, 10000, 100000, 1000000])
    fig_sampled_vacf(E)
    fig_derivative(E)
    if "--skip-time-average" not in sys.argv:
        print("time averages:")
        fig_time_average()
    print("wrote", sorted(p.name for p in OUT.glob("m2_*.png")))
