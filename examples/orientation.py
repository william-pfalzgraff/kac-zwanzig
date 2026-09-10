"""Orientation figures on linear scales.

  figures/orient_C_Cdot_K.png   C(t), dC/dt and the voles kernel for Ohmic (eta = 1, 5) and Debye (lambda = 0.6)
  figures/scaling_omega_c.png   omega_c only rescales time: C depends on eta/(M omega_c) alone

Writes PNG figures to examples/output/.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import kac_zwanzig as kz

OUT = pathlib.Path(__file__).resolve().parent / "output"
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


def fig_orientation():
    t = np.linspace(0, 15, 1501)
    cases = [
        ("Ohmic, η = 1", kz.System(kz.ohmic(eta=1.0, N=2000)), SERIES[0]),
        ("Ohmic, η = 5", kz.System(kz.ohmic(eta=5.0, N=2000)), SERIES[1]),
        ("Debye, λ = 0.6", kz.System(kz.debye(lmbda=0.6, N=4000)), SERIES[2]),
    ]
    fig, ax = plt.subplots(3, 1, figsize=(6.5, 9.6), sharex=True, gridspec_kw={"hspace": 0.1})
    for label, s, col in cases:
        ax[0].plot(t, s.vacf(t), color=col, label=label)
        ax[1].plot(t, s.vacf_dot(t), color=col, label=label)
        ax[2].plot(t, s.memory_kernel(t), color=col, label=label)
    for a in ax:
        a.axhline(0, color=INK2, lw=0.6)
    ax[0].set_ylabel("C(t)")
    ax[0].set_title("M = k_BT = ω_c = 1;  finite baths (N = 2000 Ohmic, 4000 Debye)", loc="left")
    ax[0].legend(loc="upper right")
    ax[1].set_ylabel("dC/dt")
    ax[2].set_ylabel("memory kernel  Γ(t) = K(t)/M")
    ax[2].set_xlabel("t  (units of 1/ω_c)")
    ax[2].annotate("Γ(0):  Ohmic 2ηω_c/π = 0.64, 3.18;   Debye 2λ = 1.2", xy=(0.36, 0.85),
                   xycoords="axes fraction", color=INK2, fontsize=9)
    fig.savefig(OUT / "orient_C_Cdot_K.png")
    plt.close(fig)


def fig_scaling():
    """Same eta/(M omega_c) = 5, three different omega_c: identical once time is measured in 1/omega_c."""
    combos = [(1.0, 0.2, SERIES[0]), (5.0, 1.0, SERIES[1]), (25.0, 5.0, SERIES[2])]
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.8), gridspec_kw={"hspace": 0.35})
    t = np.linspace(0, 30, 3001)
    for eta, wc, col in combos:
        s = kz.System(kz.ohmic(eta=eta, omega_c=wc, N=2000))
        ax[0].plot(t, s.vacf(t), color=col, label=f"η = {eta:g}, ω_c = {wc:g}")
        x = np.linspace(0, 12, 1201)
        ax[1].plot(x, s.vacf(x / wc), color=col, lw=2.6 - 0.8 * combos.index((eta, wc, col)),
                   label=f"η = {eta:g}, ω_c = {wc:g}")
    ax[0].axhline(0, color=INK2, lw=0.6)
    ax[0].set_xlabel("t  (absolute units, M = 1)")
    ax[0].set_ylabel("C(t)")
    ax[0].set_title("Three baths with the same η/(Mω_c) = 5", loc="left")
    ax[0].legend(loc="upper right")
    ax[1].axhline(0, color=INK2, lw=0.6)
    ax[1].set_xlabel("ω_c t")
    ax[1].set_ylabel("C")
    ax[1].set_title("Plotted against ω_c t they coincide: ω_c only sets the clock", loc="left")
    ax[1].legend(loc="upper right")
    fig.savefig(OUT / "scaling_omega_c.png")
    plt.close(fig)
    s1 = kz.System(kz.ohmic(eta=1.0, omega_c=0.2, N=2000))
    s2 = kz.System(kz.ohmic(eta=5.0, omega_c=1.0, N=2000))
    x = np.linspace(0, 12, 121)
    print("max |C(eta=1, wc=0.2)(t/0.2) - C(eta=5, wc=1)(t)| =", np.abs(s1.vacf(x / 0.2) - s2.vacf(x)).max())


if __name__ == "__main__":
    fig_orientation()
    fig_scaling()
    print("wrote orient_C_Cdot_K.png, scaling_omega_c.png")
