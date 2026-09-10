"""Figures for the frequency-grid notes (milestone 3).

  figures/notes_ohmic_grids.png     Ohmic: uniform-midpoint vs uniform-endpoint vs inverse-CDF, error vs t and vs N
  figures/notes_debye_grids.png     Debye: tan grid vs uniform grid with a cutoff; kernel and VACF errors
  figures/notes_images.png          the Poisson-summation picture: K_N - K_inf against the image sum

Writes PNG figures to examples/output/.
"""
import pathlib

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


def fig_ohmic_grids():
    t = np.linspace(0, 50, 4001)
    Kinf = kz.ohmic(N=10).kernel_continuum(t)
    fig, ax = plt.subplots(3, 1, figsize=(6.5, 11.8), gridspec_kw={"hspace": 0.85})
    # A: error vs t at N = 2000 for three grids
    for label, b, col in (("uniform, midpoints", kz.ohmic(N=2000), SERIES[0]),
                          ("uniform, endpoints (j·Δω)", kz.ohmic(N=2000, midpoint=False), SERIES[3]),
                          ("inverse-CDF", kz.ohmic(N=2000, grid="invcdf"), SERIES[1])):
        ax[0].semilogy(t, np.abs(b.kernel(t) - Kinf) + 1e-13, color=col, lw=0.9, label=label)
    ax[0].semilogy(t, Kinf, color=INK, ls="--", lw=1.0, label="K_∞ itself")
    ax[0].set_ylabel("|K_N − K_∞|")
    ax[0].set_xlabel("t")
    ax[0].set_title("Ohmic, N = 2000: the endpoint grid carries a constant bias Δω η/π", loc="left")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.5)
    ax[0].set_ylim(1e-9, 1)
    # B: max error vs N
    Ns = [250, 500, 1000, 2000, 4000, 8000, 16000]
    for label, mk, kw, col in (("uniform, midpoints", "o", dict(grid="uniform"), SERIES[0]),
                               ("uniform, endpoints", "^", dict(grid="uniform", midpoint=False), SERIES[3]),
                               ("inverse-CDF", "s", dict(grid="invcdf"), SERIES[1])):
        errs = [np.abs(kz.ohmic(N=N, **kw).kernel(t) - Kinf).max() for N in Ns]
        ax[1].loglog(Ns, errs, mk + "-", color=col, ms=5, lw=1.0, label=label)
    ax[1].loglog(Ns, [2.2 / n for n in Ns], color=INK2, ls=":", lw=0.9, label="ripple estimate K(0)√(ω_c t)/N, t = 50")
    ax[1].loglog(Ns, [2 * (2 / np.pi) / (1 + (2 * np.pi * n / 25) ** 2) for n in Ns], color=INK, ls="--", lw=0.9,
                 label="first Poisson images, 2K_∞(2π/Δω)")
    ax[1].set_xlabel("N   (ω_max = 25 ω_c for the uniform grids)")
    ax[1].set_ylabel("max |K_N − K_∞|, t ≤ 50")
    ax[1].set_title("Convergence with N: second order (images) vs first order (bias, ripple)", loc="left")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.5)
    # C: window: the uniform grid's periodic images appear at t = 2π/Δω
    b = kz.ohmic(N=400)                       # Δω = 0.0625 -> period 100.5
    tt = np.linspace(0, 160, 6401)
    ax[2].plot(tt, b.kernel(tt), color=SERIES[0], lw=0.9, label="K_N, N = 400, Δω = 0.0625")
    ax[2].plot(tt, b.kernel_continuum(tt), color=INK, lw=1.0, ls="--", label="K_∞")
    per = 2 * np.pi / (b.omega[1] - b.omega[0])
    ax[2].axvline(per, color=INK2, lw=0.8, ls=":")
    ax[2].axvline(per / 2, color=INK2, lw=0.8, ls=":")
    ax[2].annotate("π/Δω: symmetry axis", xy=(per / 2, 0.3), ha="center", color=INK2, fontsize=9)
    ax[2].annotate("2π/Δω: negative image", xy=(per, -0.5), ha="center", color=INK2, fontsize=9)
    ax[2].set_ylim(-0.75, 0.75)
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("K(t)")
    ax[2].set_title("Uniform grid: period 2π/Δω, first image negative (alternating signs)", loc="left")
    ax[2].legend(loc="upper right", fontsize=8.5)
    fig.savefig(OUT / "notes_ohmic_grids.png")
    plt.close(fig)


def fig_debye_grids():
    lm, wc, M = 0.6, 1.0, 1.0
    t = np.linspace(0, 20, 2001)
    Kinf = kz.debye(lmbda=lm, N=10).kernel_continuum(t)
    Cinf = kz.debye_vacf_continuum(t, lm, wc, M)
    cases = [("tan grid (inverse-CDF), N = 4000", kz.debye(lmbda=lm, N=4000), SERIES[1]),
             ("uniform, ω_max = 100, N = 4000 (Δω = 0.025)",
              kz.debye(lmbda=lm, N=4000, grid="uniform", omega_max=100), SERIES[0]),
             ("uniform, ω_max = 400, N = 4000 (Δω = 0.1)",
              kz.debye(lmbda=lm, N=4000, grid="uniform", omega_max=400), SERIES[2])]
    fig, ax = plt.subplots(3, 1, figsize=(6.5, 12.2), gridspec_kw={"hspace": 0.9})
    for label, b, col in cases:
        ax[0].semilogy(t, np.abs(b.kernel(t) - Kinf) + 1e-13, color=col, lw=0.9, label=label)
    ax[0].semilogy(t, Kinf, color=INK, ls="--", lw=1.0, label=r"K_∞ = 2λ$e^{-\omega_c t}$")
    ax[0].set_ylabel("|K_N − K_∞|")
    ax[0].set_xlabel("t")
    ax[0].set_title(f"Debye λ = {lm}: kernel error, three discretizations", loc="left")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.5)
    ax[0].set_ylim(1e-7, 1)
    for label, b, col in cases:
        s = kz.System(b, M=M, kT=1.0)
        ax[1].semilogy(t, np.abs(s.vacf(t) - Cinf) + 1e-13, color=col, lw=0.9, label=label)
    ax[1].set_ylabel("|C_N − C_∞|")
    ax[1].set_xlabel("t")
    ax[1].set_title("VACF error: the particle filters out most of the kernel ripple", loc="left")
    ax[1].set_ylim(1e-9, 1e-2)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.5)
    # C: the cusp region: cutoff grids miss (2/pi) K0 omega_c/omega_max at t = 0
    tt = np.linspace(0, 0.1, 1001)
    for label, b, col in cases:
        ax[2].plot(tt, b.kernel(tt) - 2 * lm * np.exp(-wc * tt), color=col, lw=1.0, label=label)
    for W, col in ((100, SERIES[0]), (400, SERIES[2])):
        ax[2].axhline(-(2 / np.pi) * 2 * lm * np.arctan(wc / W), color=col, lw=0.7, ls=":")
    ax[2].axhline(0, color=INK2, lw=0.6)
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("K_N − K_∞ near the cusp")
    ax[2].set_title("Cutoff grids miss (2/π)K(0)·arctan(ω_c/ω_max) at t = 0 (dotted); the tan grid does not",
                    loc="left",
                    fontsize=10)
    ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.5)
    fig.savefig(OUT / "notes_debye_grids.png")
    plt.close(fig)


def fig_images():
    """K_N - K_inf for the midpoint-uniform Ohmic grid against the alternating Poisson image sum."""
    b = kz.ohmic(N=400)                       # Δω = 0.0625, period 100.5
    dw = b.omega[1] - b.omega[0]
    P = 2 * np.pi / dw
    t = np.linspace(0, 60, 2401)
    err = b.kernel(t) - b.kernel_continuum(t)
    images = np.zeros_like(t)
    for m in range(1, 200):
        images += (-1) ** m * (b.kernel_continuum(t - m * P) + b.kernel_continuum(t + m * P))
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.4), sharex=True, gridspec_kw={"hspace": 0.12})
    ax[0].plot(t, err, color=SERIES[0], lw=1.2, label="K_N − K_∞ (computed)")
    ax[0].plot(t, images, color=INK, lw=1.0, ls="--", label="Σ_{m≠0} (−1)^m K_∞(t − 2πm/Δω)  (Poisson images)")
    ax[0].set_ylabel("error")
    ax[0].set_title("Midpoint-uniform Ohmic grid, N = 400, Δω = 0.0625: error = alternating images", loc="left")
    ax[0].legend(loc="lower left", fontsize=8.5)
    ax[1].semilogy(t, np.abs(err - images) + 1e-17, color=SERIES[1], lw=0.9, label="|computed − images|: residual")
    ax[1].semilogy(t, np.abs(images) + 1e-17, color=INK, lw=0.9, ls="--", label="|images|")
    ax[1].set_ylabel("magnitude")
    ax[1].set_xlabel("t")
    ax[1].legend(loc="lower right", fontsize=8.5)
    fig.savefig(OUT / "notes_images.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_ohmic_grids()
    fig_debye_grids()
    fig_images()
    print("wrote", sorted(p.name for p in OUT.glob("notes_*.png")))
