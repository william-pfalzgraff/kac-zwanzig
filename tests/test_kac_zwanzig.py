"""Test suite for kac_zwanzig: exact identities (T1-T3), sampling and propagation (T4), the estimators'
noise formulas, the bootstrap, the random force and the finite differences.  Run ``python -m pytest -q``.
"""
import numpy as np
import pytest

import kac_zwanzig as kz

_trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2 has only trapz

M, KT = 1.3, 0.7          # deliberately not 1, to catch unit slips


@pytest.fixture(scope="module")
def small():
    """Ohmic, uniform grid, N=200: fast enough for double sums."""
    return kz.System(kz.ohmic(eta=1.5, omega_c=1.0, N=200), M=M, kT=KT)


@pytest.fixture(scope="module")
def small_invcdf():
    return kz.System(kz.ohmic(eta=1.5, omega_c=1.0, N=200, grid="invcdf"), M=M, kT=KT)


@pytest.fixture(scope="module")
def trapped():
    return kz.System(kz.ohmic(eta=1.5, omega_c=1.0, N=200), M=M, kT=KT, Omega=0.8)


@pytest.fixture(scope="module")
def production():
    """The planned working case: Ohmic, uniform midpoint grid, N=2000."""
    return kz.System(kz.ohmic(eta=1.0, omega_c=1.0, N=2000), M=1.0, kT=1.0)


def convolution_closed_form(system, t):
    """int_0^t K_N(t-s) C(s) ds  via  int_0^t cos w(t-s) cos nu s ds = (w sin wt - nu sin nu t)/(w^2 - nu^2),
    plus the trap's constant  M Omega^2 int_0^t C(s) ds."""
    w, k = system.bath.omega, system.bath.k
    nu, a2 = system.spectral_measure()
    W, NU = np.meshgrid(w, nu, indexing="ij")
    den = W ** 2 - NU ** 2
    coincident = den == 0.0          # nu_k == omega_j to the last bit (tiny-weight high modes): use the limit
    den_safe = np.where(coincident, 1.0, den)
    out = np.empty_like(t)
    for i, tt in enumerate(t):
        integ = (W * np.sin(W * tt) - NU * np.sin(NU * tt)) / den_safe
        integ = np.where(coincident, (np.sin(W * tt) + W * tt * np.cos(W * tt)) / (2 * W), integ)
        out[i] = k @ integ @ a2
    return out + system.M * system.Omega ** 2 * system.vacf_integral(t)


# ---------------------------------------------------------------- T1: spectral measure identities
@pytest.mark.parametrize("fx", ["small", "small_invcdf", "trapped"])
def test_T1_spectral_measure(fx, request):
    s = request.getfixturevalue(fx)
    nu, a2 = s.spectral_measure()
    assert abs(a2.sum() - 1.0) < 1e-12
    assert s.vacf(0.0) == pytest.approx(1.0, abs=1e-12)
    assert s.vacf_dot(0.0) == pytest.approx(0.0, abs=1e-12)
    # C''(0) = -(K0 + M Omega^2)/M = -Gamma(0)
    assert s.vacf_ddot(0.0) == pytest.approx(-s.memory_kernel_0, rel=1e-10)
    if s.Omega == 0:
        assert nu[0] == 0.0 and np.all(nu[1:] > 0)
        assert a2[0] == pytest.approx(s.zero_mode_weight_formula(), rel=1e-9)
    else:
        assert np.all(nu > 0)
    # interlacing: exactly one nonzero mode in each (w_j, w_{j+1}) and one above w_N
    w = s.bath.omega
    counts = np.histogram(nu[nu > 0], bins=np.concatenate([[0.0], w, [np.inf]]))[0]
    assert np.all(counts[1:] == 1) and counts[0] == (0 if s.Omega == 0 else 1)


# ---------------------------------------------------------------- T2: the master equation holds exactly
@pytest.mark.parametrize("fx", ["small", "small_invcdf", "trapped"])
def test_T2_vie_identity_closed_form(fx, request):
    s = request.getfixturevalue(fx)
    t = np.linspace(0.0, 40.0, 161)
    resid = s.M * s.vacf_dot(t) + convolution_closed_form(s, t)
    assert np.abs(resid).max() < 1e-11


def test_T2_vie_identity_production(production):
    t = np.linspace(0.0, 30.0, 31)
    resid = production.M * production.vacf_dot(t) + convolution_closed_form(production, t)
    assert np.abs(resid).max() < 1e-10


def test_T2_memory_kernel_sign(small):
    """dC/dt = -int_0^t Gamma(t-s) C(s) ds with Gamma = K_N/M (free particle), checked by quadrature on a fine grid."""
    dt = 1e-3
    t = np.arange(0, 20 + dt / 2, dt)
    C, G = small.vacf(t), small.memory_kernel(t)
    conv = np.array([_trapezoid(G[i::-1] * C[: i + 1], dx=dt) for i in range(0, t.size, 500)])
    assert np.abs(-conv - small.vacf_dot(t[::500])).max() < 1e-5
    assert small.memory_kernel_0 == pytest.approx(small.bath.K0 / small.M)
    assert small.memory_kernel(0.0) == pytest.approx(small.memory_kernel_0)


# ---------------------------------------------------------------- T3: finite bath vs continuum
def test_T3_ohmic_uniform_matches_continuum():
    t = np.linspace(0, 50, 2001)
    for N, tol in ((2000, 1e-5), (20000, 1e-7)):
        b = kz.ohmic(eta=1.0, omega_c=1.0, N=N)
        assert np.abs(b.kernel(t) - b.kernel_continuum(t)).max() < tol
        # midpoint-rule error in K0 is -(dw^2/24) K0 (= the alternating Poisson images), 6.5e-6 relative at N=2000
        assert b.K0 == pytest.approx(2 / np.pi, rel=2e-5)


def test_T3_ohmic_invcdf_is_worse_but_exact_in_K0():
    t = np.linspace(0, 50, 2001)
    b = kz.ohmic(eta=1.0, omega_c=1.0, N=2000, grid="invcdf")
    err = np.abs(b.kernel(t) - b.kernel_continuum(t)).max()
    assert 1e-4 < err < 5e-3
    assert b.K0 == pytest.approx(2 / np.pi, rel=1e-14)


def test_T3_uniform_endpoint_grid_biases_K0():
    b_mid = kz.ohmic(N=2000)
    b_end = kz.ohmic(N=2000, midpoint=False)
    dw = b_mid.params["omega_max"] / 2000
    assert abs(b_mid.K0 - 2 / np.pi) < 1e-5
    # endpoint grid = trapezoid rule minus its half-weight at 0: bias K(0) dw/2 (1 - dw/6 + ...)
    assert (2 / np.pi - b_end.K0) == pytest.approx(dw / np.pi, rel=1e-2)


def test_T3_debye_kernel_and_vacf_converge():
    lm, wc = 0.6, 1.0
    t = np.linspace(0, 20, 801)
    errs_K, errs_C = [], []
    for N in (250, 1000, 4000):
        b = kz.debye(lmbda=lm, omega_c=wc, N=N)
        s = kz.System(b, M=M, kT=KT)
        errs_K.append(np.abs(b.kernel(t) - b.kernel_continuum(t))[t > 0.5].max())
        errs_C.append(np.abs(s.vacf(t) - kz.debye_vacf_continuum(t, lm, wc, M)).max())
    assert errs_K[0] > errs_K[1] > errs_K[2]          # slow: ~N^-0.8 (scrambled phases of the heavy tail)
    assert errs_C[0] > errs_C[1] > errs_C[2]          # fast: ~N^-1.7 (those modes barely couple to the particle)
    assert errs_C[2] < 1e-5


def test_debye_continuum_vacf_sanity():
    for lm, wc, m in ((0.05, 1.0, 1.0), (0.6, 1.0, 1.3), (1.0 / 8.0, 1.0, 1.0)):   # over-, under-, critically damped
        h = 1e-4
        t = np.array([0.0, h, 2 * h])
        C = kz.debye_vacf_continuum(t, lm, wc, m)
        assert C[0] == pytest.approx(1.0, abs=1e-12)
        assert (C[1] - C[0]) / h == pytest.approx(0.0, abs=1e-3)
        assert (C[2] - 2 * C[1] + C[0]) / h ** 2 == pytest.approx(-2 * lm / m, rel=1e-3)


# ---------------------------------------------------------------- T4: sampling and propagation
def test_T4_direct_equals_collapsed(small):
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 51)
    alpha, beta = kz.sample_modes(small, 300, rng)
    C_d, Cacc_d, Csym_d = kz.vacf_direct(small, t, alpha, beta)
    # feed the identical samples through the collapsed path via a stub generator
    est = kz.vacf_ensemble(small, t, 300, np.random.default_rng(1), n_blocks=1, chunk=300)
    assert np.abs(est.C - C_d).max() < 1e-12
    assert np.abs(est.Cdot_acc - Cacc_d).max() < 1e-12
    assert np.abs(est.Cdot_sym - Csym_d).max() < 1e-12


def test_T4_sample_statistics(small):
    """Amplitudes have the right variances and v(0) has variance kT/M."""
    rng = np.random.default_rng(2)
    alpha, beta = kz.sample_modes(small, 200000, rng)
    sig2 = small.a2 * KT / M
    big = sig2 > 1e-3 * sig2.max()
    assert np.allclose(alpha[big].var(axis=1), sig2[big], rtol=0.03)
    assert np.allclose(beta[big & (small.nu > 0)].var(axis=1), sig2[big & (small.nu > 0)], rtol=0.03)
    assert alpha.sum(axis=0).var() == pytest.approx(KT / M, rel=0.01)


@pytest.mark.parametrize("normalization", ["sample", "exact"])
def test_T4_pulls_match_predicted_variance(small, normalization):
    rng = np.random.default_rng(3)
    t = np.arange(0, 20.0, 0.25)
    n = 20000
    est = kz.vacf_ensemble(small, t, n, rng, normalization=normalization, n_blocks=20)
    C = small.vacf(t)
    var = kz.predicted_variance(C, n, normalization)
    mask = var > 1e-12 * var.max()
    pulls = (est.C - C)[mask] / np.sqrt(var[mask])
    rms = np.sqrt((pulls ** 2).mean())
    assert 0.7 < rms < 1.4, rms
    assert np.abs(pulls).max() < 4.5
    # block standard errors agree with the prediction to within their own scatter
    ratio = est.stderr("C")[mask] / np.sqrt(var[mask])
    assert 0.6 < np.median(ratio) < 1.5
    if normalization == "sample":
        assert est.C[0] == pytest.approx(1.0, abs=1e-14)


def test_T4_derivative_estimators(small):
    rng = np.random.default_rng(4)
    t = np.arange(0, 20.0, 0.25)
    n = 20000
    est = kz.vacf_ensemble(small, t, n, rng, normalization="exact", n_blocks=20)
    Cdot = small.vacf_dot(t)
    sig = np.sqrt(kz.predicted_variance_derivative(Cdot, small.bath.K0 / small.M, n))
    assert np.abs(est.Cdot_acc - Cdot).max() < 4.5 * sig.max()
    assert np.abs(est.Cdot_sym - Cdot).max() < 4.5 * sig.max()
    # symmetrizing should not be worse on average
    assert np.sqrt(((est.Cdot_sym - Cdot) ** 2).mean()) < 1.2 * np.sqrt(((est.Cdot_acc - Cdot) ** 2).mean())


def test_T4_covariance_structure(small):
    """Noise is correlated in time as predicted, checked on the block estimates."""
    rng = np.random.default_rng(5)
    t = np.arange(0, 10.0, 0.5)
    est = kz.vacf_ensemble(small, t, 40000, rng, normalization="exact", n_blocks=40)
    C = small.vacf(t)
    pred = kz.predicted_covariance(C, 1000, "exact")          # per block of 1000
    emp = np.cov(est.blocks_C.T)
    i, j = 4, 6
    assert emp[i, j] == pytest.approx(pred[i, j], abs=4 * pred[i, i] / np.sqrt(20))


# ---------------------------------------------------------------- finite differences
def test_finite_difference_orders(small):
    dt = 0.05
    t = np.arange(0, 15 + dt / 2, dt)
    C, Cdot = small.vacf(t), small.vacf_dot(t)
    errs = [np.abs(kz.finite_difference(C, dt, order=o) - Cdot).max() for o in (2, 4, 6)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-6
    assert kz.finite_difference(C, dt, order=4)[0] == 0.0          # symmetry at the origin
    # against np.gradient in the interior
    g = np.gradient(C, dt, edge_order=2)
    assert np.abs(kz.finite_difference(C, dt, order=2)[2:-2] - g[2:-2]).max() < 1e-12


def test_fd_weights_reproduce_classics():
    assert np.allclose(kz.fd_weights([-1, 0, 1]), [-0.5, 0, 0.5])
    assert np.allclose(kz.fd_weights([-2, -1, 0, 1, 2]), [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
    assert np.allclose(kz.fd_weights([0, 1, 2]), [-1.5, 2, -0.5])


# ---------------------------------------------------------------- library conveniences
def test_from_spectral_density_matches_presets():
    J = lambda w: 1.5 * w * np.exp(-w)
    b_num = kz.Bath.from_spectral_density(J, N=200, grid="uniform", omega_max=25)
    b_pre = kz.ohmic(eta=1.5, N=200)
    assert np.allclose(b_num.omega, b_pre.omega) and np.allclose(b_num.k, b_pre.k)
    b_num2 = kz.Bath.from_spectral_density(J, N=200, grid="invcdf", omega_max=60)
    b_pre2 = kz.ohmic(eta=1.5, N=200, grid="invcdf")
    assert b_num2.K0 == pytest.approx(b_pre2.K0, rel=1e-6)
    assert np.allclose(b_num2.omega, b_pre2.omega, rtol=1e-3, atol=1e-5)
    t = np.linspace(0, 10, 11)
    assert np.abs(b_num.kernel_continuum(t) - b_pre.kernel_continuum(t)).max() < 1e-6
    b_sup = kz.Bath.from_spectral_density(lambda w: 0.5 * w ** 3 * np.exp(-w), N=300, omega_max=30)
    assert b_sup.K0 == pytest.approx(2 / np.pi, rel=1e-4)          # (2/pi) * 0.5 * int w^2 e^-w dw = 2/pi


def test_position_and_acceleration_consistent_with_velocity(small):
    h = 1e-3
    traj = small.trajectory(np.arange(0, 3, h), n_traj=2, seed=3)
    assert np.allclose(traj.q[:, 0], 0.0)
    dq = np.gradient(traj.q, h, axis=1)
    dv = np.gradient(traj.v, h, axis=1)
    assert np.abs(dq[:, 3:-3] - traj.v[:, 3:-3]).max() < 1e-4
    assert np.abs(dv[:, 3:-3] - traj.a[:, 3:-3]).max() < 5e-3
    # same amplitudes re-evaluated with the module functions
    assert np.allclose(kz.velocity(small, traj.t, traj.alpha, traj.beta), traj.v)


def test_trajectory_sampled_vacf_and_inversion_data(small):
    t = np.arange(0, 10, 0.1)
    traj = small.trajectory(t, n_traj=4, seed=5)
    assert traj.v.shape == (4, t.size) and traj.q.shape == traj.v.shape and traj.n_traj == 4
    est = small.sampled_vacf(t, 2000, seed=6)
    assert est.C[0] == pytest.approx(1.0) and est.n_traj == 2000
    data = small.inversion_data(t, est, derivative="fd4")
    assert data.dt == pytest.approx(0.1) and data.memory_kernel_0 == pytest.approx(small.memory_kernel_0)
    assert np.allclose(data.Cdot, kz.finite_difference(est.C, 0.1, order=4)) and data.source.startswith("sampled")
    data0 = small.inversion_data(t)
    assert np.allclose(data0.Cdot, small.vacf_dot(t)) and np.allclose(data0.memory_kernel, small.memory_kernel(t))
    assert data0.residual(data0.memory_kernel) == 0.0 and data0.source == "exact"
    with pytest.raises(ValueError):
        small.inversion_data(np.array([0.0, 0.1, 0.3]))
    # seeds reproduce
    assert np.allclose(small.trajectory(t, 2, seed=9).v, small.trajectory(t, 2, seed=9).v)


def test_bootstrap_matches_predicted_noise(small):
    t = np.arange(0, 20.0, 0.25)
    n = 20000
    C = small.vacf(t)
    est = small.sampled_vacf(t, n, seed=11, n_blocks=50)
    boot = est.bootstrap(400, seed=12)
    assert boot.C.shape == (400, t.size) and np.allclose(boot.C[:, 0], 1.0)
    sig = np.sqrt(kz.predicted_variance(C, n, "sample"))
    mask = sig > 1e-3 * sig.max()
    assert 0.7 < np.median(boot.std("C")[mask] / sig[mask]) < 1.4
    assert np.abs(boot.C.mean(axis=0) - est.C).max() < 3 * sig.max()          # replicate mean ~ pooled estimate
    lo, hi = boot.band("C", 0.95)
    assert np.all(lo <= est.C + 1e-12) and np.all(hi >= est.C - 1e-12)
    assert boot.covariance("C").shape == (t.size, t.size)
    est2 = small.sampled_vacf(t, n, seed=13, normalization="exact", n_blocks=50)
    boot2 = est2.bootstrap(300, seed=14)
    sig2 = np.sqrt(kz.predicted_variance(C, n, "exact"))
    assert 0.7 < np.median(boot2.std("C") / sig2) < 1.4
    # the derivative estimator gets replicates too
    assert 0.5 < np.median(boot.std("acc") / (est.stderr("acc") + 1e-30)) < 2.0
    with pytest.raises(ValueError):
        small.sampled_vacf(t, 100, seed=1, n_blocks=1).bootstrap(10)


def test_sample_state_and_random_force_statistics(small):
    rng = np.random.default_rng(21)
    n = 20000
    state = kz.sample_state(small, n, rng)
    sig2 = small.a2 * KT / M
    big = sig2 > 1e-3 * sig2.max()
    assert np.allclose(state.alpha[big].var(axis=1), sig2[big], rtol=0.03)
    assert np.allclose(state.beta[big & (small.nu > 0)].var(axis=1), sig2[big & (small.nu > 0)], rtol=0.03)
    t = np.arange(0, 10, 0.1)
    F = kz.random_force(small, t, state)
    assert F.shape == (n, t.size)
    K0 = small.bath.K0
    # fluctuation-dissipation: <F(t) F(0)> = kT K_N(t)
    FF = (F * F[:, :1]).mean(axis=0)
    assert np.abs(FF - KT * small.bath.kernel(t)).max() < 5 * 1.5 * KT * K0 / np.sqrt(n)
    # the random force is uncorrelated with the initial velocity
    v0 = state.alpha.sum(axis=0)
    assert np.abs((F * v0[:, None]).mean(axis=0)).max() < 5 * np.sqrt(KT * K0 * KT / M / n)


@pytest.mark.parametrize("fx", ["small", "trapped"])
def test_gle_holds_per_trajectory(fx, request):
    s = request.getfixturevalue(fx)
    h = 2e-3
    t = np.arange(0, 6, h)
    traj = s.trajectory(t, n_traj=3, seed=22, random_force=True)
    assert traj.F.shape == traj.v.shape and traj.state is not None
    res = kz.gle_residual(s, traj)
    assert np.abs(res).max() < 1e-3 * np.abs(traj.F).max()
    # positions integrate v; they start at the sampled q(0) only when the trap makes it meaningful
    q0_expected = kz.initial_position(s, traj.state) if s.Omega > 0 else np.zeros(3)
    assert np.allclose(traj.q[:, 0], q0_expected)
    assert np.abs(np.gradient(traj.q, h, axis=1)[:, 3:-3] - traj.v[:, 3:-3]).max() < 1e-4


def test_spectral_density_histogram_has_the_right_weight(small):
    c, Jh = small.bath.spectral_density_histogram(bins=40)
    width = c[1] - c[0]
    assert (Jh * width).sum() == pytest.approx(0.5 * np.pi * (small.bath.k * small.bath.omega).sum(), rel=1e-12)
    # one node per bin, edges aligned with the grid: the binned density equals J at the nodes exactly
    dw = small.bath.omega[1] - small.bath.omega[0]
    c1, J1 = small.bath.spectral_density_histogram(bins=np.arange(small.bath.N + 1) * dw)
    assert np.allclose(c1, small.bath.omega) and np.allclose(J1, small.bath.J(c1), rtol=1e-10)
