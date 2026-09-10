# Theory: how the package computes what it computes

This note derives the formulas behind `kac_zwanzig`. It assumes classical mechanics and
basic statistical mechanics, and no prior exposure to generalized Langevin equations.
Equation numbers (1)–(10) refer to the README; the derivations here fill in the steps
between them.

## 1. The model

A particle of mass $M$, coordinate $q$ and momentum $p$, optionally in a harmonic trap of
frequency $\Omega$, is coupled to $N$ harmonic oscillators of unit mass with coordinates
$x_j$, momenta $p_j$ and frequencies $\omega_j$:

$$
H=\frac{p^2}{2M}+\frac12 M\Omega^2q^2+\sum_{j=1}^{N}\left[\frac{p_j^2}{2}+\frac{\omega_j^2}{2}\big(x_j-g_j\,q\big)^2\right],\qquad g_j=\frac{\sqrt{k_j}}{\omega_j}.
$$

The numbers $k_j\ge 0$ are the *kernel weights*. In the notation common in the literature,
$H_{\rm bath}=\sum_j[p_j^2/2m_j+\tfrac12 m_j\omega_j^2(x_j-c_jq/m_j\omega_j^2)^2]$, one has
$k_j=c_j^2/(m_j\omega_j^2)$; bath masses can be set to one because they only rescale $x_j$
and $p_j$. Setting $M=k_BT=1$ and measuring time in units of the bath's characteristic
frequency leaves, for the free particle in an Ohmic bath, a single dimensionless parameter.

## 2. Integrating out the bath: the generalized Langevin equation

Each oscillator obeys $\ddot x_j=-\omega_j^2(x_j-g_jq)$, a driven harmonic oscillator whose
solution is

$$
x_j(t)=x_j(0)\cos\omega_jt+\frac{p_j(0)}{\omega_j}\sin\omega_jt+\omega_jg_j\int_0^t\sin\omega_j(t-s)\,q(s)\,ds .
$$

Integrating the last term by parts,

$$
\omega_jg_j\int_0^t\sin\omega_j(t-s)\,q(s)\,ds=g_j\Big[q(t)-q(0)\cos\omega_jt-\int_0^t\cos\omega_j(t-s)\,\dot q(s)\,ds\Big].
$$

The particle obeys $M\ddot q=-M\Omega^2q+\sum_j\omega_j^2g_j(x_j-g_jq)$. Substituting $x_j(t)$,
the $g_j^2\omega_j^2q(t)$ terms cancel exactly against the counter-term in the Hamiltonian
(this is why the coupling is written as a difference), and what remains is

$$
M\ddot q(t)=-M\Omega^2q(t)-\int_0^tK_N(t-s)\,\dot q(s)\,ds+F(t),
$$

$$
K_N(t)=\sum_jk_j\cos\omega_jt,\qquad
F(t)=\sum_j\sqrt{k_j}\,\omega_j\Big[y_j(0)\cos\omega_jt+\frac{p_j(0)}{\omega_j}\sin\omega_jt\Big],\qquad y_j\equiv x_j-g_jq .
$$

This is exact for any $N$, any $\Omega$, and any initial conditions: the memory kernel is
the cosine sum $K_N$, and the "random" force $F$ is a deterministic function of the bath's
initial shifted coordinates $y_j(0)$ and momenta $p_j(0)$.

**Thermal initial conditions.** In the canonical ensemble $H$ is a sum of independent
quadratic terms in $p$, $p_j$ and $y_j$, so these are independent Gaussians with
$\langle p^2\rangle=Mk_BT$, $\langle p_j^2\rangle=k_BT$, $\langle y_j^2\rangle=k_BT/\omega_j^2$.
Hence

$$
\langle F(t)F(0)\rangle=\sum_jk_j\omega_j^2\langle y_j^2\rangle\cos\omega_jt=k_BT\,K_N(t),\qquad \langle F(t)\,v(0)\rangle=0 ,
$$

the fluctuation–dissipation theorem. The random force is uncorrelated with the initial
velocity because the bath is equilibrated *in the presence of the particle*: the $y_j$,
not the bare $x_j$, are the independent variables. (For a free particle with $q(0)=0$
the two coincide.)

## 3. The equation for the correlation function

Multiply the GLE by $v(0)=p(0)/M$ and average. $\langle F(t)v(0)\rangle=0$ removes the
random force; $\langle q(t)v(0)\rangle=\int_0^t\langle v(s)v(0)\rangle ds$ handles the trap.
With $C(t)=\langle v(t)v(0)\rangle/\langle v^2\rangle$,

$$
\dot C(t)=-\int_0^t\Gamma(t-s)\,C(s)\,ds,\qquad \Gamma(t)=\frac{K_N(t)+M\Omega^2}{M}.
$$

The trap contributes a *constant* to the memory kernel, because a harmonic potential is
equivalent to a bath oscillator of zero frequency. The equation is linear and homogeneous
in $C$, so it holds for $\langle v(t)v(0)\rangle$ unnormalized as well, and $\Gamma$ does
not depend on $k_BT$. Given $C$ and $\dot C$ it is a Volterra integral equation of the first
kind for $\Gamma$; differentiating once gives $\ddot C(t)=-\Gamma(t)-\int_0^t\Gamma(t-s)\dot C(s)ds$,
a second-kind equation, which shows the solution is unique and that $\Gamma(0)=-\ddot C(0)$.

## 4. Normal modes and the particle's spectral measure

In mass-weighted coordinates $\tilde Q_0=\sqrt M\,q$, $\tilde Q_j=x_j$ (momenta
$\tilde\Pi_0=p/\sqrt M$, $\tilde\Pi_j=p_j$) the Hamiltonian is
$H=\tfrac12|\tilde\Pi|^2+\tfrac12\tilde Q^{\mathsf T}D\tilde Q$ with the arrowhead matrix

$$
D_{00}=\frac{K_0+M\Omega^2}{M},\qquad D_{0j}=D_{j0}=-\sqrt{\frac{k_j}{M}}\,\omega_j,\qquad D_{jj}=\omega_j^2,\qquad K_0=\sum_jk_j .
$$

Diagonalize $D=U\Lambda U^{\mathsf T}$ ($U$ orthogonal, $\Lambda=\mathrm{diag}\,\nu_k^2$). The
normal modes $Q=U^{\mathsf T}\tilde Q$, $P=U^{\mathsf T}\tilde\Pi$ are independent
oscillators, $H=\sum_k\tfrac12(P_k^2+\nu_k^2Q_k^2)$. Writing $a_k\equiv U_{0k}$ for the
particle's component of mode $k$:

- Row $j$ of the eigenvector equation gives $U_{jk}=\sqrt{k_j/M}\,\omega_ja_k/(\omega_j^2-\nu_k^2)$.
- Row $0$ then gives the **secular equation** $\sum_jk_j/(\nu^2-\omega_j^2)=M$ for $\nu^2\neq0$.
  Its left side decreases monotonically between consecutive poles, so there is exactly one
  root in each interval $(\omega_j^2,\omega_{j+1}^2)$ and one above $\omega_N^2$ (interlacing).
- Normalizing the eigenvector gives the weights in closed form,
  $a_k^2=\big[1+\frac1M\sum_jk_j\omega_j^2/(\omega_j^2-\nu_k^2)^2\big]^{-1}$, with
  $\sum_ka_k^2=1$ because the first row of $U$ is a unit vector.
- For the free particle $\Omega=0$, $D$ has the null vector $(\sqrt M,g_1,\dots,g_N)$: the
  rigid translation of particle and bath together. Its weight is
  $a_0^2=\big[1+\frac1M\sum_jk_j/\omega_j^2\big]^{-1}$, of order $1/N$ for the grids used here.

The package computes $(\nu_k,a_k^2)$ with a dense symmetric eigensolve (`numpy.linalg.eigh`),
which costs $O(N^3)$: under a second at $N=2000$. For $N\gtrsim10^4$ the secular equation
can be solved root by root in $O(N^2)$ instead; this is not implemented because, as the
frequency-grid analysis shows, a good grid at $N=2000$ is already closer to the continuum
than a poor grid at $N=10^5$.

## 5. Exact correlation functions

Each mode evolves as $P_k(t)=P_k(0)\cos\nu_kt-\nu_kQ_k(0)\sin\nu_kt$, and the particle's
velocity is $v=\tilde\Pi_0/\sqrt M=\frac1{\sqrt M}\sum_ka_kP_k(t)$. In the canonical
ensemble $\langle P_k^2\rangle=k_BT$, $\langle Q_k^2\rangle=k_BT/\nu_k^2$, independent, so

$$
C(t)=\sum_ka_k^2\cos\nu_kt,\qquad \dot C(t)=-\sum_ka_k^2\nu_k\sin\nu_kt,\qquad \ddot C(0)=-\sum_ka_k^2\nu_k^2=-D_{00}=-\Gamma(0).
$$

**Consistency check in closed form.** With
$\int_0^t\cos\omega(t-s)\cos\nu s\,ds=(\omega\sin\omega t-\nu\sin\nu t)/(\omega^2-\nu^2)$,
the convolution $\int_0^t\Gamma(t-s)C(s)ds$ is a double sum over bath and normal modes whose
coefficients vanish term by term by the secular equation and by the resolvent identity
$\sum_ka_k^2/(\omega_j^2-\nu_k^2)=0$. The test suite evaluates it numerically and finds
$|M\dot C+\int K_NC|<10^{-12}$ for every grid and for the trapped particle. In Laplace space
the same statement reads $\tilde C(s)=1/\big(s+\tilde\Gamma(s)\big)$ with
$\tilde K_N(s)=\sum_jk_js/(s^2+\omega_j^2)$, whose poles are again the secular equation.

**The Debye bath** has $J(\omega)=2\lambda\omega_c\omega/(\omega_c^2+\omega^2)$, hence
$K_\infty(t)=2\lambda e^{-\omega_ct}$ and $\tilde K_\infty(s)=2\lambda/(s+\omega_c)$, which is
rational; the continuum VACF is therefore elementary,
$\tilde C(s)=(s+\omega_c)/(s^2+\omega_cs+2\lambda/M)$, i.e. two exponentials with
$s_\pm=\tfrac12\big(-\omega_c\pm\sqrt{\omega_c^2-8\lambda/M}\big)$ (`debye_vacf_continuum`).
For the Ohmic bath no such closed form exists; the finite bath is the exact object.

## 6. Exact trajectories and their sampling

Thermal initial conditions are independent Gaussians in the normal modes,
$P_k\sim\mathcal N(0,k_BT)$, $Q_k\sim\mathcal N(0,k_BT/\nu_k^2)$. Substituting the mode
solution into $v=\frac1{\sqrt M}\sum_ka_kP_k(t)$,

$$
v(t)=\sum_k\big[\alpha_k\cos\nu_kt+\beta_k\sin\nu_kt\big],\qquad
\alpha_k=\frac{a_kP_k(0)}{\sqrt M},\quad\beta_k=-\frac{a_k\nu_kQ_k(0)}{\sqrt M},
$$

so $\alpha_k$ and $\beta_k$ are independent $\mathcal N(0,a_k^2k_BT/M)$: the $\nu_k^2$ in
$\beta_k$ cancels the $1/\nu_k^2$ in the variance of $Q_k$. The zero mode's $Q_0$ is the
improper flat direction of the free particle, but it multiplies $\nu_0=0$, so nothing
improper enters. Sampling the canonical ensemble *is* drawing these amplitudes; a batch of
trajectories on any output grid is two matrix products, with no time step and no
integration error. The acceleration is the term-by-term derivative and the position the
term-by-term integral (the zero mode gives a uniform drift $\alpha_0t$).

**The random force** needs the oscillators' own initial coordinates. Drawing the full mode
state $(P_k,Q_k)$ and transforming back with $\tilde Q=UQ$, $\tilde\Pi=UP$ gives
$x_j(0)$, $p_j(0)$ and $q(0)$, hence $y_j(0)=x_j(0)-g_jq(0)$ and $F(t)$ from the formula in
§2. The package verifies, trajectory by trajectory, that
$M\dot v+M\Omega^2q+\int_0^tK_N(t-s)v(s)ds=F(t)$ to the accuracy of the trapezoid rule
used for the convolution, and statistically that $\langle F(t)F(0)\rangle=k_BT\,K_N(t)$.

## 7. Sampled correlation functions and their noise

An estimate from $n$ independent trajectories is $\hat C(t)=R(t)/R(0)$ with
$R(t)=\frac1n\sum_nv_n(t)v_n(0)$ (or $R(t)/(k_BT/M)$, the "exact" normalization). Because
$v(t)$ is a Gaussian process, Isserlis' theorem gives the raw covariance
$\mathrm{Cov}[R(t),R(t')]=\langle v^2\rangle^2\,[C(t-t')+C(t)C(t')]/n$, and propagating the
ratio to first order,

$$
\mathrm{Var}\big[\hat C(t)\big]=\frac{1-C(t)^2}{n},\qquad
\mathrm{Cov}\big[\hat C(t),\hat C(t')\big]=\frac{C(t-t')-C(t)C(t')}{n}
$$

for the sample normalization, and $(1+C^2)/n$, $[C(t-t')+C(t)C(t')]/n$ for the exact one.
The noise is correlated over the correlation time of $C$ itself: it is smooth, not white,
which is why finite differencing amplifies it far less than a white-noise estimate would
suggest.

**Collapsed evaluation.** Substituting the mode expansion,
$R(t)=\sum_k[\bar\alpha_k\cos\nu_kt+\bar\beta_k\sin\nu_kt]$ with
$\bar\alpha_k=\frac1n\sum_n\alpha_{kn}v_n(0)$, so the estimator needs only the
$(N+1)\times n$ amplitude matrix, never the trajectories: cost $O(N(n+n_t))$. The same
contraction gives the derivative estimators $\langle\dot v(t)v(0)\rangle$ and
$-\langle v(t)\dot v(0)\rangle$ for free.

**Groups and bootstrap.** The trajectories are split into independent groups with a
complete raw estimate kept per group. The group spread gives the standard error of the
pooled estimate; resampling groups with replacement and re-forming the pooled ratio gives
bootstrap replicate curves that carry the time correlation above and treat the ratio
normalization exactly. Replicates can be pushed through any downstream calculation, such
as a kernel inversion, to obtain error bars on its output.

**Why not time averages.** The finite bath is integrable. Along one trajectory the time
average of $v(s+t)v(s)$ converges to $\sum_ka_k^2E_k\cos\nu_kt\cdot k_BT/M$ with $E_k$ the
(conserved) mode energies in units of $k_BT$, exponential variates fixed once and for all.
Its expectation is $C(t)$, but its variance, $(k_BT/M)^2\sum_ka_k^4\cos^2\nu_kt$, does not
shrink with the trajectory length: one trajectory is worth about $1/\sum_ka_k^4$
independent samples and no more. Real MD escapes this because it is chaotic; here the
ensemble average over independent initial conditions is the estimator that converges.

## 8. Choosing the frequencies: a summary

Both grids are quadrature rules for $K_\infty(t)=\int_0^\infty f(\omega)\cos\omega t\,d\omega$,
$f=\frac2\pi J/\omega$. The midpoint-uniform grid, $\omega_j=(j-\tfrac12)\Delta\omega$,
$k_j=f(\omega_j)\Delta\omega$, has by Poisson summation the error
$K_N-K_\infty=\sum_{m\neq0}(-1)^mK_\infty(t-2\pi m/\Delta\omega)$: a sum of smooth,
alternating images of the kernel, about $2K_\infty(2\pi/\Delta\omega)$ in the window
$t<\pi/\Delta\omega$, second order in $\Delta\omega$ when $f$ has a kink at $\omega=0$ (Ohmic) and
exponentially small when $f$ is analytic (Debye). The inverse-CDF grid places equal weights
at the quantiles of $f$; it reproduces $K(0)$ exactly and never truncates, but the highest
few modes have scrambled phases at any given $t$ and leave a ripple that decays only as
$N^{-1}$ (Ohmic) or $N^{-3/4}$ (Debye). In every case the particle's $C_N$ converges much
faster than $K_N$, because a bath mode far above the particle's response band enters
$C$ with weight $\approx k_j/(M\omega_j^2)$. The full analysis with measurements is in
`frequency_grids.pdf`.
