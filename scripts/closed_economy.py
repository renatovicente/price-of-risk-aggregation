"""The closed one-period economy (Section 3).

Primitives are payoffs, quantities and the riskless rate, not return moments.
There are n risky assets with terminal payoffs X ~ N(mu, V), V positive
definite, in exogenous supply of nq shares; a riskless asset with gross return
Rf; and m investors with initial wealth W0_a who choose share holdings z_a, so

    W_a = Rf (W0_a - p'z_a) + X'z_a.

With the preferences of equation (1) the random part of the exponent is -X'q_a,
where q_a = (theta_a + lam_a) z_a - lam_a sum_j A_aj z_j is the same bracket as
before, now in shares. The exponent is strictly convex in z_a because
dq_a/dz_a = (theta_a + lam_a) I, so the optimum is interior and solves

    V q_a = mu - Rf p.

Hence z_a = c_a V^{-1}(mu - Rf p) with the SAME system

    [ D_{theta+lam} - D_lam A ] c = 1,        theta_M = 1 / sum_a c_a,

and clearing sum_a z_a = nq gives mu - Rf p = theta_M V nq, that is

    p = ( mu - theta_M V nq ) / Rf.

The fixed point is trivial: theta_M depends only on (theta, lam, A) and not on
prices, so prices are explicit. Prices are positive exactly when
mu > theta_M V nq componentwise.

Consequences verified below:

  (i)   the first-order conditions reproduce the same c as the conditional model;
  (ii)  the conditional relation r_e = theta_M W_M Sigma w is recovered, with
        W_M = p'nq the ENDOGENOUS market value and Sigma = D_p^{-1} V D_p^{-1};
  (iii) the market price of risk is exactly theta_M W_M, and the market Sharpe
        ratio is theta_M sqrt(nq' V nq): proportional to theta_bar, with the
        constant of proportionality on the payoff side. Every comparative static
        in theta_bar is therefore a comparative static in the price of risk;
  (iv)  if the riskless asset is in zero net supply, Rf adjusts so that
        p'nq = sum_a W0_a, and the market value equals aggregate initial wealth;
  (v)   sign reversal lowers theta_bar, hence raises prices and lowers premia.
"""
import numpy as np


def demand_scales(A, theta, lam):
    m = len(A)
    return np.linalg.solve(np.diag(theta) + lam * (np.eye(m) - A), np.ones(m))


def equilibrium(A, theta, lam, mu, V, nq, Rf):
    """Equilibrium prices and share holdings."""
    c = demand_scales(A, theta, lam)
    thM = 1.0 / c.sum()
    p = (mu - thM * (V @ nq)) / Rf
    z = np.outer(c, np.linalg.solve(V, mu - Rf * p))
    return dict(c=c, thM=thM, p=p, z=z)


def riskless_rate_zero_net_supply(theta_M, mu, V, nq, W0):
    """Rf that clears the riskless market, so that p'nq = sum_a W0_a."""
    return (mu @ nq - theta_M * (nq @ V @ nq)) / W0.sum()


def objective(za, a, A, theta, lam, mu, V, nq, Rf, p, z, W0):
    """Exponent of -E[U_a], as a function of investor a's own holdings."""
    m = len(A)
    q = (theta[a] + lam) * za - lam * sum(A[a, j] * z[j] for j in range(m))
    det = (-(theta[a] + lam) * (Rf * W0[a] - Rf * p @ za)
           + lam * sum(A[a, j] * (Rf * W0[j] - Rf * p @ z[j]) for j in range(m)))
    return det - q @ mu + 0.5 * q @ V @ q


if __name__ == "__main__":
    from scipy.optimize import minimize
    rng = np.random.default_rng(4)
    m, n = 5, 3
    Bm = rng.random((n, n)); V = Bm @ Bm.T + 0.6 * np.eye(n)
    mu = 3.0 + rng.random(n); nq = 0.5 + rng.random(n); Rf = 1.03
    th = 0.3 + rng.random(m) * 1.5
    A = rng.random((m, m)); np.fill_diagonal(A, 0.0); A /= A.sum(1, keepdims=True)
    W0 = 1.0 + rng.random(m) * 3
    lam = 0.8
    eq = equilibrium(A, th, lam, mu, V, nq, Rf)
    p, z, thM = eq["p"], eq["z"], eq["thM"]

    print("(i) the first-order conditions, checked against direct optimisation")
    for a in [0, 2, 4]:
        r = minimize(objective, np.zeros(n), method="BFGS", tol=1e-14,
                     args=(a, A, th, lam, mu, V, nq, Rf, p, z, W0))
        print(f"    investor {a}: max |numerical - closed form| = "
              f"{np.abs(r.x - z[a]).max():.2e}")

    print("\n(ii) clearing, positivity, and price independence of theta_M")
    print(f"    max |sum_a z_a - nq| = {np.abs(z.sum(0) - nq).max():.2e}")
    print(f"    prices {np.round(p, 6)}, all positive: {bool((p > 0).all())}")
    print(f"    theta_M = {thM:.9f}, a function of (theta, lam, A) alone")

    print("\n(iii) the conditional relation is recovered")
    Sigma = V / np.outer(p, p)
    re = (mu - Rf * p) / p
    WM = p @ nq
    w = p * nq / WM
    print(f"    r_e                  = {np.round(re, 9)}")
    print(f"    theta_M W_M Sigma w  = {np.round(thM * WM * Sigma @ w, 9)}")
    print(f"    max difference       = {np.abs(re - thM * WM * Sigma @ w).max():.2e}")

    print("\n(iv) the price of risk and the market Sharpe ratio")
    ERM = (mu @ nq) / WM - Rf
    VarM = (nq @ V @ nq) / WM ** 2
    print(f"    (E[R_M] - Rf)/Var(R_M) = {ERM / VarM:.9f}   theta_M W_M = {thM * WM:.9f}")
    print(f"    Sharpe ratio           = {ERM / np.sqrt(VarM):.9f}"
          f"   theta_M sqrt(nq'V nq) = {thM * np.sqrt(nq @ V @ nq):.9f}")

    print("\n(v) zero net supply of the riskless asset makes Rf endogenous")
    Rf_star = riskless_rate_zero_net_supply(thM, mu, V, nq, W0)
    eq2 = equilibrium(A, th, lam, mu, V, nq, Rf_star)
    riskless = W0 - eq2["z"] @ eq2["p"]
    print(f"    Rf* = {Rf_star:.9f}   p*'nq = {eq2['p'] @ nq:.9f}"
          f"   sum_a W0_a = {W0.sum():.9f}")
    print(f"    riskless holdings sum to {riskless.sum():.2e}")

    print("\n(vi) sign reversal raises prices and lowers premia")
    Bx = np.array([[0., 1., 0.], [1., 0., 0.], [1., 0., 0.]])
    THx = np.array([0.5, 1.0, 2.0])
    nq3 = np.array([0.8, 0.5, 0.6])
    B3 = rng.random((3, 3)); V3 = B3 @ B3.T + 0.6 * np.eye(3)
    mu3 = 3.0 + rng.random(3)
    print(f"    {'lam':>7}{'theta_bar':>13}{'p_1':>11}{'p_2':>11}{'p_3':>11}"
          f"{'premium_1':>12}{'Sharpe':>10}")
    for l in [0.0, 0.5, 1.0, 5.0, 100.0]:
        e3 = equilibrium(Bx, THx, l, mu3, V3, nq3, 1.03)
        pp = e3["p"]
        print(f"    {l:>7.1f}{3 * e3['thM']:>13.6f}{pp[0]:>11.6f}{pp[1]:>11.6f}"
              f"{pp[2]:>11.6f}{(mu3[0] - 1.03 * pp[0]) / pp[0]:>12.6f}"
              f"{e3['thM'] * np.sqrt(nq3 @ V3 @ nq3):>10.6f}")
