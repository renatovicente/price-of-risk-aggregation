"""Solver for the Katz-Bonacich system of the paper, plus self-tests.

Investor a maximises U_a = -exp[-theta_a W_a - lam_a (W_a - sum_j A_aj W_j)]
with A row-stochastic (Assumption 1). Monetary demands are pi_a = c_a Sigma^-1 r_e
and the first-order conditions under Nash behaviour give equation (3),

    [ D_{theta+lam} - D_lam A ] c = 1,        theta_bar = m / sum_a c_a.

so that c is the Katz-Bonacich centrality of the influence network.

The self-tests below check the three structural facts the paper relies on:

  (i)   the matrix is strictly diagonally dominant whenever some theta_a > 0,
        which is what gives uniqueness in Theorem 1 and Proposition 3;
  (ii)  the solve agrees with the Neumann series c = sum_k (D^-1 D_lam A)^k D^-1 1,
        which is the Katz-Bonacich reading of the same object;
  (iii) the closed form of Proposition 2 is reproduced to machine precision.
"""
import numpy as np


def system_matrix(A, theta, lam):
    """M = D_{theta+lam} - D_lam A, the matrix of equation (3)."""
    m = len(A)
    theta = np.atleast_1d(theta) * np.ones(m)
    lam = np.atleast_1d(lam) * np.ones(m)
    return np.diag(theta + lam) - np.diag(lam) @ A


def bonacich(A, theta, lam):
    """Demand scales c solving equation (3)."""
    return np.linalg.solve(system_matrix(A, theta, lam), np.ones(len(A)))


def theta_bar(A, theta, lam):
    """Effective market risk-aversion coefficient, theta_bar = m / sum_a c_a."""
    return len(A) / bonacich(A, theta, lam).sum()


def dtheta_bar(A, theta, lam):
    """d(theta_bar)/d(lam) for a common lam, computed analytically.

    With M(lam) = D_theta + lam (I - A) and c = M^-1 1,
        c'       = -M^-1 (I - A) c
        theta_bar' = -m (1' c') / (1' c)^2
    """
    m = len(A)
    M = system_matrix(A, theta, lam)
    c = np.linalg.solve(M, np.ones(m))
    cp = -np.linalg.solve(M, (np.eye(m) - A) @ c)
    return -m * cp.sum() / c.sum() ** 2


def row_stochastic(m, rng):
    """Random A satisfying Assumption 1: zero diagonal, rows summing to one."""
    P = rng.random((m, m))
    np.fill_diagonal(P, 0.0)
    return P / P.sum(1, keepdims=True)


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    print("(i) strict diagonal dominance, hence uniqueness")
    for m in [5, 50, 400]:
        A = row_stochastic(m, rng)
        th = np.clip(1.0 + 0.6 * rng.normal(size=m), 0.15, None)
        lam = rng.random(m) * 3
        M = system_matrix(A, th, lam)
        slack = np.abs(np.diag(M)) - (np.abs(M).sum(1) - np.abs(np.diag(M)))
        print(f"    m={m:<4} min row slack = {slack.min():.6f}  "
              f"(equals min theta_a = {th.min():.6f})")

    print("\n(ii) the solve equals the Katz-Bonacich Neumann series")
    m = 60
    A = row_stochastic(m, rng)
    th = np.clip(1.0 + 0.5 * rng.normal(size=m), 0.15, None)
    for lam in [0.5, 2.0, 10.0]:
        D = np.diag(th + lam)
        G = np.linalg.solve(D, lam * A)              # D^-1 D_lam A
        c_series = np.linalg.solve(D, np.ones(m))
        term = c_series.copy()
        for _ in range(4000):
            term = G @ term
            c_series = c_series + term
        c_direct = bonacich(A, th, lam)
        print(f"    lam={lam:<5} max |series - solve| = "
              f"{np.abs(c_series - c_direct).max():.3e}")

    print("\n(iii) closed form of Proposition 2")
    B = np.array([[0., 1., 0.], [1., 0., 0.], [1., 0., 0.]])
    TH = np.array([0.5, 1.0, 2.0])
    print(f"    {'lam':>6}{'solver':>14}{'closed form':>14}"
          f"{'derivative':>14}{'closed form':>14}")
    for lam in [0.0, 0.132, 1 / 3, 1.0, 2.0]:
        closed = 3 * (lam + 2) * (3 * lam + 1) / (12 * lam ** 2 + 24 * lam + 7)
        dclosed = -3 * (12 * lam ** 2 + 6 * lam - 1) / (12 * lam ** 2 + 24 * lam + 7) ** 2
        print(f"    {lam:>6.3f}{theta_bar(B, TH, lam):>14.9f}{closed:>14.9f}"
              f"{dtheta_bar(B, TH, lam):>14.9f}{dclosed:>14.9f}")
    root = (np.sqrt(21) - 3) / 12
    print(f"\n    sign change of the derivative at (sqrt(21)-3)/12 = {root:.9f}")
    print(f"    theta_bar'(0)   = {dtheta_bar(B, TH, 0.0):.9f}   (= 3/49 = {3/49:.9f})")
    print(f"    theta_bar'(root)= {dtheta_bar(B, TH, root):.3e}   (zero)")
    print(f"    theta_bar(1/3)  = {theta_bar(B, TH, 1/3):.9f}   (= 6/7 = {6/7:.9f})")
