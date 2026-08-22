"""General characterisation of the network effect (Theorems 2 and 3).

Two limits pin down the whole comparative static in lambda, with a common
comparison intensity and A row stochastic.

MARGINAL EFFECT AT ZERO. With t = D_theta^{-1} 1 the vector of risk tolerances,

    theta_bar'(0) = m t'(I - A) t / (1't)^2.

If A is doubly stochastic this is a Dirichlet form,

    t'(I - A) t = (1/2) sum_ij A_ij (t_i - t_j)^2 >= 0,

so the marginal effect is non-negative and vanishes exactly when tolerance is
constant across every link. For merely row-stochastic A the sign is free.

INTENSE COMPARISON. Let C_1..C_r be the recurrent classes of A, pi^(k) the
stationary distribution on C_k, and h_k(a) the probability of absorption into
C_k starting from a. Then

    lim_{lam->inf} theta_bar(lam) = m / sum_k (1'h_k) / (pi^(k)' theta).

With a single recurrent class this is pi'theta. The three results of the paper
are the three cases:

    mean field    pi -> uniform            limit = E[theta]   (arithmetic)
    reversal      pi concentrated on the   limit = pi'theta < H(theta)
                  least risk averse
    homophily     one class per block,     limit = H(theta) = theta_bar(0)
                  theta constant within

UNIFORM BOUND. c_a <= 1/theta_min for every lam, which is what makes the
mean-field passage to the limit rigorous: if c* = max_a c_a is attained at a*,
row a* gives (theta_a* + lam) c* = 1 + lam sum_j A_a*j c_j <= 1 + lam c*, hence
theta_a* c* <= 1.
"""
import numpy as np
from scipy.linalg import block_diag


def capacities(A, theta, lam):
    m = len(A)
    return np.linalg.solve(np.diag(theta) + lam * (np.eye(m) - A), np.ones(m))


def theta_bar(A, theta, lam):
    return len(A) / capacities(A, theta, lam).sum()


def dtheta_bar_at_zero(A, theta):
    """Theorem 2: m t'(I - A) t / (1't)^2."""
    m = len(A)
    t = 1.0 / np.asarray(theta, float)
    return m * (t @ (np.eye(m) - A) @ t) / t.sum() ** 2


def dirichlet_form(A, theta):
    """(1/2) sum_ij A_ij (t_i - t_j)^2; equals t'(I-A)t when A is doubly stochastic."""
    t = 1.0 / np.asarray(theta, float)
    return 0.5 * ((A * (t[:, None] - t[None, :]) ** 2).sum())


def stationary(A):
    """Left eigenvector of eigenvalue one, normalised to sum to one."""
    w, V = np.linalg.eig(A.T)
    p = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return p / p.sum()


def uniform_row_distance(A):
    """delta_m = max_a || A_a. - (1/m) 1 ||_1, the primitive mixing condition."""
    m = len(A)
    return np.abs(A - 1.0 / m).sum(1).max()


def sinkhorn(m, rng, iters=4000):
    """A doubly stochastic with zero diagonal, by alternating normalisation."""
    A = rng.random((m, m)) + 0.05
    np.fill_diagonal(A, 0.0)
    for _ in range(iters):
        A /= A.sum(1, keepdims=True)
        A /= A.sum(0, keepdims=True)
    return A / A.sum(1, keepdims=True)


def row_stochastic(m, rng):
    A = rng.random((m, m))
    np.fill_diagonal(A, 0.0)
    return A / A.sum(1, keepdims=True)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("(1) Theorem 2 against a central difference of the solver")
    for _ in range(4):
        m = int(rng.integers(3, 9))
        A = row_stochastic(m, rng)
        th = 0.3 + rng.random(m) * 2
        h = 1e-6
        num = (theta_bar(A, th, h) - theta_bar(A, th, -h)) / (2 * h)
        print(f"    m={m}  formula={dtheta_bar_at_zero(A, th):+.9f}"
              f"   central difference={num:+.9f}")

    print("\n(2) doubly stochastic: the Dirichlet identity, hence a non-negative effect")
    for _ in range(3):
        m = int(rng.integers(4, 9))
        A = sinkhorn(m, rng)
        th = 0.2 + rng.random(m) * 3
        t = 1.0 / th
        print(f"    m={m}  t'(I-A)t={t @ (np.eye(m) - A) @ t:.12f}"
              f"   (1/2)sum A_ij (t_i-t_j)^2={dirichlet_form(A, th):.12f}")

    B = np.array([[0., 1., 0.], [1., 0., 0.], [1., 0., 0.]])
    TH = np.array([0.5, 1.0, 2.0])
    print(f"\n    without double stochasticity the sign is free. Proposition 2's")
    print(f"    network has column sums {B.sum(0)} and a POSITIVE marginal effect,")
    print(f"    theta_bar'(0) = {dtheta_bar_at_zero(B, TH):+.9f} (= 3/49 = {3/49:.9f});")
    print(f"    it is the later decline that reverses the sign. For a NEGATIVE")
    print(f"    marginal effect take instead")
    Bneg = np.array([[0., 1., 0.], [1., 0., 0.], [0., 1., 0.]])
    THneg = np.array([1.0, 1.0, 2.0])
    tneg = 1.0 / THneg
    print(f"    A rows {Bneg.tolist()}, theta = {THneg}, columns {Bneg.sum(0)}")
    print(f"    t'(I-A)t = {tneg @ (np.eye(3) - Bneg) @ tneg:+.9f} (= -1/4)")
    print(f"    theta_bar'(0) = {dtheta_bar_at_zero(Bneg, THneg):+.9f} (= -3/25 ="
          f" {-3/25:+.9f})")
    pineg = stationary(Bneg)
    print(f"    pi = {np.round(pineg, 6)}, pi'theta = {pineg @ THneg:.6f}"
          f" < H(theta) = {3 / np.sum(1 / THneg):.6f}")
    print(f"    so this one falls monotonically:", end=" ")
    print(", ".join(f"theta_bar({l:g})={theta_bar(Bneg, THneg, l):.6f}"
                    for l in [0, 1, 1e4]))

    print("\n(3) Theorem 3, single recurrent class")
    for _ in range(3):
        m = int(rng.integers(3, 8))
        A = row_stochastic(m, rng)
        th = 0.3 + rng.random(m) * 2.5
        print(f"    m={m}  theta_bar(1e7)={theta_bar(A, th, 1e7):.9f}"
              f"   pi'theta={stationary(A) @ th:.9f}")

    print("\n(4) Theorem 3 explains the counterexample")
    pi = stationary(B)
    H = 3 / np.sum(1 / TH)
    print(f"    pi = {np.round(pi, 6)}, zero weight on the most risk averse investor")
    print(f"    pi'theta = {pi @ TH:.9f} < H(theta) = {H:.9f} = theta_bar(0)")
    for lam in [0, 1, 10, 100, 1e4, 1e7]:
        print(f"      lam={lam:<10.0f} theta_bar={theta_bar(B, TH, lam):.9f}")

    print("\n(5) Theorem 3 with transient states: weights are absorption probabilities")
    A = np.zeros((6, 6))
    A[0, 1] = A[1, 0] = 1.0                       # recurrent class C1 = {0,1}
    A[2, 3] = A[3, 2] = 1.0                       # recurrent class C2 = {2,3}
    A[4, 0], A[4, 2] = 0.7, 0.3                   # transient
    A[5, 1], A[5, 3], A[5, 4] = 0.2, 0.5, 0.3     # transient
    th = np.array([0.4, 0.9, 2.0, 1.4, 3.0, 0.3])
    T, R1, R2 = [4, 5], [0, 1], [2, 3]
    h1 = np.zeros(6)
    h1[R1] = 1.0
    h1[T] = np.linalg.solve(np.eye(len(T)) - A[np.ix_(T, T)], A[np.ix_(T, R1)].sum(1))
    h2 = 1.0 - h1
    v1 = stationary(A[np.ix_(R1, R1)]) @ th[R1]
    v2 = stationary(A[np.ix_(R2, R2)]) @ th[R2]
    pred = len(A) / (h1.sum() / v1 + h2.sum() / v2)
    print(f"    absorption into C1: {np.round(h1, 4)}, total {h1.sum():.4f}")
    print(f"    absorption into C2: {np.round(h2, 4)}, total {h2.sum():.4f}")
    print(f"    predicted={pred:.9f}   theta_bar(1e8)={theta_bar(A, th, 1e8):.9f}")

    print("\n(6) homophily is the case where the two limits coincide")
    blocks = [np.array([[0., 1.], [1., 0.]]),
              np.array([[0., 1., 0.], [1., 0., 0.], [0., 1., 0.]]),
              np.array([[0., 1.], [1., 0.]])]
    Ab = block_diag(*blocks)
    thb = np.concatenate([np.full(2, 0.5), np.full(3, 2.0), np.full(2, 1.2)])
    Hb = len(Ab) / np.sum(1 / thb)
    print(f"    theta constant within blocks: theta_bar(0)=H(theta)={Hb:.9f}"
          f"   theta_bar(1e7)={theta_bar(Ab, thb, 1e7):.9f}")
    thh = np.concatenate([[0.4, 0.9], [2.0, 0.6, 1.3], [1.1, 1.7]])
    per = [stationary(b) @ t for b, t in
           zip(blocks, [thh[:2], thh[2:5], thh[5:]])]
    predb = len(Ab) / sum(len(b) / v for b, v in zip(blocks, per))
    print(f"    theta varying within blocks:  predicted={predb:.9f}"
          f"   theta_bar(1e7)={theta_bar(Ab, thh, 1e7):.9f}")

    print("\n(7) uniform bound c_a <= 1/theta_min, and the primitive mixing condition")
    for m in [15, 40]:
        A = row_stochastic(m, rng)
        th = 0.2 + rng.random(m) * 2
        worst = max(capacities(A, th, lam).max() for lam in [0, 0.5, 2, 10, 1e4])
        print(f"    m={m:<3} max over lam of max_a c_a = {worst:.6f}"
              f"   1/theta_min = {1 / th.min():.6f}")
    print(f"    {'m':>6}{'delta_m (leave-one-out)':>26}{'2/m':>12}")
    for m in [10, 100, 1000]:
        A = (np.ones((m, m)) - np.eye(m)) / (m - 1)
        print(f"    {m:>6}{uniform_row_distance(A):>26.8f}{2 / m:>12.8f}")
