"""Sign reversal in ARBITRARILY LARGE networks (Remark 5).

Table 1 shows reversal becoming rare as m grows under a dense i.i.d.-weight
design. That is a property of MIXING, not of SIZE. When A is block diagonal the
system [D_{theta+lam} - D_lam A] c = 1 decouples: each block solves its own
system, and theta_bar = (m^-1 sum_a c_a)^-1 is the reciprocal of the
population-average capacity. Replicating an identical block leaves that average,
and hence theta_bar, unchanged. So replicating the counterexample of
Proposition 2 k times gives a theta_bar(lam) that does not depend on k.

Note the contrast with Proposition 3 (homophily): there the blocks also
decouple, but theta_a is constant WITHIN each block, which kills the mechanism.
Here the heterogeneity is internal to the block and the absence of ties is
external: same structure, opposite conclusions.
"""
import numpy as np
from scipy.linalg import block_diag

# counterexample of Proposition 2
B  = np.array([[0.,1.,0.],[1.,0.,0.],[1.,0.,0.]])
TH = np.array([0.5,1.0,2.0])

def theta_bar(A, th, lam):
    m=len(A)
    return m/np.linalg.solve(np.diag(th)+lam*(np.eye(m)-A), np.ones(m)).sum()

def replicate(k):
    return block_diag(*[B]*k), np.tile(TH,k)

def mix(A, eps):
    """A_eps = (1-eps) A_block + eps A_complete, the complete one leave-one-out.

    Both terms are already row stochastic, so A_eps is too; the assertion below
    records that rather than renormalising.
    """
    m=len(A)
    A=(1-eps)*A + eps*(np.ones((m,m))-np.eye(m))/(m-1)
    assert np.allclose(A.sum(1), 1.0)
    return A

if __name__=="__main__":
    print("Block of Proposition 2 replicated k times, lam = 1\n")
    print(f"{'k':>4}{'m':>6}{'theta_bar(0)':>15}{'theta_bar(1)':>15}{'reversed':>10}")
    for k in [1,2,5,20,100,1000]:
        A,th=replicate(k)
        t0,t1=theta_bar(A,th,0.0), theta_bar(A,th,1.0)
        print(f"{k:>4}{3*k:>6}{t0:>15.6f}{t1:>15.6f}{str(bool(t1<t0)):>10}")
    print("\n-> exactly 6/7 and 0.837209 for every k: the system decouples.\n")

    print("Robustness: blocks linked by a mixing fraction eps, m = 30\n")
    print(f"{'eps':>6}{'theta_bar(0)':>16}{'theta_bar(1)':>16}"
          f"{'to 4 dp':>11}{'reversed':>10}")
    A0,th=replicate(10)
    for eps in [0.0,0.01,0.05,0.10,0.20,0.50]:
        t0,t1=theta_bar(mix(A0,eps),th,0.0), theta_bar(mix(A0,eps),th,1.0)
        print(f"{eps:>6.2f}{t0:>16.7f}{t1:>16.7f}{t1:>11.4f}{str(bool(t1<t0)):>10}")
    print("\n-> reversal survives to about eps = 0.1; what destroys it is mixing,")
    print("   not the number of investors.")
