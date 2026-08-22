"""Theorem 1 (neutrality) and Proposition 1 (mean-field aggregation).

CARA economy with a normalised peer benchmark (A row-stochastic, Assumption 1)
and Nash behaviour. Monetary demands are pi_a = c_a Sigma^-1 r_e with

    [ D_{theta+lam} - D_lam A ] c = 1.

Verified here:

  (1) NEUTRALITY (exact; any row-stochastic A, any profile lam_a): if
      theta_a = theta for every a, then c = (1/theta) 1 and theta_bar = theta.
      Relative wealth concerns do not move the effective coefficient.

  (2) AGGREGATION (mean field): with heterogeneous theta_a,
          theta_bar(lam) = H(theta + lam) - lam,   H(x) = 1/E[1/x].

  (3) d(theta_bar)/d(lam) = CV^2[1/(theta+lam)] >= 0, with equality if and only
      if theta is degenerate. Comparison raises the effective coefficient at a
      rate equal to the squared coefficient of variation of shifted risk
      tolerance.

  (4) theta_bar interpolates from the HARMONIC mean (lam = 0) to the ARITHMETIC
      mean (lam -> infinity), with theta_bar = E[theta] - Var(theta)/lam + O(lam^-2).
"""
import numpy as np

def H(x): return 1.0/np.mean(1.0/np.asarray(x,float))

def solve_c(A, theta, lam):
    m=len(A)
    theta=np.atleast_1d(theta)*np.ones(m); lam=np.atleast_1d(lam)*np.ones(m)
    return np.linalg.solve(np.diag(theta+lam)-np.diag(lam)@A, np.ones(m))

def theta_bar(A, theta, lam):
    c=solve_c(A,theta,lam); return len(A)/c.sum()

def theta_bar_meanfield(theta, lam): return H(np.asarray(theta)+lam)-lam

def dtheta_dlam(theta, lam):
    t=1.0/(np.asarray(theta)+lam)          # risk tolerance
    return t.var()/t.mean()**2             # CV^2

def row_stochastic(m, rng):
    P=rng.random((m,m)); np.fill_diagonal(P,0.0); return P/P.sum(1,keepdims=True)

if __name__=="__main__":
    rng=np.random.default_rng(1)
    print("(1) neutrality: homogeneous theta, arbitrary lambda_a")
    m=400; A=row_stochastic(m,rng)
    for lab,lam in [("homogeneous",0.7),("random",rng.random(m)*2),
                    ("decreasing in wealth",np.clip(0.5*((1-rng.random(m))**(-1/1.5))**-1,0,5))]:
        print(f"    lambda {lab:<24} theta_bar = {theta_bar(A,1.0,lam):.8f}")

    print("\n(2)-(4) heterogeneous theta")
    print(f"{'cv':>5}{'lam':>7}{'network':>12}{'H(th+l)-l':>13}{'d/dlam':>10}{'CV^2':>10}")
    for cv in [0.4,0.8]:
        r=np.random.default_rng(1); th=np.clip(1.0+cv*r.normal(size=1200),0.15,None)
        Ah=row_stochastic(1200,r)
        for lam in [0.0,0.5,2.0]:
            h=1e-4
            d=((theta_bar_meanfield(th,lam+h))-(theta_bar_meanfield(th,lam-h)))/(2*h)
            print(f"{cv:>5.1f}{lam:>7}{theta_bar(Ah,th,lam):>12.6f}"
                  f"{theta_bar_meanfield(th,lam):>13.6f}{d:>10.5f}{dtheta_dlam(th,lam):>10.5f}")
        print(f"      H(theta)={H(th):.5f}   E[theta]={th.mean():.5f}"
              f"   theta_bar(1e4)={theta_bar_meanfield(th,1e4):.5f}")
