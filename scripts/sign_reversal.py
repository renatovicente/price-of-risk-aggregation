"""Frequency of negative marginal and cumulative effects (Table 1).

Two distinct measures, which should not be conflated:

  cumulative : theta_bar(lam) < theta_bar(0)
  marginal   : d theta_bar / d lam < 0  at lam

The derivative is analytic, not a finite difference. With
M(lam) = D_theta + lam (I - A)  and  c = M^{-1} 1,

    c'   = -M^{-1} (I - A) c
    tb'  = -m * (1' c') / (1' c)^2

Ensemble: m is held FIXED within each row, so that comparison across rows
identifies the effect of population size. Off-diagonal entries of A are uniform
on (0,1) with zero diagonal, normalised by rows; theta_a = max{1 + sigma Z_a,
0.15} with Z_a standard normal. Seeds are m*1000 + 10*sigma.
"""
import numpy as np

def solve(A, th, lam):
    m=len(A); I=np.eye(m)
    M=np.diag(th)+lam*(I-A)
    c=np.linalg.solve(M,np.ones(m))
    cp=-np.linalg.solve(M,(I-A)@c)          # dc/dlam
    tb=m/c.sum()
    tbp=-m*cp.sum()/c.sum()**2              # d(theta_bar)/dlam
    return tb, tbp

def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

def sweep(reps, m, sigma, lam, seed=0):
    rng=np.random.default_rng(seed); cum=0; marg=0
    for _ in range(reps):
        A=rng.random((m,m)); np.fill_diagonal(A,0.0); A/=A.sum(1,keepdims=True)
        th=np.clip(1.0+sigma*rng.normal(size=m),0.15,None)
        tb0,_=solve(A,th,0.0); tb,tbp=solve(A,th,lam)
        cum += tb<tb0; marg += tbp<0
    return cum, marg, reps

if __name__=="__main__":
    lam=0.5; reps=20000
    print(f"lambda = {lam}, {reps} draws per row, m held FIXED\n")
    print(f"{'m':>5}{'sigma':>7}{'cumulative':>13}{'95% CI':>20}{'marginal':>12}{'95% CI':>20}")
    for m in [3,6,10,20,60]:
        for sigma in [0.4,0.8]:
            k1,k2,n=sweep(reps,m,sigma,lam,seed=m*1000+int(sigma*10))
            l1,h1=wilson(k1,n); l2,h2=wilson(k2,n)
            print(f"{m:>5}{sigma:>7.1f}{k1/n:>13.4f}{f'[{l1:.4f}, {h1:.4f}]':>20}"
                  f"{k2/n:>12.4f}{f'[{l2:.4f}, {h2:.4f}]':>20}")
