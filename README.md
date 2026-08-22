# Relative Wealth Concerns, Network Aggregation, and Effective Risk Aversion

Replication package for *Relative Wealth Concerns, Network Aggregation, and
Effective Risk Aversion* (Renato Vicente, Instituto de Matemática, Estatística e
Ciência da Computação, Universidade de São Paulo).

The paper studies a CARA economy in which investor `a` compares his wealth to a
weighted average `Σ_j A_aj W_j` over a network of peers. Equilibrium demand
scales solve the Katz–Bonacich system

```
[ D_{θ+λ} − D_λ A ] c = 1,        θ̄ = m / Σ_a c_a
```

Every numerical claim in the paper comes from the four scripts below. They are
self-contained: no data files, no network access, no local imports, and all
random draws are seeded.

## Reproduce everything

```
pip install -r requirements.txt
./reproduce.sh
```

Runtime is about fifteen seconds on a laptop.

## What each script produces

| Script | Produces | Where it appears |
|---|---|---|
| `scripts/network.py` | solves the Katz–Bonacich system; `θ̄(λ)`; critical `λ` | eq. (3), Section 3 |
| `scripts/neutrality.py` | verifies Theorem 1 and the mean-field limit `H(θ+λ) − λ` | Theorem 1, Proposition 1 |
| `scripts/sign_reversal.py` | frequency of cumulative and marginal sign reversal, Wilson intervals | Table 1 |
| `scripts/block_reversal.py` | block replication and robustness to mixing | Remark 5 |
| `scripts/characterization.py` | Theorems 2 and 3, the Dirichlet identity, the uniform bound | Theorems 2-3, Lemma 1 |
| `scripts/closed_economy.py` | equilibrium prices, market clearing, the Sharpe identity | Section 3 |

## Notes on the numerical design

**Table 1.** Two distinct events are reported and should not be conflated: the
cumulative event `θ̄(0.5) < θ̄(0)` and the marginal event `θ̄'(0.5) < 0`. The
derivative is analytic, not a finite difference — with
`M(λ) = D_θ + λ(I − A)` and `c = M⁻¹1`,

```
c' = −M⁻¹ (I − A) c,        θ̄' = −m (1ᵀc') / (1ᵀc)²
```

The population size is held fixed within each row, so that comparison across
rows identifies the effect of size. Off-diagonal entries of `A` are uniform on
(0,1) before row normalisation, and `θ_a = max{1 + σ Z_a, 0.15}` with `Z_a`
standard normal. Seeds are `m*1000 + 10σ`, set in `sign_reversal.py`.

**Theorems 2 and 3.** The two limits that organise the paper. At zero intensity
`theta_bar'(0) = m t'(I-A)t / (1't)^2` with `t = 1/theta`; when `A` is doubly
stochastic this equals the Dirichlet form `(1/2) sum_ij A_ij (t_i-t_j)^2`, so
the effect is non-negative. As intensity grows, `theta_bar` converges to a
stationary average `pi'theta`, weighted by influence rather than population.
`characterization.py` checks both against the solver, exhibits the counterexample
where `pi = (1/2, 1/2, 0)` puts zero weight on the most risk averse investor,
and covers the reducible case where class weights are absorption probabilities.

**Remark 5.** The decay of reversal with `m` in Table 1 is a property of the
dense i.i.d.-weight ensemble, not of population size. `block_reversal.py`
verifies that replicating the three-investor counterexample `k` times leaves
`θ̄(λ)` unchanged up to `m = 3000`, and that the reversal survives weak ties
across blocks up to a mixing fraction of about `ε = 0.1`.

## Requirements

`numpy` and `scipy` only; see `requirements.txt`. Reported runs used numpy
2.4.4 and scipy 1.18.0 under Python 3.14, but the scripts use no recent API and
run under considerably older versions.

## Paper

The author's manuscript version, in three builds from one source:

| File | Contents |
|---|---|
| `paper/heterogeneous.pdf` | the paper alone, 16 pages |
| `paper/supplement.pdf` | the supplement alone, 3 pages |
| `paper/submission.pdf` | both in one PDF, 19 pages |

The bodies live in `paper/body-paper.tex` and `paper/body-supplement.tex`; the
three `.tex` files at top level are thin shells around them. The supplement's
references to the paper go through a `\paperref` macro that prints literal text
in the standalone build and a real `\ref` in the combined one.

## License

Code is released under the MIT License (`LICENSE`). The manuscript is the
author's own work and is not covered by that license.
