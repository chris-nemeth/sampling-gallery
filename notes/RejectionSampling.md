# Rejection Sampling

**In one line.** Draw from a simple proposal q, accept each draw with probability π(x)/(M·q(x)) — the survivors are *exact* independent samples from π.

## How it works

Choose an envelope constant M with π(x) ≤ M·q(x) everywhere. Draw x ~ q and u ~ U(0,1); keep x if u < π(x)/(M·q(x)). Accepted points need no burn-in, no mixing diagnostics, and no autocorrelation corrections — every kept point is an independent draw from π. The price is the acceptance rate, which is exactly 1/M when π is normalised: all the difficulty hides in M.

This demo computes M numerically as the maximum of π/q over a grid, times a 1.2 safety factor, so the envelope is always valid for the target you picked.

## What to watch

The grey circle is the proposal's 1-sd contour; orange dots are rejected draws, and accepted points accumulate on the plot. The panel reports the acceptance rate and the current envelope M. Because accepted draws are i.i.d., the effective sample size equals the number of samples — the only sampler in the gallery where that is exactly true.

## Tuning

- **Proposal sd** — the whole story. Too wide wastes proposals in the tails; too *narrow* is catastrophic: if q has thinner tails than π, the ratio π/q blows up and M becomes enormous. Try sd = 1.5 on the banana: M jumps to ~10⁶ and the acceptance rate hits zero (the panel explains when this happens). This is the honest lesson — rejection sampling collapses in exactly the situations where MCMC is needed.

## References

- von Neumann (1951). Various techniques used in connection with random digits.
- Robert & Casella (2004). *Monte Carlo Statistical Methods*, ch. 2.
