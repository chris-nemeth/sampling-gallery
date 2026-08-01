# Random Walk Metropolis

**In one line.** Propose a blind Gaussian hop from the current point; accept with probability min(1, π(x′)/π(x)) — the simplest MCMC there is, and the baseline every other sampler here is trying to beat.

## How it works

From x, propose x′ = x + σξ with ξ ~ N(0, I). Accept with probability min(1, π(x′)/π(x)); otherwise stay put (and count the current point again). The chain's stationary distribution is exactly π for *any* σ — correctness is free, efficiency is not. Uphill moves are always accepted; downhill moves survive with probability equal to the density ratio, which is what lets the chain explore rather than optimise.

## What to watch

The proposal animates as a hop from the current point: green flashes are acceptances, red are rejections. On the banana, watch the chain inch along the ridge — each accepted hop is small relative to the ridge's length, so the ESS in the stats box grows much more slowly than the step count. That gap *is* the autocorrelation cost of random-walk exploration, and it is the number to compare against the gradient-based samplers.

## Tuning

- **Proposal σ** — the classic trade-off. Small σ: nearly everything is accepted but the chain barely moves. Large σ: proposals land in the tails and nearly everything is rejected. The theory says aim for ~23% acceptance in high dimensions (up to ~44% in 1–2D); try both extremes and watch the ESS, not the acceptance rate alone.

## References

- Metropolis, Rosenbluth, Rosenbluth, Teller & Teller (1953). Equation of state calculations by fast computing machines. *J. Chem. Phys.*
- Hastings (1970). Monte Carlo sampling methods using Markov chains. *Biometrika*.
- Roberts, Gelman & Gilks (1997). Weak convergence and optimal scaling of random walk Metropolis. *Ann. Appl. Probab.*
