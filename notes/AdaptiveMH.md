# Adaptive Metropolis–Hastings

**In one line.** Random walk Metropolis that learns its own proposal covariance from the chain's history — the proposal shape moulds itself to the target.

## How it works

After a short warm-up, the Gaussian proposal covariance is set to the empirical covariance of the samples so far, scaled by the dimension-dependent factor (2.38)²/d that is optimal for Gaussian targets (Haario, Saksman & Tamminen 2001). Adaptation that goes on forever technically breaks the Markov property, but *diminishing adaptation* — the empirical covariance changes less and less as the chain grows — preserves ergodicity.

## What to watch

The proposal contour starts circular and then stretches: on the **ridge** target (correlation 0.95) it tilts to align with the ridge within a few hundred steps, and the acceptance rate and ESS climb together. Compare the same target with plain random walk Metropolis, whose isotropic proposal wastes most of its hops perpendicular to the ridge.

The limitation shows on the **banana**: a single global covariance cannot follow a curved ridge, so the fitted ellipse ends up broad and diagonal — better than isotropic, far from ideal. That failure motivates locally-adaptive samplers (H2MC, Riemannian HMC).

## Tuning

- **Initial σ** — matters only before adaptation kicks in; a poor start delays but does not prevent adaptation.
- Watch for the adaptation to *stabilise*: the ellipse should stop changing visibly. If it keeps drifting, the chain hasn't yet seen enough of the target for a stable covariance estimate.

## References

- Haario, Saksman & Tamminen (2001). An adaptive Metropolis algorithm. *Bernoulli*.
- Roberts & Rosenthal (2009). Examples of adaptive MCMC. *J. Comput. Graph. Statist.*
