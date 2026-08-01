# Nested Sampling

**In one line.** Maintain a population of live points and repeatedly replace the *worst* one with a new point of higher likelihood — compressing inward through nested likelihood shells, and computing the evidence along the way.

## How it works

Nested sampling (Skilling 2006) reframes integration: the evidence Z = ∫ L(θ)π(θ)dθ becomes a one-dimensional integral over prior volume, Z = ∫ L(X)dX, where X(λ) is the prior mass with likelihood above λ. Keep N live points drawn from the prior; at each iteration, remove the lowest-likelihood point (a "dead point", contributing its likelihood times the shell of prior volume ~1/N it represents) and replace it with a fresh draw from the prior *constrained to higher likelihood*. Posterior samples fall out for free by weighting dead points by their evidence contributions.

The hard step is sampling the constrained prior. This implementation uses **RadFriends** (Buchner 2014): the allowed region is the union of balls around the current live points, with radius chosen by leave-one-out cross-validation — robust to multimodality without hand-tuned proposals.

## What to watch

The shaded region is the RadFriends constraint contracting around the live points as the likelihood floor rises — the visual essence of the method: *inward compression through nested shells*. Dead points accumulate as the posterior sample. On the **multimodal** target, watch the region split naturally into islands, one per mode — no special handling needed.

## Tuning

- **Live points N** — resolution of the compression: each step removes ~1/N of the remaining prior volume, so more live points give finer (slower) compression and better multimodal coverage.

## References

- Skilling (2006). Nested sampling for general Bayesian computation. *Bayesian Analysis*.
- Buchner (2014). A statistical test for nested sampling algorithms. *Statist. Comput.* (RadFriends).
