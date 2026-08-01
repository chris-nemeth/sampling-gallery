# Zig-Zag Sampler

**In one line.** A continuous-time, non-reversible sampler: the particle moves in straight lines with velocity components ±1, and each coordinate flips direction at random times governed by the gradient — no rejections, ever.

## How it works

The Zig-Zag process is a piecewise-deterministic Markov process (PDMP). Between events, x(t) = x + vt with v ∈ {−1, +1}². Coordinate i flips its velocity at the first arrival of an inhomogeneous Poisson process with rate λᵢ(x, v) = max(0, −vᵢ·∂ᵢ log π(x)) — you flip only when moving *downhill* in that coordinate, and the more steeply downhill, the sooner. This dynamics has π as its invariant distribution while being non-reversible, which typically reduces backtracking compared with reversible chains. Positions recorded at equal time intervals form an unweighted sample.

Event times are simulated either by fine time-discretisation (robust everywhere, small O(dt) bias) or by **Poisson thinning** with a bound on the rate (exact where the bound is valid) — the method toggle exposes the real implementation issue for PDMPs: rate bounds are target-specific work.

## What to watch

The path is the signature zig-zag: axis-diagonal straight lines with kinks at flip events. On the **donut**, watch flips cluster on the ring's inner and outer walls where the gradient is steep. Non-reversibility is visible directly: the particle sweeps *through* regions rather than dithering back and forth.

## Tuning

- **Sample interval** — how often the continuous path is recorded; it changes the sample spacing, not the path.
- **Event method** — discretisation vs thinning; compare their speed and remember only thinning is exact.

## References

- Bierkens, Fearnhead & Roberts (2019). The Zig-Zag process and super-efficient sampling for Bayesian analysis of big data. *Ann. Statist.*
- Fearnhead, Bierkens, Pollock & Roberts (2018). Piecewise deterministic Markov processes for continuous-time Monte Carlo. *Statist. Sci.*
