# Slice Sampling

**In one line.** Sample uniformly from the region *under* the density curve: draw a height, find the horizontal slice at that height, and sample a point from it — step sizes adapt themselves.

## How it works

Given x, draw a height y ~ U(0, π(x)); the "slice" is the set {x′ : π(x′) > y}. Sampling uniformly from the slice leaves π invariant. Since the slice's shape is unknown, Neal's procedure brackets it: **step out** an interval of width w around x until both ends leave the slice, then sample uniformly inside, **shrinking** the interval toward x on each point that misses. The shrinkage guarantees termination, and acceptance is automatic — every iteration moves (like Gibbs, there is no reject step). This demo applies the univariate procedure to each coordinate in turn.

## What to watch

The horizontal bracket is the stepped-out interval; orange dots are shrinkage rejections; the accepted point closes the sweep. On the **squiggle**, watch the bracket span several wiggles — the slice at low heights is disconnected, and stepping out lets the sampler hop between pieces, something a local random walk does slowly.

## Tuning

- **Width w** — remarkably forgiving, which is slice sampling's selling point. Too small costs stepping-out expansions; too large costs shrinkage steps; both are linear costs, not efficiency cliffs. Compare with the acceptance cliff of a badly-scaled Metropolis σ.

The coordinate-wise sweep shares Gibbs sampling's weakness: on the **ridge** target, axis-aligned moves are short because the slice along each axis is narrow.

## References

- Neal (2003). Slice sampling. *Ann. Statist.*
