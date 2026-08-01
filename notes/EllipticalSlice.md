# Elliptical Slice Sampling

**In one line.** For targets of the form Gaussian prior × likelihood: draw one auxiliary Gaussian point, slice-sample the *angle* around the ellipse it defines with the current state — no step size, no rejections.

## How it works

Write π(x) ∝ N(x; 0, Σ)·L(x). Draw ν ~ N(0, Σ) and consider the ellipse x·cos θ + ν·sin θ: every point on it is a valid "prior rotation" of the current state, so only the likelihood needs slice-sampling. Draw a level log y = log L(x) + log u, take θ ~ U(0, 2π) with bracket [θ − 2π, θ], and shrink the bracket toward θ = 0 on each rejection. Since θ = 0 *is* the current state, the loop must terminate — every iteration accepts, and there are no free parameters.

The gallery's targets are arbitrary densities, so each is split as N(0, σ²I) × residual likelihood. The sampler is exact for *any* σ; σ only controls the geometry.

## What to watch

The pale ellipse is the current proposal locus; the orange ring is ν; orange dots are shrinkage rejections collapsing toward the accepted blue point. On the **flower**, watch the ellipses arc across petals — the prior draw carries the state to genuinely distant regions.

## Tuning

- **Prior sd σ** — the one knob, and it's instructive. With σ matched to the target's scale, the residual likelihood is nearly flat, the first angle is usually accepted, and moves are huge. With σ mismatched, the likelihood is informative and shrinkage bites (watch the rejected dots multiply). In real use — Gaussian process posteriors — the prior is given and there is *nothing* to tune, which is the method's celebrated property.

## References

- Murray, Adams & MacKay (2010). Elliptical slice sampling. *AISTATS*.
