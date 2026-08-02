# Barker Proposal

**In one line.** Draw a random Gaussian perturbation, then use the local gradient to skew the direction of each coordinate before applying a Metropolis correction.

## How it works

The Barker proposal begins by drawing a symmetric Gaussian perturbation ξ ~ N(0, σ²I). Rather than moving directly by ξ, each coordinate is flipped independently according to the local gradient. For coordinate *i*, the sign is chosen with probability P(bᵢ = 1) = 1 / (1 + exp(−ξᵢ∇ᵢ log π(x))), and the proposal becomes y = x + b ⊙ ξ, where ⊙ denotes the element-wise product.

The gradient therefore determines **which directions are more likely**, not how far to move. Once the Gaussian perturbation has been sampled, there are 2ᵈ possible proposals corresponding to all combinations of coordinate-wise sign flips. The Metropolis–Hastings acceptance step then corrects the first-order approximation used to construct the proposal, ensuring that the chain targets the exact distribution π.

## What to watch

Each proposal begins with a single Gaussian perturbation. In two dimensions, this produces four candidate moves; in general there are 2ᵈ candidates. The blue circles indicate the probability of selecting each candidate, while the highlighted point is the proposal that was actually sampled.

Notice that Barker behaves differently from MALA. MALA shifts the centre of its Gaussian proposal using the gradient, whereas Barker keeps the proposal symmetric and instead biases the direction of each coordinate. The gradient therefore acts as a **steering mechanism** rather than a deterministic drift.

On multimodal targets Barker remains a local sampler: the gradient improves local exploration, but moving between distant modes still relies on drawing sufficiently large Gaussian perturbations.

## Tuning

- **Proposal scale σ** — controls the typical jump length. Small values produce short, conservative moves with high acceptance. Larger values explore more aggressively but reduce the acceptance probability. In general, however, Barker is remarkably robust to this tuning parameter by design.

## References

- Livingstone & Zanella (2022). The Barker proposal: Combining robustness and efficiency in gradient-based MCMC. *JRSS-B*.
- Hird, Livingstone & Zanella (2020). A fresh take on ‘Barker dynamics’ for MCMC. *Proceedings of Monte Carlo and Quasi-Monte Carlo Methods 2020*.