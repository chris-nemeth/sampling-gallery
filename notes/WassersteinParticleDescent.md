# Wasserstein Particle Descent

**In one line.** Follow the Wasserstein gradient flow of KL(q‖π) directly: attraction is the plain score ∇log π, and the entropy-driven repulsion is estimated from the particles' own density — three closely related estimators, one card.

## How it works

The KL objective's Wasserstein gradient flow moves mass along v(x) = ∇log π(x) − ∇log q(x), where q is the current particle density. The whole family question is how to estimate the unknowable ∇log q from the particles:

- **GFSD** smooths q with a kernel density estimate: ∇log q̂(xᵢ) = ∑ⱼ∇ᵢk(xᵢ,xⱼ) / ∑ⱼk(xᵢ,xⱼ).
- **Blob** derives the force from a regularised entropy functional, adding a second, symmetrising repulsion term — typically the best-behaved of the three.
- **GFSF** estimates the score by solving a small kernel ridge system (Stein's identity inverted with an ℓ² regulariser).

SVGD belongs to the same family but *smooths the velocity field* in an RKHS rather than smoothing q; these methods keep the raw score attraction, so their forces are more local and their repulsion comes from an explicit density estimate. Liu et al. (2019) is the unifying treatment.

## What to watch

Flip between modes on the **donut** with the force decomposition on: the attraction arrows are identical (the score doesn't change); the repulsion differs — that *is* the difference between the methods. Compare against SVGD from the same initialisation (init selector): SVGD's kernel-averaged attraction is smoother; WPD's raw-score attraction is sharper near the ring. The **KSD²** metric makes the race quantitative.

## Tuning

- **Mode** — Blob / GFSD / GFSF as above.
- **Bandwidth / median heuristic** — now it's a *density estimation* bandwidth: too small and repulsion turns spiky (particles crystallise), too large and the cloud over-smooths.
- **Step size & AdaGrad** — as in SVGD.

## References

- Liu, Zhuo, Cheng, Zhang, Zhu & Carin (2019). Understanding and accelerating particle-based variational inference. *ICML*.
- Chen, Zhang, Wang, Li & Chen (2018). A unified particle-optimization framework for scalable Bayesian sampling. *UAI*.
