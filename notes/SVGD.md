# Stein Variational Gradient Descent

**In one line.** Move a whole cloud of particles together: each particle feels an attraction toward high density and a kernel repulsion from its neighbours — deterministic transport toward π, no acceptance step.

## How it works

SVGD updates every particle along the velocity field

φ(x) = (1/n) ∑ⱼ [ k(xⱼ, x)·∇log π(xⱼ) + ∇_{xⱼ} k(xⱼ, x) ],

which is the steepest-descent direction of KL(q‖π) within the unit ball of an RKHS. The first term is a kernel-weighted average of the gradient — **attraction**; the second is the derivative of the kernel itself — **repulsion**, pushing particles apart so the cloud spreads to cover π rather than collapsing to the mode. With a single particle the repulsion vanishes and SVGD is gradient *ascent* on log π: the spread of the approximation lives entirely in the interaction.

The kernel is an RBF whose bandwidth follows the median heuristic (median pairwise distance² / log n) unless fixed by hand.

## What to watch

Use the **Show forces** toggle: blue arrows are attraction, orange are repulsion, black their sum. At equilibrium the two nearly cancel — the cloud stands still with forces balanced. Watch the **KSD²** metric fall as the cloud converges: the kernel Stein discrepancy is a genuine measure of fit to π that needs no reference samples.

Two instructive failures: set **initialisation** to "single mode" on the multimodal target — the cloud fattens the mode it starts in and only slowly (or never) discovers the other, since no particle receives a signal from an unvisited region. And shrink the bandwidth: repulsion localises, particles clump, and the marginals develop gaps.

## Tuning

- **Bandwidth / median heuristic** — the interaction range; the median trick usually works, which is why it's the default.
- **Step size & AdaGrad** — plain gradient-descent tuning; compare Coin SVGD, which removes it.
- **Particles n** — the resolution of the approximation; variance is slightly underestimated at small n.

## References

- Liu & Wang (2016). Stein variational gradient descent. *NeurIPS*.
- Liu (2017). Stein variational gradient descent as gradient flow. *NeurIPS*.
