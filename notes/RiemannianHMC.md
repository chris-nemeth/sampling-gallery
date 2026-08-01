# Riemannian-Manifold HMC

**In one line.** HMC on a position-dependent metric G(x): the sampler's notion of distance stretches and shrinks with the local curvature of the target, so one step size serves both the funnel's mouth and its neck.

## How it works

Replace the fixed mass matrix with a metric G(x) built from local curvature. Girolami & Calderhead's original uses the Fisher information, which needs a statistical model; for the gallery's bare densities we use Betancourt's **SoftAbs** metric: eigendecompose the Hessian of U = −log π and smooth each eigenvalue through λ·coth(αλ), which behaves like |λ| where curvature is strong and levels off at 1/α where it vanishes. The Hamiltonian becomes non-separable,

H(x, p) = U(x) + ½ log det G(x) + ½ pᵀG(x)⁻¹p,

so the leapfrog update is implicit — the momentum half-step and position step are each solved by a few fixed-point iterations (the "generalised leapfrog") — and momenta are drawn as p ~ N(0, G(x)).

## What to watch

The ellipses along the trajectory are the local metric G⁻¹ — the shape of a "unit step" at that point. Run it on the **funnel**: the ellipses are wide in the mouth and needle-thin in the neck, and the sampler visits both freely. Compare plain HMC on the same target: its samples avoid the neck and the y-mean biases upward, while RMHMC reproduces the exact N(2, 3) marginal. On the **banana**, the ellipses rotate to track the ridge direction.

## Tuning

- **Leapfrog ε** — far more forgiving than in plain HMC because the metric rescales the dynamics; the ΔH readout stays small even at ε = 0.25 on the funnel.
- **SoftAbs α** — how faithfully G follows the Hessian. Large α (≈ 5+) tracks |Hessian| closely but stresses the implicit integrator where eigenvalues pass through zero (watch ΔH grow); α ≈ 1 is a good default.
- **Steps L** — as in HMC.

The cost is honest: every step solves implicit equations requiring Hessians, visible in the slower step rate. Curvature information isn't free.

## References

- Girolami & Calderhead (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. *JRSS B*.
- Betancourt (2013). A general metric for Riemannian manifold Hamiltonian Monte Carlo. *GSI*.
