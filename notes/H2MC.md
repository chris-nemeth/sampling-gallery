# Hessian–Hamiltonian Monte Carlo

**In one line.** Use the local gradient *and* Hessian to shape a long anisotropic Gaussian proposal that approximates Hamiltonian flow — curvature-aware moves without integrating a trajectory.

## How it works

Around the current point, approximate log π by its second-order Taylor expansion. Under that quadratic model, Hamiltonian dynamics can be solved *in closed form* (the solution decomposes along the Hessian's eigenvectors into sinusoids for negative curvature and hyperbolic functions for positive), and marginalising the momentum turns the endpoint into a Gaussian proposal whose covariance stretches along flat directions and shrinks along curved ones. A standard MH correction accounts for the approximation. The method comes from gradient-domain light-transport rendering (Li et al. 2015), where the same anisotropy problem arises.

## What to watch

The proposal contour is the local Gaussian: on the **banana**, watch it rotate and elongate as the chain moves around the curve — each proposal is tailored to where the chain *is*, unlike Adaptive MH's single global covariance. Compare the two on the banana; then compare with Riemannian HMC, which uses the same curvature information inside the dynamics rather than a one-shot Gaussian.

## Tuning

- **σ** — overall proposal scale.
- **L** — the pseudo-integration time: how far along the closed-form flow the proposal reaches. Larger L stretches proposals further along flat directions.

## References

- Li, Lehtinen, Ramamoorthi, Jakob & Durand (2015). Anisotropic Gaussian mutations for Metropolis light transport through Hessian–Hamiltonian dynamics. *ACM TOG (SIGGRAPH Asia)*.
