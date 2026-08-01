# Hamiltonian Monte Carlo

**In one line.** Give the current point a random momentum and let it coast along the log-density landscape like a frictionless puck — long, informed moves with high acceptance.

## How it works

Augment x with a momentum p ~ N(0, I) and simulate Hamiltonian dynamics for the energy H(x, p) = U(x) + ½|p|², where U = −log π. The dynamics conserve H exactly, so a perfect simulation would always be accepted; the leapfrog integrator (L steps of size ε) makes an O(ε²) energy error which the Metropolis step corrects: accept with probability min(1, exp(H₀ − H₁)). Gradients steer the trajectory into the high-density region, which is why HMC scales so much better with dimension than random-walk methods.

## What to watch

Each proposal shows the full leapfrog trajectory with the initial momentum arrow. On the banana the trajectories curve to follow the ridge — the gradient doing the steering. Watch the ESS per step against random walk Metropolis: the trajectories cost L gradient evaluations each, but buy nearly-independent samples.

Two classic failure modes are worth provoking. Push ε up until trajectories **diverge** (the energy error blows up, everything is rejected). And on the **funnel**, note that *no* fixed ε works: steps small enough for the neck are wasteful in the mouth — the failure that motivates Riemannian HMC.

## Tuning

- **Leapfrog ε** — sets the energy error; the acceptance rate falls as ε grows, sharply once the integrator hits its stability limit.
- **Steps L** — total integration time εL should be long enough to decorrelate but not so long the trajectory U-turns back (the waste NUTS was invented to eliminate).
- **Adapt step size** — dual averaging tunes ε automatically toward a target acceptance rate.

## References

- Duane, Kennedy, Pendleton & Roweth (1987). Hybrid Monte Carlo. *Phys. Lett. B*.
- Neal (2011). MCMC using Hamiltonian dynamics. *Handbook of MCMC*.
