# Metropolis-Adjusted Langevin Algorithm

**In one line.** A random-walk proposal with a gradient nudge — half diffusion, half hill-climb — corrected by a Metropolis step.

## How it works

The Langevin diffusion dX = ½∇log π(X)dt + dW has π as its stationary distribution. Discretising with step ε gives the proposal x′ = x + (ε²/2)∇log π(x) + εξ, ξ ~ N(0, I). Discretisation introduces bias, so MALA applies a Metropolis correction with the *asymmetric* proposal density (the gradient term makes q(x′|x) ≠ q(x|x′) — forgetting the correction's asymmetry is a classic implementation bug). The result is exact for any ε, with the gradient pulling proposals toward high density.

## What to watch

Each proposal is drawn with its gradient arrow: the deterministic drift plus the random scatter around it. Compare with random walk Metropolis at the same step size — the drift buys a visibly higher acceptance rate at equal moves. On the **swiss roll**, watch the drift follow the spiral arm.

MALA remains a *local* sampler: on the multimodal target it explores one mode thoroughly and crosses between modes rarely — gradients point uphill, and uphill is away from the saddle.

## Tuning

- **Step size ε** — optimal acceptance is ≈ 57% (higher than random walk's 23%, because proposals are informed). Too large and the discretised drift overshoots, tanking acceptance in high-curvature regions.

## References

- Roberts & Tweedie (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*.
- Roberts & Rosenthal (1998). Optimal scaling of discrete approximations to Langevin diffusions. *JRSS B*.
