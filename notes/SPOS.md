# Stochastic Particle-Optimization Sampling

**In one line.** SVGD's interacting drift plus Langevin noise — the bridge between deterministic particle transport and stochastic-gradient MCMC, and a cure for particle collapse.

## How it works

Each particle takes the SVGD step *and* an independent Gaussian kick:

xᵢ ← xᵢ + ε·φ(xᵢ) + √(2ε/β)·ξᵢ, ξᵢ ~ N(0, I).

With β → ∞ this is SVGD; with the interaction removed it is ULA run in parallel. The noise matters for more than aesthetics: deterministic SVGD with finitely many particles can *collapse* — in high dimensions the repulsion can fail to hold the cloud open (the "variance collapse" phenomenon) — while the Langevin term keeps every particle individually ergodic. Zhang et al. prove non-asymptotic convergence for the interacting stochastic system.

## What to watch

Set **initialisation** to "single mode" on the **multimodal** target and race SPOS against SVGD: SVGD's cloud stays where the signal is, while SPOS particles diffuse across the valley and seed the other mode — stochastic exploration doing what deterministic transport cannot. Then crank β up and watch SPOS turn back into SVGD before your eyes. The particle cloud is fuzzier than SVGD's at equilibrium — that's the noise floor, the price of exploration.

## Tuning

- **Noise β** — the interpolation dial: low β ≈ noisy Langevin cloud, high β ≈ deterministic SVGD.
- **Step size, bandwidth, particles** — as in SVGD.

## References

- Zhang, Zhang, Carin & Chen (2020). Stochastic particle-optimization sampling and the non-asymptotic convergence theory. *AISTATS*.
- Ba, Erdogdu, Ghassemi, Sun, Suzuki, Wu & Zhang (2021). Understanding the variance collapse of SVGD in high dimensions. *ICLR* (the collapse phenomenon).
