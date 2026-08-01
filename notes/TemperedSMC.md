# Tempered Sequential Monte Carlo

**In one line.** March a whole weighted particle cloud from an easy reference distribution to the target along a temperature path, reweighting, resampling, and rejuvenating as you go.

## How it works

Define π_β ∝ μ₀^(1−β) π^β with μ₀ a broad Gaussian and β: 0 → 1. Each step: (1) choose the next increment δ *adaptively* — here by bisection so that the effective sample size of the incremental weights stays near a target fraction of N; (2) reweight the particles by (π/μ₀)^δ; (3) if the cumulative ESS drops below N/2, **resample** (systematically) to cull dead particles and duplicate good ones; (4) **rejuvenate** every particle with a few random-walk MH steps targeting the current π_β. Once β = 1 the cloud is sampling the target, and keeps rejuvenating.

**Annealed importance sampling** (Neal 2001) is the same anneal with a *fixed* ladder and *no resampling or interaction*: each particle carries its weight to the end. The mode toggle runs both so the difference is visible rather than theoretical.

## What to watch

Particle size shows weight. In SMC mode, watch β climb (5–11 adaptive steps on most targets), the ESS sag and then snap back to N at each resample. In AIS mode the ESS decays monotonically and never recovers — the weights are the output. On the **multimodal** target, notice the anneal populating *both* modes from the start: tempering discovers modes globally, where a single chain must find them by luck.

## Tuning

- **Particles N** — more particles, smoother everything.
- **Rejuvenation moves / σ** — too little rejuvenation and resampled duplicates stay duplicated.
- **AIS levels** — coarser ladders decay the ESS faster; try 10 vs 100.

## References

- Del Moral, Doucet & Jasra (2006). Sequential Monte Carlo samplers. *JRSS B*.
- Neal (2001). Annealed importance sampling. *Statist. Comput.*
