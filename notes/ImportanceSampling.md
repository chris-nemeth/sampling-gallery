# Importance Sampling

**In one line.** Keep *every* draw from the proposal q, but weight it by w = π(x)/q(x) — estimates are weighted averages, and the weights tell you exactly how much trouble you're in.

## How it works

For x₁,…,xₙ ~ q, the self-normalised estimator of E_π[f] is ∑ wᵢ f(xᵢ) / ∑ wᵢ with wᵢ = π(xᵢ)/q(xᵢ) (unnormalised densities are fine — the normalising constant cancels). Nothing is rejected, so the method never "fails" visibly; instead it fails through **weight degeneracy**: a few draws carry almost all the weight and the effective sample size collapses. The panel reports the Kish effective sample size, (∑w)²/∑w², shown as *weighted ESS*: equal weights give n, one dominant weight gives ≈ 1.

**SIR** (sampling-importance-resampling) resamples the weighted cloud into an equally-weighted one — useful as a building block (it is the resampling step inside SMC), at the cost of duplicating high-weight points.

**PSIS** (Pareto-smoothed importance sampling) fits a generalised Pareto distribution to the largest weights and replaces them with the fit's expected order statistics, stabilising the heavy right tail. Its shape estimate k̂ is a diagnostic you can trust: k̂ < 0.7 means the weights are usable; k̂ ≥ 0.7 means no amount of smoothing will save this proposal.

## What to watch

Dot size is proportional to weight. Move the proposal mean off-centre (say muX = 3 on the banana) and watch a handful of big dots dominate while the weighted ESS collapses; the marginal histograms still converge to the truth, just slowly. Then enable PSIS and watch the ESS partially recover — and k̂ report whether to believe it.

## Tuning

- **Proposal sd** — err wide: an over-dispersed q gives bounded weights (k̂ < 0), while an under-dispersed one gives unbounded weights (try sd = 1 → k̂ ≈ 0.9, flagged unreliable).
- **Proposal mean x/y** — misplacing the proposal is the cleanest way to provoke degeneracy.

## References

- Kish (1965). *Survey Sampling* (the ESS formula).
- Vehtari, Simpson, Gelman, Yao & Gabry (2024). Pareto smoothed importance sampling. *JMLR*.
