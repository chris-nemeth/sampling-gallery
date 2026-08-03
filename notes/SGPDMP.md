# Stochastic-Gradient PDMP

**In one line.** Zig-Zag and Bouncy Particle dynamics running on *noisy* gradient estimates — non-reversible, rejection-free sampling for the setting where the exact gradient is a luxury you can't afford.

## How it works

Exact PDMPs need the true gradient inside their event rates, and simulating those events exactly requires target-specific bounds. The stochastic-gradient scheme (the paper's O(h) approximation) sidesteps both: draw a noisy gradient estimate ĝ and **freeze it for a short window of length h**. While frozen, the event rates are constant, so event times are exact exponential draws — no thinning, no bounds. At each event, and at every window boundary, a fresh estimate is drawn.

- **SG Zig-Zag**: coordinate i flips its velocity at rate max(0, vᵢ·ĝᵢ) + λ_ref.
- **SG Bouncy Particle**: the velocity reflects in a freshly drawn ĝ at rate max(0, v·ĝ), with full velocity refreshes at rate λ_ref.

There is no dataset in this demo, so — as in the SGLD entry — a tunable N(0, σ²I) perturbation of the true gradient stands in for minibatch noise. The sampler is *approximate*: the invariant measure carries an O(h) bias that also grows with the gradient noise. In our checks on the correlated banana, SG Zig-Zag at h = 0.05 sits essentially on the truth, and SG-BPS converges cleanly as h shrinks (exact to two decimals by h = 0.02).

## What to watch

Two arrows track the current position: the **true** gradient in blue and the **frozen noisy estimate** in orange. The jitter between them is the entire idea of the paper — the dynamics only ever see the orange one. The path shows the usual PDMP signature: straight flight with kinks at events.

Then provoke the bias: crank **gradient noise σ** or the **window h** on the standard Gaussian and watch the sample cloud inflate past the contours, exactly as ULA's does with its step size. Zig-Zag is noticeably more robust to both knobs than the Bouncy Particle here — reflection directions inherit the gradient noise directly, and on curved targets those errors compound.

## Tuning

- **Gradient noise σ** — stands in for minibatch size; σ = 0 recovers the (still h-discretised) noise-free scheme.
- **Refresh window h** — the bias/cost dial: smaller h means more gradient evaluations per unit of trajectory and less bias.
- **Refresh rate λ** — ergodicity insurance, as in the exact BPS; also adds random flips to Zig-Zag.

## References

- Fearnhead, Grazzi, Nemeth & Roberts (2024). Stochastic gradient piecewise deterministic Monte Carlo samplers. arXiv:2406.19051.
- Bierkens, Fearnhead & Roberts (2019). The Zig-Zag process and super-efficient sampling for Bayesian analysis of big data. *Ann. Statist.*
