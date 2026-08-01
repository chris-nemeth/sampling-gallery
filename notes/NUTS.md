# No-U-Turn Sampler

**In one line.** HMC that decides its own trajectory length: it keeps doubling the path until it starts to double back, then samples a point from the whole tree — the self-tuning configuration behind Stan.

## How it works

HMC's efficiency hinges on the integration time: too short is a random walk, too long wastes gradients as the trajectory loops back. NUTS integrates both forwards and backwards in time, repeatedly *doubling* the trajectory, and stops when the ends start moving toward each other (the U-turn criterion ⟨x⁺ − x⁻, p⟩ < 0). A point is then drawn from the trajectory in a way that preserves detailed balance (slice or multinomial sampling across the tree). Combined with dual-averaging step-size adaptation, it removes both tuning parameters — at the price of a genuinely intricate recursive algorithm.

The gallery's default is the *efficient* dual-averaged variant used by Stan; the naive tree and fixed step size are available as advanced options to show what each refinement buys.

## What to watch

The trajectory tree is drawn each step: it grows by doublings and stops at different lengths in different parts of the target — long paths across the banana's ridge, short careful ones in tight regions. That per-step adaptivity is exactly what fixed-L HMC lacks. On the donut, watch trajectories wrap around the ring and stop just before returning to their start.

## Tuning

- Nothing, by design — that is the point. The advanced options let you break it deliberately: fix the step size too large and watch divergences; switch to the naive tree to see extra rejected work.

## References

- Hoffman & Gelman (2014). The No-U-Turn Sampler. *JMLR*.
- Betancourt (2017). A conceptual introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.
