# Ensemble MCMC

**In one line.** Run a population of walkers and let each propose moves using the *others'* positions — the ensemble's own geometry becomes the proposal, giving affine invariance for free.

## How it works

The **stretch move** (Goodman & Weare 2010): to move walker xᵢ, pick another walker xⱼ and propose y = xⱼ + z(xᵢ − xⱼ), where z has density ∝ 1/√z on [1/a, a]. Accept with probability min(1, z^(d−1)·π(y)/π(xᵢ)). Because the proposal is built from differences of walker positions, an affine change of coordinates transforms the proposals the same way as the target: the algorithm performs *identically* on a round Gaussian and a squashed one. This is the move behind the *emcee* package beloved in astrophysics.

The **differential evolution** mode (ter Braak 2006) instead proposes y = xᵢ + γ(xⱼ − xₖ) + small noise, using the difference of two other walkers as the step direction — same ensemble idea, different move.

## What to watch

The faint dots are the other walkers; the grey line anchors the current proposal to its helper walker. Run it on the **ridge**: acceptance is ~70% and identical to what you get on the round standard Gaussian — that's affine invariance, demonstrated rather than claimed. Compare random walk Metropolis on the same pair of targets.

## Tuning

- **Walkers** — more walkers give a richer proposal geometry; too few and the ensemble can collapse into a subspace.
- **Stretch a** — the step-size analogue; 2 is the standard choice.
- **DE γ** — 2.38/√(2d) is the classical guideline.

## References

- Goodman & Weare (2010). Ensemble samplers with affine invariance. *CAMCoS*.
- ter Braak (2006). A Markov chain Monte Carlo version of the genetic algorithm differential evolution. *Statist. Comput.*
- Foreman-Mackey, Hogg, Lang & Goodman (2013). emcee: the MCMC hammer. *PASP*.
