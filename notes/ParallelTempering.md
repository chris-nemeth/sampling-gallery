# Parallel Tempering

**In one line.** Run several chains at once on flattened copies of the target π^β, and occasionally swap their states — hot chains roam between modes and pass their discoveries down to the cold chain you keep.

## How it works

A temperature ladder β₁ = 1 > β₂ > … > β_K (geometric here) defines targets π^βₖ; each level runs its own random-walk MH with step size scaled as σ/√β. Periodically, adjacent levels propose to exchange states, accepted with probability min(1, exp((βⱼ − βⱼ₊₁)(log π(xⱼ₊₁) − log π(xⱼ)))) — a valid MH move on the joint product distribution, so the cold chain still targets π exactly. Multimodality that traps a single chain becomes traversable at high temperature, and swaps ferry those excursions to β = 1.

## What to watch

Orange dots are the hot chains (flatter and more adventurous as β falls); the orange arc flashes on successful swaps; the cold chain's samples accumulate as usual. On the **multimodal** target, watch a mode switch happen: a hot chain wanders across the valley, then a cascade of swaps hands that state down the ladder. Turn the number of levels down to 1 and watch mode-switching stop.

## Tuning

- **Levels K** and **β_min** — the ladder must be dense enough that adjacent tempered targets overlap; the swap-acceptance readout is the diagnostic (aim for ~20–40%). Too-coarse a ladder shows up as near-zero swap acceptance.
- **σ** — the per-level base step size.

## References

- Geyer (1991). Markov chain Monte Carlo maximum likelihood. *Computing Science and Statistics*.
- Swendsen & Wang (1986). Replica Monte Carlo simulation of spin-glasses. *Phys. Rev. Lett.*
- Earl & Deem (2005). Parallel tempering: theory, applications, and new perspectives. *PCCP*.
