# Gibbs Sampling

**In one line.** Update one coordinate at a time by sampling it exactly from its full conditional distribution, holding the others fixed — no proposals, no rejections.

## How it works

Cycle through the coordinates; for each, draw xᵢ ~ π(xᵢ | x₋ᵢ). Every such draw leaves π invariant, so the composition does too, with acceptance probability 1 throughout. In models with conjugate structure the conditionals are known distributions and Gibbs is extremely convenient; in this 2-D demo each conditional is sampled exactly by inverse-CDF along a fine grid of the density slice.

## What to watch

The moves are strictly axis-aligned — the chain traces a staircase. On the **standard** Gaussian it mixes essentially instantly. On the **ridge** (correlation 0.95) the staircase becomes tiny: each conditional is a narrow band around the ridge, so consecutive samples barely move, and the ESS collapses despite every "acceptance" succeeding. That is the canonical Gibbs failure — high posterior correlation — and the visual argument for reparametrisation, blocking, or joint samplers.

## Tuning

Nothing to tune — which is the appeal. The efficiency is entirely a property of the target's correlation structure in the chosen coordinates.

## References

- Geman & Geman (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. *IEEE PAMI*.
- Casella & George (1992). Explaining the Gibbs sampler. *Amer. Statist.*
