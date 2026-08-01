# Quasi-Monte Carlo

**In one line.** Replace random draws with a *low-discrepancy* point set that fills space evenly — same importance-weighting machinery, lower-variance estimates.

## How it works

Pseudorandom points clump and leave gaps; their integration error shrinks like n^(−1/2). Low-discrepancy sequences are designed so that every initial segment covers the unit square as evenly as possible, giving errors closer to n^(−1) (up to log factors) for smooth integrands. This demo uses the Halton sequence (radical-inverse in bases 2 and 3), mapped through the Gaussian proposal by the inverse normal CDF, with importance weights π/q exactly as in importance sampling.

A deterministic point set gives no honest error bars, so the sequence is *randomised* by a Cranley–Patterson rotation: one uniform shift (mod 1) applied to the whole sequence, redrawn on every reset. The result is unbiased while keeping the low discrepancy.

## What to watch

Flip **Sequence** between "Halton (QMC)" and "Pseudorandom (MC)" and compare the point patterns at the same sample count: the Halton cloud looks combed — no clumps, no holes — while the pseudorandom one has both. The weighted ESS is similar; the gain is in the *stability* of the estimates (in repeated runs the QMC mean wanders less; over many verification runs its RMS error was ~15% lower on the banana).

## Tuning

- **Proposal sd** — same trade-off as importance sampling; the stratification does not rescue a bad proposal.
- **Batch size** — larger batches make the space-filling pattern easier to see.

## References

- Niederreiter (1992). *Random Number Generation and Quasi-Monte Carlo Methods*. SIAM.
- Cranley & Patterson (1976). Randomization of number theoretic methods for multiple integration. *SIAM J. Numer. Anal.*
