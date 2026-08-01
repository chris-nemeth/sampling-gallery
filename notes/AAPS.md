# Apogee-to-Apogee Path Sampler

**In one line.** Cut the leapfrog path into segments at its *apogees* — the points where the trajectory tips from climbing the potential to descending — and propose from a random window of segments: HMC-like efficiency with remarkable robustness to tuning.

## How it works

An apogee occurs where p·∇U flips from positive to negative: a local maximum of the potential along the path. The stretch between consecutive apogees is a *segment*. Crucially, the set of apogees is a property of the path itself — start the integration from any point on the path and you recover the same segments (*segment invariance*), which is what makes the following valid.

Each iteration draws a fresh momentum, samples c ~ U{0,…,K}, and integrates backwards past c apogees and forwards past K−c, so the current segment sits uniformly at random inside a window of K+1 segments. A proposal is then drawn from the whole path with probability ∝ w(z, z′) — here the paper's recommended Scheme 3, w = ‖x′ − x‖²·exp(−H(z′)), which favours distant, high-density points — and accepted with the ratio of weight sums seen from the current point versus the proposal. Unlike NUTS, there is no recursion and no U-turn bookkeeping: the whole algorithm is a loop and a weighted draw.

## What to watch

Segments alternate blue and orange, with grey dots at the apogees; the black dot is the current point, blue the proposal. Watch how often the proposal lands several segments away — those are the long jumps that decorrelate the chain. The path is rebuilt from scratch each iteration (fresh momentum, fresh window placement).

## Tuning

- **Extra segments K** — efficiency is famously flat in K: values from 2 to 20 all work on the banana (the sweet spot in our tests was K ≈ 8–12). Compare with HMC's sharp sensitivity to L.
- **Leapfrog ε** — the acceptance mechanism uses weighted sums rather than a single endpoint energy, so it degrades gracefully as ε grows — until the integrator's stability limit, where the path blows up and the panel says so.

## References

- Sherlock, Urbas & Ludkin (2023). The apogee to apogee path sampler. *J. Comput. Graph. Statist.*
