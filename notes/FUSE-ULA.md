# Tuning-Free ULA (FUSE)

**In one line.** ULA whose step size sets *itself* from the history of distances moved and gradients seen — no manual step size tuning required.

## How it works

ULA's one knob is its step size, and the right value depends on the target's scale, which you don't know in advance. FUSE replaces the fixed step with a "distance-over-gradients" schedule adapted from parameter-free optimisation:

η_t = max(r_ε, max_{s≤t} d_s) / √(∑_{s≤t} ‖∇log π(x_s)‖²),

where d_s is the distance travelled from a reference point and r_ε is a small initial movement scale. Early on the numerator grows as the chain travels (step size ramps up); the accumulating gradient energy in the denominator then tempers it. The schedule inherits parameter-free optimisation's guarantee: it tracks the scale the problem itself reveals, with only mild dependence on r_ε.

This entry is the single-chain special case of the paper's general construction, which develops tuning-free schedules for samplers viewed as optimisation on the space of probability measures.

## What to watch

Reset on different targets and watch the first few dozen steps: the step size ramps from cautious to confident at a rate set by the target, not by you. Compare against plain ULA, where you would have to re-tune ε when switching from the standard Gaussian to the banana. The step size can be read off the movement scale of the chain.

## Tuning

- **r_ε** — the initial movement scale; the schedule's dependence on it is deliberately weak (try 10× larger and smaller).
- **λ** — the noise scale of the underlying flow; λ = 1 corresponds to standard ULA.

## References

- Sharrock & Nemeth (2025). Tuning-free sampling via optimization on the space of probability measures. arXiv:2510.25315.
- Ivgi, Hinder & Carmon (2023). DoG is SGD's best friend: a parameter-free dynamic step size schedule. *ICML* (the optimisation ancestor).
