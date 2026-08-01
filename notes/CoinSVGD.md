# Coin Stein Variational Gradient Descent

**In one line.** SVGD without a learning rate: each particle coordinate is a gambler placing coin-betting wagers on the drift, and the step size *emerges* from the history of bets.

## How it works

The drift is exactly SVGD's φ(x) — same kernel, same attraction/repulsion. What changes is the optimiser. Instead of x ← x + ε·φ(x) with a tuned ε, each coordinate of each particle runs a parameter-free coin-betting scheme (COCOB-style): track the largest drift seen L, the running drift sum S, the absolute sum A, and a "wealth" from past bets, reward ← max(reward + (x − x₀)·c, 0); then bet

x = x₀ + S / (L(A + L)) · (L + reward).

Early on, small wealth means cautious steps; as bets pay off (drift keeps pointing the same way), wealth compounds and steps grow — an automatic warm-up-then-accelerate schedule, per coordinate, with nothing to tune. The construction inherits parameter-free online-learning guarantees: the scheme is competitive with the best fixed step size in hindsight.

## What to watch

Reset and watch the first ~50 iterations next to plain SVGD: Coin SVGD starts cautiously, then visibly accelerates as wealth accumulates — no AdaGrad, no ε slider anywhere in its panel. On the banana it typically reaches the ridge shape *sooner* than default-tuned SVGD at equal iteration counts. The **KSD²** metric gives the honest comparison: same kernel, same diagnostic, different optimiser.

## Tuning

- **Bandwidth / median heuristic** — the kernel is still yours to choose.
- There is deliberately no step size. If you feel the urge to tune something, that is the point of the paper.

## References

- Sharrock & Nemeth (2023). Coin sampling: gradient-based Bayesian inference without learning rates. *ICML*.
- Orabona & Tommasi (2017). Training deep networks without learning rates through coin betting. *NeurIPS*.
