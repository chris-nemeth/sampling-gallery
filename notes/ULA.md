# Unadjusted Langevin Algorithm

**In one line.** The Langevin discretisation x′ = x + (ε²/2)∇log π + εξ, *without* the Metropolis correction — fast, biased, and the reason "unadjusted" is a technical term and not an insult.

## How it works

Drop MALA's accept/reject step and just iterate the discretised diffusion. The chain no longer targets π exactly: its stationary distribution is a smeared version of π with bias of order ε (in Wasserstein distance, under regularity). In exchange, every step moves, no density evaluations are needed for a correction, and the algorithm scales to settings where an MH step is impractical — it is the backbone of stochastic-gradient MCMC and of gradient-flow views of sampling.

## What to watch

Run ULA and MALA side by side on the **standard** Gaussian with a large step size: ULA's cloud is visibly *wider* than the contours — that's the discretisation bias, not slow mixing. Shrink ε and the cloud tightens to the truth while the movement per step drops: the bias–speed dial in action. The marginal histograms make the inflation easy to see.

## Tuning

- **Step size ε** — the only knob, and it is exactly the bias/speed trade-off. There is no "safe" ε that is also fast; that tension is what FUSE-ULA automates and what MALA's correction eliminates at the cost of rejections.

## References

- Roberts & Tweedie (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*.
- Durmus & Moulines (2017). Nonasymptotic convergence analysis for the unadjusted Langevin algorithm. *Ann. Appl. Probab.*
