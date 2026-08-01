# Stochastic Gradient Langevin Dynamics

**In one line.** ULA driven by *noisy* gradients — the algorithm that made Bayesian inference run on minibatches, and the honest picture of what gradient noise does to a sampler.

## How it works

In large-data problems the log-posterior gradient is a sum over N observations; SGLD replaces it with an unbiased minibatch estimate and runs the unadjusted Langevin update. Welling & Teh's observation: as the step size decays, the *injected* Langevin noise (scale ε) dominates the *gradient* noise (scale ε²... in the drift term), so the algorithm transitions from noisy optimisation to approximate posterior sampling without ever computing an accept/reject step on the full data.

There is no dataset in this demo, so a tunable Gaussian perturbation of the true gradient stands in for minibatch noise — which makes the noise level a slider you can experiment with, rather than a fact of your data.

**Control variates** (SGLD-CV) anchor the gradient estimate at the posterior mode: the noisy estimate is re-centred by the difference of stochastic and exact gradients at the mode, so estimator variance shrinks near the mode — visibly calmer trajectories exactly where the chain spends its time.

## What to watch

Increase **gradient noise** and watch the sample cloud inflate beyond the contours — gradient noise acts like extra temperature. Then enable **control variates**: the cloud tightens near the mode while behaving similarly in the tails, which is precisely the CV guarantee.

## Tuning

- **Step size ε** — as in ULA, bias vs speed.
- **Gradient noise** — stands in for minibatch size: more noise ≈ smaller batches.
- **Control variates** — toggle and compare cloud width at equal noise.

## References

- Welling & Teh (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
- Baker, Fearnhead, Fox & Nemeth (2019). Control variates for stochastic gradient MCMC. *Statist. Comput.*
