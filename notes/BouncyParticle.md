# Bouncy Particle Sampler

**In one line.** A particle travels in straight lines and *bounces* off the log-density gradient like light off a mirror — plus occasional velocity refreshments to guarantee it explores everywhere.

## How it works

Like Zig-Zag, a piecewise-deterministic process: x(t) = x + vt with a unit velocity v. Bounce events arrive at rate λ(x, v) = max(0, −v·∇log π(x)) — again, only when heading downhill. At a bounce, the velocity reflects in the hyperplane of the gradient: v ← v − 2(v·ĝ)ĝ, preserving speed but redirecting along the contour. Reflection alone can trap the dynamics on certain targets (it preserves too much structure), so the velocity is also **refreshed** to a fresh random direction at a constant rate — the mixing guarantee.

## What to watch

Straight flight paths with mirror-like kinks at bounces. On the **banana**, the particle skims along the ridge, bouncing gently between its walls — very long moves per gradient evaluation. Turn the **refresh rate** down and watch the trajectories become beautiful and suspicious: near-periodic orbits that circulate without exploring (on the standard Gaussian this is easiest to see). Turn it up and the paths shorten toward diffusive behaviour. The sweet spot is the whole game.

## Tuning

- **Refresh rate** — the key knob: too low risks non-ergodic orbiting, too high throws away the non-reversible advantage.
- **Event method** — as with Zig-Zag, discretised vs thinned event simulation.

## References

- Bouchard-Côté, Vollmer & Doucet (2018). The bouncy particle sampler: a nonreversible rejection-free MCMC method. *JASA*.
- Peters & de With (2012). Rejection-free Monte Carlo sampling for general potentials. *Phys. Rev. E*.
