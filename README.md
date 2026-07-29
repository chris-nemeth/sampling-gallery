# MCMC Visualisations

**Live demo: https://chris-nemeth.github.io/MCMC-Visualisations/**

An interactive gallery of Markov-chain Monte Carlo (and related) sampling
algorithms — animated, tunable, and explained in plain language. Static,
buildless HTML: open it in a browser, or serve the folder.

## Pages
- `index.html` — landing atlas: all algorithms grouped by family, with
  animated thumbnails and plain-language descriptions. Links into the sampler.
- `sampler.html` — the interactive app, driven by the project's real MCMC
  engine (`lib/`, `main/`, `algorithms/`): every one of the 20 samplers in the
  rail runs its genuine algorithm. Six target distributions (banana / donut /
  Gaussian / mixture / funnel / squiggle), per-sampler tunable parameters,
  live diagnostics (steps / acceptance / mean / ESS), a 2D view with proposal
  arrows, gradients and trajectories rendered live, and a rotating 3D
  density-surface view with the chain and step geometry lifted onto it.
- `classic.html` — the original interface (`app.html`), kept for continuity:
  a full-window 2D visualizer with a lil-gui control panel.
- `styles.css` — Broadsheet design tokens + component classes.
- `docs/*.gif` — animation thumbnails used by the landing page.

## Samplers
Random Walk MH, Adaptive MH, HMC, HMC with dual averaging, NUTS (naive,
efficient, and dual-averaging), MALA, ULA, SGLD, SGLD-CV, tuning-free ULA
(FUSE), H2MC, Microcanonical HMC, Gibbs, DE-MCMC-Z, Zig-Zag, Bouncy Particle
Sampler, SVGD, and Nested Sampling (RadFriends).

## Run locally
Open `index.html` in any modern browser, or from the folder root:

    python3 -m http.server

then visit http://localhost:8000/

## Notes
- No build step; all libraries are vendored in `lib/` (fonts load from Google
  Fonts when online; the pages remain functional offline).
- Every sampler runs the real, tested implementation from `algorithms/` —
  there are no placeholder fall-backs.
- Redesigned in the Broadsheet system; based on the original demo by Chi Feng:
  https://github.com/chi-feng/mcmc-demo
