# MCMC Visualisations — redesigned UI

A Broadsheet-styled redesign of the MCMC interactive gallery. Two static,
buildless HTML pages — open either directly in a browser, or serve the folder.

## Files
- `gallery.html` — landing atlas: all algorithms grouped by family, with
  animated thumbnails and plain-language descriptions. Links into the sampler.
- `sampler.html` — the interactive app. Driven by the project's real MCMC
  engine (`lib/`, `main/`, `algorithms/`): every sampler in the rail runs its
  genuine algorithm. Live sampling on four targets (banana / donut / Gaussian /
  mixture), a control rail, per-sampler tunable parameters, and live
  diagnostics (steps / acceptance / running mean) over a 2D density heatmap.
- `styles.css` — Broadsheet design tokens + component classes, used by both
  `gallery.html` and `sampler.html`.
- `docs/*.gif` — animation thumbnails used by `gallery.html`.

## Use
Open `gallery.html` in any modern browser, or from the folder root:

    python3 -m http.server

then visit http://localhost:8000/gallery.html

## Notes
- No build step, no dependencies, no network required (fonts load from Google
  Fonts when online; the pages remain functional offline).
- Every sampler runs the project's real, tested implementation from
  `algorithms/` (RWMH, HMC, NUTS, MALA, ULA, SGLD, Gibbs, SVGD) — there are no
  placeholder fall-backs.
- Redesigned in the Broadsheet system; original demo:
  https://github.com/chi-feng/mcmc-demo
