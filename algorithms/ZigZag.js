"use strict";

// The Zig-Zag sampler: a non-reversible, rejection-free piecewise-deterministic
// Markov process (PDMP). Bierkens, Fearnhead & Roberts (2019), "The Zig-Zag
// process and super-efficient sampling for Bayesian analysis of big data",
// Annals of Statistics, arXiv:1607.03188.
//
// State is a position x and a velocity v with components in {-1, +1}. Between
// events the particle moves in a straight line x(t) = x + v t. Each coordinate
// i carries an independent Poisson clock with switching rate
//
//     lambda_i(x, v) = max(0,  v_i * d_i U(x))  =  max(0, -v_i * d_i log pi(x))
//
// (U = -log pi is the potential). When coordinate i's clock fires, v_i flips.
//
// One step() advances continuous time by a fixed "sample interval" tStep,
// applying any switches that occur inside that window, and records the position
// at the end of the interval. Because the samples are equally spaced in
// continuous time they form a time-uniform (unweighted) approximation of the
// invariant measure -- no importance weights needed.
//
// The next switch time within a window is found either by fine time
// discretization (robust on every target, small O(dt) bias) or by Poisson
// thinning (exact when the rate is affine along the ray, e.g. Gaussian targets;
// uses a numerically bounded envelope otherwise). The method is selectable in
// the control panel.
MCMC.registerAlgorithm("ZigZag", {
  description: "Zig-Zag Sampler",

  about: function () {
    window.open("https://arxiv.org/abs/1607.03188");
  },

  init: function (self) {
    self.method = "Discretized";
    self.tStep = 0.5; // continuous time advanced per step (sample spacing)
    self.dt = 0.02; // max micro-step for the discretized method
    self.thinHorizon = 0.2; // look-ahead window for the thinning envelope
  },

  reset: function (self) {
    self.x = MCMC.algorithms["ZigZag"].smartInit(self);
    self.v = MCMC.algorithms["ZigZag"].randomSigns(self.dim);
    self.chain = [self.x.copy()];
  },

  // Initialise from the best of a few N(0, I) draws: some targets (e.g. the
  // flower) have near-zero-mass regions around the origin where switching
  // rates blow up, and starting a PDMP inside one makes early mixing terrible.
  smartInit: function (self) {
    var best = MultivariateNormal.getSample(self.dim);
    if (!self.logDensity) return best;
    var bestLp = self.logDensity(best);
    for (var k = 1; k < 10; k++) {
      var c = MultivariateNormal.getSample(self.dim);
      var lp = self.logDensity(c);
      if (lp > bestLp) {
        best = c;
        bestLp = lp;
      }
    }
    return best;
  },

  attachUI: function (self, folder) {
    folder.add(self, "method", ["Discretized", "Thinning"]).name("Event simulation");
    folder.add(self, "tStep", 0.1, 2).step(0.1).name("Sample interval");
    folder.add(self, "dt", 0.005, 0.1).step(0.005).name("Δt (discretized)");
    folder.open();
  },

  // velocity with each component drawn uniformly from {-1, +1}
  randomSigns: function (dim) {
    var v = zeros(dim, 1);
    for (var i = 0; i < dim; i++) v[i] = Math.random() < 0.5 ? -1 : 1;
    return v;
  },

  // Find the next coordinate switch along the ray x + t v for t in [0, maxTime],
  // by fine time-stepping the superposed process. The micro-step adapts to the
  // local rate (lambda_tot * step <= 0.5) so that stiff regions are resolved
  // rather than degenerating into flip-every-step jitter; a switch fires with
  // probability 1 - exp(-lambda_tot * step) and the coordinate is then chosen
  // proportionally to its rate (exact superposition decomposition). Returns
  // { dt, coord, x } with coord = -1 if the window elapsed without a switch.
  nextSwitchDiscretized: function (self, x, v, maxTime) {
    var t = 0;
    var xc = x.copy();
    var guard = 0;
    while (t < maxTime && guard++ < 50000) {
      var g = self.gradLogDensity(xc);
      var r0 = Math.max(0, -v[0] * g[0]);
      var r1 = Math.max(0, -v[1] * g[1]);
      var lam = r0 + r1;
      var step = Math.min(self.dt, maxTime - t, lam > 0 ? 0.5 / lam : self.dt);
      var fired = Math.random() < 1 - Math.exp(-lam * step);
      xc = xc.add(v.scale(step));
      t += step;
      if (fired) return { dt: t, coord: Math.random() * lam < r0 ? 0 : 1, x: xc };
    }
    return { dt: maxTime, coord: -1, x: xc };
  },

  // Find the next coordinate switch by Poisson thinning of the superposed rate
  // Lambda(s) = sum_i max(0, -v_i d_i log pi(x + s v)). Over each look-ahead
  // window the envelope M is the (safety-inflated) maximum of Lambda sampled at
  // several points; a candidate time ~ Exp(M) is accepted with prob Lambda/M.
  // Exact when Lambda is affine in s (its window-max is attained at an endpoint,
  // which is sampled); otherwise the envelope is a numerical estimate.
  nextSwitchThinning: function (self, x, v, maxTime) {
    var t = 0;
    var xc = x.copy();
    var guard = 0;
    while (t < maxTime && guard++ < 10000) {
      var h = Math.min(self.thinHorizon, maxTime - t);
      // envelope: max of the superposed rate over the window, sampled at nb
      // points and inflated. Exact when the rate is convex along the ray
      // (e.g. Gaussian targets); the short horizon + dense sampling keep the
      // estimate a valid bound on the stiff targets too (audited empirically).
      var nb = 8;
      var M = 0;
      for (var k = 0; k < nb; k++) {
        var s = (h * k) / (nb - 1);
        var g = self.gradLogDensity(xc.add(v.scale(s)));
        var lam = 0;
        for (var i = 0; i < self.dim; i++) lam += Math.max(0, -v[i] * g[i]);
        if (lam > M) M = lam;
      }
      M = 1.3 * M + 1e-6;
      var tau = -Math.log(Math.random()) / M;
      if (tau > h) {
        // no candidate in this window; slide to its end and re-bound
        xc = xc.add(v.scale(h));
        t += h;
        continue;
      }
      xc = xc.add(v.scale(tau));
      t += tau;
      // evaluate the true coordinate rates at the candidate point
      var g2 = self.gradLogDensity(xc);
      var rates = zeros(self.dim, 1);
      var Lam = 0;
      for (var i = 0; i < self.dim; i++) {
        rates[i] = Math.max(0, -v[i] * g2[i]);
        Lam += rates[i];
      }
      if (Math.random() < Lam / M) {
        // accept: pick the switching coordinate proportional to its rate
        var u = Math.random() * Lam;
        var acc = 0;
        var coord = self.dim - 1;
        for (var i = 0; i < self.dim; i++) {
          acc += rates[i];
          if (u <= acc) {
            coord = i;
            break;
          }
        }
        return { dt: t, coord: coord, x: xc };
      }
      // thinned out: continue from the candidate point
    }
    return { dt: maxTime, coord: -1, x: xc };
  },

  // append points along the straight segment a -> b for smooth animation
  pushSegment: function (traj, a, b) {
    var d = b.subtract(a).norm();
    var n = Math.max(1, Math.ceil(d / 0.2));
    for (var k = 1; k <= n; k++) traj.push(a.add(b.subtract(a).scale(k / n)));
  },

  step: function (self, visualizer) {
    var alg = MCMC.algorithms["ZigZag"];
    var x = self.x.copy();
    var v = self.v.copy();
    var v0 = v.copy();
    var trajectory = [x.copy()];
    var tRemaining = self.tStep;
    var guard = 0;

    while (tRemaining > 1e-12 && guard++ < 10000) {
      var res =
        self.method === "Thinning"
          ? alg.nextSwitchThinning(self, x, v, tRemaining)
          : alg.nextSwitchDiscretized(self, x, v, tRemaining);
      alg.pushSegment(trajectory, x, res.x);
      x = res.x;
      tRemaining -= res.dt;
      if (res.coord >= 0) v[res.coord] = -v[res.coord];
      else break; // window elapsed with no switch
    }

    self.x = x;
    self.v = v;
    self.chain.push(x.copy());

    visualizer.queue.push({
      type: "proposal",
      proposal: x.copy(),
      trajectory: trajectory,
      initialMomentum: v0.scale(0.6), // draw the initial velocity direction
    });
    // PDMPs are rejection-free: every move is "accepted"
    visualizer.queue.push({ type: "accept", proposal: x.copy() });
  },
});
