"use strict";

// The Bouncy Particle Sampler (BPS): a non-reversible, rejection-free
// piecewise-deterministic Markov process. Bouchard-Cote, Vollmer & Doucet
// (2018), "The Bouncy Particle Sampler: A Nonreversible Rejection-Free Markov
// Chain Monte Carlo Method", JASA, arXiv:1510.02451.
//
// State is a position x and a continuous velocity v. Between events the particle
// moves in a straight line x(t) = x + v t. Two competing Poisson processes act:
//
//   * bounces at rate  lambda(x, v) = max(0, <v, grad U(x)>)
//                                   = max(0, -<v, grad log pi(x)>),
//     at which the velocity reflects off the gradient hyperplane:
//         v  <-  v - 2 (<v, g> / ||g||^2) g,     g = grad log pi(x)
//     (equivalently reflecting about grad U, since U = -log pi);
//
//   * refreshments at a constant rate lambda_ref, at which v is resampled from
//     N(0, I). Refreshment is required for ergodicity -- without it BPS can be
//     reducible (set the rate to 0 to see this).
//
// As with the Zig-Zag sampler, one step() advances continuous time by a fixed
// "sample interval" tStep and records the end position, giving a time-uniform
// (unweighted) approximation of the invariant measure. The next event time is
// found either by discretization (robust everywhere) or by Poisson thinning
// (exact when <v, grad U> is affine along the ray, e.g. Gaussian targets).
MCMC.registerAlgorithm("BouncyParticle", {
  description: "Bouncy Particle Sampler",

  about: function () {
    window.open("https://arxiv.org/abs/1510.02451");
  },

  init: function (self) {
    self.method = "Discretized";
    self.tStep = 0.5; // continuous time advanced per step (sample spacing)
    self.dt = 0.02; // max micro-step for the discretized method
    self.refreshRate = 1.0; // constant velocity-refreshment rate
    self.thinHorizon = 0.2; // look-ahead window for the thinning envelope
  },

  reset: function (self) {
    self.x = MCMC.algorithms["BouncyParticle"].smartInit(self);
    self.v = MultivariateNormal.getSample(self.dim); // v ~ N(0, I)
    self.chain = [self.x.copy()];
  },

  // Initialise from the best of a few N(0, I) draws: some targets (e.g. the
  // flower) have near-zero-mass regions around the origin where bounce rates
  // blow up, and starting a PDMP inside one makes early mixing terrible.
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
    folder.add(self, "refreshRate", 0, 5).step(0.1).name("Refresh rate");
    folder.open();
  },

  // reflect v off the gradient hyperplane: v - 2 (<v,g>/||g||^2) g
  reflect: function (v, g) {
    var gg = g.norm2();
    if (gg < 1e-12) return v.copy(); // no gradient direction to bounce off
    var vg = v.dot(g);
    return v.subtract(g.scale((2 * vg) / gg));
  },

  // Next event (bounce or refresh) along x + t v for t in [0, maxTime], by fine
  // time-stepping the superposed process. The micro-step adapts to the local
  // total rate (Lambda * step <= 0.5) so stiff regions are resolved rather than
  // degenerating into event-every-step jitter; an event fires with probability
  // 1 - exp(-Lambda * step) and is classified bounce/refresh proportionally to
  // the component rates (exact superposition decomposition).
  // Returns { dt, type: "bounce" | "refresh" | "none", x }.
  nextEventDiscretized: function (self, x, v, maxTime) {
    var t = 0;
    var xc = x.copy();
    var guard = 0;
    while (t < maxTime && guard++ < 50000) {
      var g = self.gradLogDensity(xc);
      var lamB = Math.max(0, -v.dot(g)); // = max(0, <v, grad U>)
      var lam = lamB + self.refreshRate;
      var step = Math.min(self.dt, maxTime - t, lam > 0 ? 0.5 / lam : self.dt);
      var fired = Math.random() < 1 - Math.exp(-lam * step);
      xc = xc.add(v.scale(step));
      t += step;
      if (fired) return { dt: t, type: Math.random() * lam < lamB ? "bounce" : "refresh", x: xc };
    }
    return { dt: maxTime, type: "none", x: xc };
  },

  // Next event by Poisson thinning of the superposed rate
  // Lambda(s) = max(0, -<v, grad log pi(x + s v)>) + refreshRate.
  // The envelope M is the safety-inflated maximum of Lambda over the look-ahead
  // window (sampled at several points); a candidate ~ Exp(M) is accepted with
  // prob Lambda/M, and classified as refresh with prob refreshRate/Lambda.
  nextEventThinning: function (self, x, v, maxTime) {
    var t = 0;
    var xc = x.copy();
    var guard = 0;
    while (t < maxTime && guard++ < 10000) {
      var h = Math.min(self.thinHorizon, maxTime - t);
      // envelope: max of the total rate over the window, sampled at nb points
      // and inflated. Exact when the rate is convex along the ray (e.g.
      // Gaussian targets); the short horizon + dense sampling keep the
      // estimate a valid bound on the stiff targets too (audited empirically).
      var nb = 8;
      var M = 0;
      for (var k = 0; k < nb; k++) {
        var s = (h * k) / (nb - 1);
        var g = self.gradLogDensity(xc.add(v.scale(s)));
        var lam = Math.max(0, -v.dot(g)) + self.refreshRate;
        if (lam > M) M = lam;
      }
      M = 1.3 * M + 1e-6;
      var tau = -Math.log(Math.random()) / M;
      if (tau > h) {
        xc = xc.add(v.scale(h));
        t += h;
        continue;
      }
      xc = xc.add(v.scale(tau));
      t += tau;
      var g2 = self.gradLogDensity(xc);
      var lamB = Math.max(0, -v.dot(g2));
      var Lam = lamB + self.refreshRate;
      if (Math.random() < Lam / M) {
        var type = Math.random() < self.refreshRate / Lam ? "refresh" : "bounce";
        return { dt: t, type: type, x: xc };
      }
      // thinned out: continue from the candidate point
    }
    return { dt: maxTime, type: "none", x: xc };
  },

  // append points along the straight segment a -> b for smooth animation
  pushSegment: function (traj, a, b) {
    var d = b.subtract(a).norm();
    var n = Math.max(1, Math.ceil(d / 0.2));
    for (var k = 1; k <= n; k++) traj.push(a.add(b.subtract(a).scale(k / n)));
  },

  step: function (self, visualizer) {
    var alg = MCMC.algorithms["BouncyParticle"];
    var x = self.x.copy();
    var v = self.v.copy();
    var v0 = v.copy();
    var trajectory = [x.copy()];
    var tRemaining = self.tStep;
    var guard = 0;

    while (tRemaining > 1e-12 && guard++ < 10000) {
      var res =
        self.method === "Thinning"
          ? alg.nextEventThinning(self, x, v, tRemaining)
          : alg.nextEventDiscretized(self, x, v, tRemaining);
      alg.pushSegment(trajectory, x, res.x);
      x = res.x;
      tRemaining -= res.dt;
      if (res.type === "bounce") v = alg.reflect(v, self.gradLogDensity(x));
      else if (res.type === "refresh") v = MultivariateNormal.getSample(self.dim);
      else break; // window elapsed with no event
    }

    self.x = x;
    self.v = v;
    self.chain.push(x.copy());

    visualizer.queue.push({
      type: "proposal",
      proposal: x.copy(),
      trajectory: trajectory,
      initialMomentum: v0.scale(0.4), // draw the initial velocity direction
    });
    // PDMPs are rejection-free: every move is "accepted"
    visualizer.queue.push({ type: "accept", proposal: x.copy() });
  },
});
