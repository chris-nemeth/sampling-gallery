"use strict";

// Stochastic-gradient PDMPs (Fearnhead, Grazzi, Nemeth & Roberts 2024,
// arXiv:2406.19051): piecewise-deterministic samplers driven by NOISY
// gradient estimates. The O(1) scheme freezes a stochastic gradient estimate
// over a short window of length h; within the window the event rates are
// constant, so event times are exact exponential draws — no thinning bounds
// needed. At each event (and at window boundaries) a fresh estimate is drawn.
//
//  - SG Zig-Zag: coordinate i flips its velocity at rate
//        max(0, v_i ĝ_i) + λ_ref,        ĝ = frozen noisy grad of U
//  - SG Bouncy Particle: reflect v in a freshly drawn ĝ at rate
//        max(0, v · ĝ), with velocity refreshes v ~ N(0, I) at rate λ_ref.
//
// There is no dataset in these demos, so — as in the SGLD entry — a tunable
// N(0, σ²I) perturbation of the true gradient stands in for minibatch noise.
// The sampler is approximate: its invariant measure inflates as h or the
// gradient noise grow, which is exactly what the demo lets you provoke.
MCMC.registerAlgorithm("SGPDMP", {
  description: "Stochastic-Gradient PDMP",

  about: function () {
    window.open("https://arxiv.org/abs/2406.19051");
  },

  init: function (self) {
    self.dynamics = "Zig-Zag";
    self.gradientNoise = 0.5; // sd of the simulated minibatch noise
    self.h = 0.05; // gradient refresh window (the O(h) bias knob)
    self.refreshRate = 0.5; // λ_ref: ZZ random flips / BPS velocity refresh
    self.tStep = 0.5; // continuous time advanced per step (sample spacing)
  },

  reset: function (self) {
    self.x = MultivariateNormal.getSample(self.dim);
    self.chain = [self.x.copy()];
    if (self.dynamics === "Zig-Zag") {
      self.v = zeros(self.dim, 1);
      for (var k = 0; k < self.dim; k++) self.v[k] = Math.random() < 0.5 ? -1 : 1;
    } else {
      self.v = MultivariateNormal.getSample(self.dim);
    }
    self.gHat = null; // frozen noisy gradient (drawn on first step)
    self.winLeft = 0; // time left before the next scheduled gradient refresh
    self.nEvents = 0;
    self.tTotal = 0;
  },

  attachUI: function (self, folder) {
    folder
      .add(self, "dynamics", ["Zig-Zag", "Bouncy Particle"])
      .name("Dynamics")
      .onChange(function () {
        sim.reset();
      });
    folder.add(self, "gradientNoise", 0, 4).step(0.1).name("Gradient noise σ");
    folder.add(self, "h", 0.02, 0.5).step(0.01).name("Refresh window h");
    folder.add(self, "refreshRate", 0, 2).step(0.05).name("Refresh rate λ");
    folder.add(self, "tStep", 0.1, 2).step(0.1).name("Sample interval");
    folder.open();
  },

  step: function (self, visualizer) {
    var isZZ = self.dynamics === "Zig-Zag";
    var gradU = function (x) {
      return self.gradLogDensity(x).scale(-1);
    };
    // stochastic gradient: true gradient plus simulated minibatch noise
    var noisyGrad = function (x) {
      var g = gradU(x);
      if (self.gradientNoise > 0) g.increment(MultivariateNormal.getSample(self.dim).scale(self.gradientNoise));
      return g;
    };
    var expTime = function (rate) {
      return rate > 1e-12 ? -Math.log(Math.random()) / rate : Infinity;
    };

    if (!self.gHat) {
      self.gHat = noisyGrad(self.x);
      self.winLeft = self.h;
    }

    var trajectory = [self.x.copy()];
    var tRemaining = self.tStep;
    var guard = 3000;

    while (tRemaining > 1e-12 && guard-- > 0) {
      // exact exponential clocks against the frozen (constant) rates
      var tauEv = Infinity,
        evK = -1,
        evKind = 0; // 1 = flip/bounce, 2 = refresh (BPS)
      if (isZZ) {
        for (var k = 0; k < self.dim; k++) {
          var rate = Math.max(0, self.v[k] * self.gHat[k]) + self.refreshRate;
          var t = expTime(rate);
          if (t < tauEv) {
            tauEv = t;
            evK = k;
            evKind = 1;
          }
        }
      } else {
        var tB = expTime(Math.max(0, self.v.dot(self.gHat)));
        var tR = expTime(self.refreshRate);
        if (tB < tR) {
          tauEv = tB;
          evKind = 1;
        } else {
          tauEv = tR;
          evKind = 2;
        }
      }

      var dt = Math.min(tauEv, self.winLeft, tRemaining);
      self.x.increment(self.v.scale(dt)); // scale() returns a new vector
      trajectory.push(self.x.copy());
      self.tTotal += dt;
      tRemaining -= dt;
      self.winLeft -= dt;

      if (dt === tauEv) {
        // event: refresh the gradient estimate at the event position first
        self.gHat = noisyGrad(self.x);
        self.winLeft = self.h;
        self.nEvents++;
        if (isZZ) {
          self.v[evK] *= -1;
        } else if (evKind === 1) {
          // reflect v in the fresh noisy gradient
          var g = self.gHat;
          var vg = self.v.dot(g);
          var g2 = g.dot(g);
          if (g2 > 1e-12) self.v.increment(g.copy().scale((-2 * vg) / g2));
        } else {
          self.v = MultivariateNormal.getSample(self.dim);
        }
      } else if (self.winLeft <= 1e-12) {
        // scheduled refresh of the frozen gradient estimate
        self.gHat = noisyGrad(self.x);
        self.winLeft = self.h;
      }
    }

    self.chain.push(self.x.copy());

    // overlay: the frozen noisy gradient (orange) against the true gradient
    // (blue) at the current position — the jitter between them is the point
    var gTrue = gradU(self.x);
    var mkArrow = function (g, color) {
      var n = g.norm();
      if (!(n > 1e-9)) return null;
      var s = 0.9 / n;
      return {
        from: [self.x[0], self.x[1]],
        to: [self.x[0] - g[0] * s, self.x[1] - g[1] * s], // -grad U = uphill in π
        color: color,
        lw: 1.5,
        alpha: 0.85,
        arrow: true,
      };
    };
    var arrows = [];
    var a1 = mkArrow(gTrue, "#0072b2");
    var a2 = mkArrow(self.gHat, "#e69f00");
    if (a1) arrows.push(a1);
    if (a2) arrows.push(a2);
    visualizer.queue.push({
      type: "overlay",
      clear: true,
      segments: arrows,
      metrics: [{ k: "Events/time", v: (self.nEvents / Math.max(self.tTotal, 1e-9)).toFixed(1) }],
    });
    visualizer.queue.push({
      type: "proposal",
      proposal: self.x.copy(),
      trajectory: trajectory,
    });
    // PDMPs are rejection-free: every move is "accepted"
    visualizer.queue.push({ type: "accept", proposal: self.x.copy() });
  },
});
