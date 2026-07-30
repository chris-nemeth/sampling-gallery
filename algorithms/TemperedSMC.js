"use strict";

// Tempered Sequential Monte Carlo sampler (Del Moral, Doucet & Jasra 2006):
// a particle cloud moves from an easy reference distribution to the target
// along the geometric path
//     log pi_beta(x) = (1 - beta) log mu0(x) + beta log p(x),   beta: 0 -> 1,
// with mu0 = N(0, 3^2 I). Each step (a) picks the next temperature increment
// adaptively so the incremental weights keep the effective sample size near
// 0.8 N, (b) reweights, (c) resamples systematically when the ESS drops below
// N/2, and (d) rejuvenates every particle with a few random-walk Metropolis
// moves targeting pi_beta. Weights live in log space throughout. Once beta
// reaches 1 the cloud simply keeps rejuvenating, i.e. sampling the target.
MCMC.registerAlgorithm("TemperedSMC", {
  description: "Tempered Sequential Monte Carlo",

  about: function () {
    window.open("https://academic.oup.com/jrsssb/article/68/3/411/7110641");
  },

  init: function (self) {
    self.nParticles = 200;
    self.mcmcSteps = 3; // rejuvenation moves per tempering step
    self.sigma = 0.5; // rejuvenation random-walk proposal sd
    self.essTarget = 0.8; // adaptive tempering aims for ESS ~ essTarget * N
    self.essThreshold = 0.5; // resample when ESS < essThreshold * N
  },

  reset: function (self) {
    self.mu0 = new MultivariateNormal(zeros(self.dim, 1), eye(self.dim).scale(9));
    var N = Math.max(10, Math.round(self.nParticles));
    self.xs = [];
    self.logW = [];
    for (var i = 0; i < N; i++) {
      self.xs.push(self.mu0.getSample());
      self.logW.push(0);
    }
    self.beta = 0;
    self.lastResampled = false;
    MCMC.algorithms["TemperedSMC"].publish(self);
  },

  attachUI: function (self, folder) {
    folder
      .add(self, "nParticles", 50, 500)
      .step(10)
      .name("Particles N")
      .onChange(function () {
        sim.reset();
      });
    folder.add(self, "mcmcSteps", 1, 10).step(1).name("Rejuvenation moves");
    folder.add(self, "sigma", 0.1, 2).step(0.05).name("Rejuvenation σ");
    folder.open();
  },

  // Kish ESS of log-weights (log-space, max-subtracted)
  ess: function (logW) {
    var mx = -Infinity;
    for (var i = 0; i < logW.length; i++) if (logW[i] > mx) mx = logW[i];
    if (!isFinite(mx)) return 0;
    var sw = 0,
      sw2 = 0;
    for (var i = 0; i < logW.length; i++) {
      var w = Math.exp(logW[i] - mx);
      sw += w;
      sw2 += w * w;
    }
    return (sw * sw) / sw2;
  },

  // normalized weights from log-weights
  normWeights: function (logW) {
    var mx = -Infinity;
    for (var i = 0; i < logW.length; i++) if (logW[i] > mx) mx = logW[i];
    var ws = new Array(logW.length);
    var sw = 0;
    for (var i = 0; i < logW.length; i++) {
      ws[i] = isFinite(mx) ? Math.exp(logW[i] - mx) : 1;
      sw += ws[i];
    }
    for (var i = 0; i < logW.length; i++) ws[i] /= sw;
    return ws;
  },

  systematicResample: function (self, ws) {
    var n = ws.length;
    var idx = new Array(n);
    var u = Math.random() / n;
    var cum = ws[0];
    var j = 0;
    for (var i = 0; i < n; i++) {
      var pos = u + i / n;
      while (cum < pos && j < n - 1) {
        j++;
        cum += ws[j];
      }
      idx[i] = j;
    }
    var xs = new Array(n);
    for (var i = 0; i < n; i++) xs[i] = self.xs[idx[i]].copy();
    self.xs = xs;
    for (var i = 0; i < n; i++) self.logW[i] = 0;
  },

  // expose the weighted cloud to the UI (weighted mean / Kish ESS / histograms)
  publish: function (self) {
    var ws = MCMC.algorithms["TemperedSMC"].normWeights(self.logW);
    self.chain = [];
    self.chain_weights = [];
    for (var i = 0; i < self.xs.length; i++) {
      self.chain.push(self.xs[i].copy());
      self.chain_weights.push(ws[i]);
    }
    return ws;
  },

  step: function (self, visualizer) {
    var alg = MCMC.algorithms["TemperedSMC"];
    var N = self.xs.length;
    var safeLogP = function (p) {
      var v = self.logDensity(p);
      return isFinite(v) ? v : -Infinity;
    };
    self.lastResampled = false;

    // 1. adaptive temperature increment: bisect delta so the ESS of the
    // INCREMENTAL weights u_i = exp(delta * d_i) is ~ essTarget * N (or take
    // the full remaining step). Adapting on the increments alone — not the
    // updated cumulative weights — matters: between resamples the cumulative
    // ESS can already sit below the target, in which case no delta could
    // reach it and the bisection would collapse to delta -> 0, stalling beta.
    if (self.beta < 1) {
      var d = new Array(N); // incremental log-weight per unit delta
      for (var i = 0; i < N; i++) d[i] = safeLogP(self.xs[i]) - self.mu0.logDensity(self.xs[i]);
      var essAt = function (delta) {
        var lw = new Array(N);
        for (var i = 0; i < N; i++) lw[i] = delta * d[i];
        return alg.ess(lw);
      };
      var target = self.essTarget * N;
      var delta = 1 - self.beta;
      if (essAt(delta) < target) {
        var lo = 0,
          hi = delta;
        for (var it = 0; it < 30; it++) {
          var mid = (lo + hi) / 2;
          if (essAt(mid) < target) hi = mid;
          else lo = mid;
        }
        delta = (lo + hi) / 2;
      }
      for (var i = 0; i < N; i++) self.logW[i] += delta * d[i];
      self.beta = Math.min(1, self.beta + delta);
    }

    // 2. resample when the weights have degenerated (or on reaching the target)
    var kish = alg.ess(self.logW);
    if (kish < self.essThreshold * N || (self.beta >= 1 && kish < 0.999 * N)) {
      alg.systematicResample(self, alg.normWeights(self.logW));
      self.lastResampled = true;
      kish = N;
    }

    // 3. rejuvenation: random-walk MH targeting pi_beta
    var logPi = function (p) {
      return (1 - self.beta) * self.mu0.logDensity(p) + self.beta * safeLogP(p);
    };
    for (var m = 0; m < self.mcmcSteps; m++) {
      for (var i = 0; i < N; i++) {
        var prop = self.xs[i].add(MultivariateNormal.getSample(self.dim).scale(self.sigma));
        var logA = logPi(prop) - logPi(self.xs[i]);
        if (isFinite(logA) && Math.log(Math.random()) < logA) self.xs[i] = prop;
      }
    }

    // 4. publish + draw the weighted cloud
    var ws = alg.publish(self);
    var wmax = 1e-300;
    for (var i = 0; i < N; i++) if (ws[i] > wmax) wmax = ws[i];
    var points = [];
    for (var i = 0; i < N; i++) {
      points.push({
        center: [self.xs[i][0], self.xs[i][1]],
        radius: 0.03 + 0.12 * (ws[i] / wmax),
        fill: "#0088b0",
        alpha: 0.75,
      });
    }
    visualizer.queue.push({
      type: "overlay",
      clear: true,
      points: points,
      labels: [
        self.beta >= 1 ? "β = 1.00 — sampling the target" : "β = " + self.beta.toFixed(2),
        "ESS " + kish.toFixed(0) + " / " + N + (self.lastResampled ? " (resampled)" : ""),
      ],
      histograms: true,
    });
  },
});
