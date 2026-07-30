"use strict";

// Quasi-Monte Carlo importance sampling: draws come from a LOW-DISCREPANCY
// sequence (a randomized Halton sequence in bases 2 and 3, mapped through the
// Gaussian proposal by the inverse normal CDF) instead of pseudorandom
// numbers, and carry importance weights p/q exactly as in importance
// sampling. The space-filling pattern of the point set is the visual payload:
// QMC stratifies the proposal far more evenly than i.i.d. draws, which lowers
// the variance of the resulting estimates. Toggle the sequence to
// "Pseudorandom" to compare like for like. The Cranley-Patterson random shift
// (re-drawn on every reset) makes the sequence a randomized QMC point set, so
// the weighted estimates remain unbiased.
MCMC.registerAlgorithm("QuasiMonteCarlo", {
  description: "Quasi-Monte Carlo",

  about: function () {
    window.open("https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method");
  },

  init: function (self) {
    self.sequence = "Halton (QMC)";
    self.proposalSd = 2.5;
    self.batchSize = 10;
  },

  reset: function (self) {
    self.proposalDist = new MultivariateNormal(
      zeros(self.dim, 1),
      eye(self.dim).scale(self.proposalSd * self.proposalSd)
    );
    self._idx = 0;
    // Cranley-Patterson rotation: a fresh uniform shift each reset randomizes
    // the deterministic sequence without disturbing its low discrepancy
    self._shift = [Math.random(), Math.random()];
    self._xs = [];
    self._lws = [];
    self.chain = [];
    self.chain_weights = [];
  },

  attachUI: function (self, folder) {
    folder
      .add(self, "sequence", ["Halton (QMC)", "Pseudorandom (MC)"])
      .name("Sequence")
      .onChange(function () {
        sim.reset();
      });
    folder
      .add(self, "proposalSd", 0.5, 5)
      .step(0.1)
      .name("Proposal sd")
      .onChange(function () {
        sim.reset();
      });
    folder.add(self, "batchSize", 1, 50).step(1).name("Batch size");
    folder.open();
  },

  // radical-inverse (van der Corput) sequence in the given base
  halton: function (i, base) {
    var f = 1,
      r = 0;
    while (i > 0) {
      f /= base;
      r += f * (i % base);
      i = Math.floor(i / base);
    }
    return r;
  },

  // Acklam's rational approximation to the inverse normal CDF (|err| < 1e-9)
  invNorm: function (p) {
    var a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    var b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    var c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    var d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    var pl = 0.02425;
    var q, r;
    if (p < pl) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p <= 1 - pl) {
      q = p - 0.5;
      r = q * q;
      return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  },

  step: function (self, visualizer) {
    var alg = MCMC.algorithms["QuasiMonteCarlo"];
    var qmc = self.sequence === "Halton (QMC)";
    var batch = [];
    for (var b = 0; b < self.batchSize; b++) {
      var u0, u1;
      if (qmc) {
        self._idx++;
        u0 = (alg.halton(self._idx, 2) + self._shift[0]) % 1;
        u1 = (alg.halton(self._idx, 3) + self._shift[1]) % 1;
      } else {
        u0 = Math.random();
        u1 = Math.random();
      }
      // clamp away from 0/1 to keep the inverse CDF finite
      u0 = Math.min(Math.max(u0, 1e-12), 1 - 1e-12);
      u1 = Math.min(Math.max(u1, 1e-12), 1 - 1e-12);
      var x = zeros(self.dim, 1);
      x[0] = alg.invNorm(u0) * self.proposalSd;
      x[1] = alg.invNorm(u1) * self.proposalSd;
      var lp = self.logDensity(x);
      var lw = isFinite(lp) ? lp - self.proposalDist.logDensity(x) : -Infinity;
      self._xs.push(x);
      self._lws.push(lw);
      batch.push(x);
    }
    while (self._xs.length > 3000) {
      self._xs.shift();
      self._lws.shift();
    }

    // normalized weights over the whole history (log-space, max-subtracted)
    var n = self._xs.length;
    var lwMax = -Infinity;
    for (var i = 0; i < n; i++) if (self._lws[i] > lwMax) lwMax = self._lws[i];
    var ws = new Array(n);
    var sw = 0,
      sw2 = 0,
      wmax = 0;
    for (var i = 0; i < n; i++) {
      var w = Math.exp(self._lws[i] - lwMax);
      ws[i] = w;
      sw += w;
      sw2 += w * w;
      if (w > wmax) wmax = w;
    }
    var kish = sw > 0 ? (sw * sw) / sw2 : 0;

    self.chain = [];
    self.chain_weights = [];
    for (var i = 0; i < n; i++) {
      self.chain.push(self._xs[i].copy());
      self.chain_weights.push(ws[i]);
    }

    // overlay: proposal ellipse + the whole weighted point set (its evenness
    // is the point of the demo) with the current batch highlighted
    var points = [];
    for (var i = 0; i < n; i++) {
      points.push({
        center: [self._xs[i][0], self._xs[i][1]],
        radius: 0.012 + 0.05 * (ws[i] / (wmax || 1)),
        fill: "#0088b0",
        alpha: 0.45,
      });
    }
    for (var b = 0; b < batch.length; b++) {
      points.push({ center: [batch[b][0], batch[b][1]], radius: 0.05, fill: "#0072b2", alpha: 0.9 });
    }
    visualizer.queue.push({
      type: "overlay",
      clear: true,
      ellipses: [{ center: [0, 0], cov: self.proposalDist.cov }],
      points: points,
      metrics: [{ k: "Sequence", v: qmc ? "Halton" : "Random" }],
      histograms: true,
    });
  },
});
