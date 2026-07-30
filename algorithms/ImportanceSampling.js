"use strict";

// Self-normalized importance sampling with a Gaussian proposal. Every draw is
// kept and carries a weight w = p(x)/q(x); the UI consumes chain_weights
// natively (weighted mean, Kish effective sample size, weighted histograms),
// so weight degeneracy is directly visible as a collapsing ESS. The optional
// SIR mode (sampling-importance-resampling) displays an equally-weighted
// systematic resample of the weighted history instead.
MCMC.registerAlgorithm("ImportanceSampling", {
  description: "Importance Sampling",

  about: function () {
    window.open("https://en.wikipedia.org/wiki/Importance_sampling");
  },

  init: function (self) {
    self.proposalSd = 2.5;
    self.muX = 0; // move the proposal off-centre to provoke degeneracy
    self.muY = 0;
    self.batchSize = 10;
    self.resampleSIR = false;
  },

  reset: function (self) {
    var mean = zeros(self.dim, 1);
    mean[0] = self.muX;
    mean[1] = self.muY;
    self.proposalDist = new MultivariateNormal(
      mean,
      eye(self.dim).scale(self.proposalSd * self.proposalSd)
    );
    self._xs = []; // raw draws
    self._lws = []; // raw log-weights (unnormalized)
    self.chain = [];
    self.chain_weights = [];
  },

  attachUI: function (self, folder) {
    folder
      .add(self, "proposalSd", 0.5, 5)
      .step(0.1)
      .name("Proposal sd")
      .onChange(function () {
        sim.reset();
      });
    folder
      .add(self, "muX", -3, 3)
      .step(0.25)
      .name("Proposal mean x")
      .onChange(function () {
        sim.reset();
      });
    folder
      .add(self, "muY", -3, 3)
      .step(0.25)
      .name("Proposal mean y")
      .onChange(function () {
        sim.reset();
      });
    folder.add(self, "batchSize", 1, 50).step(1).name("Batch size");
    folder.add(self, "resampleSIR").name("Resample (SIR)");
    folder.open();
  },

  // systematic resampling of indices by (normalized) weights
  systematicResample: function (weights) {
    var n = weights.length;
    var total = 0;
    for (var i = 0; i < n; i++) total += weights[i];
    var idx = new Array(n);
    var u = Math.random() / n;
    var cum = weights[0] / total;
    var j = 0;
    for (var i = 0; i < n; i++) {
      var pos = u + i / n;
      while (cum < pos && j < n - 1) {
        j++;
        cum += weights[j] / total;
      }
      idx[i] = j;
    }
    return idx;
  },

  step: function (self, visualizer) {
    // draw a batch from q and compute log-weights
    var batch = [];
    for (var b = 0; b < self.batchSize; b++) {
      var x = self.proposalDist.getSample();
      var lp = self.logDensity(x);
      var lw = isFinite(lp) ? lp - self.proposalDist.logDensity(x) : -Infinity;
      self._xs.push(x);
      self._lws.push(lw);
      batch.push({ x: x, lw: lw });
    }
    // cap the history (keep arrays aligned)
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
    var maxShare = sw > 0 ? (100 * wmax) / sw : 0;

    // expose to the UI: weighted history, or an equally-weighted SIR resample
    if (self.resampleSIR) {
      var idx = MCMC.algorithms["ImportanceSampling"].systematicResample(ws);
      self.chain = [];
      self.chain_weights = [];
      for (var i = 0; i < n; i++) {
        self.chain.push(self._xs[idx[i]].copy());
        self.chain_weights.push(1);
      }
    } else {
      self.chain = [];
      self.chain_weights = [];
      for (var i = 0; i < n; i++) {
        self.chain.push(self._xs[i].copy());
        self.chain_weights.push(ws[i]);
      }
    }

    // overlay: proposal ellipse + this batch sized by (batch-)normalized weight
    var bmax = 1e-300;
    for (var b = 0; b < batch.length; b++) {
      var w = Math.exp(batch[b].lw - lwMax);
      batch[b].w = w;
      if (w > bmax) bmax = w;
    }
    var points = [];
    for (var b = 0; b < batch.length; b++) {
      points.push({
        center: [batch[b].x[0], batch[b].x[1]],
        radius: 0.03 + 0.22 * (batch[b].w / bmax),
        fill: "#0088b0",
        alpha: 0.8,
      });
    }
    visualizer.queue.push({
      type: "overlay",
      clear: true,
      ellipses: [{ center: [self.muX, self.muY], cov: self.proposalDist.cov }],
      points: points,
      labels: [
        "Kish ESS " + kish.toFixed(0) + " of " + n + (self.resampleSIR ? " (SIR resampled)" : ""),
        "max weight share " + maxShare.toFixed(1) + "%",
      ],
      histograms: true,
    });
  },
});
