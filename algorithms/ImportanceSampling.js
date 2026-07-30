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
    self.paretoSmooth = false; // PSIS: smooth the weight tail, report khat
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
    folder.add(self, "paretoSmooth").name("Pareto smoothing (PSIS)");
    folder.open();
  },

  // Pareto-smoothed importance sampling (Vehtari, Simpson, Gelman, Yao &
  // Gabry, arXiv:1507.02646): fit a generalized Pareto distribution to the
  // largest M weights and replace them by the expected order statistics of
  // the fit (capped at the raw maximum). Stabilises the heavy right tail of
  // the weight distribution; the shape estimate khat is the reliability
  // diagnostic (khat < 0.7: reliable).
  // Returns { ws: smoothed weights (same array, modified), khat }.
  psisSmooth: function (ws) {
    var n = ws.length;
    var M = Math.ceil(Math.min(0.2 * n, 3 * Math.sqrt(n)));
    if (M < 5) return { ws: ws, khat: NaN };
    var idx = ws.map(function (w, i) {
      return i;
    });
    idx.sort(function (a, b) {
      return ws[a] - ws[b];
    });
    var top = idx.slice(n - M); // indices of the M largest weights, ascending
    var u = ws[idx[n - M - 1]]; // threshold: largest weight NOT smoothed
    var y = top.map(function (i) {
      return Math.max(0, ws[i] - u);
    });
    var ymax = y[y.length - 1];
    if (!(ymax > 0)) return { ws: ws, khat: NaN };

    // Zhang & Stephens (2009) posterior-mean fit of the GPD, in the sign
    // convention of the PSIS paper / loo package: khat = xi = mean log(1 - theta y)
    // (positive = heavy tail), sigma = -xi / theta.
    var m = 30 + Math.floor(Math.sqrt(M));
    var yq = y[Math.max(0, Math.floor(M / 4 + 0.5) - 1)] || ymax / 2;
    if (!(yq > 0)) yq = ymax / 2;
    var xiOf = function (theta) {
      var s = 0;
      for (var i = 0; i < M; i++) s += Math.log(Math.max(1e-12, 1 - theta * y[i]));
      return s / M;
    };
    var thetas = [],
      logls = [],
      maxl = -Infinity;
    for (var j = 1; j <= m; j++) {
      var theta = 1 / ymax + (1 - Math.sqrt(m / (j - 0.5))) / (3 * yq);
      var xi = xiOf(theta);
      var ll = M * (Math.log(-theta / xi) - xi - 1);
      if (!isFinite(ll)) ll = -Infinity;
      thetas.push(theta);
      logls.push(ll);
      if (ll > maxl) maxl = ll;
    }
    var wsum = 0,
      thetaHat = 0;
    for (var j = 0; j < m; j++) {
      var wgt = Math.exp(logls[j] - maxl);
      wsum += wgt;
      thetaHat += wgt * thetas[j];
    }
    thetaHat /= wsum;
    var khat = xiOf(thetaHat);
    var sigma = -khat / thetaHat;

    // replace the top-M weights by expected GPD order statistics, capped
    var qGPD = function (p) {
      if (Math.abs(khat) < 1e-6) return -sigma * Math.log(1 - p);
      return (sigma / khat) * (Math.pow(1 - p, -khat) - 1);
    };
    var cap = u + ymax;
    for (var j = 0; j < M; j++) {
      var smoothed = u + qGPD((j + 0.5) / M);
      ws[top[j]] = Math.min(isFinite(smoothed) ? smoothed : cap, cap);
    }
    return { ws: ws, khat: khat };
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
    // optional PSIS: smooth the weight tail, then recompute the summaries
    var khat = NaN;
    if (self.paretoSmooth) {
      var fit = MCMC.algorithms["ImportanceSampling"].psisSmooth(ws);
      khat = fit.khat;
      sw = 0;
      sw2 = 0;
      wmax = 0;
      for (var i = 0; i < n; i++) {
        sw += ws[i];
        sw2 += ws[i] * ws[i];
        if (ws[i] > wmax) wmax = ws[i];
      }
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
    // full weighted history: small dots sized by weight, so the weighted
    // cloud accumulates visibly on the plot
    for (var i = 0; i < n; i++) {
      points.push({
        center: [self._xs[i][0], self._xs[i][1]],
        radius: 0.012 + 0.05 * (ws[i] / (wmax || 1)),
        fill: "#0088b0",
        alpha: 0.45,
      });
    }
    // this step's batch, highlighted
    for (var b = 0; b < batch.length; b++) {
      points.push({
        center: [batch[b].x[0], batch[b].x[1]],
        radius: 0.03 + 0.1 * (batch[b].w / bmax),
        fill: "#0072b2",
        alpha: 0.9,
      });
    }
    visualizer.queue.push({
      type: "overlay",
      clear: true,
      ellipses: [{ center: [self.muX, self.muY], cov: self.proposalDist.cov }],
      points: points,
      metrics: (function () {
        var m = [{ k: "Max w share", v: maxShare.toFixed(1) + "%" }];
        if (self.paretoSmooth) m.push({ k: "Pareto k̂", v: isFinite(khat) ? khat.toFixed(2) : "—" });
        return m;
      })(),
      labels: (function () {
        var l = [];
        if (self.resampleSIR) l.push("SIR: showing an equally-weighted resample of the weighted history");
        if (self.paretoSmooth && isFinite(khat) && khat >= 0.7)
          l.push("k̂ ≥ 0.7: the weight tail is too heavy — PSIS estimates are unreliable");
        return l.length ? l : null;
      })(),
      histograms: true,
    });
  },
});
