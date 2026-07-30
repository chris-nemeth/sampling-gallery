"use strict";

// Parallel tempering / replica exchange (Geyer 1991): K random-walk
// Metropolis chains target the tempered densities p(x)^beta_k on a geometric
// ladder beta_k = betaMin^(k/(K-1)), from beta_0 = 1 (the cold chain, which is
// the reported sample) down to betaMin. Hot chains see a flattened landscape
// and cross between modes freely; adjacent-pair swap moves let those crossings
// propagate down the ladder to the cold chain. The exchange is what makes
// multimodal targets (Mix, Flower) mix where a single random walk sticks.
MCMC.registerAlgorithm("ParallelTempering", {
  description: "Parallel Tempering",

  about: function () {
    window.open("https://en.wikipedia.org/wiki/Parallel_tempering");
  },

  init: function (self) {
    self.nChains = 4;
    self.betaMin = 0.1;
    self.sigma = 0.6; // cold-chain random-walk proposal sd (scaled by beta^-1/2 up the ladder)
  },

  reset: function (self) {
    var alg = MCMC.algorithms["ParallelTempering"];
    var K = Math.max(2, Math.round(self.nChains));
    self.betas = [];
    for (var k = 0; k < K; k++) self.betas.push(Math.pow(self.betaMin, k / (K - 1)));
    self.xs = [];
    for (var k = 0; k < K; k++) self.xs.push(alg.smartInit(self));
    self.nSwapAttempt = 0;
    self.nSwapAccept = 0;
    self.chain = [self.xs[0].copy()];
  },

  // best-of-10 N(0, I) draws — avoids planting chains in near-zero-mass
  // regions with pathological geometry (see ZigZag.smartInit)
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
    folder
      .add(self, "nChains", 2, 8)
      .step(1)
      .name("Chains K")
      .onChange(function () {
        sim.reset();
      });
    folder
      .add(self, "betaMin", 0.02, 0.5)
      .step(0.01)
      .name("β (hottest)")
      .onChange(function () {
        sim.reset();
      });
    folder.add(self, "sigma", 0.1, 2).step(0.05).name("Proposal σ (cold)");
    folder.open();
  },

  step: function (self, visualizer) {
    var K = self.betas.length;
    var safeLogP = function (p) {
      var v = self.logDensity(p);
      return isFinite(v) ? v : -Infinity;
    };

    // 1. within-chain random-walk MH updates targeting p^beta_k
    var coldProposal = null;
    var coldAccepted = false;
    for (var k = 0; k < K; k++) {
      var beta = self.betas[k];
      var x = self.xs[k];
      var prop = x.add(MultivariateNormal.getSample(self.dim).scale(self.sigma / Math.sqrt(beta)));
      var logA = beta * (safeLogP(prop) - safeLogP(x));
      var acc = Math.log(Math.random()) < logA;
      if (acc) self.xs[k] = prop;
      if (k === 0) {
        coldProposal = prop.copy();
        coldAccepted = acc;
      }
    }

    // 2. one adjacent-pair swap attempt
    var swapSeg = null;
    var j = Math.floor(Math.random() * (K - 1)); // pair (j, j+1)
    self.nSwapAttempt++;
    var logS =
      (self.betas[j] - self.betas[j + 1]) * (safeLogP(self.xs[j + 1]) - safeLogP(self.xs[j]));
    if (Math.log(Math.random()) < logS) {
      self.nSwapAccept++;
      var tmp = self.xs[j];
      self.xs[j] = self.xs[j + 1];
      self.xs[j + 1] = tmp;
      swapSeg = {
        from: [self.xs[j][0], self.xs[j][1]],
        to: [self.xs[j + 1][0], self.xs[j + 1][1]],
        color: "#e69f00",
        lw: 2,
        alpha: 0.9,
      };
    }

    // 3. visualize: hot chains + swap, then the cold chain's usual move events
    var points = [];
    for (var k = 1; k < K; k++) {
      var a = 0.9 - (0.55 * (k - 1)) / Math.max(1, K - 2); // hotter = fainter
      points.push({
        center: [self.xs[k][0], self.xs[k][1]],
        radius: 0.09,
        fill: "rgba(230,159,0," + a.toFixed(2) + ")",
        color: "#e69f00",
        lw: 1,
      });
    }
    var swapRate = self.nSwapAttempt > 0 ? (100 * self.nSwapAccept) / self.nSwapAttempt : 0;
    visualizer.queue.push({
      type: "overlay",
      clear: true,
      points: points,
      segments: swapSeg ? [swapSeg] : [],
      labels: [
        "swap acceptance " + swapRate.toFixed(0) + "%",
        "β ladder: " + self.betas.map(function (b) { return b.toFixed(2); }).join(", "),
      ],
    });
    visualizer.queue.push({ type: "proposal", proposal: coldProposal });
    visualizer.queue.push({
      type: coldAccepted ? "accept" : "reject",
      proposal: coldProposal,
    });

    self.chain.push(self.xs[0].copy());
  },
});
