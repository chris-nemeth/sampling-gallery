"use strict";

// Coin SVGD (Sharrock & Nemeth, ICML 2023; arXiv:2301.11294): Stein
// variational gradient descent WITHOUT a learning rate. The SVGD drift for
// each particle is fed to a coin-betting scheme (COCOB-style, after Orabona &
// Tommasi): each coordinate of each particle is a gambler whose bet on the
// next position scales with the wealth accumulated from past drifts,
//     L      <- max(L, |c|)                      (largest drift seen)
//     S      <- S + c;   A <- A + |c|            (drift sum, absolute sum)
//     reward <- max(reward + (x - x0) c, 0)      (wealth from past bets)
//     x      <- x0 + S / (L (A + L)) * (L + reward),
// so the effective step size emerges from the drift history instead of being
// tuned. The drift itself (kernel, median-heuristic bandwidth) is identical
// to the gallery's SVGD entry — the only difference is the optimizer.
MCMC.registerAlgorithm("CoinSVGD", {
  description: "Coin Stein Variational Gradient Descent",

  about: function () {
    window.open("https://arxiv.org/abs/2301.11294");
  },

  init: function (self) {
    self.n = 200; // number of particles
    self.h = 0.15; // kernel bandwidth
    self.use_median = true;
    self.particleInit = ParVI.initNames[0];
    self.showForces = "total";
    self.reset(self);
  },

  newParticle: function (self) {
    var L = Float64Array.zeros(self.dim, 1);
    for (var k = 0; k < self.dim; k++) L[k] = 1e-10; // max |drift| per coordinate
    return {
      x0: ParVI.initCloud(self.particleInit, self.dim), // betting anchor
      L: L,
      gradSum: Float64Array.zeros(self.dim, 1),
      absGradSum: Float64Array.zeros(self.dim, 1),
      reward: Float64Array.zeros(self.dim, 1),
    };
  },

  reset: function (self) {
    var alg = MCMC.algorithms["CoinSVGD"];
    self.chain = [];
    self.coin = [];
    self.gradx = [];
    self.gradLogDensities = [];
    self.iter = 0;
    for (var i = 0; i < self.n; i++) {
      var c = alg.newParticle(self);
      self.coin.push(c);
      self.chain.push(c.x0.copy());
      self.gradx.push(Float64Array.zeros(self.dim, 1));
      self.gradLogDensities.push(0);
    }
  },

  attachUI: function (self, folder) {
    folder
      .add(self, "particleInit", ParVI.initNames)
      .name("Initialization")
      .onChange(function () {
        sim.reset();
      });
    folder
      .add(self, "showForces", ["total", "attraction", "repulsion", "both"])
      .name("Show forces");
    folder.add(self, "use_median").name("Median heuristic").listen();
    folder
      .add(self, "h", 0.05, 2)
      .step(0.05)
      .name("bandwidth")
      .listen()
      .onChange(function () {
        self.use_median = false;
      });
    folder.add(self, "n", 10, 400).step(1).name("numParticles");
    folder.open();
  },

  step: function (self, visualizer) {
    var alg = MCMC.algorithms["CoinSVGD"];
    // resize the particle set if the slider moved
    while (self.chain.length < self.n) {
      var c = alg.newParticle(self);
      self.coin.push(c);
      self.chain.push(c.x0.copy());
      self.gradx.push(Float64Array.zeros(self.dim, 1));
      self.gradLogDensities.push(0);
    }
    if (self.n < self.chain.length) {
      self.chain = self.chain.slice(0, self.n);
      self.coin = self.coin.slice(0, self.n);
      self.gradx = self.gradx.slice(0, self.n);
      self.gradLogDensities = self.gradLogDensities.slice(0, self.n);
    }

    var n = self.chain.length;

    for (var i = 0; i < n; i++) {
      self.gradLogDensities[i] = self.gradLogDensity(self.chain[i]);
      for (var k = 0; k < self.dim; k++) self.gradx[i][k] = 0;
    }

    // pairwise squared distances (same kernel conventions as SVGD.js)
    var dist2 = new Float64Array(n * n);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < i; j++) {
        var delta = 0;
        for (var k = 0; k < self.dim; k++) delta += Math.pow(self.chain[i][k] - self.chain[j][k], 2);
        dist2[i * n + j] = delta;
        dist2[j * n + i] = delta;
      }
    }
    if (self.use_median) {
      var dist2copy = new Float64Array(dist2);
      dist2copy.sort();
      var median = dist2copy[Math.floor(dist2copy.length / 2)];
      self.h = median / Math.log(n);
    }

    // SVGD drift phi(x_i) = (1/n) sum_j [ k(x_j, x_i) grad log p(x_j) + grad_{x_j} k ]
    // with the attraction / repulsion terms kept separate for display
    var attract = new Array(n),
      repel = new Array(n);
    for (var i = 0; i < n; i++) {
      attract[i] = Float64Array.zeros(self.dim, 1);
      repel[i] = Float64Array.zeros(self.dim, 1);
      for (var j = 0; j < n; j++) {
        var rbf = Math.exp(-dist2[i * n + j] / self.h);
        for (var k = 0; k < self.dim; k++) {
          attract[i][k] += self.gradLogDensities[j][k] * rbf;
          repel[i][k] += ((self.chain[i][k] - self.chain[j][k]) * 2 * rbf) / self.h;
        }
      }
      for (var k = 0; k < self.dim; k++) {
        attract[i][k] /= n;
        repel[i][k] /= n;
        self.gradx[i][k] = attract[i][k] + repel[i][k];
      }
    }

    // coin-betting update: position is set from the betting state, and the
    // displayed arrow is the actual move x_new - x_old
    for (var i = 0; i < n; i++) {
      var c = self.coin[i];
      var x = self.chain[i];
      for (var k = 0; k < self.dim; k++) {
        var g = self.gradx[i][k];
        var ag = Math.abs(g);
        if (ag > c.L[k]) c.L[k] = ag;
        c.gradSum[k] += g;
        c.absGradSum[k] += ag;
        c.reward[k] = Math.max(c.reward[k] + (x[k] - c.x0[k]) * g, 0);
        var xNew = c.x0[k] + (c.gradSum[k] / (c.L[k] * (c.absGradSum[k] + c.L[k]))) * (c.L[k] + c.reward[k]);
        self.gradx[i][k] = xNew - x[k];
      }
    }

    // snapshot for the event: the queue is consumed asynchronously and the
    // particles are mutated in place below
    var xSnap = new Array(n),
      gSnap = new Array(n);
    for (var i = 0; i < n; i++) {
      xSnap[i] = self.chain[i].copy();
      gSnap[i] = self.gradx[i].copy();
    }
    var forces = null;
    if (self.showForces === "attraction") forces = [{ gradx: attract, rgb: "0,114,178" }];
    else if (self.showForces === "repulsion") forces = [{ gradx: repel, rgb: "213,94,0" }];
    else if (self.showForces === "both")
      forces = [
        { gradx: attract, rgb: "0,114,178" },
        { gradx: repel, rgb: "213,94,0" },
      ];
    visualizer.queue.push({
      type: "svgd-step",
      x: xSnap,
      gradx: gSnap,
      forces: forces,
      h: self.h,
    });
    visualizer.queue.push({
      type: "overlay",
      clear: false,
      metrics: [{ k: "KSD", v: ParVI.ksd(xSnap, self.gradLogDensities, self.h, self.dim).toFixed(3) }],
    });

    for (var i = 0; i < n; i++) self.chain[i].increment(self.gradx[i]);
    self.iter++;
  },
});
