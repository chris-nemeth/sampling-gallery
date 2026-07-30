"use strict";

// Wasserstein Particle Descent: particle discretizations of the Wasserstein
// gradient flow of KL(q || pi), following the unifying treatment of Liu et
// al. (ICML 2019, "Understanding and Accelerating Particle-Based Variational
// Inference"). Every particle moves with velocity
//     v(x_i) = grad log pi(x_i) - xi(x_i),
// where xi approximates grad log q at the particle — the entropy force. The
// modes differ only in how xi is built from the kernel matrix:
//   Blob — smoothed density AND smoothed test functions (two-term repulsion);
//   GFSD — plain kernel density estimate, xi = grad log q_hat;
//   GFSF — solves the Stein identity Sum_i [xi_i k_il + grad_i k_il] = 0 for
//          the score values (a ridge-regularized n x n solve).
// Unlike SVGD, the attraction is the RAW target score (not kernel-smoothed),
// so isolated particles always feel the full pull of the target.
MCMC.registerAlgorithm("WassersteinParticleDescent", {
  description: "Wasserstein Particle Descent",

  about: function () {
    window.open("https://proceedings.mlr.press/v97/liu19i.html");
  },

  init: function (self) {
    self.mode = "Blob";
    self.n = 200;
    self.epsilon = 0.01;
    self.h = 0.15;
    self.use_median = true;
    self.use_adagrad = true;
    self.alphaAda = 0.9;
    self.fudge_factor = 1e-2;
    self.particleInit = ParVI.initNames[0];
    self.showForces = "total";
    self.reset(self);
  },

  reset: function (self) {
    self.chain = [];
    self.gradx = [];
    self.historical_grad = [];
    self.gradLogDensities = [];
    for (var i = 0; i < self.n; i++) {
      self.chain.push(ParVI.initCloud(self.particleInit, self.dim));
      self.gradx.push(Float64Array.zeros(self.dim, 1));
      self.historical_grad.push(Float64Array.zeros(self.dim, 1));
      self.gradLogDensities.push(0);
    }
  },

  attachUI: function (self, folder) {
    folder.add(self, "mode", ["Blob", "GFSD", "GFSF"]).name("Entropy force");
    folder
      .add(self, "particleInit", ParVI.initNames)
      .name("Initialization")
      .onChange(function () {
        sim.reset();
      });
    folder.add(self, "showForces", ["total", "attraction", "repulsion", "both"]).name("Show forces");
    folder.add(self, "use_median").name("Median heuristic").listen();
    folder
      .add(self, "h", 0.05, 2)
      .step(0.05)
      .name("bandwidth")
      .listen()
      .onChange(function () {
        self.use_median = false;
      });
    folder.add(self, "use_adagrad").name("Adagrad");
    folder.add(self, "epsilon", 0.001, 0.1).step(0.001).name("stepsize");
    folder.add(self, "n", 10, 400).step(1).name("numParticles");
    folder.open();
  },

  step: function (self, visualizer) {
    while (self.chain.length < self.n) {
      self.chain.push(ParVI.initCloud(self.particleInit, self.dim));
      self.gradx.push(Float64Array.zeros(self.dim, 1));
      self.historical_grad.push(Float64Array.zeros(self.dim, 1));
      self.gradLogDensities.push(0);
    }
    if (self.n < self.chain.length) {
      self.chain = self.chain.slice(0, self.n);
      self.gradx = self.gradx.slice(0, self.n);
      self.historical_grad = self.historical_grad.slice(0, self.n);
      self.gradLogDensities = self.gradLogDensities.slice(0, self.n);
    }
    var n = self.chain.length;
    var d = self.dim;

    for (var i = 0; i < n; i++) self.gradLogDensities[i] = self.gradLogDensity(self.chain[i]);

    // kernel matrix (same conventions as the SVGD entry)
    var dist2 = new Float64Array(n * n);
    for (var i = 0; i < n; i++)
      for (var j = 0; j < i; j++) {
        var delta = 0;
        for (var k = 0; k < d; k++) delta += Math.pow(self.chain[i][k] - self.chain[j][k], 2);
        dist2[i * n + j] = delta;
        dist2[j * n + i] = delta;
      }
    if (self.use_median) {
      var dist2copy = new Float64Array(dist2);
      dist2copy.sort();
      self.h = dist2copy[Math.floor(dist2copy.length / 2)] / Math.log(n);
    }
    var K = new Float64Array(n * n);
    for (var i = 0; i < n * n; i++) K[i] = Math.exp(-dist2[i] / self.h);

    // entropy force xi ~ grad log q at each particle
    var xi = new Array(n);
    for (var i = 0; i < n; i++) xi[i] = Float64Array.zeros(d, 1);
    var rowSum = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var s = 0;
      for (var j = 0; j < n; j++) s += K[i * n + j];
      rowSum[i] = s;
    }
    if (self.mode === "GFSF") {
      // solve (K + lambda I) Xi = B,  B_l = (2/h) sum_i (x_i - x_l) k_il
      var B = new Float64Array(n * d);
      for (var l = 0; l < n; l++)
        for (var i = 0; i < n; i++) {
          var kv = K[i * n + l];
          for (var k = 0; k < d; k++) B[l * d + k] += ((2 / self.h) * (self.chain[i][k] - self.chain[l][k]) * kv);
        }
      var X = ParVI.solve(K, B, n, d, 0.01);
      for (var i = 0; i < n; i++) for (var k = 0; k < d; k++) xi[i][k] = X[i * d + k];
    } else {
      for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
          var kv = K[i * n + j];
          for (var k = 0; k < d; k++) {
            var gk = (-(2 / self.h) * (self.chain[i][k] - self.chain[j][k])) * kv; // grad_{x_i} k_ij
            xi[i][k] += gk / rowSum[i]; // grad log KDE (GFSD term)
            if (self.mode === "Blob") xi[i][k] += gk / rowSum[j]; // smoothed test-function term
          }
        }
      }
    }

    // velocity = raw score - entropy force
    var attract = new Array(n),
      repel = new Array(n);
    for (var i = 0; i < n; i++) {
      attract[i] = self.gradLogDensities[i].copy();
      repel[i] = xi[i].scale(-1);
      for (var k = 0; k < d; k++) self.gradx[i][k] = attract[i][k] + repel[i][k];
    }

    // adagrad + step size (as in the SVGD entry)
    if (self.use_adagrad) {
      for (var i = 0; i < n; i++)
        for (var k = 0; k < d; k++)
          self.historical_grad[i][k] =
            self.alphaAda * self.historical_grad[i][k] + (1 - self.alphaAda) * Math.pow(self.gradx[i][k], 2);
      for (var i = 0; i < n; i++)
        for (var k = 0; k < d; k++) self.gradx[i][k] /= self.fudge_factor + Math.sqrt(self.historical_grad[i][k]);
    }
    for (var i = 0; i < n; i++) for (var k = 0; k < d; k++) self.gradx[i][k] *= self.epsilon;

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
    visualizer.queue.push({ type: "svgd-step", x: xSnap, gradx: gSnap, forces: forces, h: self.h });
    visualizer.queue.push({
      type: "overlay",
      clear: false,
      metrics: [
        { k: "Mode", v: self.mode },
        { k: "KSD", v: ParVI.ksd(xSnap, self.gradLogDensities, self.h, d).toFixed(3) },
      ],
    });

    for (var i = 0; i < n; i++) self.chain[i].increment(self.gradx[i]);
  },
});
