"use strict";

// Stochastic Particle-Optimization Sampling (Zhang, Zhang, Carin & Chen,
// AISTATS 2020): interacting particles that combine the SVGD transport with
// Langevin noise,
//   dx_i = [ tau grad log pi(x_i) + phi_svgd(x_i) ] dt + sqrt(2 tau) dW_i,
// with tau = 1/beta. Both pieces separately preserve the target — the
// Langevin part exactly, the SVGD part by the Stein identity — so SPOS
// bridges deterministic particle transport (tau = 0 recovers plain SVGD)
// and noisy Langevin exploration, and the noise counters SVGD's
// deterministic particle collapse.
MCMC.registerAlgorithm("SPOS", {
  description: "Stochastic Particle-Optimization Sampling",

  about: function () {
    window.open("https://proceedings.mlr.press/v108/zhang20d.html");
  },

  init: function (self) {
    self.n = 200;
    self.epsilon = 0.01;
    self.tau = 0.1; // noise temperature 1/beta; 0 recovers SVGD
    self.h = 0.15;
    self.use_median = true;
    self.particleInit = ParVI.initNames[0];
    self.showForces = "total";
    self.reset(self);
  },

  reset: function (self) {
    self.chain = [];
    self.gradx = [];
    self.gradLogDensities = [];
    for (var i = 0; i < self.n; i++) {
      self.chain.push(ParVI.initCloud(self.particleInit, self.dim));
      self.gradx.push(Float64Array.zeros(self.dim, 1));
      self.gradLogDensities.push(0);
    }
  },

  attachUI: function (self, folder) {
    folder.add(self, "tau", 0, 1).step(0.01).name("Noise τ = 1/β");
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
    folder.add(self, "epsilon", 0.001, 0.1).step(0.001).name("stepsize");
    folder.add(self, "n", 10, 400).step(1).name("numParticles");
    folder.open();
  },

  step: function (self, visualizer) {
    while (self.chain.length < self.n) {
      self.chain.push(ParVI.initCloud(self.particleInit, self.dim));
      self.gradx.push(Float64Array.zeros(self.dim, 1));
      self.gradLogDensities.push(0);
    }
    if (self.n < self.chain.length) {
      self.chain = self.chain.slice(0, self.n);
      self.gradx = self.gradx.slice(0, self.n);
      self.gradLogDensities = self.gradLogDensities.slice(0, self.n);
    }
    var n = self.chain.length;
    var d = self.dim;

    for (var i = 0; i < n; i++) self.gradLogDensities[i] = self.gradLogDensity(self.chain[i]);

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

    // drift: tau * raw score (Langevin part) + SVGD transport
    var attract = new Array(n),
      repel = new Array(n);
    for (var i = 0; i < n; i++) {
      attract[i] = Float64Array.zeros(d, 1);
      repel[i] = Float64Array.zeros(d, 1);
      for (var j = 0; j < n; j++) {
        var rbf = Math.exp(-dist2[i * n + j] / self.h);
        for (var k = 0; k < d; k++) {
          attract[i][k] += self.gradLogDensities[j][k] * rbf;
          repel[i][k] += ((self.chain[i][k] - self.chain[j][k]) * 2 * rbf) / self.h;
        }
      }
      for (var k = 0; k < d; k++) {
        attract[i][k] = attract[i][k] / n + self.tau * self.gradLogDensities[i][k];
        repel[i][k] /= n;
        self.gradx[i][k] = self.epsilon * (attract[i][k] + repel[i][k]);
      }
    }

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
      metrics: [{ k: "KSD", v: ParVI.ksd(xSnap, self.gradLogDensities, self.h, d).toFixed(3) }],
    });

    // move: deterministic drift + Langevin noise sqrt(2 eps tau)
    var noiseSd = Math.sqrt(2 * self.epsilon * self.tau);
    for (var i = 0; i < n; i++) {
      self.chain[i].increment(self.gradx[i]);
      if (self.tau > 0) self.chain[i].increment(MultivariateNormal.getSample(d).scale(noiseSd));
    }
  },
});
