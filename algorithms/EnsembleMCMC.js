"use strict";

// Ensemble MCMC: an interacting ensemble of K walkers where each walker's
// proposal is built from the OTHERS' current positions, so the proposal scale
// and orientation adapt to the target automatically. Two modes:
//
//  * "Stretch move" (Goodman & Weare 2010; the emcee sampler): pick another
//    walker x_j, draw z from g(z) ∝ 1/sqrt(z) on [1/a, a], and propose
//        y = x_j + z (x_i - x_j),
//    accepted with probability min(1, z^{d-1} p(y)/p(x_i)). The move is
//    AFFINE-INVARIANT: performance is unchanged under any linear map of the
//    target, so highly correlated targets (Ridge) come for free.
//
//  * "Differential evolution" (ter Braak 2006; ter Braak & Vrugt 2008): propose
//        y = x_i + gamma (x_j - x_k) + e,   e ~ N(0, 1e-4 I),
//    a symmetric proposal whose scale/orientation is the ensemble's own
//    spread; plain Metropolis acceptance.
//
// The reported chain is walker 0 (a valid marginal chain, so acceptance, ESS
// and histograms behave normally); the rest of the ensemble and the proposal
// construction are drawn as an overlay.
MCMC.registerAlgorithm("EnsembleMCMC", {
  description: "Ensemble MCMC",

  about: function () {
    window.open(
      this && this.method === "Differential evolution"
        ? "https://link.springer.com/article/10.1007/s11222-008-9104-9"
        : "https://msp.org/camcos/2010/5-1/p04.xhtml"
    );
  },

  init: function (self) {
    self.method = "Stretch move";
    self.nWalkers = 10;
    self.a = 2; // stretch-move scale parameter
    self.gamma = 1.19; // DE scale, default 2.38 / sqrt(2 d) with d = 2
  },

  reset: function (self) {
    var K = Math.max(4, Math.round(self.nWalkers));
    self.xs = [];
    for (var i = 0; i < K; i++) self.xs.push(MultivariateNormal.getSample(self.dim).scale(1.5));
    self.nProp = 0;
    self.nAcc = 0;
    self.chain = [self.xs[0].copy()];
  },

  attachUI: function (self, folder) {
    folder
      .add(self, "method", ["Stretch move", "Differential evolution"])
      .name("Move type")
      .onChange(function () {
        sim.reset();
      });
    folder
      .add(self, "nWalkers", 4, 32)
      .step(1)
      .name("Walkers K")
      .onChange(function () {
        sim.reset();
      });
    folder.add(self, "a", 1.2, 4).step(0.1).name("Stretch a");
    folder.add(self, "gamma", 0.1, 2).step(0.05).name("DE γ");
    folder.open();
  },

  step: function (self, visualizer) {
    var K = self.xs.length;
    var safeLogP = function (p) {
      var v = self.logDensity(p);
      return isFinite(v) ? v : -Infinity;
    };
    var pickOther = function (i, avoid) {
      var j;
      do {
        j = Math.floor(Math.random() * K);
      } while (j === i || j === avoid);
      return j;
    };

    var walker0 = null; // geometry of walker 0's move, for the overlay
    for (var i = 0; i < K; i++) {
      var x = self.xs[i];
      var prop, logA, helperFrom = null;
      if (self.method === "Stretch move") {
        var j = pickOther(i, -1);
        // z ~ g(z) ∝ 1/sqrt(z) on [1/a, a]: z = ((a-1)u + 1)^2 / a
        var u = Math.random();
        var z = Math.pow((self.a - 1) * u + 1, 2) / self.a;
        prop = self.xs[j].add(x.subtract(self.xs[j]).scale(z));
        // acceptance includes the z^{d-1} volume factor (d = 2)
        logA = (self.dim - 1) * Math.log(z) + safeLogP(prop) - safeLogP(x);
        helperFrom = self.xs[j];
      } else {
        var j2 = pickOther(i, -1);
        var k2 = pickOther(i, j2);
        var e = MultivariateNormal.getSample(self.dim).scale(0.01);
        prop = x.add(self.xs[j2].subtract(self.xs[k2]).scale(self.gamma)).add(e);
        logA = safeLogP(prop) - safeLogP(x);
        helperFrom = self.xs[j2];
      }
      self.nProp++;
      var acc = Math.log(Math.random()) < logA;
      if (acc) {
        self.nAcc++;
        self.xs[i] = prop;
      }
      if (i === 0) walker0 = { prop: prop.copy(), acc: acc, helper: helperFrom.copy() };
    }

    // overlay: the rest of the ensemble + walker 0's proposal construction
    var points = [];
    for (var i = 1; i < K; i++)
      points.push({
        center: [self.xs[i][0], self.xs[i][1]],
        radius: 0.07,
        fill: "rgba(0,136,176,0.45)",
        color: "#0088b0",
        lw: 1,
      });
    var accRate = self.nProp > 0 ? (100 * self.nAcc) / self.nProp : 0;
    visualizer.queue.push({
      type: "overlay",
      clear: true,
      points: points,
      segments: [
        // the stretch line / DE difference vector anchoring walker 0's proposal
        { from: [walker0.helper[0], walker0.helper[1]], to: [walker0.prop[0], walker0.prop[1]], color: "#999", lw: 1, alpha: 0.7 },
      ],
      labels: [self.method, "ensemble acceptance " + accRate.toFixed(0) + "%"],
    });
    visualizer.queue.push({ type: "proposal", proposal: walker0.prop });
    visualizer.queue.push({ type: walker0.acc ? "accept" : "reject", proposal: walker0.prop });
    self.chain.push(self.xs[0].copy());
  },
});
