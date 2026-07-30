"use strict";

// Elliptical slice sampling (Murray, Adams & MacKay 2010): designed for
// targets of the form p(x) ∝ N(x; 0, Σ) L(x). A fresh prior draw ν and the
// current state x define an ellipse x cos θ + ν sin θ; slice sampling on the
// angle θ (with bracket shrinkage toward θ = 0, where the proposal equals the
// current state) always terminates with an accepted move — no rejections and
// no step-size parameter. The gallery's targets are arbitrary densities, so
// we split them as p(x) ∝ N(x; 0, σ²I) × [p(x)/N(x; 0, σ²I)] and treat the
// second factor as the likelihood; the sampler is exact for any prior scale
// σ, but σ controls how large the proposal ellipses are — the one knob.
MCMC.registerAlgorithm("EllipticalSlice", {
  description: "Elliptical Slice Sampling",

  about: function () {
    window.open("https://arxiv.org/abs/1001.0175");
  },

  init: function (self) {
    self.priorSd = 2.5; // scale of the surrogate Gaussian prior
  },

  reset: function (self) {
    self.priorDist = new MultivariateNormal(
      zeros(self.dim, 1),
      eye(self.dim).scale(self.priorSd * self.priorSd)
    );
    self.chain = [self.priorDist.getSample()];
    self.nShrinkTotal = 0;
    self.nSteps = 0;
  },

  attachUI: function (self, folder) {
    folder
      .add(self, "priorSd", 1, 5)
      .step(0.1)
      .name("Prior sd")
      .onChange(function () {
        sim.reset();
      });
    folder.open();
  },

  step: function (self, visualizer) {
    // residual log-likelihood of the prior × likelihood split
    var logLik = function (p) {
      var v = self.logDensity(p) - self.priorDist.logDensity(p);
      return isFinite(v) ? v : -Infinity;
    };

    var x = self.chain.last();
    // 1. auxiliary prior draw defining the ellipse through x and ν
    var nu = self.priorDist.getSample();
    // 2. slice level under the likelihood alone
    var logy = logLik(x) + Math.log(Math.random());
    // 3. initial angle and bracket [θ − 2π, θ]
    var theta = 2 * Math.PI * Math.random();
    var thetaMin = theta - 2 * Math.PI;
    var thetaMax = theta;

    var onEllipse = function (t) {
      return x.copy().scale(Math.cos(t)).add(nu.copy().scale(Math.sin(t)));
    };

    // 4-6. propose on the ellipse, shrink the bracket toward θ = 0 on
    // rejection; θ = 0 is the current state, so the loop must terminate
    var rejects = [];
    var prop = onEllipse(theta);
    var guard = 100;
    while (logLik(prop) <= logy && guard-- > 0) {
      rejects.push([prop[0], prop[1]]);
      if (theta < 0) thetaMin = theta;
      else thetaMax = theta;
      theta = thetaMin + (thetaMax - thetaMin) * Math.random();
      prop = onEllipse(theta);
    }
    if (guard <= 0) prop = x.copy(); // numerical fallback: stay put

    self.nShrinkTotal += rejects.length;
    self.nSteps++;

    // overlay: the full ellipse as a polyline, the prior draw ν anchoring it,
    // shrinkage rejects, and the accepted point
    var segments = [];
    var K = 48;
    var prev = onEllipse(0);
    for (var k = 1; k <= K; k++) {
      var cur = onEllipse((2 * Math.PI * k) / K);
      segments.push({ from: [prev[0], prev[1]], to: [cur[0], cur[1]], color: "#0088b0", lw: 1, alpha: 0.35 });
      prev = cur;
    }
    var points = [{ center: [nu[0], nu[1]], radius: 0.09, fill: "rgba(230,159,0,0.35)", color: "#e69f00", lw: 1 }];
    for (var i = 0; i < rejects.length; i++)
      points.push({ center: rejects[i], radius: 0.05, fill: "rgba(213,94,0,0.5)" });
    points.push({ center: [prop[0], prop[1]], radius: 0.07, fill: "#0072b2" });

    visualizer.queue.push({
      type: "overlay",
      clear: true,
      segments: segments,
      points: points,
      metrics: [{ k: "Shrinks/step", v: (self.nShrinkTotal / self.nSteps).toFixed(1) }],
    });
    visualizer.queue.push({ type: "proposal", proposal: prop.copy() });
    visualizer.queue.push({ type: "accept", proposal: prop.copy() });

    self.chain.push(prop.copy());
  },
});
