"use strict";

MCMC.registerAlgorithm("SGLD-CV", {
  description: "Stochastic Gradient Langevin Dynamics with Control Variates",

  about: function () {
    window.open("https://arxiv.org/abs/1706.05439");
  },

  init: function (self) {
    // Smaller step than SGLD: the distance-scaled noise below makes the
    // unadjusted (rejection-free) dynamics more prone to diverge on stiff
    // targets, so we use a conservative default step size.
    self.sigma = 0.3;
    // As in SGLD, there is no data to minibatch here, so we inject synthetic
    // gradient noise. The control variate anchors the estimator at the mode,
    // so the noise variance is proportional to the SQUARED distance from the
    // mode and vanishes there -- unlike plain SGLD, whose noise is constant.
    self.gradientNoise = 1;
  },

  reset: function (self) {
    // The control variate is evaluated at the posterior mode, found here by
    // gradient ascent with a few random restarts.
    self.mode = MCMC.algorithms["SGLD-CV"].findMode(self);
    // Baker et al. initialize the chain at the mode (the optimum is already
    // computed for the control variate); this also starts the chain in the
    // low-noise region.
    self.chain = [self.mode.copy()];
  },

  // gradient ascent on the log density to locate the posterior mode
  findMode: function (self) {
    var best = null,
      bestVal = -Infinity;
    for (var r = 0; r < 5; r++) {
      var x = MultivariateNormal.getSample(self.dim).scale(2);
      for (var i = 0; i < 1000; i++) {
        var g = self.gradLogDensity(x);
        var gn = g.norm();
        if (gn < 1e-8) break;
        // clip the step so large gradients don't cause overshoot
        x = x.add(g.scale(Math.min(0.1, 1.0 / gn)));
      }
      var val = self.logDensity(x);
      if (isFinite(val) && val > bestVal) {
        bestVal = val;
        best = x;
      }
    }
    return best === null ? zeros(self.dim) : best;
  },

  attachUI: function (self, folder) {
    folder.add(self, "sigma", 0.1, 0.5).step(0.05).name("Proposal σ");
    folder.add(self, "gradientNoise", 0, 5).step(0.25).name("Gradient noise");
    folder.open();
  },

  step: function (self, visualizer) {
    var x = self.chain.last();
    // control-variate estimator: exact gradient plus noise whose standard
    // deviation grows with the distance from the mode (zero at the mode).
    // Baker et al.'s variance bound is local, so we cap the distance well
    // outside the plotting region; this also prevents the rejection-free
    // dynamics from running away when noise pushes a point far off-screen.
    var dist = Math.min(x.subtract(self.mode).norm(), 6);
    var noise = MultivariateNormal.getSample(self.dim).scale(self.gradientNoise * dist);
    var gradient = self.gradLogDensity(x).add(noise);
    var Zdist = new MultivariateNormal(zeros(self.dim), eye(self.dim).scale(self.sigma * self.sigma));
    var Z = Zdist.getSample();
    var proposal = x.add(Z).add(gradient.scale((self.sigma * self.sigma) / 2));

    visualizer.queue.push({
      type: "proposal",
      proposal: proposal.copy(),
      proposalCov: Zdist.cov.copy(),
      gradient: gradient.scale((self.sigma * self.sigma) / 2),
      mode: self.mode.copy(),
    });

    // no Metropolis-Hastings correction: always move to the proposal
    self.chain.push(proposal);
    visualizer.queue.push({ type: "accept", proposal: proposal.copy() });
  },
});
