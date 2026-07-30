"use strict";

MCMC.registerAlgorithm("SGLD", {
  description: "Stochastic Gradient Langevin Dynamics",

  about: function () {
    window.open(
      this && this.controlVariates
        ? "https://arxiv.org/abs/1706.05439"
        : "https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf"
    );
  },

  init: function (self) {
    self.sigma = 0.5;
    // there is no data to minibatch in these examples, so gradient noise
    // N(0, gradientNoise^2 I) stands in for the stochasticity of minibatched gradients
    self.gradientNoise = 2;
    // control variates (Baker et al. 2019, SGLD-CV): anchor the gradient
    // estimator at the posterior mode, so the injected noise is proportional
    // to the distance from the mode and vanishes there
    self.controlVariates = false;
  },

  reset: function (self) {
    if (self.controlVariates) {
      // the control variate is evaluated at the posterior mode, found by
      // gradient ascent with random restarts; Baker et al. initialise the
      // chain there too (the optimum is already computed, and it starts the
      // chain in the low-noise region)
      self.mode = MCMC.algorithms["SGLD"].findMode(self);
      self.chain = [self.mode.copy()];
    } else {
      self.chain = [MultivariateNormal.getSample(self.dim)];
    }
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
    folder.add(self, "sigma", 0.1, 1).step(0.05).name("Proposal σ");
    folder.add(self, "gradientNoise", 0, 10).step(0.25).name("Gradient noise");
    folder
      .add(self, "controlVariates")
      .name("Control variates (CV)")
      .onChange(function () {
        sim.reset();
      });
    folder.open();
  },

  step: function (self, visualizer) {
    var x = self.chain.last();
    var noise;
    if (self.controlVariates) {
      // control-variate estimator: noise standard deviation grows with the
      // distance from the mode (zero at the mode). The variance bound is
      // local, so the distance is capped well outside the plotting region;
      // this also stops the rejection-free dynamics running away when noise
      // pushes a point far off-screen.
      var dist = Math.min(x.subtract(self.mode).norm(), 6);
      noise = MultivariateNormal.getSample(self.dim).scale(self.gradientNoise * dist);
    } else {
      noise = MultivariateNormal.getSample(self.dim).scale(self.gradientNoise);
    }
    var gradient = self.gradLogDensity(x).add(noise);
    var Zdist = new MultivariateNormal(zeros(self.dim), eye(self.dim).scale(self.sigma * self.sigma));
    var Z = Zdist.getSample();
    var proposal = x.add(Z).add(gradient.scale((self.sigma * self.sigma) / 2));
    // trust region: the rejection-free dynamics can diverge on stiff targets
    // when the (noisy) drift is large — especially with distance-scaled
    // control-variate noise — so cap the per-step displacement. Only active
    // in the blow-up regime; typical steps are far smaller.
    var d = proposal.subtract(x);
    var dn = d.norm();
    if (dn > 3) proposal = x.add(d.scale(3 / dn));

    var proposalEvent = {
      type: "proposal",
      proposal: proposal.copy(),
      proposalCov: Zdist.cov.copy(),
      gradient: gradient.scale((self.sigma * self.sigma) / 2),
    };
    if (self.controlVariates) proposalEvent.mode = self.mode.copy();
    visualizer.queue.push(proposalEvent);

    // no Metropolis-Hastings correction: always move to the proposal
    self.chain.push(proposal);
    visualizer.queue.push({ type: "accept", proposal: proposal.copy() });
  },
});
