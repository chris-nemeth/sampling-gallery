"use strict";

// FUSE (Functional Upper Bound Step Size Estimator) applied to ULA.
// Automatic, tuning-free step size for gradient-based samplers without an
// MH accept/reject step. Sharrock & Nemeth (2025), "Tuning-Free Sampling via
// Optimization on the Space of Probability Measures", arXiv:2510.25315.
//
// The fixed ULA step is replaced by the DoG-style ("distance over gradients")
// schedule (single-chain / n = 1 special case of the paper's ensemble form):
//
//     eta_t = max(r_eps, max_{s<=t} d_s) / sqrt( sum_{s=1}^t g_s^2 )
//
// with drift = gradLogDensity, g_s^2 = ||drift(x_s)||^2, and d_s the distance
// of the current half-step from the initial half-step reference.
MCMC.registerAlgorithm("FUSE-ULA", {
  description: "Tuning-free ULA (FUSE step size)",

  about: function () {
    window.open("https://arxiv.org/abs/2510.25315");
  },

  init: function (self) {
    // initial movement size; the schedule has only a mild dependence on it
    self.r_eps = 0.1;
    // noise scale of the Wasserstein gradient flow (lambda = 1 for ULA)
    self.lambda = 1;
  },

  reset: function (self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
    // FUSE schedule state
    self.G = 0; // cumulative gradient energy, sum_s g_s^2
    self.r_bar = self.r_eps; // running max of max(r_eps, d_s)
    self.half_ref = null; // first half-step point (set on the first step)
    self.eta = self.r_eps; // most recent step size (for display)
  },

  attachUI: function (self, folder) {
    folder.add(self, "r_eps", 0.001, 1).step(0.001).name("r&epsilon;");
    folder.open();
  },

  step: function (self, visualizer) {
    var x = self.chain.last();
    var drift = self.gradLogDensity(x);
    var eta;

    if (self.half_ref === null) {
      // first step: eta_0 = r_eps, and G / r_bar stay at their initial values
      eta = self.r_eps;
      var half0 = x.add(drift.scale(eta));
      self.half_ref = half0.copy();
    } else {
      // g_t^2 = ||drift(x_t)||^2, accumulate before computing eta
      self.G += drift.norm2();
      eta = self.r_bar / Math.sqrt(self.G + 1e-16);
      var half = x.add(drift.scale(eta));
      // d_{t+1} = ||half_ref - half||; r_bar_{t+1} = max(r_bar, max(r_eps, d))
      var d = self.half_ref.subtract(half).norm();
      self.r_bar = Math.max(self.r_bar, Math.max(self.r_eps, d));
    }
    self.eta = eta;

    // Euler-Maruyama update: x_{t+1} = x_t + eta*drift + sqrt(2*lambda*eta)*xi
    var meanStep = drift.scale(eta);
    var noise = MultivariateNormal.getSample(self.dim).scale(Math.sqrt(2 * self.lambda * eta));
    var proposal = x.add(meanStep).add(noise);

    visualizer.queue.push({
      type: "proposal",
      proposal: proposal.copy(),
      gradient: meanStep,
      proposalCov: eye(self.dim).scale(2 * self.lambda * eta),
      stepSize: eta,
    });

    // no Metropolis-Hastings correction: always move to the proposal
    self.chain.push(proposal);
    visualizer.queue.push({ type: "accept", proposal: proposal.copy() });
  },
});
