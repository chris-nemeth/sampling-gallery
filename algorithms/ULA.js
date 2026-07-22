"use strict";

MCMC.registerAlgorithm("ULA", {
  description: "Unadjusted Langevin algorithm",

  about: function () {
    window.open("http://projecteuclid.org/euclid.bj/1178291835");
  },

  init: function (self) {
    self.sigma = 0.5;
  },

  reset: function (self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function (self, folder) {
    folder.add(self, "sigma", 0.1, 1).step(0.05).name("Proposal σ");
    folder.open();
  },

  step: function (self, visualizer) {
    var gradient = self.gradLogDensity(self.chain.last());
    var Zdist = new MultivariateNormal(zeros(self.dim), eye(self.dim).scale(self.sigma * self.sigma));
    var Z = Zdist.getSample();
    var proposal = self.chain
      .last()
      .add(Z)
      .add(gradient.scale((self.sigma * self.sigma) / 2));

    visualizer.queue.push({
      type: "proposal",
      proposal: proposal.copy(),
      proposalCov: Zdist.cov.copy(),
      gradient: gradient.scale((self.sigma * self.sigma) / 2),
    });

    // no Metropolis-Hastings correction: always move to the proposal
    self.chain.push(proposal);
    visualizer.queue.push({ type: "accept", proposal: proposal.copy() });
  },
});
