"use strict";

MCMC.registerAlgorithm("SGLD", {
  description: "Stochastic Gradient Langevin Dynamics",

  about: function () {
    window.open("https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf");
  },

  init: function (self) {
    self.sigma = 0.5;
    // there is no data to minibatch in these examples, so gradient noise
    // N(0, gradientNoise^2 I) stands in for the stochasticity of minibatched gradients
    self.gradientNoise = 2;
  },

  reset: function (self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function (self, folder) {
    folder.add(self, "sigma", 0.1, 1).step(0.05).name("Proposal &sigma;");
    folder.add(self, "gradientNoise", 0, 10).step(0.25).name("Gradient noise");
    folder.open();
  },

  step: function (self, visualizer) {
    var noise = MultivariateNormal.getSample(self.dim).scale(self.gradientNoise);
    var gradient = self.gradLogDensity(self.chain.last()).add(noise);
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
