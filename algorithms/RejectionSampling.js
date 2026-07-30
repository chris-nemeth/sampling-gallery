"use strict";

MCMC.registerAlgorithm("RejectionSampling", {
  description: "Rejection Sampling",

  about: () => {
    window.open("https://en.wikipedia.org/wiki/Rejection_sampling");
  },

  init: (self) => {
    self.proposalSd = 3; // std deviation of the Gaussian proposal q = N(0, s^2 I)
    self.batchSize = 10; // proposals drawn per animation step
  },

  reset: (self) => {
    self.chain = [];
    // accepted draws are i.i.d., so every sample carries equal weight; setting
    // chain_weights makes the stats panel report the correct ESS (= n)
    self.chain_weights = [];
    self.nProposed = 0;
    self.nAccepted = 0;

    // Proposal distribution q = N(0, s^2 I)
    const s = self.proposalSd;
    self.proposalDist = new MultivariateNormal(zeros(self.dim, 1), eye(self.dim).scale(s * s));

    // Envelope constant M such that p(x) <= M q(x). The targets are arbitrary
    // unnormalized log densities, so an exact analytic bound is unavailable —
    // we approximate it NUMERICALLY: take the max of log p(x) - log q(x) over
    // a 200x200 grid covering [-6.5, 6.5]^2 (slightly beyond the display
    // window) and inflate by a factor 1.2 as a safety margin against the
    // ratio peaking between grid points.
    const grid = 200;
    const lo = -6.5;
    const hi = 6.5;
    const x = zeros(2, 1);
    let logM = -Infinity;
    for (let i = 0; i < grid; i++) {
      x[0] = lo + ((hi - lo) * i) / (grid - 1);
      for (let j = 0; j < grid; j++) {
        x[1] = lo + ((hi - lo) * j) / (grid - 1);
        const ratio = self.logDensity(x) - self.proposalDist.logDensity(x);
        if (isFinite(ratio) && ratio > logM) logM = ratio;
      }
    }
    if (!isFinite(logM)) logM = 0; // fallback if the target was non-finite everywhere
    self.logM = logM + Math.log(1.2);
  },

  attachUI: (self, folder) => {
    folder
      .add(self, "proposalSd", 1.5, 5)
      .step(0.1)
      .name("Proposal sd")
      .onChange(() => {
        // envelope M depends on q, so recompute it via a full reset
        sim.reset();
      });
    folder.add(self, "batchSize", 1, 50).step(1).name("Batch size");
    folder.open();
  },

  step: (self, visualizer) => {
    const points = [];
    const accepted = [];
    for (let b = 0; b < self.batchSize; b++) {
      const proposal = self.proposalDist.getSample();
      self.nProposed++;
      // accept x ~ q with probability p(x) / (M q(x))
      const logRatio = self.logDensity(proposal) - self.proposalDist.logDensity(proposal) - self.logM;
      if (isFinite(logRatio) && Math.log(Math.random()) < logRatio) {
        self.nAccepted++;
        self.chain.push(proposal.copy());
        self.chain_weights.push(1);
        if (self.chain.length > 5000) {
          self.chain.shift();
          self.chain_weights.shift();
        }
        accepted.push([proposal[0], proposal[1]]);
        points.push({ center: [proposal[0], proposal[1]], radius: 0.06, fill: "#0072b2" });
      } else {
        points.push({ center: [proposal[0], proposal[1]], radius: 0.045, fill: "#d55e00", alpha: 0.85 });
      }
    }

    const acceptance = self.nProposed > 0 ? (100 * self.nAccepted) / self.nProposed : 0;
    const M = Math.exp(self.logM);
    const Mstr = M >= 1000 ? M.toExponential(2) : M.toPrecision(3);
    visualizer.queue.push({
      type: "overlay",
      clear: true,
      ellipses: [{ center: [0, 0], cov: self.proposalDist.cov }],
      points: points,
      samples: accepted,
      labels:
        self.nProposed > 200 && acceptance < 0.2
          ? [
              "acceptance " + acceptance.toFixed(2) + "% (M = " + Mstr + ")",
              "proposal is narrower than the target:",
              "the envelope M must be enormous, so",
              "almost every draw is rejected",
            ]
          : ["acceptance " + acceptance.toFixed(1) + "% (M = " + Mstr + ")"],
      histograms: true,
    });
  },
});
