"use strict";

MCMC.registerAlgorithm("HamiltonianMC", {
  description: "Hamiltonian Monte Carlo",

  about: () => {
    window.open("https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo");
  },

  init: (self) => {
    self.leapfrogSteps = 37;
    self.dt = 0.1;
    // optional dual-averaging step-size adaptation (Hoffman & Gelman 2011,
    // Algorithm 5): tunes the leapfrog step towards a target acceptance rate
    // delta during the first M_adapt iterations, then freezes it
    self.adaptStepSize = false;
    self.delta = 0.65;
    self.M_adapt = 200;

    self.joint = (theta, r) => {
      return Math.exp(self.logDensity(theta) - r.norm2() / 2);
    };

    self.leapFrog = (theta, r, epsilon) => {
      const r_ = r.add(self.gradLogDensity(theta).scale(epsilon / 2));
      const theta_ = theta.add(r_.scale(epsilon));
      r_.increment(self.gradLogDensity(theta_).scale(epsilon / 2));
      return { theta: theta_, r: r_ };
    };

    // Algorithm 4: double/halve epsilon until one leapfrog step from the
    // ORIGINAL state crosses acceptance probability 1/2
    self.findReasonableEpsilon = (theta) => {
      let epsilon = 0.1;
      const r = MultivariateNormal.getSample(self.dim);
      let result = self.leapFrog(theta, r, epsilon);
      const a = 2 * (self.joint(result.theta, result.r) / self.joint(theta, r) > 0.5 ? 1 : 0) - 1;
      while (Math.pow(self.joint(result.theta, result.r) / self.joint(theta, r), a) > Math.pow(2.0, -a)) {
        epsilon = Math.pow(2, a) * epsilon;
        result = self.leapFrog(theta, r, epsilon);
      }
      // clamp: unlike dual-averaging HMC with a fixed path LENGTH, we keep a
      // fixed number of leapfrog steps, so a large initial epsilon would send
      // the whole trajectory off to infinity before adaptation can react
      return Math.min(0.5, Math.max(1e-3, epsilon));
    };
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
    // dual-averaging state (initialised even when adaptation is off, so the
    // toggle can be flipped mid-session and reset cleanly)
    self.epsilon = [self.adaptStepSize ? self.findReasonableEpsilon(self.chain.last()) : self.dt];
    self.mu = Math.log(10 * self.epsilon[0]);
    self.epsilon_bar = [1.0];
    self.H_bar = [0.0];
    self.gamma = 0.2;
    self.t0 = 10;
    self.kappa = 0.75;
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.05, 0.5).step(0.025).name("Leapfrog Δt");
    folder
      .add(self, "adaptStepSize")
      .name("Adapt step size")
      .onChange(() => {
        sim.reset();
      });
    folder
      .add(self, "delta", 0.2, 0.95)
      .step(0.05)
      .name("δ (target accept)")
      .onChange(() => {
        sim.reset();
      });
    folder.open();
  },

  step: (self, visualizer) => {
    const dt = self.adaptStepSize ? self.epsilon.last() : self.dt;
    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    // use leapfrog integration to find proposal
    const q = q0.copy();
    const p = p0.copy();
    const trajectory = [q.copy()];
    for (let i = 0; i < self.leapfrogSteps; i++) {
      p.increment(self.gradLogDensity(q).scale(dt / 2));
      q.increment(p.scale(dt));
      p.increment(self.gradLogDensity(q).scale(dt / 2));
      trajectory.push(q.copy());
    }

    // add integrated trajectory to visualizer animation queue
    const proposalEvent = {
      type: "proposal",
      proposal: q,
      trajectory: trajectory,
      initialMomentum: p0,
    };
    if (self.adaptStepSize) {
      proposalEvent.epsilon = ((dt * 1000) | 0) / 1000;
      proposalEvent.alpha = self.delta - self.H_bar.last();
    }
    visualizer.queue.push(proposalEvent);

    // calculate acceptance ratio
    const H0 = -self.logDensity(q0) + p0.norm2() / 2;
    const H = -self.logDensity(q) + p.norm2() / 2;
    const logAcceptRatio = -H + H0;
    // a diverged trajectory (non-finite H) counts as a certain rejection so
    // it cannot poison the dual-averaging statistics
    let alpha = Math.min(1, Math.exp(logAcceptRatio));
    if (!isFinite(alpha)) alpha = 0;

    // accept or reject proposal
    if (Math.random() < alpha) {
      self.chain.push(q.copy());
      visualizer.queue.push({ type: "accept", proposal: q });
    } else {
      self.chain.push(q0.copy());
      visualizer.queue.push({ type: "reject", proposal: q });
    }

    // dual-averaging step-size update (Algorithm 5)
    const m = self.chain.length;
    if (self.adaptStepSize && m <= self.M_adapt) {
      self.H_bar.push((1 - 1 / (m + self.t0)) * self.H_bar.last() + (1 / (m + self.t0)) * (self.delta - alpha));
      let log_epsilon = self.mu - (Math.sqrt(m) / self.gamma) * self.H_bar.last();
      log_epsilon = Math.min(Math.max(log_epsilon, -4.5), Math.log(2));
      self.epsilon.push(Math.exp(log_epsilon));
      self.epsilon_bar.push(
        Math.exp(
          Math.pow(m, -self.kappa) * log_epsilon + (1 - Math.pow(m, -self.kappa)) * Math.log(self.epsilon_bar.last())
        )
      );
    } else {
      self.epsilon.push(self.epsilon_bar.last());
    }
  },
});
