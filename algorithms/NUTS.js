"use strict";

// The No-U-Turn Sampler (Hoffman & Gelman, 2011, arXiv:1111.4246), unified:
// the default is the efficient (Algorithm 3) tree with dual-averaging
// step-size adaptation (Algorithm 6) — the configuration used by Stan.
// Advanced options expose the naive tree of Algorithm 2 (which stores every
// candidate state, useful for seeing the doubling construction) and allow
// switching adaptation off in favour of a fixed leapfrog step size.
MCMC.registerAlgorithm("NUTS", {
  description: "No-U-Turn Sampler",

  about: function () {
    window.open("http://arxiv.org/abs/1111.4246");
  },

  init: function (self) {
    self.treeMethod = "Efficient";
    self.adaptStepSize = true;
    self.delta = 0.65; // target acceptance statistic for dual averaging
    self.M_adapt = 200;
    self.dt = 0.1; // fixed leapfrog step, used when adaptation is off
    self.Delta_max = 1000;

    self.joint = function (q, p) {
      return Math.exp(self.logDensity(q) - p.norm2() / 2);
    };

    self.leapFrog = function (theta, r, epsilon) {
      var r_ = r.add(self.gradLogDensity(theta).scale(epsilon / 2));
      var theta_ = theta.add(r_.scale(epsilon));
      r_.increment(self.gradLogDensity(theta_).scale(epsilon / 2));
      return { theta: theta_, r: r_ };
    };

    // Algorithm 4: double/halve epsilon until one leapfrog step from the
    // ORIGINAL state crosses acceptance probability 1/2
    self.findReasonableEpsilon = function (theta) {
      var epsilon = 0.1;
      var r = MultivariateNormal.getSample(self.dim);
      var result = self.leapFrog(theta, r, epsilon);
      var a = 2 * (self.joint(result.theta, result.r) / self.joint(theta, r) > 0.5 ? 1 : 0) - 1;
      while (Math.pow(self.joint(result.theta, result.r) / self.joint(theta, r), a) > Math.pow(2.0, -a)) {
        epsilon = Math.pow(2, a) * epsilon;
        result = self.leapFrog(theta, r, epsilon);
      }
      return Math.max(0.1, epsilon);
    };
  },

  reset: function (self) {
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

  attachUI: function (self, folder) {
    folder
      .add(self, "adaptStepSize")
      .name("Adapt step size")
      .onChange(function () {
        sim.reset();
      });
    folder
      .add(self, "delta", 0.2, 0.95)
      .step(0.05)
      .name("δ (target accept)")
      .onChange(function () {
        sim.reset();
      });
    folder.add(self, "dt", 0.025, 0.6).step(0.025).name("Δt (if not adapting)");
    folder.add(self, "treeMethod", ["Efficient", "Naive"]).name("Tree (advanced)");
    folder.open();
  },

  // Notation adopted from http://arxiv.org/pdf/1111.4246v1.pdf
  step: function (self, visualizer) {
    var trajectory = [];
    var dt = self.adaptStepSize ? self.epsilon.last() : self.dt;

    // BuildTree from Algorithm 3 (efficient: O(depth) memory, multinomial-
    // style sampling along the tree), with the acceptance statistic of
    // Algorithm 6 accumulated for dual averaging. theta0/p0 are the global
    // initial state of the whole step; qFrom is this node's pre-step position
    // (used only to draw the leapfrog segment).
    function buildTreeEfficient(q, p, u, v, j, theta0, p0) {
      var q = q.copy(),
        p = p.copy(),
        qFrom = q.copy();
      if (j == 0) {
        // base case - take one leapfrog step in the direction v
        p.increment(self.gradLogDensity(q).scale((v * dt) / 2));
        q.increment(p.scale(v * dt));
        p.increment(self.gradLogDensity(q).scale((v * dt) / 2));
        var n_ = u < Math.exp(self.logDensity(q) - p.norm2() / 2) ? 1 : 0;
        var s_ = u < Math.exp(self.Delta_max + self.logDensity(q) - p.norm2() / 2) ? 1 : 0;
        trajectory.push({
          type: n_ == 1 ? "accept" : "reject",
          from: qFrom.copy(),
          to: q.copy(),
        });
        var alpha = Math.min(1, Math.exp(self.logDensity(q) - p.norm2() / 2 - self.logDensity(theta0) + p0.norm2() / 2));
        if (!isFinite(alpha)) alpha = 0; // diverged leaf: certain rejection, must not poison adaptation
        return { q_p: q, p_p: p, q_m: q, p_m: p, q_: q, n_: n_, s_: s_, alpha_: alpha, n_alpha_: 1 };
      } else {
        // recursion - build the left and right subtrees
        var result = buildTreeEfficient(q, p, u, v, j - 1, theta0, p0);
        var q_m = result.q_m,
          p_m = result.p_m,
          q_p = result.q_p,
          p_p = result.p_p,
          q_ = result.q_,
          n_ = result.n_,
          s_ = result.s_,
          alpha_ = result.alpha_,
          n_alpha_ = result.n_alpha_;
        if (s_ == 1) {
          var result2;
          if (v == -1) {
            result2 = buildTreeEfficient(q_m, p_m, u, v, j - 1, theta0, p0);
            q_m = result2.q_m;
            p_m = result2.p_m;
          } else {
            result2 = buildTreeEfficient(q_p, p_p, u, v, j - 1, theta0, p0);
            q_p = result2.q_p;
            p_p = result2.p_p;
          }
          if (Math.random() < result2.n_ / (n_ + result2.n_)) q_ = result2.q_;
          alpha_ = alpha_ + result2.alpha_;
          n_alpha_ = n_alpha_ + result2.n_alpha_;
          s_ =
            s_ * result2.s_ * (q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0) * (q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0);
          n_ = n_ + result2.n_;
        }
        return { q_p: q_p, p_p: p_p, q_m: q_m, p_m: p_m, q_: q_, n_: n_, s_: s_, alpha_: alpha_, n_alpha_: n_alpha_ };
      }
    }

    // BuildTree from Algorithm 2 (naive: stores the full candidate set C and
    // samples uniformly from it), with the same acceptance statistic added so
    // dual averaging works for either tree.
    function buildTreeNaive(q, p, u, v, j, theta0, p0) {
      var q = q.copy(),
        p = p.copy(),
        qFrom = q.copy();
      if (j == 0) {
        p.increment(self.gradLogDensity(q).scale((v * dt) / 2));
        q.increment(p.scale(v * dt));
        p.increment(self.gradLogDensity(q).scale((v * dt) / 2));
        var C_prime = [];
        if (u < Math.exp(self.logDensity(q) - p.norm2() / 2)) {
          C_prime.push([q.copy(), p.copy()]);
          trajectory.push({ type: "accept", from: qFrom.copy(), to: q.copy() });
        } else {
          trajectory.push({ type: "reject", from: qFrom.copy(), to: q.copy() });
        }
        var s_prime = u < Math.exp(self.Delta_max + self.logDensity(q) - p.norm2() / 2) ? 1 : 0;
        var alpha = Math.min(1, Math.exp(self.logDensity(q) - p.norm2() / 2 - self.logDensity(theta0) + p0.norm2() / 2));
        if (!isFinite(alpha)) alpha = 0; // diverged leaf: certain rejection, must not poison adaptation
        return { q_plus: q, p_plus: p, q_minus: q, p_minus: p, C_prime: C_prime, s_prime: s_prime, alpha_: alpha, n_alpha_: 1 };
      } else {
        var result = buildTreeNaive(q, p, u, v, j - 1, theta0, p0);
        var q_minus = result.q_minus,
          p_minus = result.p_minus,
          C_prime = result.C_prime,
          s_prime = result.s_prime,
          q_plus = result.q_plus,
          p_plus = result.p_plus,
          alpha_ = result.alpha_,
          n_alpha_ = result.n_alpha_;
        var result2;
        if (v == -1) {
          result2 = buildTreeNaive(q_minus, p_minus, u, v, j - 1, theta0, p0);
          q_minus = result2.q_minus;
          p_minus = result2.p_minus;
        } else {
          result2 = buildTreeNaive(q_plus, p_plus, u, v, j - 1, theta0, p0);
          q_plus = result2.q_plus;
          p_plus = result2.p_plus;
        }
        var I1 = q_plus.subtract(q_minus).dot(p_minus) >= 0 ? 1 : 0;
        var I2 = q_plus.subtract(q_minus).dot(p_plus) >= 0 ? 1 : 0;
        s_prime = s_prime * result2.s_prime * I1 * I2;
        for (var i = 0; i < result2.C_prime.length; ++i) C_prime.push(result2.C_prime[i]);
        alpha_ = alpha_ + result2.alpha_;
        n_alpha_ = n_alpha_ + result2.n_alpha_;
        return { q_plus: q_plus, p_plus: p_plus, q_minus: q_minus, p_minus: p_minus, C_prime: C_prime, s_prime: s_prime, alpha_: alpha_, n_alpha_: n_alpha_ };
      }
    }

    var theta0 = self.chain.last().copy();
    var p0 = MultivariateNormal.getSample(self.dim);
    var u = Math.random() * Math.exp(self.logDensity(theta0) - p0.norm2() / 2);

    var q = theta0.copy(),
      q_m = theta0.copy(),
      q_p = theta0.copy(),
      p_m = p0.copy(),
      p_p = p0.copy(),
      j = 0,
      n = 1,
      s = 1,
      alpha = 1,
      n_alpha = 1;

    if (self.treeMethod === "Naive") {
      var C = [[theta0.copy(), p0.copy()]];
      while (s == 1) {
        var v = Math.sign(Math.random() - 0.5);
        var result;
        if (v == -1) {
          trajectory.push({ type: "left" });
          result = buildTreeNaive(q_m, p_m, u, v, j, theta0, p0);
          q_m = result.q_minus;
          p_m = result.p_minus;
        } else {
          trajectory.push({ type: "right" });
          result = buildTreeNaive(q_p, p_p, u, v, j, theta0, p0);
          q_p = result.q_plus;
          p_p = result.p_plus;
        }
        if (result.s_prime == 1) {
          for (var i = 0; i < result.C_prime.length; ++i) C.push(result.C_prime[i]);
        }
        alpha = result.alpha_;
        n_alpha = result.n_alpha_;
        var I1 = q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0;
        var I2 = q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0;
        s = result.s_prime * I1 * I2;
        j = j + 1;
      }
      q = C[Math.floor(Math.random() * C.length)][0];
    } else {
      while (s == 1) {
        var v = Math.sign(Math.random() - 0.5);
        var result;
        if (v == -1) {
          result = buildTreeEfficient(q_m, p_m, u, v, j, theta0, p0);
          q_m = result.q_m;
          p_m = result.p_m;
        } else {
          result = buildTreeEfficient(q_p, p_p, u, v, j, theta0, p0);
          q_p = result.q_p;
          p_p = result.p_p;
        }
        if (result.s_ == 1 && Math.random() < result.n_ / n) q = result.q_.copy();
        alpha = result.alpha_;
        n_alpha = result.n_alpha_;
        s = result.s_ * (q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0) * (q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0);
        n = n + result.n_;
        j = j + 1;
      }
    }

    var proposalEvent = {
      type: "proposal",
      proposal: q,
      nuts_trajectory: trajectory,
      initialMomentum: p0,
    };
    if (self.adaptStepSize) {
      proposalEvent.epsilon = ((self.epsilon.last() * 1000) | 0) / 1000;
      proposalEvent.alpha = self.delta - self.H_bar.last();
    }
    visualizer.queue.push(proposalEvent);

    self.chain.push(q.copy());

    // dual-averaging step-size update (Algorithm 6)
    var m = self.chain.length;
    if (self.adaptStepSize && m <= self.M_adapt) {
      self.H_bar.push(
        (1 - 1 / (m + self.t0)) * self.H_bar.last() + (1 / (m + self.t0)) * (self.delta - alpha / Math.max(1, n_alpha))
      );
      var log_epsilon = self.mu - (Math.sqrt(m) / self.gamma) * self.H_bar.last();
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

    visualizer.queue.push({ type: "accept", proposal: q });
  },
});
