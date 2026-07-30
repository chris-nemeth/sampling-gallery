"use strict";

// Riemannian-manifold HMC (Girolami & Calderhead 2011) with the SoftAbs
// metric (Betancourt 2013). The mass matrix becomes position-dependent,
//     G(x) = Q softabs(Lambda) Q^T,   where  -hess log pi(x) = Q Lambda Q^T
// and softabs(l) = l coth(alpha l) smooths each eigenvalue to a positive
// value (~|l| for large |l|, ~1/alpha near 0), so the sampler is defined for
// any smooth target — no statistical model needed for a Fisher metric. The
// Hamiltonian is non-separable,
//     H(x, p) = U(x) + 1/2 log det G(x) + 1/2 p^T G(x)^{-1} p,
// so each leapfrog step is the implicit "generalized leapfrog": the momentum
// half-step and the position step are solved by fixed-point iteration, and
// momenta are drawn from N(0, G(x)). The local metric is drawn as ellipses
// along the trajectory — watch them stretch with the target's curvature on
// the funnel and banana, where fixed-metric HMC struggles.
MCMC.registerAlgorithm("RiemannianHMC", {
  description: "Riemannian-Manifold HMC",

  about: function () {
    window.open("https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00765.x");
  },

  init: function (self) {
    self.epsilon = 0.14; // leapfrog step size
    self.L = 20; // leapfrog steps per iteration
    self.alpha = 1.0; // SoftAbs sharpness: larger tracks |Hessian| more closely
    self.fixedPointIters = 6;
  },

  reset: function (self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function (self, folder) {
    folder.add(self, "epsilon", 0.02, 0.5).step(0.01).name("Leapfrog ε");
    folder.add(self, "L", 5, 50).step(1).name("Leapfrog steps L");
    folder.add(self, "alpha", 0.2, 10).step(0.1).name("SoftAbs α");
    folder.open();
  },

  // SoftAbs metric at x from the closed-form eigendecomposition of the 2x2
  // Hessian of U = -log pi. Returns G, G^{-1} and G^{1/2} as symmetric
  // [xx, xy, yy] triples plus log det G.
  metric: function (self, x) {
    var Hm = self.hessLogDensity(x); // Hessian of log pi
    var a = -Hm[0],
      b = -(Hm[1] + Hm[2]) / 2,
      c = -Hm[3];
    var tr = a + c;
    var disc = Math.sqrt((a - c) * (a - c) + 4 * b * b);
    var l1 = (tr + disc) / 2,
      l2 = (tr - disc) / 2;
    // eigenvector for l1 (unit); the second is its perpendicular
    var v1x, v1y;
    if (Math.abs(b) > 1e-14) {
      v1x = l1 - c;
      v1y = b;
    } else if (a >= c) {
      v1x = 1;
      v1y = 0;
    } else {
      v1x = 0;
      v1y = 1;
    }
    var nrm = Math.sqrt(v1x * v1x + v1y * v1y);
    v1x /= nrm;
    v1y /= nrm;
    var v2x = -v1y,
      v2y = v1x;
    var softabs = function (l) {
      var al = self.alpha * l;
      if (Math.abs(al) < 1e-4) return 1 / self.alpha; // l coth(al) -> 1/alpha
      return l / Math.tanh(al);
    };
    var s1 = softabs(l1),
      s2 = softabs(l2);
    var comb = function (w1, w2) {
      // w1 v1 v1^T + w2 v2 v2^T as [xx, xy, yy]
      return [w1 * v1x * v1x + w2 * v2x * v2x, w1 * v1x * v1y + w2 * v2x * v2y, w1 * v1y * v1y + w2 * v2y * v2y];
    };
    return {
      G: comb(s1, s2),
      Ginv: comb(1 / s1, 1 / s2),
      sqrtG: comb(Math.sqrt(s1), Math.sqrt(s2)),
      logdet: Math.log(s1) + Math.log(s2),
    };
  },

  step: function (self, visualizer) {
    var alg = MCMC.algorithms["RiemannianHMC"];
    var eps = self.epsilon;

    var U = function (x) {
      var v = self.logDensity(x);
      return isFinite(v) ? -v : Infinity;
    };
    var mul = function (S, p) {
      // symmetric [xx, xy, yy] times vector
      var r = zeros(self.dim, 1);
      r[0] = S[0] * p[0] + S[1] * p[1];
      r[1] = S[1] * p[0] + S[2] * p[1];
      return r;
    };
    var quad = function (S, p) {
      return S[0] * p[0] * p[0] + 2 * S[1] * p[0] * p[1] + S[2] * p[1] * p[1];
    };
    // the position-dependent part of H beyond U(x)
    var psi = function (x, p) {
      var m = alg.metric(self, x);
      return 0.5 * m.logdet + 0.5 * quad(m.Ginv, p);
    };
    var Hfun = function (x, p) {
      return U(x) + psi(x, p);
    };
    // dH/dx = grad U(x) + finite-difference gradient of psi in x
    var gradHx = function (x, p) {
      var g = self.gradLogDensity(x).scale(-1);
      var d = 1e-4;
      for (var k = 0; k < self.dim; k++) {
        var xp = x.copy(),
          xm = x.copy();
        xp[k] += d;
        xm[k] -= d;
        g[k] += (psi(xp, p) - psi(xm, p)) / (2 * d);
      }
      return g;
    };

    var x = self.chain.last().copy();
    var m0 = alg.metric(self, x);
    // momentum ~ N(0, G(x))
    var p = mul(m0.sqrtG, MultivariateNormal.getSample(self.dim));
    var H0 = Hfun(x, p);

    var trajectory = [x.copy()];
    var ellipses = [{ center: [x[0], x[1]], cov: matrix([[m0.Ginv[0] * 0.12, m0.Ginv[1] * 0.12], [m0.Ginv[1] * 0.12, m0.Ginv[2] * 0.12]]) }];
    var diverged = false;

    var xc = x.copy(),
      pc = p.copy();
    for (var l = 0; l < self.L; l++) {
      // generalized leapfrog: implicit momentum half-step
      var ph = pc.copy();
      for (var it = 0; it < self.fixedPointIters; it++) ph = pc.subtract(gradHx(xc, ph).scale(eps / 2));
      // implicit position step
      var v0 = mul(alg.metric(self, xc).Ginv, ph);
      var xn = xc.copy();
      for (var it = 0; it < self.fixedPointIters; it++)
        xn = xc.add(v0.add(mul(alg.metric(self, xn).Ginv, ph)).scale(eps / 2));
      // explicit momentum half-step
      var pn = ph.subtract(gradHx(xn, ph).scale(eps / 2));
      if (!isFinite(Hfun(xn, pn)) || !isFinite(xn[0]) || !isFinite(xn[1])) {
        diverged = true;
        break;
      }
      xc = xn;
      pc = pn;
      trajectory.push(xc.copy());
      // metric ellipse at a few points along the trajectory
      if ((l + 1) % Math.max(1, Math.floor(self.L / 5)) === 0) {
        var mm = alg.metric(self, xc);
        ellipses.push({
          center: [xc[0], xc[1]],
          cov: matrix([[mm.Ginv[0] * 0.12, mm.Ginv[1] * 0.12], [mm.Ginv[1] * 0.12, mm.Ginv[2] * 0.12]]),
        });
      }
    }

    var H1 = diverged ? Infinity : Hfun(xc, pc);
    var logA = H0 - H1;
    var accepted = isFinite(logA) && Math.log(Math.random()) < logA;

    visualizer.queue.push({
      type: "overlay",
      clear: true,
      ellipses: ellipses,
      metrics: [{ k: "ΔH", v: isFinite(H1) ? Math.abs(H1 - H0).toFixed(2) : "∞" }],
      labels: diverged ? ["trajectory diverged — reduce the step size ε"] : null,
    });
    visualizer.queue.push({
      type: "proposal",
      proposal: xc.copy(),
      trajectory: trajectory,
      initialMomentum: p.copy(),
    });
    if (accepted) {
      visualizer.queue.push({ type: "accept", proposal: xc.copy() });
      self.chain.push(xc.copy());
    } else {
      visualizer.queue.push({ type: "reject", proposal: xc.copy() });
      self.chain.push(self.chain.last().copy());
    }
  },
});
