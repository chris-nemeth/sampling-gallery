"use strict";

// Apogee-to-Apogee Path Sampler (Sherlock, Urbas & Ludkin, JCGS 2023;
// arXiv:2112.08187): an HMC-like sampler that is robust to tuning. A leapfrog
// path is segmented at its APOGEES — the points where the trajectory switches
// from moving uphill to downhill in the potential U = -log pi, i.e. where
// p . grad U flips from positive to negative. One iteration draws a fresh
// momentum, places the current segment uniformly at random inside a window of
// K+1 segments (c ~ U{0..K} segments behind, K-c ahead), integrates forward
// and backward until the window is covered, then proposes a point z' from the
// whole path with probability proportional to the paper's recommended weight
// (Scheme 3)
//     w(z, z') = ||x' - x||^2 pi~(z'),      pi~(z) = exp(-H(z)),
// accepting with probability min(1, sum_z w(z_curr, z) / sum_z w(z_prop, z))
// — Eq. (6) of the paper after the common factors cancel. Segment invariance
// (the same path, apogees and segments arise from any starting point on it)
// is what makes this valid.
MCMC.registerAlgorithm("AAPS", {
  description: "Apogee-to-Apogee Path Sampler",

  about: function () {
    window.open("https://arxiv.org/abs/2112.08187");
  },

  init: function (self) {
    self.epsilon = 0.25; // leapfrog step size
    self.K = 8; // extra apogee-to-apogee segments in the window
    self.Delta = 1000; // path stability threshold on max H - min H (paper Eq. 7)
  },

  reset: function (self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function (self, folder) {
    folder.add(self, "epsilon", 0.05, 1.5).step(0.05).name("Leapfrog ε");
    folder.add(self, "K", 0, 20).step(1).name("Extra segments K");
    folder.open();
  },

  step: function (self, visualizer) {
    var eps = self.epsilon;
    var U = function (x) {
      var v = self.logDensity(x);
      return isFinite(v) ? -v : Infinity;
    };
    var gradU = function (x) {
      return self.gradLogDensity(x).scale(-1);
    };
    var H = function (x, p) {
      return U(x) + 0.5 * p.dot(p);
    };

    // integrate one direction from (x0, p0) until nApo apogees are crossed;
    // points beyond the last apogee are outside the window and excluded.
    // seg records how many apogees have been crossed before the point.
    var collect = function (x0, p0, nApo, cap) {
      var pts = [];
      var x = x0.copy(),
        p = p0.copy();
      var up = p.dot(gradU(x)) > 0;
      var crossed = 0;
      while (pts.length < cap) {
        p = p.subtract(gradU(x).scale(eps / 2));
        x = x.add(p.scale(eps));
        p = p.subtract(gradU(x).scale(eps / 2));
        var h = H(x, p);
        if (!isFinite(h)) return { pts: pts, complete: false };
        var up2 = p.dot(gradU(x)) > 0;
        if (up && !up2) {
          crossed++;
          if (crossed >= nApo) return { pts: pts, complete: true };
        }
        pts.push({ x: x.copy(), H: h, seg: crossed });
        up = up2;
      }
      return { pts: pts, complete: false };
    };

    var xCurr = self.chain.last().copy();
    var p0 = MultivariateNormal.getSample(self.dim);
    var c = Math.floor(Math.random() * (self.K + 1));
    var fwd = collect(xCurr, p0, self.K - c + 1, 2000);
    var bwd = collect(xCurr, p0.scale(-1), c + 1, 2000);

    // time-ordered path: reversed backward part, the current point, forward part
    var pts = [];
    for (var i = bwd.pts.length - 1; i >= 0; i--)
      pts.push({ x: bwd.pts[i].x, H: bwd.pts[i].H, seg: -bwd.pts[i].seg });
    var curIdx = pts.length;
    pts.push({ x: xCurr.copy(), H: H(xCurr, p0), seg: 0 });
    for (var i = 0; i < fwd.pts.length; i++) pts.push(fwd.pts[i]);
    var n = pts.length;

    var minH = Infinity,
      maxH = -Infinity;
    for (var i = 0; i < n; i++) {
      if (pts[i].H < minH) minH = pts[i].H;
      if (pts[i].H > maxH) maxH = pts[i].H;
    }
    // path rejection: numerically unstable trajectory (Eq. 7) or runaway path
    var reject = !fwd.complete || !bwd.complete || maxH - minH > self.Delta;

    var propIdx = curIdx;
    if (!reject) {
      // Scheme 3 weights, seen from the current point (common exp(minH) cancels)
      var ws = new Array(n);
      var Wcurr = 0;
      for (var i = 0; i < n; i++) {
        var d = pts[i].x.subtract(xCurr);
        ws[i] = d.dot(d) * Math.exp(-(pts[i].H - minH));
        Wcurr += ws[i];
      }
      if (Wcurr > 0) {
        var u = Math.random() * Wcurr;
        var cum = 0;
        for (var i = 0; i < n; i++) {
          cum += ws[i];
          if (u <= cum) {
            propIdx = i;
            break;
          }
        }
        // the same sum, seen from the proposal (segment invariance)
        var Wprop = 0;
        for (var i = 0; i < n; i++) {
          var d = pts[i].x.subtract(pts[propIdx].x);
          Wprop += d.dot(d) * Math.exp(-(pts[i].H - minH));
        }
        if (Math.random() >= Math.min(1, Wcurr / Wprop)) propIdx = curIdx; // rejected
      }
    }
    var accepted = propIdx !== curIdx;

    // overlay: the path with segments in alternating colours, apogee
    // boundaries marked, the current point and the proposal
    var segments = [];
    for (var i = 1; i < n; i++) {
      var odd = ((pts[i].seg % 2) + 2) % 2 === 1;
      segments.push({
        from: [pts[i - 1].x[0], pts[i - 1].x[1]],
        to: [pts[i].x[0], pts[i].x[1]],
        color: odd ? "#e69f00" : "#0088b0",
        lw: 1.5,
        alpha: 0.6,
      });
    }
    var points = [];
    for (var i = 1; i < n; i++)
      if (pts[i].seg !== pts[i - 1].seg)
        points.push({ center: [pts[i].x[0], pts[i].x[1]], radius: 0.055, fill: "#555", alpha: 0.8 });
    points.push({ center: [xCurr[0], xCurr[1]], radius: 0.07, fill: "#000" });
    points.push({ center: [pts[propIdx].x[0], pts[propIdx].x[1]], radius: 0.08, fill: "#0072b2" });

    visualizer.queue.push({
      type: "overlay",
      clear: true,
      segments: segments,
      points: points,
      metrics: [{ k: "Path pts", v: String(n) }],
      labels: reject ? ["path unstable: the leapfrog energy error blew up — reduce the step size ε"] : null,
    });
    visualizer.queue.push({ type: "proposal", proposal: pts[propIdx].x.copy() });
    visualizer.queue.push({ type: accepted ? "accept" : "reject", proposal: pts[propIdx].x.copy() });

    self.chain.push(pts[propIdx].x.copy());
  },
});
