"use strict";

MCMC.registerAlgorithm("SliceSampling", {
  description: "Slice Sampling",

  about: function () {
    window.open(
      "https://projecteuclid.org/journals/annals-of-statistics/volume-31/issue-3/Slice-sampling/10.1214/aos/1056562461.full"
    );
  },

  init: function (self) {
    self.w = 1.0; // initial slice width for stepping out
    self.m = 100; // cap on total stepping-out expansions
  },

  reset: function (self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function (self, folder) {
    folder.add(self, "w", 0.25, 4).step(0.05).name("Width w");
    folder.open();
  },

  step: function (self, visualizer) {
    // treat non-finite log densities as -Infinity (outside the slice)
    var safeLogDensity = function (p) {
      var v = self.logDensity(p);
      return isFinite(v) ? v : -Infinity;
    };

    var x = self.chain.last().copy();
    var overlay = { type: "overlay", clear: true, segments: [], points: [], histograms: false };

    // one full sweep: univariate slice sampling on each coordinate in turn
    for (var i = 0; i < self.dim; i++) {
      // slice level: log y = log f(x) + log U(0,1)
      var logy = safeLogDensity(x) + Math.log(Math.random());

      // --- stepping out (Neal 2003, Fig. 3) ---
      var probe = x.copy();
      var L = x[i] - self.w * Math.random();
      var R = L + self.w;
      var J = Math.floor(self.m * Math.random());
      var K = self.m - 1 - J;
      probe[i] = L;
      while (J > 0 && safeLogDensity(probe) > logy) {
        L -= self.w;
        probe[i] = L;
        J--;
      }
      probe[i] = R;
      while (K > 0 && safeLogDensity(probe) > logy) {
        R += self.w;
        probe[i] = R;
        K--;
      }

      // final stepping-out bracket, drawn along the coordinate axis
      var bracketFrom = [x[0], x[1]];
      var bracketTo = [x[0], x[1]];
      bracketFrom[i] = L;
      bracketTo[i] = R;
      overlay.segments.push({ from: bracketFrom, to: bracketTo, color: "#999", lw: 1.5, alpha: 0.8 });

      // --- shrinkage (Neal 2003, Fig. 5) ---
      var Lbar = L;
      var Rbar = R;
      var accepted = x[i]; // fallback: stay put if the guard trips
      for (var iter = 0; iter < 100; iter++) {
        var xi = Lbar + Math.random() * (Rbar - Lbar);
        probe[i] = xi;
        if (safeLogDensity(probe) > logy) {
          accepted = xi;
          break;
        }
        overlay.points.push({ center: [probe[0], probe[1]], radius: 0.04, fill: "#d55e00", alpha: 0.9 });
        if (xi < x[i]) {
          Lbar = xi;
        } else {
          Rbar = xi;
        }
      }
      x[i] = accepted;
      overlay.points.push({ center: [x[0], x[1]], radius: 0.05, fill: "#0072b2" });
    }

    visualizer.queue.push(overlay);
    visualizer.queue.push({ type: "proposal", proposal: x.copy() });
    visualizer.queue.push({ type: "accept", proposal: x.copy() });
    self.chain.push(x.copy());
  },
});
