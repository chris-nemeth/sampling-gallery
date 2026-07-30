"use strict";

// Shared helpers for the particle-based variational inference (ParVI) family:
// SVGD, Coin SVGD, Wasserstein Particle Descent and SPOS. Loaded before them.
var ParVI = {
  // particle initialization options, selectable per algorithm: exposes how
  // deterministic particle flows depend on their start (e.g. a single-mode
  // start on the multimodal target hides the other modes from SVGD)
  initNames: ["Standard normal", "Overdispersed (sd 3)", "Concentrated (sd 0.3)", "Single mode (1.5, 1.5)"],
  initCloud: function (which, dim) {
    var x = MultivariateNormal.getSample(dim);
    if (which === "Overdispersed (sd 3)") return x.scale(3);
    if (which === "Concentrated (sd 0.3)") return x.scale(0.3);
    if (which === "Single mode (1.5, 1.5)") {
      x = x.scale(0.3);
      x[0] += 1.5;
      x[1] += 1.5;
      return x;
    }
    return x;
  },

  // kernel Stein discrepancy with the same RBF convention as the SVGD entry,
  // k(x,y) = exp(-||x-y||^2 / h). Returns the square root of the V-statistic
  //   KSD^2 = (1/n^2) sum_ij k_ij [ s_i.s_j + (2/h)(x_i-x_j).(s_i-s_j)
  //                                 + 2d/h - 4||x_i-x_j||^2/h^2 ]
  // where s = grad log pi. A principled convergence diagnostic for particle
  // flows: it decreases toward 0 as the cloud approaches the target.
  ksd: function (xs, scores, h, dim) {
    var n = xs.length;
    var acc = 0;
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var d2 = 0,
          ss = 0,
          ds = 0;
        for (var k = 0; k < dim; k++) {
          var dx = xs[i][k] - xs[j][k];
          d2 += dx * dx;
          ss += scores[i][k] * scores[j][k];
          ds += dx * (scores[i][k] - scores[j][k]);
        }
        acc += Math.exp(-d2 / h) * (ss + (2 / h) * ds + (2 * dim) / h - (4 * d2) / (h * h));
      }
    }
    var v = acc / (n * n);
    return v > 0 ? Math.sqrt(v) : 0;
  },

  // solve (A + lambda I) X = B for symmetric positive-definite-ish A (n x n),
  // B is n x d, by Gaussian elimination with partial pivoting. Used by GFSF.
  solve: function (A, B, n, d, lambda) {
    var M = new Float64Array(n * n);
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) M[i * n + j] = A[i * n + j] + (i === j ? lambda : 0);
    var X = new Float64Array(n * d);
    for (var i = 0; i < n * d; i++) X[i] = B[i];
    for (var col = 0; col < n; col++) {
      var piv = col;
      for (var r = col + 1; r < n; r++) if (Math.abs(M[r * n + col]) > Math.abs(M[piv * n + col])) piv = r;
      if (piv !== col) {
        for (var j = col; j < n; j++) {
          var t = M[col * n + j];
          M[col * n + j] = M[piv * n + j];
          M[piv * n + j] = t;
        }
        for (var j = 0; j < d; j++) {
          var t = X[col * d + j];
          X[col * d + j] = X[piv * d + j];
          X[piv * d + j] = t;
        }
      }
      var p = M[col * n + col];
      if (Math.abs(p) < 1e-12) continue;
      for (var r = 0; r < n; r++) {
        if (r === col) continue;
        var f = M[r * n + col] / p;
        if (f === 0) continue;
        for (var j = col; j < n; j++) M[r * n + j] -= f * M[col * n + j];
        for (var j = 0; j < d; j++) X[r * d + j] -= f * X[col * d + j];
      }
    }
    for (var r = 0; r < n; r++) {
      var p = M[r * n + r];
      if (Math.abs(p) > 1e-12) for (var j = 0; j < d; j++) X[r * d + j] /= p;
    }
    return X;
  },
};
