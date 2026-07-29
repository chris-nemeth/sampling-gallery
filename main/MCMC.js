"use strict";

const MCMC = {
  algorithmNames: [],
  algorithms: {},
  targetNames: [],
  targets: {},
  registerAlgorithm: (name, methods) => {
    MCMC.algorithmNames.push(name);
    MCMC.algorithms[name] = methods;
  },
  computeMean: (chain) => {
    const mean = chain[0].copy();
    for (let i = 1; i < chain.length; ++i) mean.increment(chain[i]);
    return mean.scale(1.0 / chain.length);
  },
  computeAutocorrelation: (chain, lag) => {
    const mean = MCMC.computeMean(chain);
    const autocovariance = zeros(lag, 1);
    for (let k = 0; k <= lag; ++k)
      for (let i = k; i < chain.length; ++i)
        autocovariance[k] += chain[i].subtract(mean).dot(chain[i - k].subtract(mean));
    return autocovariance.scale(1.0 / autocovariance[0]);
  },
};

// Banana distribution
var bananaDist = new MultivariateNormal(
  matrix([[0], [4]]),
  matrix([
    [1, 0.5],
    [0.5, 1],
  ])
);
MCMC.targetNames.push("banana");
MCMC.targets["banana"] = {
  xmin: -6,
  xmax: 6,
  logDensity: (x) => {
    const a = 2,
      b = 0.2;
    const y = zeros(2, 1);
    y[0] = x[0] / a;
    y[1] = x[1] * a + a * b * (x[0] * x[0] + a * a);
    return bananaDist.logDensity(y);
  },
  gradLogDensity: (x) => {
    const a = 2,
      b = 0.2;
    const y = zeros(2, 1);
    y[0] = x[0] / a;
    y[1] = x[1] * a + a * b * (x[0] * x[0] + a * a);
    const grad = bananaDist.gradLogDensity(y);
    const gradx0 = grad[0] / a + grad[1] * a * b * 2 * x[0];
    const gradx1 = grad[1] * a;
    grad[0] = gradx0;
    grad[1] = gradx1;
    return grad;
  },
};

// Donut
MCMC.targetNames.push("donut");
MCMC.targets["donut"] = {
  xmin: -6,
  xmax: 6,
  radius: 2.6,
  sigma2: 0.033,
  logDensity: (x) => {
    const r = x.norm();
    return -Math.pow(r - MCMC.targets.donut.radius, 2) / MCMC.targets.donut.sigma2;
  },
  gradLogDensity: (x) => {
    const r = x.norm();
    if (r == 0) return zeros(2);
    return matrix([
      [(x[0] * (MCMC.targets.donut.radius / r - 1) * 2) / MCMC.targets.donut.sigma2],
      [(x[1] * (MCMC.targets.donut.radius / r - 1) * 2) / MCMC.targets.donut.sigma2],
    ]);
  },
};

// Bivariate normal distribution with no correlation
MCMC.targetNames.push("standard");
const dist = new MultivariateNormal(zeros(2, 1), eye(2));
MCMC.targets["standard"] = {
  xmin: -6,
  xmax: 6,
  logDensity: (x) => {
    return dist.logDensity(x);
  },
  gradLogDensity: (x) => {
    return dist.gradLogDensity(x);
  },
};

// Mixture distribution with three components
const mixtureComponents = [
  new MultivariateNormal(matrix([[-1.5], [-1.5]]), eye(2).scale(0.8)),
  new MultivariateNormal(matrix([[1.5], [1.5]]), eye(2).scale(0.8)),
  new MultivariateNormal(matrix([[-2], [2]]), eye(2).scale(0.5)),
];
MCMC.targetNames.push("multimodal");
MCMC.targets["multimodal"] = {
  xmin: -6,
  xmax: 6,
  logDensity: (x) => {
    return Math.log(
      Math.exp(mixtureComponents[0].logDensity(x)) +
        Math.exp(mixtureComponents[1].logDensity(x)) +
        Math.exp(mixtureComponents[2].logDensity(x))
    );
  },
  gradLogDensity: (x) => {
    const p1 = Math.exp(mixtureComponents[0].logDensity(x));
    const p2 = Math.exp(mixtureComponents[1].logDensity(x));
    const p3 = Math.exp(mixtureComponents[2].logDensity(x));
    return mixtureComponents[0]
      .gradLogDensity(x)
      .scale(p1)
      .add(mixtureComponents[1].gradLogDensity(x).scale(p2))
      .add(mixtureComponents[2].gradLogDensity(x).scale(p3))
      .scale(1 / (p1 + p2 + p3));
  },
};
// fillin to get last element of array
if (!Array.prototype.last) {
  Array.prototype.last = function () {
    return this[this.length - 1];
  };
}

// "funnel" distribution from Neal, Radford M. 2003. “Slice Sampling.” Annals of Statistics 31 (3): 705–67
const f = (x, m, s) => -0.5 * Math.log(2.0 * Math.PI) - Math.log(s) - 0.5 * Math.pow((x - m) / s, 2);
const dfdx = (x, m, s) => -(x - m) / Math.pow(s, 2);
const dfds = (x, m, s) => (Math.pow(x - m, 2) - Math.pow(s, 2)) / Math.pow(s, 3);
MCMC.targetNames.push("funnel");
MCMC.targets["funnel"] = {
  xmin: -6,
  xmax: 6,
  logDensity: (x_) => {
    const x = [x_[1] - 2, x_[0]];
    const m0 = 0,
      s0 = 3;
    const m1 = 0,
      s1 = Math.exp(x[0] / 2);
    return f(x[0], m0, s0) + f(x[1], m1, s1);
  },
  gradLogDensity: (x_) => {
    const x = [x_[1] - 2, x_[0]];
    const m0 = 0,
      s0 = 3;
    const m1 = 0,
      s1 = Math.exp(x[0] / 2);
    return matrix([
      [dfdx(x[1], m1, Math.exp(x[0] / 2))],
      [dfdx(x[0], m0, s0) + 0.5 * Math.exp(x[0] / 2) * dfds(x[1], m1, Math.exp(x[0] / 2))],
    ]);
  },
};

// "Flower" target from Sejdinovic et al., "Kernel Adaptive Metropolis-
// Hastings" (ICML 2014), Appendix B: the F(r0, A, omega, sigma) family
//   F(x) = exp( -( ||x|| - r0 - A cos(omega * atan2(x2, x1)) )^2 / (2 sigma^2) )
// i.e. a ring of radius r0 with a periodic petal perturbation. Parameters are
// rescaled from the paper's to fit the [-6, 6] display.
MCMC.targetNames.push("flower");
MCMC.targets["flower"] = {
  xmin: -6,
  xmax: 6,
  r0: 3,
  A: 1.6,
  omega: 6,
  sigma2: 0.0625, // sigma = 0.25
  logDensity: (x) => {
    const t = MCMC.targets.flower;
    const r = x.norm();
    const f = r - t.r0 - t.A * Math.cos(t.omega * Math.atan2(x[1], x[0]));
    return -(f * f) / (2 * t.sigma2);
  },
  gradLogDensity: (x) => {
    const t = MCMC.targets.flower;
    const r = x.norm();
    if (r < 1e-9) return zeros(2, 1);
    const theta = Math.atan2(x[1], x[0]);
    const f = r - t.r0 - t.A * Math.cos(t.omega * theta);
    // d(theta)/dx = -y/r^2, d(theta)/dy = x/r^2
    const w = (t.A * t.omega * Math.sin(t.omega * theta)) / (r * r);
    return matrix([
      [(-f / t.sigma2) * (x[0] / r - w * x[1])],
      [(-f / t.sigma2) * (x[1] / r + w * x[0])],
    ]);
  },
};

// Swiss-roll target: a Gaussian tube of constant width sigma around the
// Archimedean spiral r = s*theta. For a point at polar (r, phi) the nearest
// arm lies at radial distance d = s * wrap(r/s - phi), where wrap maps to
// (-pi, pi] (choosing the closest of the spiral's turns), so
//   log p = -d^2 / (2 sigma^2) - r^2 / (2 tau^2).
// The Gaussian envelope N(0, tau^2 I) keeps the target proper — a constant-
// width tube along an infinite spiral is not integrable (chains would drift
// outward along the arm forever); it plays the same role as the
// N(x_{3:d}; 0, I) factor in the flower target.
MCMC.targetNames.push("swissroll");
MCMC.targets["swissroll"] = {
  xmin: -6,
  xmax: 6,
  s: 0.32, // spiral growth rate: arms are 2*pi*s ~ 2.0 apart
  sigma2: 0.0625, // sigma = 0.25 (tube thickness)
  tau2: 9, // tau = 3 (radial envelope keeping the target proper)
  wrap: (a) => {
    // map angle difference to (-pi, pi]
    a = a % (2 * Math.PI);
    if (a > Math.PI) a -= 2 * Math.PI;
    if (a <= -Math.PI) a += 2 * Math.PI;
    return a;
  },
  logDensity: (x) => {
    const t = MCMC.targets.swissroll;
    const r = x.norm();
    const d = t.s * t.wrap(r / t.s - Math.atan2(x[1], x[0]));
    return -(d * d) / (2 * t.sigma2) - (r * r) / (2 * t.tau2);
  },
  gradLogDensity: (x) => {
    const t = MCMC.targets.swissroll;
    const r = x.norm();
    if (r < 1e-9) return zeros(2, 1);
    const delta = t.wrap(r / t.s - Math.atan2(x[1], x[0]));
    // d(delta)/dx = x/(s r) + y/r^2 ; d(delta)/dy = y/(s r) - x/r^2
    const k = (-t.s * t.s * delta) / t.sigma2;
    return matrix([
      [k * (x[0] / (t.s * r) + x[1] / (r * r)) - x[0] / t.tau2],
      [k * (x[1] / (t.s * r) - x[0] / (r * r)) - x[1] / t.tau2],
    ]);
  },
};

// Strongly correlated Gaussian ("ridge", correlation 0.95): the classic
// adaptive-MCMC test case — isotropic random-walk proposals crawl along the
// narrow diagonal while preconditioned/adaptive samplers glide.
const ridgeDist = new MultivariateNormal(
  zeros(2, 1),
  matrix([
    [4, 3.8],
    [3.8, 4],
  ])
);
MCMC.targetNames.push("ridge");
MCMC.targets["ridge"] = {
  xmin: -6,
  xmax: 6,
  logDensity: (x) => {
    return ridgeDist.logDensity(x);
  },
  gradLogDensity: (x) => {
    return ridgeDist.gradLogDensity(x);
  },
};

// Squiggle distribution
const squiggleDist = new MultivariateNormal(
  matrix([[0], [0]]),
  matrix([
    [2, 0.25],
    [0.25, 0.5],
  ])
);
MCMC.targetNames.push("squiggle");
MCMC.targets["squiggle"] = {
  xmin: -6,
  xmax: 6,
  logDensity: (x) => {
    const y = zeros(2, 1);
    y[0] = x[0];
    y[1] = x[1] + Math.sin(5 * x[0]);
    return squiggleDist.logDensity(y);
  },
  gradLogDensity: (x) => {
    const y = zeros(2, 1);
    y[0] = x[0];
    y[1] = x[1] + Math.sin(5 * x[0]);
    const grad = squiggleDist.gradLogDensity(y);
    const gradx0 = grad[0] + grad[1] * 5 * Math.cos(5 * x[0]);
    const gradx1 = grad[1];
    grad[0] = gradx0;
    grad[1] = gradx1;
    return grad;
  },
};
