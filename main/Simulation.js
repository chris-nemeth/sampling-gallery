"use strict";

class Simulation {
  constructor() {
    this.mcmc = {
      initialized: false,
      hasAlgorithm: false,
      hasTarget: false,
      dim: 2,
    };
    this.delay = 250;
    this.tweeningDelay = 0;
    this.autoplay = true;
    this.stepCount = 0;
    this.acceptCount = 0;
    this.rejectCount = 0;
    this.lastStatsUpdate = 0;
  }
  setAlgorithm(algorithmName) {
    this.hasAlgorithm = true;
    this.algorithm = algorithmName;
    this.mcmc.initialized = false;
    this.mcmc.description = MCMC.algorithms[algorithmName].description;
    this.mcmc.init = MCMC.algorithms[algorithmName].init;
    this.mcmc.reset = MCMC.algorithms[algorithmName].reset;
    this.mcmc.step = MCMC.algorithms[algorithmName].step;
    this.mcmc.attachUI = MCMC.algorithms[algorithmName].attachUI;
    this.mcmc.about = MCMC.algorithms[algorithmName].about;
    document.getElementById("info").innerHTML = this.mcmc.description;
    if (this.hasAlgorithm && this.hasTarget) {
      if (this.mcmc.initialized == false) this.mcmc.init(this.mcmc);
      this.mcmc.reset(this.mcmc);
      this.mcmc.initialized = true;
      this.visualizer.resize();
    }
    this.resetStats();
  }
  setTarget(targetName) {
    this.hasTarget = true;
    this.target = targetName;
    this.mcmc.logDensity = MCMC.targets[targetName].logDensity;
    this.mcmc.gradLogDensity = MCMC.targets[targetName].gradLogDensity;

    // update visualizer extents
    const options = { ...MCMC.targets[targetName] };
    this.visualizer.xmin = options.xmin;
    this.visualizer.xmax = options.xmax;
    this.visualizer.resize();

    // TODO: actually derive Hessians
    // in the meantime, use finite difference :sadface:
    var grad = this.mcmc.gradLogDensity,
      N = this.mcmc.dim;
    var h = 1e-8;
    this.mcmc.hessLogDensity = function (x) {
      var hess = zeros(N, N);
      var Delta = eye(N, N).scale(h);
      for (let i = 0; i < N; ++i) {
        for (let j = 0; j < N; ++j) {
          hess[i * N + j] =
            (grad(x.add(Delta.col(j)))[i] - grad(x)[i]) / (2 * h) +
            (grad(x.add(Delta.col(i)))[j] - grad(x)[j]) / (2 * h);
        }
      }
      return hess;
    };

    // update contours
    const xmin = this.visualizer.xmin;
    const xmax = this.visualizer.xmax;
    const ymin = this.visualizer.ymin;
    const ymax = this.visualizer.ymax;

    const nx = 480,
      ny = 256,
      nz = 7;
    this.computeContours(this.mcmc.logDensity, xmin, xmax, ymin, ymax, nx, ny, nz);

    if (this.mcmc.initialized) this.mcmc.reset(this.mcmc);
    if (this.hasAlgorithm && this.hasTarget) {
      if (this.mcmc.initialized == false) this.mcmc.init(this.mcmc);
      this.mcmc.reset(this.mcmc);
      this.mcmc.initialized = true;
      this.visualizer.resize();
    }
    this.resetStats();
  }
  computeContours(logDensity, xmin, xmax, ymin, ymax, nx, ny, nz) {
    // get contours
    var x = linspace(xmin, xmax, nx);
    var y = linspace(ymin, ymax, ny);
    var data = [];
    var point = zeros(2, 1);
    var min = 1e10,
      max = 0;
    for (let i = 0; i < nx; ++i) {
      data.push([]);
      point[0] = x[i];
      for (let j = 0; j < ny; ++j) {
        point[1] = y[j];
        var val = Math.exp(logDensity(point));
        data[i].push(val);
        if (val > max) max = val;
        if (val < min) min = val;
      }
    }
    var z = linspace(min + 0.01 * (max - min), max - 0.02 * (max - min), nz);
    var c = new Conrec();
    c.contour(data, 0, nx - 1, 0, ny - 1, x, y, nz, z);
    var contours = c.contourList();
    this.mcmc.contours = [];
    this.mcmc.contourData = data;
    this.mcmc.contourLevels = z;
    for (let i = 0; i < contours.length; ++i) {
      var contour = [];
      for (let j = 0; j < contours[i].length; ++j) contour.push([contours[i][j].x, contours[i][j].y]);
      this.mcmc.contours.push(contour);
    }

    // numerically integrate to get marginal densities
    this.mcmc.xgrid = x;
    this.mcmc.ygrid = y;
    this.mcmc.marginals = [zeros(nx), zeros(ny)];
    for (let i = 0; i < nx; ++i) {
      for (let j = 0; j < ny; ++j) this.mcmc.marginals[0][i] += data[i][j];
      this.mcmc.marginals[0][i];
    }
    this.mcmc.marginals[0] = this.mcmc.marginals[0].scale(1.0 / this.mcmc.marginals[0].maxCoeff());
    for (let j = 0; j < ny; ++j) {
      for (let i = 0; i < nx; ++i) this.mcmc.marginals[1][j] += data[i][j];
      this.mcmc.marginals[1][j];
    }
    this.mcmc.marginals[1] = this.mcmc.marginals[1].scale(1.0 / this.mcmc.marginals[1].maxCoeff());

    // stash the grid so the heatmap can be recoloured (colormap change)
    // without recomputing the (expensive) density evaluation
    this.mcmc.densityData = data;
    this.mcmc.densityMin = min;
    this.mcmc.densityMax = max;
    this.mcmc.densityNx = nx;
    this.mcmc.densityNy = ny;
    this.buildDensityImage();
  }
  // rasterise the density grid into mcmc.densityCanvas using the visualizer's
  // current colormap; alpha ramps with density so low-probability regions fade
  // to the page background
  buildDensityImage() {
    const data = this.mcmc.densityData;
    if (!data) return;
    const nx = this.mcmc.densityNx,
      ny = this.mcmc.densityNy,
      min = this.mcmc.densityMin,
      max = this.mcmc.densityMax;
    const name = this.visualizer ? this.visualizer.colormap : "viridis";
    var buffer = document.createElement("canvas");
    buffer.width = nx;
    buffer.height = ny;
    var context = buffer.getContext("2d");
    var image = context.createImageData(nx, ny);
    for (let j = 0; j < ny; ++j) {
      for (let i = 0; i < nx; ++i) {
        var base = 4 * ((ny - 1 - j) * nx + i);
        var t = (data[i][j] - min) / (max - min);
        var rgb = Colormaps.get(name, t);
        image.data[base] = rgb[0] | 0;
        image.data[base + 1] = rgb[1] | 0;
        image.data[base + 2] = rgb[2] | 0;
        image.data[base + 3] = (Math.sqrt(t) * 255) | 0;
      }
    }
    context.putImageData(image, 0, 0);
    this.mcmc.densityCanvas = buffer;
  }
  reset() {
    this.mcmc.reset(this.mcmc);
    this.visualizer.resize();
    this.resetStats();
  }
  step() {
    if (this.visualizer.queue.length == 0) {
      this.mcmc.step(this.mcmc, this.visualizer);
      this.stepCount++;
      for (let i = 0; i < this.visualizer.queue.length; i++) {
        const type = this.visualizer.queue[i].type;
        if (type == "accept") this.acceptCount++;
        else if (type == "reject") this.rejectCount++;
      }
    }
    if (this.visualizer.animateProposal == false) {
      while (this.visualizer.queue.length > 0) this.visualizer.dequeue();
    } else {
      this.visualizer.dequeue();
    }
    this.visualizer.render();
    this.updateStats();
  }
  resetStats() {
    this.stepCount = 0;
    this.acceptCount = 0;
    this.rejectCount = 0;
    this.lastStatsUpdate = 0;
    this.updateStats(true);
  }
  updateStats(force) {
    const el = document.getElementById("stats");
    if (!el) return;
    const chain = this.mcmc.chain;
    if (!this.visualizer || !this.visualizer.showDiagnostics || !chain || chain.length == 0) {
      el.style.display = "none";
      return;
    }
    // throttle: recomputing over the full chain every frame is wasteful at delay=0
    const now = performance.now();
    if (!force && now - this.lastStatsUpdate < 200) return;
    this.lastStatsUpdate = now;
    el.style.display = "inline-block";

    const hasWeights = this.mcmc.hasOwnProperty("chain_weights");
    let m0 = 0,
      m1 = 0,
      wsum = 0,
      w2sum = 0;
    for (let i = 0; i < chain.length; i++) {
      const w = hasWeights ? this.mcmc.chain_weights[i] : 1;
      m0 += w * chain[i][0];
      m1 += w * chain[i][1];
      wsum += w;
      w2sum += w * w;
    }
    m0 /= wsum;
    m1 /= wsum;

    const proposals = this.acceptCount + this.rejectCount;
    const lines = [];
    lines.push("steps  " + this.stepCount);
    lines.push("accept " + (proposals > 0 ? ((100 * this.acceptCount) / proposals).toFixed(0) + "%" : "n/a"));
    lines.push("mean   (" + m0.toFixed(2) + ", " + m1.toFixed(2) + ")");
    if (hasWeights) {
      // Kish effective sample size for weighted samples (nested sampling)
      lines.push("ESS    " + ((wsum * wsum) / w2sum).toFixed(0) + " (weighted)");
    } else if (proposals > 0 && chain.length > 10) {
      const tau = this.integratedAutocorrTime(chain);
      lines.push("ESS    " + (chain.length / tau).toFixed(0) + " (\u03C4 = " + tau.toFixed(1) + ")");
    } else {
      // chain is not a Markov chain (e.g. SVGD particles) or too short
      lines.push("ESS    n/a");
    }
    el.textContent = lines.join("\n");
  }
  // integrated autocorrelation time over a recent window, with Sokal's
  // adaptive truncation: sum lags while k < c * tau(k) (c = 5). This avoids
  // the tail-chopping bias of a hard threshold — a fixed cutoff at rho = 0.05
  // underestimates tau by ~10-15% on strongly correlated chains.
  integratedAutocorrTime(chain) {
    const W = Math.min(chain.length, 2000);
    const start = chain.length - W;
    let m0 = 0,
      m1 = 0;
    for (let i = start; i < chain.length; i++) {
      m0 += chain[i][0];
      m1 += chain[i][1];
    }
    m0 /= W;
    m1 /= W;
    let c0 = 0;
    for (let i = start; i < chain.length; i++)
      c0 += Math.pow(chain[i][0] - m0, 2) + Math.pow(chain[i][1] - m1, 2);
    c0 /= W;
    if (c0 <= 0) return 1;
    let tau = 1;
    const maxLag = Math.min(500, W - 2);
    for (let k = 1; k <= maxLag; k++) {
      let ck = 0;
      for (let i = start + k; i < chain.length; i++)
        ck += (chain[i][0] - m0) * (chain[i - k][0] - m0) + (chain[i][1] - m1) * (chain[i - k][1] - m1);
      ck /= W - k;
      const rho = ck / c0;
      if (rho <= 0) break; // noise floor
      tau += 2 * rho;
      if (k >= 5 * tau) break; // Sokal window
    }
    return tau;
  }
  animate() {
    var self = this;
    if (this.autoplay || this.visualizer.tweening) this.step();
    if (this.visualizer.tweening) {
      setTimeout(function () {
        requestAnimationFrame(function () {
          self.animate();
        });
      }, self.tweeningDelay);
    } else {
      setTimeout(function () {
        requestAnimationFrame(function () {
          self.animate();
        });
      }, self.delay);
    }
  }
}
var viz, sim, gui;

function getUrlVars() {
  var vars = [],
    pair;
  var pairs = window.location.search.substr(1).split("&");
  for (let i = 0; i < pairs.length; i++) {
    pair = pairs[i].split("=");
    vars.push(pair[0]);
    vars[pair[0]] = pair[1] && decodeURIComponent(pair[1].replace(/\+/g, " "));
  }
  return vars;
}

window.onload = function () {
  viz = new Visualizer(
    document.getElementById("plotCanvas"),
    document.getElementById("xHistCanvas"),
    document.getElementById("yHistCanvas")
  );
  sim = new Simulation();
  sim.visualizer = viz;
  viz.simulation = sim;

  var algorithm = MCMC.algorithmNames[0];
  var target = MCMC.targetNames[0];
  var seed = Math.seedrandom();

  function parseBool(value) {
    return value == "true";
  }

  if (window.location.search != "") {
    var queryParams = getUrlVars();

    if ("algorithm" in queryParams && MCMC.algorithmNames.indexOf(queryParams["algorithm"]) > -1) {
      algorithm = queryParams["algorithm"];
    }
    if ("target" in queryParams && MCMC.targetNames.indexOf(queryParams["target"]) > -1) {
      target = queryParams["target"];
    }
    if ("seed" in queryParams) {
      // reseed
      seed = Math.seedrandom(queryParams["seed"]);
    }
    let config = [
      ["delay", parseInt, sim, "sim"],
      ["tweeningDelay", parseInt, sim, "sim"],
      ["autoplay", parseBool, sim, "sim"],
      ["animateProposal", parseBool, viz, "viz"],
      ["showSamples", parseBool, viz, "viz"],
      ["showHistograms", parseBool, viz, "viz"],
      ["showDiagnostics", parseBool, viz, "viz"],
      ["histBins", parseInt, viz, "viz"],
    ];
    for (let i = 0; i < config.length; i++) {
      let param = config[i][0],
        parse = config[i][1],
        obj = config[i][2],
        objName = config[i][3];
      if (param in queryParams) {
        let value = parse(queryParams[param]);
        obj[param] = value;
      }
    }
  }

  sim.setAlgorithm(algorithm);
  sim.setTarget(target);

  sim.mcmc.init(sim.mcmc);
  viz.setTheme(viz.theme);
  window.onresize = function () {
    viz.resize();
  };

  gui = new lil.GUI({ width: 300 });

  // the algorithm-options folder is rebuilt whenever the algorithm changes;
  // keep a reference so it can be destroyed cleanly (lil-gui folder.destroy())
  var algoFolder = null;
  function rebuildAlgoOptions() {
    if (algoFolder) algoFolder.destroy();
    algoFolder = gui.addFolder("Algorithm Options");
    sim.mcmc.attachUI(sim.mcmc, algoFolder);
    algoFolder.add(sim.mcmc, "about").name("About this algorithm");
    algoFolder.open();
  }

  var f1 = gui.addFolder("Simulation options");
  f1.add(sim, "algorithm", MCMC.algorithmNames)
    .name("Algorithm")
    .onChange(function (value) {
      sim.setAlgorithm(value);
      rebuildAlgoOptions();
    });
  f1.add(sim, "target", MCMC.targetNames)
    .name("Target distribution")
    .onChange(function (value) {
      sim.setTarget(value);
    });
  f1.add(sim, "autoplay").name("Autoplay");
  f1.add(sim, "delay", 0, 1000)
    .name("Autoplay delay")
    .onChange(function (value) {
      if (value == 0) {
        viz.animateProposal = false;
      } else {
        viz.animateProposal = true;
      }
    });
  f1.add(sim, "tweeningDelay", 0, 200).name("Tweening delay");
  f1.add(sim, "step").name("Step");
  f1.add(sim, "reset").name("Reset");
  f1.open();

  var f2 = gui.addFolder("Visualization Options");
  f2.add(viz, "colormap", Colormaps.names)
    .name("Colormap")
    .onChange(function (value) {
      sim.buildDensityImage();
      viz.redrawDensity();
    });
  f2.add(viz, "animateProposal").name("Animate proposal").listen();
  f2.add(viz, "showTargetDensity").name("Show target");
  f2.add(viz, "showSamples").name("Show samples");
  f2.add(viz, "showHistograms").name("Show histogram");
  f2.add(viz, "showDiagnostics")
    .name("Show diagnostics")
    .onChange(function (value) {
      sim.updateStats(true);
    });
  f2.add(viz, "histBins", 20, 200)
    .step(1)
    .name("Histogram bins")
    .onChange(function (value) {
      viz.drawHistograms();
      viz.render();
    });
  f2.add(viz, "theme", ["light", "dark"])
    .name("Theme")
    .onChange(function (value) {
      viz.setTheme(value);
    });
  f2.add(viz, "trails").name("Sample trails");
  f2.add({ savePNG: function () { viz.savePNG(); } }, "savePNG").name("Save PNG");
  f2.add({ recordGIF: function () { viz.recordGIF(3); } }, "recordGIF").name("Record GIF (3s)");
  f2.open();

  rebuildAlgoOptions();

  sim.animate();
};
