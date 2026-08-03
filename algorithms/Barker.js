"use strict";

MCMC.registerAlgorithm("Barker", {
  description: "Metropolis-Hastings with Barker proposal",

  about: function () {window.open("https://academic.oup.com/jrsssb/article/84/2/496/7056132"); },

  init: function (self) {
    self.sigma = 0.5;
  },

  reset: function (self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function (self, folder) {
    folder.add(self, "sigma", 0.1, 1).step(0.05).name("Proposal σ");
    folder.open();
  },

  step: function (self, visualizer) {

    var x = self.chain.last();

    // draw centred Gaussian increment
    var Zdist = new MultivariateNormal(
      zeros(self.dim),
      eye(self.dim).scale(self.sigma * self.sigma)
    );

    var z = Zdist.getSample();
    
    var gradx = self.gradLogDensity(x);


    var corners = [];
    var cornerProb = [];
    
    if (self.dim == 2) {
    
        var p1 = 1.0 / (1.0 + Math.exp(-z[0] * gradx[0]));
        var p2 = 1.0 / (1.0 + Math.exp(-z[1] * gradx[1]));
    
        var c1 = x.copy();
        c1[0] += z[0];
        c1[1] += z[1];
    
        var c2 = x.copy();
        c2[0] += z[0];
        c2[1] -= z[1];
    
        var c3 = x.copy();
        c3[0] -= z[0];
        c3[1] += z[1];
    
        var c4 = x.copy();
        c4[0] -= z[0];
        c4[1] -= z[1];
    
        corners = [c1,c2,c3,c4];
    
        cornerProb = [
            p1*p2,
            p1*(1-p2),
            (1-p1)*p2,
            (1-p1)*(1-p2)
        ];
    }

    // construct proposal coordinate-wise
    var y = x.copy();

    for (var i = 0; i < self.dim; i++) {

      var p =
        1.0 /
        (1.0 + Math.exp(-z[i] * gradx[i]));

      var b = (Math.random() < p) ? 1.0 : -1.0;

      y[i] += b * z[i];
    }

    //----------------------------------------------------
    // Barker proposal density q(y|x)
    //----------------------------------------------------

    var grady = self.gradLogDensity(y);
    
    var logAcceptRatio =
        self.logDensity(y) -
        self.logDensity(x);
    
    for (var i = 0; i < self.dim; i++) {
    
        logAcceptRatio +=
            Math.log(1 + Math.exp((x[i] - y[i]) * gradx[i])) -
            Math.log(1 + Math.exp((y[i] - x[i]) * grady[i]));
    }

    visualizer.queue.push({
        type: "proposal",
        proposalType: "barker",
        proposal: y.copy(),
        gradient: gradx.copy(),
        corners: corners,
        cornerProb: cornerProb
    });

    if (Math.random() < Math.exp(logAcceptRatio)) {

      self.chain.push(y);

      visualizer.queue.push({
        type: "accept",
        proposal: y.copy()
      });

    } else {

      self.chain.push(x.copy());

      visualizer.queue.push({
        type: "reject",
        proposal: y.copy()
      });

    }
  }
});
