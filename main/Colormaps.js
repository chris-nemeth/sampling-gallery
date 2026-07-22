"use strict";

// Perceptually-uniform colormaps for the target-density heatmap. Each map is a
// list of evenly-spaced RGB anchor points; Colormaps.get(name, t) returns a
// linearly-interpolated [r, g, b] for t in [0, 1].
const Colormaps = {
  names: ["viridis", "magma", "plasma", "grayscale", "classic"],

  data: {
    viridis: [
      [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
      [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37],
    ],
    magma: [
      [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
      [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
    ],
    plasma: [
      [13, 8, 135], [75, 3, 161], [125, 3, 168], [168, 34, 150], [203, 70, 121],
      [229, 107, 93], [248, 148, 65], [253, 195, 40], [240, 249, 33],
    ],
    grayscale: [
      [210, 210, 210], [20, 20, 20],
    ],
    // the original demo look: a single flat blue (kept for continuity)
    classic: [
      [102, 153, 187], [102, 153, 187],
    ],
  },

  get: function (name, t) {
    const stops = Colormaps.data[name] || Colormaps.data.viridis;
    if (stops.length === 1) return stops[0].slice();
    if (t <= 0) return stops[0].slice();
    if (t >= 1) return stops[stops.length - 1].slice();
    const n = stops.length - 1;
    const s = t * n;
    const i = Math.floor(s);
    const f = s - i;
    const a = stops[i];
    const b = stops[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  },
};
