/* viz-charts.js - D3-based chart rendering for C-ML Training View */
/* global d3 */

"use strict";

const VizCharts = (() => {

  // ── Shared config ──────────────────────────────────────────
  const MARGIN = { top: 10, right: 15, left: 50, bottom: 30 };

  /* Read a themed token so charts track the shell's light/dark state and the
     user's theme toggle, with a fallback for offline/isolated rendering. */
  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  /* Categorical palette, assigned by ENTITY (not by chart or by rank) so a
     series keeps one colour across loss and accuracy. Blue/orange/aqua are the
     first three slots of the validated dark categorical order: on the chassis
     surface they clear the lightness band, the chroma floor, adjacent-pair CVD
     separation (worst deltaE 9.4), the normal-vision floor, and 3:1 contrast. */
  const SERIES = {
    training:   "#3987e5",
    testing:    "#d95926",
    validation: "#199e70",
  };


  // ── Series utilities ───────────────────────────────────────

  /* Exponential moving average with bias correction, matching TensorBoard's
     smoothing so a familiar 0-1 weight behaves the way people expect. Without
     the correction the first points sag toward zero and look like a dip that
     is not in the data. */
  function smoothSeries(values, weight) {
    if (!weight || weight <= 0) return values.slice();
    const out = new Array(values.length);
    let last = 0, numAccum = 0;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v == null || !isFinite(v)) { out[i] = v; continue; }
      last = last * weight + (1 - weight) * v;
      numAccum++;
      out[i] = last / (1 - Math.pow(weight, numAccum));
    }
    return out;
  }

  /* Cap a series at `maxPoints` by uniform stride, always keeping the first and
     last sample. Long runs otherwise push tens of thousands of SVG nodes into
     the DOM and the tab stops responding. */
  function downsample(data, maxPoints) {
    const n = data.length;
    if (!maxPoints || n <= maxPoints) return data;
    const stride = Math.ceil(n / maxPoints);
    const out = [];
    for (let i = 0; i < n; i += stride) out.push(data[i]);
    if (out[out.length - 1] !== data[n - 1]) out.push(data[n - 1]);
    return out;
  }

  /* A log axis cannot show <= 0, which loss legitimately reaches. Clamp to the
     smallest positive value present so the toggle degrades instead of blanking
     the chart. */
  function positiveFloor(values) {
    const pos = values.filter(v => v != null && isFinite(v) && v > 0);
    return pos.length ? Math.min(...pos) : 1e-6;
  }

  // ── Distribution chart (percentile bands over epochs) ──────

  /* Renders min/max and p25/p75 as nested bands with the median on top. This is
     the view a single gradient-norm scalar cannot give you: a handful of
     exploding channels widen the max band while the quartiles stay flat, and a
     layer that has stopped learning collapses the whole envelope toward zero. */
  function createDistributionChart(container, dist, opts) {
    const o = opts || {};
    container.innerHTML = "";
    if (!dist || !dist.p50 || dist.p50.length === 0) return;

    const rect = container.getBoundingClientRect();
    const W = rect.width || 400;
    const H = rect.height || 260;
    const w = W - MARGIN.left - MARGIN.right;
    const h = H - MARGIN.top - MARGIN.bottom;
    if (w <= 0 || h <= 0) return;

    const n = dist.p50.length;
    let idx = downsample(d3.range(n), o.maxPoints || 600);

    const at = (arr, i) => (arr && arr[i] != null && isFinite(arr[i])) ? arr[i] : 0;
    const lo = idx.map(i => at(dist.min, i));
    const hi = idx.map(i => at(dist.max, i));

    const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
    const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    const x = d3.scaleLinear().domain([1, n]).range([0, w]);
    const yMin = Math.min(...lo), yMax = Math.max(...hi);
    const pad = (yMax - yMin) * 0.05 || 1e-6;
    const y = d3.scaleLinear().domain([yMin - pad, yMax + pad]).range([h, 0]).nice();

    g.append("g").attr("class", "grid").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).tickSize(-h).tickFormat(""));
    g.append("g").attr("class", "grid")
      .call(d3.axisLeft(y).tickSize(-w).tickFormat(""));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(Math.min(n, 10)).tickFormat(d3.format("d")));
    g.append("g").attr("class", "axis")
      .call(d3.axisLeft(y).ticks(6).tickFormat(d3.format(".2~e")));

    const color = o.color || SERIES.training;
    const band = (loArr, hiArr, opacity) => {
      const area = d3.area()
        .x((d, k) => x(idx[k] + 1))
        .y0((d, k) => y(loArr[k]))
        .y1((d, k) => y(hiArr[k]))
        .curve(d3.curveMonotoneX);
      g.append("path").datum(idx).attr("fill", color).attr("opacity", opacity).attr("d", area);
    };

    band(lo, hi, 0.15);                                             // full range
    band(idx.map(i => at(dist.p25, i)), idx.map(i => at(dist.p75, i)), 0.35); // IQR

    const median = d3.line()
      .x((d, k) => x(idx[k] + 1))
      .y((d, k) => y(at(dist.p50, d)))
      .curve(d3.curveMonotoneX);
    g.append("path").datum(idx).attr("fill", "none").attr("stroke", color)
      .attr("stroke-width", 2).attr("d", median);
  }

  // ── createChart (loss or accuracy) ─────────────────────────
  function createChart(container, chartData, opts) {
    const { type, visible } = opts; // type: "loss" | "accuracy"
    const isAccuracy = type === "accuracy";
    const smoothing = opts.smoothing || 0;
    const logScale = !!opts.logScale && !isAccuracy; // accuracy is bounded, log adds nothing

    // Clear previous
    container.innerHTML = "";

    // Cap the point count before anything touches the DOM.
    chartData = downsample(chartData, opts.maxPoints || 1000);

    // Smooth a copy: the raw series stays available to the tooltip so hovering
    // still reports the measured value, not the filtered one.
    if (smoothing > 0) {
      const keys = isAccuracy
        ? ["trainingAccuracy", "testingAccuracy", "validationAccuracy"]
        : ["trainingLoss", "testingLoss", "validationLoss"];
      const smoothed = {};
      keys.forEach(k => { smoothed[k] = smoothSeries(chartData.map(d => d[k]), smoothing); });
      chartData = chartData.map((d, i) => {
        const c = Object.assign({}, d);
        keys.forEach(k => { c["raw_" + k] = d[k]; c[k] = smoothed[k][i]; });
        return c;
      });
    }

    const rect = container.getBoundingClientRect();
    const W = rect.width || 400;
    const H = rect.height || 280;
    const M = Object.assign({}, MARGIN, { right: 72 });
    const w = W - M.left - M.right;
    const h = H - M.top - M.bottom;

    if (w <= 0 || h <= 0) return;

    const svg = d3.select(container)
      .append("svg")
      .attr("width", W)
      .attr("height", H);

    const g = svg.append("g")
      .attr("transform", `translate(${M.left},${M.top})`);

    // Data keys
    const trainKey = isAccuracy ? "trainingAccuracy" : "trainingLoss";
    const testKey  = isAccuracy ? "testingAccuracy"  : "testingLoss";
    const valKey   = isAccuracy ? "validationAccuracy": "validationLoss";

    // Collect visible extents
    let yVals = [];
    if (visible.training)   yVals = yVals.concat(chartData.map(d => d[trainKey]).filter(v => v != null));
    if (visible.testing)    yVals = yVals.concat(chartData.map(d => d[testKey]).filter(v => v != null));
    if (visible.validation) yVals = yVals.concat(chartData.map(d => d[valKey]).filter(v => v != null));
    if (yVals.length === 0) yVals = [0, 1];

    const yMin = d3.min(yVals);
    const yMax = d3.max(yVals);
    const yPad = (yMax - yMin) * 0.05 || 0.1;

    // Scales
    const x = d3.scaleLinear()
      .domain([1, d3.max(chartData, d => d.epoch) || 1])
      .range([0, w]);

    const y = logScale
      ? d3.scaleLog()
          .domain([positiveFloor(yVals), yMax * 1.1 || 1])
          .range([h, 0])
          .clamp(true)
      : d3.scaleLinear()
          .domain([Math.max(0, yMin - yPad), yMax + yPad])
          .range([h, 0])
          .nice();

    // Grid
    g.append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).tickSize(-h).tickFormat(""));

    g.append("g")
      .attr("class", "grid")
      .call(d3.axisLeft(y).tickSize(-w).tickFormat(""));

    // Axes
    g.append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(Math.min(chartData.length, 10)).tickFormat(d3.format("d")));

    const yTicks = y.ticks(6);
    const pctFmt = d3.format(`.${d3.precisionFixed(yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1)}f`);
    const yAxis = isAccuracy
      ? d3.axisLeft(y).ticks(6).tickFormat(v => `${pctFmt(v)}%`)
      : logScale
        ? d3.axisLeft(y).ticks(6, "~g")
        : d3.axisLeft(y).ticks(6);

    g.append("g")
      .attr("class", "axis")
      .call(yAxis);

    // Line generator
    const line = d3.line()
      .defined(d => d != null)
      .curve(d3.curveMonotoneX)
      .x((d, i) => x(chartData[i].epoch))
      .y(d => y(d));

    // Helper to draw a series as a thin 2px line; points are read on hover
    // rather than marked on every sample.
    function drawSeries(key, color, isDashed) {
      const vals = chartData.map(d => d[key]);
      const defined = vals.map((v, i) => ({ v, i })).filter(d => d.v != null);
      if (defined.length === 0) return;

      const path = g.append("path")
        .datum(vals)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("d", line);

      if (isDashed) path.attr("stroke-dasharray", "6 4");
    }

    // Draw visible series. Colour follows the entity, so a series reads the
    // same in the loss and accuracy charts; testing stays dashed as a print and
    // colour-vision cue alongside the hue.
    const trainColor = SERIES.training;
    const testColor  = SERIES.testing;
    const valColor   = SERIES.validation;

    if (visible.training)   drawSeries(trainKey, trainColor, false);
    if (visible.testing)    drawSeries(testKey,  testColor,  true);
    if (visible.validation) drawSeries(valKey,   valColor,   false);

    // Label each curve's latest measured value at the point where it ends.
    // Labels that would overlap are pushed apart vertically, keeping order.
    const ends = [];
    [[visible.training, trainKey, trainColor],
     [visible.testing, testKey, testColor],
     [visible.validation, valKey, valColor]].forEach(([on, key, color]) => {
      if (!on) return;
      for (let i = chartData.length - 1; i >= 0; i--) {
        const v = chartData[i][key];
        if (v == null || !isFinite(v)) continue;
        const raw = chartData[i]["raw_" + key];
        const shown = raw != null && isFinite(raw) ? raw : v;
        ends.push({ color, px: x(chartData[i].epoch), py: y(v), ly: y(v),
                    text: isAccuracy ? `${shown.toFixed(1)}%` : shown.toPrecision(4) });
        break;
      }
    });
    ends.sort((a, b) => a.ly - b.ly);
    for (let k = 1; k < ends.length; k++) {
      ends[k].ly = Math.max(ends[k].ly, ends[k - 1].ly + 15);
    }
    const overflow = ends.length ? ends[ends.length - 1].ly - h : 0;
    if (overflow > 0) ends.forEach(e => { e.ly -= overflow; });
    // A leader in the series colour ties a displaced label back to its curve.
    ends.forEach(e => {
      g.append("path").attr("fill", "none").attr("stroke", e.color).attr("stroke-width", 1)
        .attr("d", `M${e.px + 4},${e.py} L${e.px + 10},${e.ly} L${e.px + 13},${e.ly}`);
      g.append("circle").attr("cx", e.px).attr("cy", e.py).attr("r", 3.5)
        .attr("fill", e.color);
      g.append("text").attr("class", "end-label")
        .attr("x", e.px + 16).attr("y", e.ly).text(e.text);
    });

    // ── Tooltip ──────────────────────────────────────────────
    const tooltipClass = isAccuracy ? "accuracy" : "loss";
    let tooltip = container.querySelector(".chart-tooltip");
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.className = `chart-tooltip ${tooltipClass}`;
      tooltip.style.display = "none";
      container.appendChild(tooltip);
    }

    const bisect = d3.bisector(d => d.epoch).left;

    const overlay = g.append("rect")
      .attr("width", w)
      .attr("height", h)
      .attr("fill", "none")
      .attr("pointer-events", "all");

    // Active dots
    const activeDots = [];
    const surfaceRing = cssVar("--bg-inset", "#070a0f");
    function addActiveDot(color) {
      const dot = g.append("circle")
        .attr("r", 5)
        .attr("fill", color)
        .attr("stroke", surfaceRing)
        .attr("stroke-width", 2)
        .style("display", "none");
      activeDots.push(dot);
      return dot;
    }

    const trainDot = visible.training   ? addActiveDot(trainColor) : null;
    const testDot  = visible.testing    ? addActiveDot(testColor)  : null;
    const valDot   = visible.validation ? addActiveDot(valColor)   : null;

    overlay
      .on("mousemove", function(event) {
        const [mx] = d3.pointer(event);
        const epoch = x.invert(mx);
        const idx = Math.min(bisect(chartData, epoch), chartData.length - 1);
        const d = chartData[idx];
        if (!d) return;

        let lines = [`<b>Epoch: ${d.epoch}</b>`];
        if (visible.training && d[trainKey] != null) {
          const v = isAccuracy ? `${d[trainKey].toFixed(2)}%` : d[trainKey].toFixed(6);
          lines.push(`Training: ${v}`);
          trainDot.attr("cx", x(d.epoch)).attr("cy", y(d[trainKey])).style("display", null);
        }
        if (visible.testing && d[testKey] != null) {
          const v = isAccuracy ? `${d[testKey].toFixed(2)}%` : d[testKey].toFixed(6);
          lines.push(`Testing: ${v}`);
          testDot.attr("cx", x(d.epoch)).attr("cy", y(d[testKey])).style("display", null);
        }
        if (visible.validation && d[valKey] != null) {
          const v = isAccuracy ? `${d[valKey].toFixed(2)}%` : d[valKey].toFixed(6);
          lines.push(`Validation: ${v}`);
          valDot.attr("cx", x(d.epoch)).attr("cy", y(d[valKey])).style("display", null);
        }

        tooltip.innerHTML = lines.join("<br>");
        tooltip.style.display = "block";

        // Position tooltip near cursor
        const tipW = tooltip.offsetWidth;
        const tipH = tooltip.offsetHeight;
        let tx = mx + M.left + 12;
        let ty = event.offsetY - tipH / 2;
        if (tx + tipW > W) tx = mx + M.left - tipW - 12;
        if (ty < 0) ty = 4;
        if (ty + tipH > H) ty = H - tipH - 4;
        tooltip.style.left = tx + "px";
        tooltip.style.top = ty + "px";
      })
      .on("mouseleave", function() {
        tooltip.style.display = "none";
        activeDots.forEach(d => d.style("display", "none"));
      });
  }

  // ── ResizeObserver wrapper ─────────────────────────────────
  function observeResize(container, renderFn) {
    const ro = new ResizeObserver(() => renderFn());
    ro.observe(container);
    return ro;
  }

  return { createChart, createDistributionChart, smoothSeries, downsample,
           positiveFloor, observeResize, SERIES };
})();
