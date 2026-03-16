/* viz-charts.js — D3-based chart rendering for C-ML Training View */
/* global d3 */

"use strict";

const VizCharts = (() => {

  // ── Shared config ──────────────────────────────────────────
  const MARGIN = { top: 10, right: 15, left: 50, bottom: 30 };
  const COLORS = {
    training:   "#4a90e2",
    testing:    "#f59e0b",
    validation: "#ef4444",
    trainAcc:   "#10b981",
    testAcc:    "#f59e0b",
    valAcc:     "#ef4444",
    grid:       "#374151",
    axis:       "#6b7280",
  };

  // ── createChart (loss or accuracy) ─────────────────────────
  function createChart(container, chartData, opts) {
    const { type, visible } = opts; // type: "loss" | "accuracy"
    const isAccuracy = type === "accuracy";

    // Clear previous
    container.innerHTML = "";

    const rect = container.getBoundingClientRect();
    const W = rect.width || 400;
    const H = rect.height || 280;
    const w = W - MARGIN.left - MARGIN.right;
    const h = H - MARGIN.top - MARGIN.bottom;

    if (w <= 0 || h <= 0) return;

    const svg = d3.select(container)
      .append("svg")
      .attr("width", W)
      .attr("height", H);

    const g = svg.append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

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

    const y = d3.scaleLinear()
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

    const yAxis = isAccuracy
      ? d3.axisLeft(y).ticks(6).tickFormat(v => `${v.toFixed(1)}%`)
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

    // Helper to draw a series
    function drawSeries(key, color, isDashed, showDots) {
      const vals = chartData.map(d => d[key]);
      const defined = vals.map((v, i) => ({ v, i })).filter(d => d.v != null);
      if (defined.length === 0) return;

      const path = g.append("path")
        .datum(vals)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 2.5)
        .attr("d", line);

      if (isDashed) path.attr("stroke-dasharray", "4 4");

      if (showDots) {
        g.selectAll(null)
          .data(defined)
          .enter()
          .append("circle")
          .attr("cx", d => x(chartData[d.i].epoch))
          .attr("cy", d => y(d.v))
          .attr("r", 5)
          .attr("fill", color)
          .attr("stroke", "#111827")
          .attr("stroke-width", 1);
      }
    }

    // Draw visible series
    const trainColor = isAccuracy ? COLORS.trainAcc : COLORS.training;
    const testColor  = isAccuracy ? COLORS.testAcc  : COLORS.testing;
    const valColor   = isAccuracy ? COLORS.valAcc   : COLORS.validation;

    if (visible.training)   drawSeries(trainKey, trainColor, false, false);
    if (visible.testing)    drawSeries(testKey,  testColor,  true,  true);
    if (visible.validation) drawSeries(valKey,   valColor,   false, false);

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
    function addActiveDot(color) {
      const dot = g.append("circle")
        .attr("r", 5)
        .attr("fill", color)
        .attr("stroke", "white")
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
        let tx = mx + MARGIN.left + 12;
        let ty = event.offsetY - tipH / 2;
        if (tx + tipW > W) tx = mx + MARGIN.left - tipW - 12;
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

  return { createChart, observeResize };
})();
