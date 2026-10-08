/* Theme token access for code that draws on canvas or into SVG attributes,
 * neither of which can read CSS variables. Shared by the Visualizer and the
 * Experiment tracker so both resolve colours from theme.css the same way. */
"use strict";

/** Current value of a theme token, e.g. tok("--accent"). Read from <body>
 *  because the Experiment tracker sets data-theme there. */
function tok(name) {
  return getComputedStyle(document.body || document.documentElement).getPropertyValue(name).trim();
}

/** Replace every "var(--x)" inside a string, array or plain object with its
 *  resolved value. */
function resolveTokens(value) {
  if (typeof value === "string") return value.replace(/var\((--[\w-]+)\)/g, (m, n) => tok(n) || m);
  if (Array.isArray(value)) return value.map(resolveTokens);
  if (value && typeof value === "object") {
    const out = {};
    for (const k in value) out[k] = resolveTokens(value[k]);
    return out;
  }
  return value;
}

/** Categorical colour for the i-th series. Slots are used in fixed order and
 *  never cycled: past the eighth, series share a neutral and rely on labels. */
function seriesColor(i) {
  return i < 8 ? tok("--cat-" + (i + 1)) : tok("--muted-2");
}
