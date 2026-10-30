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

/** Categorical colour for the i-th series. The first eight are the validated
 *  (CVD/contrast-checked) slots in fixed order; past the eighth, hues are
 *  synthesized by golden-angle rotation so large run comparisons stay
 *  distinguishable instead of collapsing to one neutral. Lightness tracks the
 *  theme so synthesized hues keep contrast on both surfaces. */
function seriesColor(i) {
  if (i < 8) return tok("--cat-" + (i + 1));
  const hue = ((i - 8) * 137.508) % 360;
  const light = (document.body || document.documentElement).getAttribute("data-theme") === "light";
  return `hsl(${hue.toFixed(1)}, 60%, ${light ? "40%" : "64%"})`;
}

/** Shared light/dark state for the shell and both app frames. Persisted to
 *  localStorage; the `storage` event fires in every other same-origin frame, so
 *  a change in one is picked up by the rest with no messaging. */
function cmlTheme() {
  try { return localStorage.getItem("cml-theme") === "light" ? "light" : "dark"; } catch { return "dark"; }
}
/** Apply a theme to this document (defaults to the stored one). */
function applyTheme(t) { (document.body || document.documentElement).setAttribute("data-theme", t || cmlTheme()); }
/** Persist and apply a theme; other frames react via their storage listener. */
function setTheme(t) { try { localStorage.setItem("cml-theme", t); } catch (e) { /* private mode */ } applyTheme(t); }
/** Run fn(theme) when another frame changes the shared theme. */
function onThemeChange(fn) { window.addEventListener("storage", e => { if (e.key === "cml-theme") fn(cmlTheme()); }); }
