/* Shell router: top-level views map to two isolated iframes.
 *   experiments -> exp.html   (keeps its own second-level sub-tabs)
 *   graph/training/kernels -> viz.html  (one frame, switched via postMessage)
 * Frames are lazy-loaded on first activation. Deep-linkable via the URL hash. */
(function () {
  "use strict";

  // top-level view -> which iframe + (for viz) the app's internal tab id
  var VIEWS = {
    experiments: { frame: "experiments", src: "exp.html?embed=1" },
    graph:       { frame: "viz", src: "viz.html?embed=1", vizTab: "graph" },
    training:    { frame: "viz", src: "viz.html?embed=1", vizTab: "training" },
    kernels:     { frame: "viz", src: "viz.html?embed=1", vizTab: "codegen" },
  };
  var ORDER = ["experiments", "graph", "training", "kernels"];

  var frameExp = document.getElementById("frame-experiments");
  var frameViz = document.getElementById("frame-viz");
  var navItems = Array.prototype.slice.call(document.querySelectorAll(".shell-nav-item"));
  var loaded = { experiments: false, viz: false };
  var current = null;

  function ensureLoaded(which, vizTab) {
    if (loaded[which]) return;
    if (which === "experiments") { frameExp.src = VIEWS.experiments.src; }
    // Carry the target tab in the initial hash so the viz app boots straight
    // into it (no Graph→target flash); the ready-handshake re-applies it too.
    else if (which === "viz")    { frameViz.src = VIEWS.graph.src + "#" + (vizTab || "graph"); }
    loaded[which] = true;
  }

  function postVizTab(tab) {
    try { frameViz.contentWindow.postMessage({ type: "cml-view", tab: tab }, "*"); }
    catch (e) { /* frame not ready yet; viz reads the hash on load instead */ }
  }

  function show(view) {
    if (!VIEWS[view]) view = "experiments";
    current = view;
    var v = VIEWS[view];

    navItems.forEach(function (b) { b.classList.toggle("active", b.dataset.view === view); });

    ensureLoaded(v.frame === "viz" ? "viz" : "experiments", v.vizTab);
    frameExp.classList.toggle("active", v.frame === "experiments");
    frameViz.classList.toggle("active", v.frame === "viz");

    if (v.frame === "viz") {
      // pass the target internal tab via the hash (read on first load) AND postMessage
      if (frameViz.contentWindow && frameViz.contentWindow.location) {
        try { frameViz.contentWindow.location.hash = v.vizTab; } catch (e) {}
      }
      postVizTab(v.vizTab);
    }
    if (location.hash.slice(1) !== view) history.replaceState(null, "", "#" + view);
    document.title = "C-ML · " + view.charAt(0).toUpperCase() + view.slice(1);
  }

  navItems.forEach(function (b) {
    b.addEventListener("click", function () { show(b.dataset.view); });
  });

  // keyboard 1..4 (ignored while typing inside a frame is handled by each app)
  document.addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
    var i = parseInt(e.key, 10);
    if (i >= 1 && i <= ORDER.length) show(ORDER[i - 1]);
  });

  window.addEventListener("hashchange", function () {
    var v = location.hash.slice(1);
    if (v && v !== current) show(v);
  });

  // viz frame announces when it's ready so we can (re)apply the pending tab
  window.addEventListener("message", function (e) {
    if (e && e.data && e.data.type === "cml-viz-ready" && current && VIEWS[current].vizTab) {
      postVizTab(VIEWS[current].vizTab);
    }
  });

  // server heartbeat -> connection pill
  var pill = document.getElementById("shell-conn");
  var label = pill.querySelector(".shell-conn-label");
  function ping() {
    fetch("/status", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
      .then(function () { pill.classList.remove("down"); pill.classList.add("live"); label.textContent = "live"; })
      .catch(function () { pill.classList.remove("live"); pill.classList.add("down"); label.textContent = "offline"; });
  }
  ping(); setInterval(ping, 4000);

  // initial view from hash, default experiments
  show(location.hash.slice(1) || "experiments");
})();
