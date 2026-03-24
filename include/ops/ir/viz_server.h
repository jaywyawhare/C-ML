/*
 * viz_server — embedded HTTP server for interactive IR graph visualization.
 *
 * Starts a lightweight single-threaded HTTP server that serves an HTML/JS
 * page rendering the computation graph.  The graph JSON is regenerated on
 * each page load, so re-loading the browser reflects the latest IR state.
 *
 * Usage:
 *   VizServer* srv = viz_server_start(ir, 8080);
 *   // ... browser opens http://localhost:8080 ...
 *   viz_server_update(srv, new_ir);   // hot-swap graph
 *   viz_server_stop(srv);
 */

#ifndef CML_OPS_IR_VIZ_SERVER_H
#define CML_OPS_IR_VIZ_SERVER_H

#include "ops/ir/ir.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VizServer VizServer;

/* ---- Server lifecycle ---- */

/*
 * Start the server on the given TCP port (0 = pick a free port).
 * ir may be NULL; call viz_server_update() later to set the graph.
 * Returns NULL on failure (port in use, etc.).
 */
VizServer* viz_server_start(CMLGraph_t ir, int port);

/*
 * Replace the graph shown by the server.  Thread-safe; takes a snapshot
 * of the graph state at call time.
 */
void viz_server_update(VizServer* srv, CMLGraph_t ir);

/*
 * Poll the server — accept and handle any pending HTTP connections.
 * Call this from your main loop (non-blocking; returns immediately if idle).
 * Returns number of requests handled.
 */
int viz_server_poll(VizServer* srv);

/*
 * Stop the server and free all resources.
 */
void viz_server_stop(VizServer* srv);

/* Actual port the server is listening on (useful when port=0 was requested). */
int viz_server_port(const VizServer* srv);

/* ---- Serialization helpers (also used by AOT export) ---- */

/*
 * Serialize the IR graph to a JSON string.
 * Caller must free() the returned string.
 * Returns NULL on failure.
 *
 * Format:
 * {
 *   "nodes": [ { "id":0, "op":"ADD", "shape":[2,3], "dtype":"float32" }, ... ],
 *   "edges": [ { "src":0, "dst":1, "slot":0 }, ... ],
 *   "schedule": [ { "kernel":0, "nodes":[0,1] }, ... ]   // optional
 * }
 */
char* viz_graph_to_json(CMLGraph_t ir);

/*
 * Write an HTML file containing the embedded graph viewer and JSON data.
 * Useful for offline inspection without a running server.
 * Returns 0 on success.
 */
int viz_export_html(CMLGraph_t ir, const char* path);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_VIZ_SERVER_H */
