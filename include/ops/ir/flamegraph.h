#ifndef CML_OPS_IR_FLAMEGRAPH_H
#define CML_OPS_IR_FLAMEGRAPH_H

/* Execution flamegraph: per-fused-kernel timing of a graph execution.
 *
 * Each executed IR node (a fused chain is already a single node) is timed and
 * recorded as one span carrying its op, a coarse "kind" (matmul/conv/elemwise/
 * reduce/fused/...), the fwd-vs-bwd phase, and the output work-size. The viz
 * dashboard folds the flat span list into a phase -> kind -> kernel tree and
 * renders it as a flamegraph (width = time), which makes it visible at a glance
 * whether fusion actually collapsed a hot chain into one wide bar.
 *
 * Opt-in via the FLAMEGRAPH env var (truthy), or implied by VIZ / the PROFILE
 * flag, so there is ZERO timing overhead when none of them is set. When on,
 * span capture wraps the per-node dispatch in the CPU execution loop at
 * fused-kernel granularity. */

#include "ops/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

struct IRNode;

/* Cached check of the FLAMEGRAPH env var (truthy and not "0"/"false"). */
int cml_flame_enabled(void);

/* Clear the accumulated spans (call at the start of a step you want to profile). */
void cml_flame_reset(void);

/* Append one kernel span. `ms` is the measured wall time for this node. No-op
 * when disabled. Op/kind strings are static; `node` is only read, not retained. */
void cml_flame_record(const struct IRNode* node, double ms);

/* Monotonic millisecond clock used to bracket a span (CLOCK_MONOTONIC). */
double cml_flame_now_ms(void);

/* Write the accumulated spans to `path` as flamegraph.json. Returns 0 on success,
 * -1 on error or when disabled / no spans. Does not clear the buffer. */
int cml_flame_export(const char* path);

/* Number of spans currently buffered. */
int cml_flame_num_spans(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_FLAMEGRAPH_H */
