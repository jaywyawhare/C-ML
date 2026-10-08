#ifndef CML_OPS_IR_REWRITE_TRACE_H
#define CML_OPS_IR_REWRITE_TRACE_H

/* Per-match record of the IR graph-rewriter, for the dashboard's rewrite
 * stepper. Each successful rule application appends one event (which rule
 * fired, on which op, what it produced, how long it took). Off by default and
 * cheap when off; enabled by VIZ (or REWRITE_TRACE) so normal runs pay nothing.
 * The matcher records; the compiler exports the collected events to JSON. */

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Whether match recording is active (VIZ or REWRITE_TRACE set, export allowed). */
bool cml_rewrite_trace_enabled(void);

/** Start a named rewrite pass; subsequent records are grouped under it. */
void cml_rewrite_trace_begin_pass(const char* pass_name);

/** Record one successful rule application: the rule's name, the matched node's
 *  op string, the output names before and after, and the emit duration (us). */
void cml_rewrite_trace_record(const char* rule, const char* op, const char* from, const char* to,
                              double dur_us);

/** Drop all recorded events and passes (call before a fresh compile). */
void cml_rewrite_trace_reset(void);

/** Serialize the recorded passes/events to a JSON string (caller frees), or
 *  NULL if nothing was recorded. Schema: {"passes":[{"name","matches":[...] }]}. */
char* cml_rewrite_trace_export_json(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_REWRITE_TRACE_H */
