/*
 * schedule_indexing — tensor index arithmetic for the lowering pipeline.
 *
 * When a kernel accesses a tensor whose ShapeTracker has multiple views or
 * a non-trivial stride/mask, the codegen must emit correct index arithmetic.
 * This module computes IndexMap objects: for each loop variable (one per
 * output dimension) it produces a SymExpr giving the flat storage offset,
 * and optionally a boolean validity guard (for padded / masked tensors).
 */

#ifndef CML_OPS_IR_SCHEDULE_INDEXING_H
#define CML_OPS_IR_SCHEDULE_INDEXING_H

#include "symbolic/symbolic.h"
#include "tensor/shape_tracker.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * IndexMap — result of resolving a ShapeTracker through a set of loop vars
 * ------------------------------------------------------------------------- */

typedef struct IndexMap {
    SymExpr* flat_index;  /* expression: maps loop vars → flat storage index */
    SymExpr* valid;       /* boolean guard, or NULL when always valid */
    int      num_vars;    /* number of loop variables this was built from */
} IndexMap;

IndexMap* index_map_create(SymExpr* flat_index, SymExpr* valid, int num_vars);
void      index_map_free(IndexMap* im);
IndexMap* index_map_copy(const IndexMap* im);

/* -------------------------------------------------------------------------
 * LoopVar — symbolic loop variable with range [begin, end)
 * ------------------------------------------------------------------------- */

typedef struct LoopVar {
    SymExpr* expr;       /* the variable itself (a SYM_VAR node) */
    int64_t  begin;
    int64_t  end;
} LoopVar;

/* Create n loop variables named "idx0", "idx1", …, "idx(n-1)"
 * with ranges given by shape[0..n-1].  Returns array of n LoopVars;
 * caller must call loop_vars_free(). */
LoopVar* loop_vars_create(const int* shape, int n);
void     loop_vars_free(LoopVar* vars, int n);

/* -------------------------------------------------------------------------
 * Index map construction
 * ------------------------------------------------------------------------- */

/*
 * Build an IndexMap for tensor t's ShapeTracker, evaluated with the given
 * loop variables.  loop_vars must have one entry per dim of the logical shape.
 *
 * Returns NULL on error.
 */
IndexMap* schedule_build_index_map(const ShapeTracker* st,
                                   const LoopVar* loop_vars,
                                   int num_vars);

/*
 * Same, but also inlines constant folding and range simplification
 * using symbolic evaluation bounds.  Typically used just before codegen.
 */
IndexMap* schedule_build_index_map_simplified(const ShapeTracker* st,
                                              const LoopVar* loop_vars,
                                              int num_vars);

/* -------------------------------------------------------------------------
 * Index expression → C / kernel code string
 * ------------------------------------------------------------------------- */

/*
 * Render an IndexMap to a C expression string.
 * var_names[i] is the C variable name for loop_vars[i].
 * valid_expr_buf: receives the validity guard expression (or "1" if no mask).
 * Returns 0 on success.
 */
int index_map_to_c(const IndexMap* im,
                   const char* const* var_names, int num_vars,
                   char* index_buf, size_t index_buf_size,
                   char* valid_buf, size_t valid_buf_size);

/* -------------------------------------------------------------------------
 * Utilities
 * ------------------------------------------------------------------------- */

/* Compose two IndexMaps: apply outer(inner(x)). */
IndexMap* index_map_compose(const IndexMap* outer, const IndexMap* inner);

/* Debug dump to stderr. */
void index_map_print(const IndexMap* im);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_SCHEDULE_INDEXING_H */
