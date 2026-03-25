#include "ops/ir/schedule_indexing.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

IndexMap* index_map_create(SymExpr* flat_index, SymExpr* valid, int num_vars) {
    if (!flat_index) return NULL;
    IndexMap* im = calloc(1, sizeof(IndexMap));
    if (!im) return NULL;
    sym_expr_retain(flat_index);
    im->flat_index = flat_index;
    if (valid) {
        sym_expr_retain(valid);
        im->valid = valid;
    }
    im->num_vars = num_vars;
    return im;
}

void index_map_free(IndexMap* im) {
    if (!im) return;
    sym_expr_release(im->flat_index);
    if (im->valid) sym_expr_release(im->valid);
    free(im);
}

IndexMap* index_map_copy(const IndexMap* im) {
    if (!im) return NULL;
    return index_map_create(im->flat_index, im->valid, im->num_vars);
}

LoopVar* loop_vars_create(const int* shape, int n) {
    if (!shape || n <= 0) return NULL;
    LoopVar* vars = malloc((size_t)n * sizeof(LoopVar));
    if (!vars) return NULL;
    for (int i = 0; i < n; ++i) {
        char name[16];
        snprintf(name, sizeof(name), "idx%d", i);
        vars[i].expr  = sym_var(name, 0, (int64_t)shape[i] - 1);
        vars[i].begin = 0;
        vars[i].end   = shape[i];
        if (!vars[i].expr) {
            for (int j = 0; j < i; ++j) sym_expr_release(vars[j].expr);
            free(vars);
            return NULL;
        }
    }
    return vars;
}

void loop_vars_free(LoopVar* vars, int n) {
    if (!vars) return;
    for (int i = 0; i < n; ++i)
        sym_expr_release(vars[i].expr);
    free(vars);
}

static SymExpr* view_flat_index(const STView* v, const LoopVar* vars, int nvars) {
    SymExpr* acc = sym_const((int64_t)v->offset);
    for (int i = 0; i < v->ndim && i < nvars; ++i) {
        if (v->strides[i] == 0) continue;
        SymExpr* stride = sym_const(v->strides[i]);
        SymExpr* term   = sym_mul(vars[i].expr, stride);
        sym_expr_release(stride);
        SymExpr* new_acc = sym_add(acc, term);
        sym_expr_release(acc);
        sym_expr_release(term);
        acc = new_acc;
    }
    return acc;
}

static SymExpr* view_valid_expr(const STView* v, const LoopVar* vars, int nvars) {
    if (!v->has_mask) return NULL;
    SymExpr* acc = NULL;
    for (int i = 0; i < v->ndim && i < nvars; ++i) {
        if (v->mask_begin[i] == 0 && v->mask_end[i] == (int64_t)v->shape[i])
            continue;
        
        SymExpr* cond = NULL;
        if (v->mask_begin[i] > 0) {
            SymExpr* begin = sym_const(v->mask_begin[i]);
            
            SymExpr* diff  = sym_add(vars[i].expr, sym_mul(sym_const(-1), begin));
            sym_expr_release(begin);
            
            cond = sym_max_expr(diff, sym_const(0));
            sym_expr_release(diff);
        }
        if (v->mask_end[i] < (int64_t)v->shape[i]) {
            SymExpr* end  = sym_const(v->mask_end[i]);
            SymExpr* diff = sym_add(end, sym_mul(sym_const(-1), vars[i].expr));
            sym_expr_release(end);
            SymExpr* guard = sym_max_expr(diff, sym_const(0));
            sym_expr_release(diff);
            if (cond) {
                SymExpr* both = sym_min_expr(cond, guard);
                sym_expr_release(cond);
                sym_expr_release(guard);
                cond = both;
            } else {
                cond = guard;
            }
        }
        if (!cond) continue;
        if (!acc) {
            acc = cond;
        } else {
            SymExpr* both = sym_min_expr(acc, cond);
            sym_expr_release(acc);
            sym_expr_release(cond);
            acc = both;
        }
    }
    return acc;
}

IndexMap* schedule_build_index_map(const ShapeTracker* st,
                                   const LoopVar* loop_vars,
                                   int num_vars) {
    if (!st || !loop_vars || num_vars <= 0) return NULL;
    const STView* v = st->views[st->num_views - 1];
    if (!v) return NULL;

    SymExpr* flat  = view_flat_index(v, loop_vars, num_vars);
    SymExpr* valid = view_valid_expr(v, loop_vars, num_vars);
    IndexMap* im   = index_map_create(flat, valid, num_vars);
    sym_expr_release(flat);
    if (valid) sym_expr_release(valid);
    return im;
}

IndexMap* schedule_build_index_map_simplified(const ShapeTracker* st,
                                               const LoopVar* loop_vars,
                                               int num_vars) {
    IndexMap* im = schedule_build_index_map(st, loop_vars, num_vars);
    if (!im) return NULL;
    SymExpr* sf = sym_simplify(im->flat_index);
    sym_expr_release(im->flat_index);
    im->flat_index = sf;
    if (im->valid) {
        SymExpr* sv = sym_simplify(im->valid);
        sym_expr_release(im->valid);
        im->valid = sv;
    }
    return im;
}

int index_map_to_c(const IndexMap* im,
                   const char* const* var_names, int num_vars,
                   char* index_buf, size_t index_buf_size,
                   char* valid_buf, size_t valid_buf_size) {
    if (!im || !var_names || !index_buf || index_buf_size == 0) return -1;
    (void)num_vars;

    
    int r = sym_expr_to_string(im->flat_index, index_buf, (int)index_buf_size);
    if (r < 0) return -1;

    if (valid_buf && valid_buf_size > 0) {
        if (!im->valid) {
            snprintf(valid_buf, valid_buf_size, "1");
        } else {
            r = sym_expr_to_string(im->valid, valid_buf, (int)valid_buf_size);
            if (r < 0) return -1;
        }
    }
    return 0;
}

IndexMap* index_map_compose(const IndexMap* outer, const IndexMap* inner) {
    if (!outer || !inner) return NULL;
    
    SymExpr* composed = sym_add(outer->flat_index, inner->flat_index);
    SymExpr* valid = NULL;
    if (outer->valid && inner->valid)
        valid = sym_min_expr(outer->valid, inner->valid);
    else if (outer->valid)
        valid = outer->valid;
    else if (inner->valid)
        valid = inner->valid;
    if (valid) sym_expr_retain(valid);
    IndexMap* im = index_map_create(composed, valid, inner->num_vars);
    sym_expr_release(composed);
    if (valid) sym_expr_release(valid);
    return im;
}

void index_map_print(const IndexMap* im) {
    if (!im) { fprintf(stderr, "IndexMap(NULL)\n"); return; }
    char buf[512] = {0};
    sym_expr_to_string(im->flat_index, buf, sizeof(buf));
    fprintf(stderr, "IndexMap: flat=%s", buf);
    if (im->valid) {
        char vbuf[512] = {0};
        sym_expr_to_string(im->valid, vbuf, sizeof(vbuf));
        fprintf(stderr, " valid=%s", vbuf);
    }
    fprintf(stderr, " vars=%d\n", im->num_vars);
}
