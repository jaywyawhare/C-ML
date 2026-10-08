#include "symbolic/symbolic.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <stdatomic.h>

static _Atomic int g_var_id_counter = 0;

/** Allocate a zeroed expression node of `type` with its reference count set to 1. */
static SymExpr* sym_alloc(SymExprType type) {
    SymExpr* e = (SymExpr*)calloc(1, sizeof(SymExpr));
    if (!e)
        return NULL;
    e->type      = type;
    e->ref_count = 1;
    return e;
}

/** Signed 64-bit minimum. */
static int64_t i64_min(int64_t a, int64_t b) { return a < b ? a : b; }
/** Signed 64-bit maximum. */
static int64_t i64_max(int64_t a, int64_t b) { return a > b ? a : b; }

/**
 * Fold a binary op over two concrete operands into `out`. Returns 1 on success and
 * 0 for division/modulo by zero or an op type that cannot be folded here.
 */
static int sym_fold_binop(SymExprType type, int64_t left, int64_t right, int64_t* out) {
    if (!out)
        return 0;
    switch (type) {
    case SYM_ADD:
        *out = left + right;
        return 1;
    case SYM_MUL:
        *out = left * right;
        return 1;
    case SYM_DIV:
        if (right == 0)
            return 0;
        *out = left / right;
        return 1;
    case SYM_MOD:
        if (right == 0)
            return 0;
        *out = left % right;
        return 1;
    case SYM_MIN:
        *out = i64_min(left, right);
        return 1;
    case SYM_MAX:
        *out = i64_max(left, right);
        return 1;
    default:
        return 0;
    }
}

/** Build a constant-valued expression node. */
SymExpr* sym_const(int64_t value) {
    SymExpr* e = sym_alloc(SYM_CONST);
    if (e)
        e->const_val = value;
    return e;
}

/** Build a named variable bounded by [vmin, vmax], tagged with a process-unique id. */
SymExpr* sym_var(const char* name, int64_t vmin, int64_t vmax) {
    if (!name)
        return NULL;
    SymExpr* e = sym_alloc(SYM_VAR);
    if (!e)
        return NULL;
    strncpy(e->var.name, name, sizeof(e->var.name) - 1);
    e->var.name[sizeof(e->var.name) - 1] = '\0';
    e->var.vmin                          = vmin;
    e->var.vmax                          = vmax;
    e->var.id                            = atomic_fetch_add(&g_var_id_counter, 1);
    return e;
}

/**
 * Build a binary-op node over `a` and `b`, constant-folding when both are constants.
 * Returns NULL on a fold that is undefined (e.g. divide by zero) or on allocation
 * failure; otherwise retains both operands.
 */
static SymExpr* sym_binop(SymExprType type, SymExpr* a, SymExpr* b) {
    if (!a || !b)
        return NULL;

    if (a->type == SYM_CONST && b->type == SYM_CONST) {
        int64_t result;
        if (sym_fold_binop(type, a->const_val, b->const_val, &result))
            return sym_const(result);
        return NULL;
    }

    SymExpr* e = sym_alloc(type);
    if (!e)
        return NULL;
    sym_expr_retain(a);
    sym_expr_retain(b);
    e->binop.left  = a;
    e->binop.right = b;
    return e;
}

/** Symbolic addition a + b. */
SymExpr* sym_add(SymExpr* a, SymExpr* b) { return sym_binop(SYM_ADD, a, b); }
/** Symbolic multiplication a * b. */
SymExpr* sym_mul(SymExpr* a, SymExpr* b) { return sym_binop(SYM_MUL, a, b); }
/** Symbolic truncating division a / b. */
SymExpr* sym_div(SymExpr* a, SymExpr* b) { return sym_binop(SYM_DIV, a, b); }
/** Symbolic remainder a % b. */
SymExpr* sym_mod(SymExpr* a, SymExpr* b) { return sym_binop(SYM_MOD, a, b); }
/** Symbolic minimum min(a, b). */
SymExpr* sym_min_expr(SymExpr* a, SymExpr* b) { return sym_binop(SYM_MIN, a, b); }
/** Symbolic maximum max(a, b). */
SymExpr* sym_max_expr(SymExpr* a, SymExpr* b) { return sym_binop(SYM_MAX, a, b); }

/**
 * Lower bound of an expression over its variables' ranges via interval arithmetic.
 * MUL/DIV test all four endpoint combinations to stay correct across negative ranges;
 * a DIV whose divisor range spans zero returns INT64_MIN as a conservative bound.
 */
int64_t sym_expr_min(const SymExpr* e) {
    if (!e)
        return 0;
    switch (e->type) {
    case SYM_CONST:
        return e->const_val;
    case SYM_VAR:
        return e->var.vmin;
    case SYM_ADD:
        return sym_expr_min(e->binop.left) + sym_expr_min(e->binop.right);
    case SYM_MUL: {
        // For MUL, consider all 4 combinations (handles negative ranges)
        int64_t a_min = sym_expr_min(e->binop.left);
        int64_t a_max = sym_expr_max(e->binop.left);
        int64_t b_min = sym_expr_min(e->binop.right);
        int64_t b_max = sym_expr_max(e->binop.right);
        int64_t p1 = a_min * b_min, p2 = a_min * b_max;
        int64_t p3 = a_max * b_min, p4 = a_max * b_max;
        return i64_min(i64_min(p1, p2), i64_min(p3, p4));
    }
    case SYM_DIV: {
        int64_t b_min = sym_expr_min(e->binop.right);
        int64_t b_max = sym_expr_max(e->binop.right);
        int64_t a_min = sym_expr_min(e->binop.left);
        int64_t a_max = sym_expr_max(e->binop.left);
        // Avoid division by zero in bounds; if range includes 0, conservative
        if (b_min <= 0 && b_max >= 0)
            return INT64_MIN;
        int64_t p1 = a_min / b_min, p2 = a_min / b_max;
        int64_t p3 = a_max / b_min, p4 = a_max / b_max;
        return i64_min(i64_min(p1, p2), i64_min(p3, p4));
    }
    case SYM_MOD: {
        int64_t b_max = sym_expr_max(e->binop.right);
        if (b_max <= 0)
            return 0;
        // Mod result is in [0, |b|-1] if a >= 0, else [-(|b|-1), 0]
        int64_t a_min = sym_expr_min(e->binop.left);
        if (a_min >= 0)
            return 0;
        return -(b_max - 1);
    }
    case SYM_MIN:
        return i64_min(sym_expr_min(e->binop.left), sym_expr_min(e->binop.right));
    case SYM_MAX:
        return i64_max(sym_expr_min(e->binop.left), sym_expr_min(e->binop.right));
    }
    return 0;
}

/**
 * Upper bound of an expression over its variables' ranges via interval arithmetic.
 * MUL/DIV test all four endpoint combinations to stay correct across negative ranges;
 * a DIV whose divisor range spans zero returns INT64_MAX as a conservative bound.
 */
int64_t sym_expr_max(const SymExpr* e) {
    if (!e)
        return 0;
    switch (e->type) {
    case SYM_CONST:
        return e->const_val;
    case SYM_VAR:
        return e->var.vmax;
    case SYM_ADD:
        return sym_expr_max(e->binop.left) + sym_expr_max(e->binop.right);
    case SYM_MUL: {
        int64_t a_min = sym_expr_min(e->binop.left);
        int64_t a_max = sym_expr_max(e->binop.left);
        int64_t b_min = sym_expr_min(e->binop.right);
        int64_t b_max = sym_expr_max(e->binop.right);
        int64_t p1 = a_min * b_min, p2 = a_min * b_max;
        int64_t p3 = a_max * b_min, p4 = a_max * b_max;
        return i64_max(i64_max(p1, p2), i64_max(p3, p4));
    }
    case SYM_DIV: {
        int64_t b_min = sym_expr_min(e->binop.right);
        int64_t b_max = sym_expr_max(e->binop.right);
        int64_t a_min = sym_expr_min(e->binop.left);
        int64_t a_max = sym_expr_max(e->binop.left);
        if (b_min <= 0 && b_max >= 0)
            return INT64_MAX;
        int64_t p1 = a_min / b_min, p2 = a_min / b_max;
        int64_t p3 = a_max / b_min, p4 = a_max / b_max;
        return i64_max(i64_max(p1, p2), i64_max(p3, p4));
    }
    case SYM_MOD: {
        int64_t b_max = sym_expr_max(e->binop.right);
        if (b_max <= 0)
            return 0;
        int64_t a_max_val = sym_expr_max(e->binop.left);
        if (a_max_val >= 0)
            return b_max - 1;
        return 0;
    }
    case SYM_MIN:
        return i64_min(sym_expr_max(e->binop.left), sym_expr_max(e->binop.right));
    case SYM_MAX:
        return i64_max(sym_expr_max(e->binop.left), sym_expr_max(e->binop.right));
    }
    return 0;
}

/**
 * Evaluate an expression to a concrete value given a variable name/value table.
 * Returns 0 on success, or -1 on an unbound variable or an undefined fold (e.g.
 * divide by zero).
 */
int sym_eval(const SymExpr* e, const char** var_names, const int64_t* values, int num_vars,
             int64_t* out) {
    if (!e || !out)
        return -1;
    switch (e->type) {
    case SYM_CONST:
        *out = e->const_val;
        return 0;
    case SYM_VAR:
        for (int i = 0; i < num_vars; i++) {
            if (var_names[i] && strcmp(var_names[i], e->var.name) == 0) {
                *out = values[i];
                return 0;
            }
        }
        return -1;
    case SYM_ADD:
    case SYM_MUL:
    case SYM_DIV:
    case SYM_MOD:
    case SYM_MIN:
    case SYM_MAX: {
        int64_t left, right;
        if (sym_eval(e->binop.left, var_names, values, num_vars, &left) != 0)
            return -1;
        if (sym_eval(e->binop.right, var_names, values, num_vars, &right) != 0)
            return -1;
        return sym_fold_binop(e->type, left, right, out) ? 0 : -1;
    }
    }
    return -1;
}

/**
 * Return a freshly-owned simplified expression: recursively folds constant subtrees
 * and applies identity rules (x+0, x*1, x*0, x/1, 0/x, x%1). Caller releases the result.
 */
SymExpr* sym_simplify(SymExpr* e) {
    if (!e)
        return NULL;

    // Constants and vars are already simplified
    if (e->type == SYM_CONST || e->type == SYM_VAR) {
        sym_expr_retain(e);
        return e;
    }

    // Recursively simplify children
    SymExpr* left  = sym_simplify(e->binop.left);
    SymExpr* right = sym_simplify(e->binop.right);
    if (!left || !right) {
        if (left)
            sym_expr_release(left);
        if (right)
            sym_expr_release(right);
        return NULL;
    }

    // Constant folding (both children are now constants)
    if (left->type == SYM_CONST && right->type == SYM_CONST) {
        int64_t result;
        bool valid = sym_fold_binop(e->type, left->const_val, right->const_val, &result);
        sym_expr_release(left);
        sym_expr_release(right);
        if (valid)
            return sym_const(result);
        return NULL;
    }

    // Identity simplifications
    switch (e->type) {
    case SYM_ADD:
        // x + 0 -> x, 0 + x -> x
        if (left->type == SYM_CONST && left->const_val == 0) {
            sym_expr_release(left);
            return right;
        }
        if (right->type == SYM_CONST && right->const_val == 0) {
            sym_expr_release(right);
            return left;
        }
        break;
    case SYM_MUL:
        // x * 1 -> x, 1 * x -> x
        if (left->type == SYM_CONST && left->const_val == 1) {
            sym_expr_release(left);
            return right;
        }
        if (right->type == SYM_CONST && right->const_val == 1) {
            sym_expr_release(right);
            return left;
        }
        // x * 0 -> 0, 0 * x -> 0
        if (left->type == SYM_CONST && left->const_val == 0) {
            sym_expr_release(right);
            return left; // already 0
        }
        if (right->type == SYM_CONST && right->const_val == 0) {
            sym_expr_release(left);
            return right; // already 0
        }
        break;
    case SYM_DIV:
        // x / 1 -> x
        if (right->type == SYM_CONST && right->const_val == 1) {
            sym_expr_release(right);
            return left;
        }
        // 0 / x -> 0
        if (left->type == SYM_CONST && left->const_val == 0) {
            sym_expr_release(right);
            return left; // already 0
        }
        // Bound fold: 0 <= x < b  =>  x / b == 0 (b a positive constant).
        if (right->type == SYM_CONST && right->const_val > 0 && sym_expr_min(left) >= 0 &&
            sym_expr_max(left) < right->const_val) {
            sym_expr_release(left);
            sym_expr_release(right);
            return sym_const(0);
        }
        break;
    case SYM_MOD:
        // x % 1 -> 0
        if (right->type == SYM_CONST && right->const_val == 1) {
            sym_expr_release(left);
            sym_expr_release(right);
            return sym_const(0);
        }
        // Bound fold: 0 <= x < b  =>  x % b == x (b a positive constant).
        if (right->type == SYM_CONST && right->const_val > 0 && sym_expr_min(left) >= 0 &&
            sym_expr_max(left) < right->const_val) {
            sym_expr_release(right);
            return left;
        }
        break;
    default:
        break;
    }

    // No simplification possible; rebuild node
    SymExpr* out = sym_alloc(e->type);
    if (!out) {
        sym_expr_release(left);
        sym_expr_release(right);
        return NULL;
    }
    out->binop.left  = left; // already retained via sym_simplify
    out->binop.right = right;
    return out;
}

/** Increment the expression's reference count. */
void sym_expr_retain(SymExpr* e) {
    if (e)
        e->ref_count++;
}

/** Drop one reference; frees the node and recursively releases children at zero. */
void sym_expr_release(SymExpr* e) {
    if (!e)
        return;
    if (--e->ref_count > 0)
        return;

    if (e->type != SYM_CONST && e->type != SYM_VAR) {
        sym_expr_release(e->binop.left);
        sym_expr_release(e->binop.right);
    }
    free(e);
}

/**
 * Render the expression into `buf`; binary ops print infix as "(a op b)" while min/max
 * print as "min(a, b)". Returns the snprintf length (bytes that would be written).
 */
int sym_expr_to_string(const SymExpr* e, char* buf, int buf_size) {
    if (!e || !buf || buf_size <= 0)
        return 0;

    switch (e->type) {
    case SYM_CONST:
        return snprintf(buf, (size_t)buf_size, "%lld", (long long)e->const_val);
    case SYM_VAR:
        return snprintf(buf, (size_t)buf_size, "%s", e->var.name);
    default: {
        const char* op;
        switch (e->type) {
        case SYM_ADD:
            op = "+";
            break;
        case SYM_MUL:
            op = "*";
            break;
        case SYM_DIV:
            op = "/";
            break;
        /* Single %: `op` is inserted through a "%s" conversion below, so it is
         * not format-processed. Written "%%" it rendered as "(a %% b)". */
        case SYM_MOD:
            op = "%";
            break;
        case SYM_MIN:
            op = "min";
            break;
        case SYM_MAX:
            op = "max";
            break;
        default:
            op = "?";
            break;
        }

        char left_buf[256], right_buf[256];
        sym_expr_to_string(e->binop.left, left_buf, sizeof(left_buf));
        sym_expr_to_string(e->binop.right, right_buf, sizeof(right_buf));

        if (e->type == SYM_MIN || e->type == SYM_MAX) {
            return snprintf(buf, (size_t)buf_size, "%s(%s, %s)", op, left_buf, right_buf);
        }
        return snprintf(buf, (size_t)buf_size, "(%s %s %s)", left_buf, op, right_buf);
    }
    }
}

/** Wrap a fixed integer extent as a concrete (non-symbolic) dimension. */
SymDim sym_dim_concrete(int value) {
    SymDim d;
    d.is_symbolic = false;
    d.concrete    = value;
    return d;
}

/** Wrap an expression as a symbolic dimension, retaining a reference to it. */
SymDim sym_dim_symbolic(SymExpr* expr) {
    SymDim d;
    d.is_symbolic = true;
    d.expr        = expr;
    if (expr)
        sym_expr_retain(expr);
    return d;
}

/** Release the expression backing a symbolic dimension, if any, and clear it. */
void sym_dim_release(SymDim* dim) {
    if (dim && dim->is_symbolic && dim->expr) {
        sym_expr_release(dim->expr);
        dim->expr = NULL;
    }
}

/** Build a shape of `ndim` concrete dimensions from a plain int array. */
SymShape* sym_shape_from_concrete(const int* dims, int ndim) {
    if (!dims || ndim <= 0)
        return NULL;

    SymShape* s = (SymShape*)calloc(1, sizeof(SymShape));
    if (!s)
        return NULL;
    s->dims = (SymDim*)calloc((size_t)ndim, sizeof(SymDim));
    if (!s->dims) {
        free(s);
        return NULL;
    }

    s->ndim      = ndim;
    s->ref_count = 1;
    for (int i = 0; i < ndim; i++) {
        s->dims[i] = sym_dim_concrete(dims[i]);
    }
    return s;
}

/**
 * Broadcast two shapes NumPy-style (right-aligned): a dim of 1 takes the other
 * operand's extent, two symbolic dims collapse to their max, and a symbolic paired
 * with a non-1 concrete keeps the symbolic. Returns NULL on an incompatible pair.
 */
SymShape* sym_shape_broadcast(const SymShape* a, const SymShape* b) {
    if (!a || !b)
        return NULL;

    int max_ndim  = a->ndim > b->ndim ? a->ndim : b->ndim;
    SymShape* out = (SymShape*)calloc(1, sizeof(SymShape));
    if (!out)
        return NULL;
    out->dims = (SymDim*)calloc((size_t)max_ndim, sizeof(SymDim));
    if (!out->dims) {
        free(out);
        return NULL;
    }
    out->ndim      = max_ndim;
    out->ref_count = 1;

    for (int i = 0; i < max_ndim; i++) {
        int ai = i - (max_ndim - a->ndim);
        int bi = i - (max_ndim - b->ndim);

        bool a_present = (ai >= 0 && ai < a->ndim);
        bool b_present = (bi >= 0 && bi < b->ndim);

        if (!a_present) {
            // Only b present
            if (b->dims[bi].is_symbolic) {
                out->dims[i] = sym_dim_symbolic(b->dims[bi].expr);
            } else {
                out->dims[i] = sym_dim_concrete(b->dims[bi].concrete);
            }
        } else if (!b_present) {
            // Only a present
            if (a->dims[ai].is_symbolic) {
                out->dims[i] = sym_dim_symbolic(a->dims[ai].expr);
            } else {
                out->dims[i] = sym_dim_concrete(a->dims[ai].concrete);
            }
        } else {
            // Both present
            bool a_sym = a->dims[ai].is_symbolic;
            bool b_sym = b->dims[bi].is_symbolic;

            if (!a_sym && !b_sym) {
                int av = a->dims[ai].concrete;
                int bv = b->dims[bi].concrete;
                if (av == bv) {
                    out->dims[i] = sym_dim_concrete(av);
                } else if (av == 1) {
                    out->dims[i] = sym_dim_concrete(bv);
                } else if (bv == 1) {
                    out->dims[i] = sym_dim_concrete(av);
                } else {
                    // Incompatible
                    sym_shape_release(out);
                    return NULL;
                }
            } else if (!a_sym && a->dims[ai].concrete == 1) {
                // concrete 1 broadcasts to symbolic
                if (b_sym) {
                    out->dims[i] = sym_dim_symbolic(b->dims[bi].expr);
                } else {
                    out->dims[i] = sym_dim_concrete(b->dims[bi].concrete);
                }
            } else if (!b_sym && b->dims[bi].concrete == 1) {
                // concrete 1 broadcasts to symbolic
                if (a_sym) {
                    out->dims[i] = sym_dim_symbolic(a->dims[ai].expr);
                } else {
                    out->dims[i] = sym_dim_concrete(a->dims[ai].concrete);
                }
            } else if (a_sym && b_sym) {
                // Both symbolic: result is max(a, b)
                SymExpr* mx  = sym_max_expr(a->dims[ai].expr, b->dims[bi].expr);
                out->dims[i] = sym_dim_symbolic(mx);
                sym_expr_release(mx); // sym_dim_symbolic retains
            } else {
                // One symbolic, one non-1 concrete => use symbolic
                SymExpr* sym_e = a_sym ? a->dims[ai].expr : b->dims[bi].expr;
                out->dims[i]   = sym_dim_symbolic(sym_e);
            }
        }
    }

    return out;
}

/**
 * Resolve every dimension to a concrete int using the variable table, writing results
 * into `out_dims`. Returns 0 on success, or -1 if a symbolic dim fails to evaluate.
 */
int sym_shape_eval(const SymShape* shape, const char** var_names, const int64_t* values,
                   int num_vars, int* out_dims) {
    if (!shape || !out_dims)
        return -1;

    for (int i = 0; i < shape->ndim; i++) {
        if (shape->dims[i].is_symbolic) {
            int64_t val;
            if (sym_eval(shape->dims[i].expr, var_names, values, num_vars, &val) != 0)
                return -1;
            out_dims[i] = (int)val;
        } else {
            out_dims[i] = shape->dims[i].concrete;
        }
    }
    return 0;
}

/** Render the shape as "(d0, d1, ...)" into `buf`; returns total chars written. */
int sym_shape_to_string(const SymShape* shape, char* buf, int buf_size) {
    if (!shape || !buf || buf_size <= 0)
        return 0;

    int written = 0;
    written += snprintf(buf + written, (size_t)(buf_size - written), "(");

    for (int i = 0; i < shape->ndim; i++) {
        if (i > 0)
            written += snprintf(buf + written, (size_t)(buf_size - written), ", ");

        if (shape->dims[i].is_symbolic) {
            written += sym_expr_to_string(shape->dims[i].expr, buf + written, buf_size - written);
        } else {
            written += snprintf(buf + written, (size_t)(buf_size - written), "%d",
                                shape->dims[i].concrete);
        }
    }

    written += snprintf(buf + written, (size_t)(buf_size - written), ")");
    return written;
}

/** Increment the shape's reference count. */
void sym_shape_retain(SymShape* shape) {
    if (shape)
        shape->ref_count++;
}

/** Drop one reference; releases each dim and frees the shape at zero. */
void sym_shape_release(SymShape* shape) {
    if (!shape)
        return;
    if (--shape->ref_count > 0)
        return;

    for (int i = 0; i < shape->ndim; i++) {
        sym_dim_release(&shape->dims[i]);
    }
    free(shape->dims);
    free(shape);
}
