/**
 * @file symbolic.c
 * @brief Symbolic shapes implementation
 */

#include "symbolic/symbolic.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

static int g_var_id_counter = 0;

// ============================================================================
// Internal helpers
// ============================================================================

static SymExpr* sym_alloc(SymExprType type) {
    SymExpr* e = (SymExpr*)calloc(1, sizeof(SymExpr));
    if (!e) return NULL;
    e->type = type;
    e->ref_count = 1;
    return e;
}

static int64_t i64_min(int64_t a, int64_t b) { return a < b ? a : b; }
static int64_t i64_max(int64_t a, int64_t b) { return a > b ? a : b; }

// ============================================================================
// Expression Creation
// ============================================================================

SymExpr* sym_const(int64_t value) {
    SymExpr* e = sym_alloc(SYM_CONST);
    if (e) e->const_val = value;
    return e;
}

SymExpr* sym_var(const char* name, int64_t vmin, int64_t vmax) {
    if (!name) return NULL;
    SymExpr* e = sym_alloc(SYM_VAR);
    if (!e) return NULL;
    strncpy(e->var.name, name, sizeof(e->var.name) - 1);
    e->var.name[sizeof(e->var.name) - 1] = '\0';
    e->var.vmin = vmin;
    e->var.vmax = vmax;
    e->var.id = g_var_id_counter++;
    return e;
}

// ============================================================================
// Arithmetic with constant folding
// ============================================================================

static SymExpr* sym_binop(SymExprType type, SymExpr* a, SymExpr* b) {
    if (!a || !b) return NULL;

    // Constant folding: if both children are constants, evaluate now
    if (a->type == SYM_CONST && b->type == SYM_CONST) {
        int64_t result;
        switch (type) {
        case SYM_ADD: result = a->const_val + b->const_val; break;
        case SYM_MUL: result = a->const_val * b->const_val; break;
        case SYM_DIV:
            if (b->const_val == 0) return NULL;
            result = a->const_val / b->const_val;
            break;
        case SYM_MOD:
            if (b->const_val == 0) return NULL;
            result = a->const_val % b->const_val;
            break;
        case SYM_MIN: result = i64_min(a->const_val, b->const_val); break;
        case SYM_MAX: result = i64_max(a->const_val, b->const_val); break;
        default: return NULL;
        }
        return sym_const(result);
    }

    SymExpr* e = sym_alloc(type);
    if (!e) return NULL;
    sym_expr_retain(a);
    sym_expr_retain(b);
    e->binop.left = a;
    e->binop.right = b;
    return e;
}

SymExpr* sym_add(SymExpr* a, SymExpr* b) { return sym_binop(SYM_ADD, a, b); }
SymExpr* sym_mul(SymExpr* a, SymExpr* b) { return sym_binop(SYM_MUL, a, b); }
SymExpr* sym_div(SymExpr* a, SymExpr* b) { return sym_binop(SYM_DIV, a, b); }
SymExpr* sym_mod(SymExpr* a, SymExpr* b) { return sym_binop(SYM_MOD, a, b); }
SymExpr* sym_min_expr(SymExpr* a, SymExpr* b) { return sym_binop(SYM_MIN, a, b); }
SymExpr* sym_max_expr(SymExpr* a, SymExpr* b) { return sym_binop(SYM_MAX, a, b); }

// ============================================================================
// Bounds Inference
// ============================================================================

int64_t sym_expr_min(const SymExpr* e) {
    if (!e) return 0;
    switch (e->type) {
    case SYM_CONST: return e->const_val;
    case SYM_VAR:   return e->var.vmin;
    case SYM_ADD:   return sym_expr_min(e->binop.left) + sym_expr_min(e->binop.right);
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
        if (b_min <= 0 && b_max >= 0) return INT64_MIN;
        int64_t p1 = a_min / b_min, p2 = a_min / b_max;
        int64_t p3 = a_max / b_min, p4 = a_max / b_max;
        return i64_min(i64_min(p1, p2), i64_min(p3, p4));
    }
    case SYM_MOD: {
        int64_t b_max = sym_expr_max(e->binop.right);
        if (b_max <= 0) return 0;
        // Mod result is in [0, |b|-1] if a >= 0, else [-(|b|-1), 0]
        int64_t a_min = sym_expr_min(e->binop.left);
        if (a_min >= 0) return 0;
        return -(b_max - 1);
    }
    case SYM_MIN: return i64_min(sym_expr_min(e->binop.left), sym_expr_min(e->binop.right));
    case SYM_MAX: return i64_max(sym_expr_min(e->binop.left), sym_expr_min(e->binop.right));
    }
    return 0;
}

int64_t sym_expr_max(const SymExpr* e) {
    if (!e) return 0;
    switch (e->type) {
    case SYM_CONST: return e->const_val;
    case SYM_VAR:   return e->var.vmax;
    case SYM_ADD:   return sym_expr_max(e->binop.left) + sym_expr_max(e->binop.right);
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
        if (b_min <= 0 && b_max >= 0) return INT64_MAX;
        int64_t p1 = a_min / b_min, p2 = a_min / b_max;
        int64_t p3 = a_max / b_min, p4 = a_max / b_max;
        return i64_max(i64_max(p1, p2), i64_max(p3, p4));
    }
    case SYM_MOD: {
        int64_t b_max = sym_expr_max(e->binop.right);
        if (b_max <= 0) return 0;
        int64_t a_max_val = sym_expr_max(e->binop.left);
        if (a_max_val >= 0) return b_max - 1;
        return 0;
    }
    case SYM_MIN: return i64_min(sym_expr_max(e->binop.left), sym_expr_max(e->binop.right));
    case SYM_MAX: return i64_max(sym_expr_max(e->binop.left), sym_expr_max(e->binop.right));
    }
    return 0;
}

// ============================================================================
// Evaluation
// ============================================================================

int64_t sym_eval(const SymExpr* e, const char** var_names, const int64_t* values, int num_vars) {
    if (!e) return 0;
    switch (e->type) {
    case SYM_CONST: return e->const_val;
    case SYM_VAR:
        for (int i = 0; i < num_vars; i++) {
            if (var_names[i] && strcmp(var_names[i], e->var.name) == 0)
                return values[i];
        }
        // Variable not found; return midpoint as fallback
        return (e->var.vmin + e->var.vmax) / 2;
    case SYM_ADD: return sym_eval(e->binop.left, var_names, values, num_vars)
                       + sym_eval(e->binop.right, var_names, values, num_vars);
    case SYM_MUL: return sym_eval(e->binop.left, var_names, values, num_vars)
                       * sym_eval(e->binop.right, var_names, values, num_vars);
    case SYM_DIV: {
        int64_t r = sym_eval(e->binop.right, var_names, values, num_vars);
        if (r == 0) return 0;
        return sym_eval(e->binop.left, var_names, values, num_vars) / r;
    }
    case SYM_MOD: {
        int64_t r = sym_eval(e->binop.right, var_names, values, num_vars);
        if (r == 0) return 0;
        return sym_eval(e->binop.left, var_names, values, num_vars) % r;
    }
    case SYM_MIN: return i64_min(sym_eval(e->binop.left, var_names, values, num_vars),
                                  sym_eval(e->binop.right, var_names, values, num_vars));
    case SYM_MAX: return i64_max(sym_eval(e->binop.left, var_names, values, num_vars),
                                  sym_eval(e->binop.right, var_names, values, num_vars));
    }
    return 0;
}

// ============================================================================
// Simplification
// ============================================================================

SymExpr* sym_simplify(SymExpr* e) {
    if (!e) return NULL;

    // Constants and vars are already simplified
    if (e->type == SYM_CONST || e->type == SYM_VAR) {
        sym_expr_retain(e);
        return e;
    }

    // Recursively simplify children
    SymExpr* left = sym_simplify(e->binop.left);
    SymExpr* right = sym_simplify(e->binop.right);
    if (!left || !right) {
        if (left) sym_expr_release(left);
        if (right) sym_expr_release(right);
        return NULL;
    }

    // Constant folding (both children are now constants)
    if (left->type == SYM_CONST && right->type == SYM_CONST) {
        int64_t result;
        bool valid = true;
        switch (e->type) {
        case SYM_ADD: result = left->const_val + right->const_val; break;
        case SYM_MUL: result = left->const_val * right->const_val; break;
        case SYM_DIV:
            if (right->const_val == 0) { valid = false; break; }
            result = left->const_val / right->const_val; break;
        case SYM_MOD:
            if (right->const_val == 0) { valid = false; break; }
            result = left->const_val % right->const_val; break;
        case SYM_MIN: result = i64_min(left->const_val, right->const_val); break;
        case SYM_MAX: result = i64_max(left->const_val, right->const_val); break;
        default: valid = false;
        }
        sym_expr_release(left);
        sym_expr_release(right);
        if (valid) return sym_const(result);
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
        break;
    case SYM_MOD:
        // x % 1 -> 0
        if (right->type == SYM_CONST && right->const_val == 1) {
            sym_expr_release(left);
            sym_expr_release(right);
            return sym_const(0);
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
    out->binop.left = left;   // already retained via sym_simplify
    out->binop.right = right;
    return out;
}

// ============================================================================
// Memory Management
// ============================================================================

void sym_expr_retain(SymExpr* e) {
    if (e) e->ref_count++;
}

void sym_expr_release(SymExpr* e) {
    if (!e) return;
    if (--e->ref_count > 0) return;

    // Free children for binary ops
    if (e->type != SYM_CONST && e->type != SYM_VAR) {
        sym_expr_release(e->binop.left);
        sym_expr_release(e->binop.right);
    }
    free(e);
}

// ============================================================================
// Debug / String Conversion
// ============================================================================

int sym_expr_to_string(const SymExpr* e, char* buf, int buf_size) {
    if (!e || !buf || buf_size <= 0) return 0;

    switch (e->type) {
    case SYM_CONST:
        return snprintf(buf, (size_t)buf_size, "%lld", (long long)e->const_val);
    case SYM_VAR:
        return snprintf(buf, (size_t)buf_size, "%s", e->var.name);
    default: {
        const char* op;
        switch (e->type) {
        case SYM_ADD: op = "+"; break;
        case SYM_MUL: op = "*"; break;
        case SYM_DIV: op = "/"; break;
        case SYM_MOD: op = "%%"; break;
        case SYM_MIN: op = "min"; break;
        case SYM_MAX: op = "max"; break;
        default: op = "?"; break;
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

// ============================================================================
// SymDim / SymShape
// ============================================================================

SymDim sym_dim_concrete(int value) {
    SymDim d;
    d.is_symbolic = false;
    d.concrete = value;
    return d;
}

SymDim sym_dim_symbolic(SymExpr* expr) {
    SymDim d;
    d.is_symbolic = true;
    d.expr = expr;
    if (expr) sym_expr_retain(expr);
    return d;
}

void sym_dim_release(SymDim* dim) {
    if (dim && dim->is_symbolic && dim->expr) {
        sym_expr_release(dim->expr);
        dim->expr = NULL;
    }
}

SymShape* sym_shape_from_concrete(const int* dims, int ndim) {
    if (!dims || ndim <= 0) return NULL;

    SymShape* s = (SymShape*)calloc(1, sizeof(SymShape));
    if (!s) return NULL;
    s->dims = (SymDim*)calloc((size_t)ndim, sizeof(SymDim));
    if (!s->dims) { free(s); return NULL; }

    s->ndim = ndim;
    s->ref_count = 1;
    for (int i = 0; i < ndim; i++) {
        s->dims[i] = sym_dim_concrete(dims[i]);
    }
    return s;
}

SymShape* sym_shape_broadcast(const SymShape* a, const SymShape* b) {
    if (!a || !b) return NULL;

    int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    SymShape* out = (SymShape*)calloc(1, sizeof(SymShape));
    if (!out) return NULL;
    out->dims = (SymDim*)calloc((size_t)max_ndim, sizeof(SymDim));
    if (!out->dims) { free(out); return NULL; }
    out->ndim = max_ndim;
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
                SymExpr* mx = sym_max_expr(a->dims[ai].expr, b->dims[bi].expr);
                out->dims[i] = sym_dim_symbolic(mx);
                sym_expr_release(mx); // sym_dim_symbolic retains
            } else {
                // One symbolic, one non-1 concrete => use symbolic
                SymExpr* sym_e = a_sym ? a->dims[ai].expr : b->dims[bi].expr;
                out->dims[i] = sym_dim_symbolic(sym_e);
            }
        }
    }

    return out;
}

int sym_shape_eval(const SymShape* shape, const char** var_names, const int64_t* values,
                   int num_vars, int* out_dims) {
    if (!shape || !out_dims) return -1;

    for (int i = 0; i < shape->ndim; i++) {
        if (shape->dims[i].is_symbolic) {
            out_dims[i] = (int)sym_eval(shape->dims[i].expr, var_names, values, num_vars);
        } else {
            out_dims[i] = shape->dims[i].concrete;
        }
    }
    return 0;
}

int sym_shape_to_string(const SymShape* shape, char* buf, int buf_size) {
    if (!shape || !buf || buf_size <= 0) return 0;

    int written = 0;
    written += snprintf(buf + written, (size_t)(buf_size - written), "(");

    for (int i = 0; i < shape->ndim; i++) {
        if (i > 0) written += snprintf(buf + written, (size_t)(buf_size - written), ", ");

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

void sym_shape_retain(SymShape* shape) {
    if (shape) shape->ref_count++;
}

void sym_shape_release(SymShape* shape) {
    if (!shape) return;
    if (--shape->ref_count > 0) return;

    for (int i = 0; i < shape->ndim; i++) {
        sym_dim_release(&shape->dims[i]);
    }
    free(shape->dims);
    free(shape);
}
