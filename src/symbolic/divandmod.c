#include "symbolic/divandmod.h"
#include <stdlib.h>
#include <limits.h>
#include <string.h>

/* =========================================================================
 * Helpers
 * ========================================================================= */

/* True if e is a SYM_CONST with value v. */
static bool is_const(const SymExpr* e, int64_t v) {
    return e && e->type == SYM_CONST && e->const_val == v;
}

/* True if e is any SYM_CONST. */
static bool is_const_any(const SymExpr* e, int64_t* out) {
    if (!e || e->type != SYM_CONST) return false;
    if (out) *out = e->const_val;
    return true;
}

/* Safe floor-divide for integers (rounds towards −∞). */
static int64_t floordiv_i64(int64_t a, int64_t b) {
    if (b == 0) return 0;
    int64_t q = a / b;
    /* Round toward −∞ if signs differ. */
    if ((a ^ b) < 0 && q * b != a) --q;
    return q;
}

static int64_t ceildiv_i64(int64_t a, int64_t b) {
    if (b == 0) return 0;
    return floordiv_i64(a + b - 1, b);
}

static int64_t mod_pos_i64(int64_t a, int64_t b) {
    if (b == 0) return 0;
    int64_t r = a % b;
    if (r < 0) r += (b > 0 ? b : -b);
    return r;
}

/* =========================================================================
 * sym_floordiv
 * ========================================================================= */

SymExpr* sym_floordiv(SymExpr* a, SymExpr* b) {
    if (!a || !b) return NULL;

    int64_t av, bv;
    /* Constant folding */
    if (is_const_any(a, &av) && is_const_any(b, &bv))
        return sym_const(floordiv_i64(av, bv));

    /* a // 1 = a */
    if (is_const(b, 1)) { sym_expr_retain(a); return a; }

    /* 0 // b = 0 */
    if (is_const(a, 0)) return sym_const(0);

    /* (a * b) // b = a  (when b is a positive constant) */
    if (is_const_any(b, &bv) && bv > 0 && a->type == SYM_MUL) {
        int64_t cv;
        if (is_const_any(a->binop.right, &cv) && cv == bv) {
            sym_expr_retain(a->binop.left);
            return a->binop.left;
        }
        if (is_const_any(a->binop.left, &cv) && cv == bv) {
            sym_expr_retain(a->binop.right);
            return a->binop.right;
        }
    }

    /* (a + b*k) // b  =  a//b + k  when a >= 0 and a < b */
    if (is_const_any(b, &bv) && bv > 0 && a->type == SYM_ADD) {
        /* Check if one child is a multiple of b. */
        SymExpr* terms[2] = { a->binop.left, a->binop.right };
        for (int t = 0; t < 2; ++t) {
            SymExpr* term = terms[t];
            SymExpr* rest = terms[1 - t];
            int64_t cv;
            bool term_is_multiple = false;
            SymExpr* quotient = NULL;
            if (is_const_any(term, &cv) && cv % bv == 0) {
                term_is_multiple = true;
                quotient = sym_const(cv / bv);
            } else if (term->type == SYM_MUL) {
                if (is_const_any(term->binop.right, &cv) && cv % bv == 0) {
                    term_is_multiple = true;
                    SymExpr* q = sym_const(cv / bv);
                    quotient = sym_mul(term->binop.left, q);
                    sym_expr_release(q);
                } else if (is_const_any(term->binop.left, &cv) && cv % bv == 0) {
                    term_is_multiple = true;
                    SymExpr* q = sym_const(cv / bv);
                    quotient = sym_mul(q, term->binop.right);
                    sym_expr_release(q);
                }
            }
            if (term_is_multiple && quotient) {
                /* Check that rest is bounded: 0 <= rest < b. */
                int64_t rmin = sym_expr_min(rest);
                int64_t rmax = sym_expr_max(rest);
                if (rmin >= 0 && rmax < bv) {
                    /* (rest + multiple) // b = multiple // b + rest // b
                     * but since rest < b: rest // b = 0.
                     * Result = quotient. */
                    return quotient;
                }
                sym_expr_release(quotient);
            }
        }
    }

    /* (a // b) // c = a // (b*c) */
    if (is_const_any(b, &bv) && bv > 0 && a->type == SYM_DIV) {
        int64_t inner_b;
        if (is_const_any(a->binop.right, &inner_b) && inner_b > 0) {
            SymExpr* combined = sym_const(bv * inner_b);
            SymExpr* result   = sym_floordiv(a->binop.left, combined);
            sym_expr_release(combined);
            return result;
        }
    }

    /* Default: create a SYM_DIV node. */
    return sym_div(a, b);
}

SymExpr* sym_floordiv_const(SymExpr* a, int64_t b) {
    SymExpr* bc = sym_const(b);
    SymExpr* r  = sym_floordiv(a, bc);
    sym_expr_release(bc);
    return r;
}

/* =========================================================================
 * sym_ceildiv
 * ========================================================================= */

SymExpr* sym_ceildiv(SymExpr* a, SymExpr* b) {
    if (!a || !b) return NULL;
    int64_t av, bv;
    if (is_const_any(a, &av) && is_const_any(b, &bv))
        return sym_const(ceildiv_i64(av, bv));
    /* ceildiv(a, b) = (a + b - 1) // b */
    SymExpr* bm1    = sym_add(b, sym_const(-1));
    SymExpr* a_plus = sym_add(a, bm1);
    sym_expr_release(bm1);
    SymExpr* result = sym_floordiv(a_plus, b);
    sym_expr_release(a_plus);
    return result;
}

SymExpr* sym_ceildiv_const(SymExpr* a, int64_t b) {
    SymExpr* bc = sym_const(b);
    SymExpr* r  = sym_ceildiv(a, bc);
    sym_expr_release(bc);
    return r;
}

/* =========================================================================
 * sym_mod_pos
 * ========================================================================= */

SymExpr* sym_mod_pos(SymExpr* a, SymExpr* b) {
    if (!a || !b) return NULL;
    int64_t av, bv;
    if (is_const_any(a, &av) && is_const_any(b, &bv))
        return sym_const(mod_pos_i64(av, bv));

    /* a % 1 = 0 */
    if (is_const(b, 1)) return sym_const(0);

    /* 0 % b = 0 */
    if (is_const(a, 0)) return sym_const(0);

    /* (a * b) % b = 0 */
    if (is_const_any(b, &bv) && bv > 0 && a->type == SYM_MUL) {
        int64_t cv;
        if ((is_const_any(a->binop.right, &cv) && cv % bv == 0) ||
            (is_const_any(a->binop.left,  &cv) && cv % bv == 0))
            return sym_const(0);
    }

    /* (a + b*k) % b = a % b  when 0 <= a < b */
    if (is_const_any(b, &bv) && bv > 0 && a->type == SYM_ADD) {
        SymExpr* terms[2] = { a->binop.left, a->binop.right };
        for (int t = 0; t < 2; ++t) {
            SymExpr* term = terms[t];
            SymExpr* rest = terms[1 - t];
            int64_t cv;
            bool divisible = false;
            if (is_const_any(term, &cv) && cv % bv == 0)
                divisible = true;
            else if (term->type == SYM_MUL) {
                if ((is_const_any(term->binop.right, &cv) && cv % bv == 0) ||
                    (is_const_any(term->binop.left,  &cv) && cv % bv == 0))
                    divisible = true;
            }
            if (divisible) {
                int64_t rmin = sym_expr_min(rest);
                int64_t rmax = sym_expr_max(rest);
                if (rmin >= 0 && rmax < bv) {
                    sym_expr_retain(rest);
                    return rest;
                }
                /* General case: (a+k*b) % b = a % b */
                return sym_mod_pos(rest, b);
            }
        }
    }

    /* If a's range fits in [0, b): a % b = a */
    if (is_const_any(b, &bv) && bv > 0) {
        int64_t amin = sym_expr_min(a);
        int64_t amax = sym_expr_max(a);
        if (amin >= 0 && amax < bv) {
            sym_expr_retain(a);
            return a;
        }
    }

    return sym_mod(a, b);
}

SymExpr* sym_mod_pos_const(SymExpr* a, int64_t b) {
    SymExpr* bc = sym_const(b);
    SymExpr* r  = sym_mod_pos(a, bc);
    sym_expr_release(bc);
    return r;
}

/* =========================================================================
 * Divisibility predicates
 * ========================================================================= */

bool sym_divisible_by(const SymExpr* expr, int64_t divisor) {
    if (!expr || divisor == 0) return false;
    if (divisor == 1) return true;
    switch (expr->type) {
        case SYM_CONST:
            return expr->const_val % divisor == 0;
        case SYM_MUL: {
            int64_t cv;
            if (is_const_any(expr->binop.right, &cv) && cv % divisor == 0) return true;
            if (is_const_any(expr->binop.left,  &cv) && cv % divisor == 0) return true;
            return sym_divisible_by(expr->binop.left, divisor) ||
                   sym_divisible_by(expr->binop.right, divisor);
        }
        case SYM_ADD:
            return sym_divisible_by(expr->binop.left,  divisor) &&
                   sym_divisible_by(expr->binop.right, divisor);
        default:
            return false;
    }
}

bool sym_divmod_exact(const SymExpr* expr, int64_t divisor) {
    /* (expr // d) * d + (expr % d) == expr always.
     * We can prove it losslessly when divisor divides expr exactly. */
    return sym_divisible_by(expr, divisor);
}

/* =========================================================================
 * Simplification passes
 * ========================================================================= */

SymExpr* sym_simplify_divmod(SymExpr* e) {
    if (!e) return NULL;
    /* Bottom-up: simplify children first. */
    if (e->type == SYM_ADD || e->type == SYM_MUL ||
        e->type == SYM_DIV || e->type == SYM_MOD ||
        e->type == SYM_MIN || e->type == SYM_MAX) {
        SymExpr* sl = sym_simplify_divmod(e->binop.left);
        SymExpr* sr = sym_simplify_divmod(e->binop.right);
        /* Reconstruct with simplified children. */
        SymExpr* rebuilt = NULL;
        switch (e->type) {
            case SYM_ADD: rebuilt = sym_add(sl, sr); break;
            case SYM_MUL: rebuilt = sym_mul(sl, sr); break;
            case SYM_DIV: rebuilt = sym_floordiv(sl, sr); break;
            case SYM_MOD: rebuilt = sym_mod_pos(sl, sr);  break;
            case SYM_MIN: rebuilt = sym_min_expr(sl, sr); break;
            case SYM_MAX: rebuilt = sym_max_expr(sl, sr); break;
            default: break;
        }
        sym_expr_release(sl);
        sym_expr_release(sr);
        if (rebuilt) return rebuilt;
    }
    sym_expr_retain(e);
    return e;
}

SymExpr* sym_simplify_full(SymExpr* e) {
    if (!e) return NULL;
    SymExpr* prev = NULL;
    sym_expr_retain(e);
    SymExpr* cur = e;
    /* Iterate until convergence (max 8 rounds). */
    for (int iter = 0; iter < 8; ++iter) {
        SymExpr* s1 = sym_simplify(cur);
        SymExpr* s2 = sym_simplify_divmod(s1);
        sym_expr_release(s1);
        /* Check if we've converged (same string representation). */
        if (prev) {
            char buf_cur[256] = {0}, buf_prv[256] = {0};
            sym_expr_to_string(s2,   buf_cur, sizeof(buf_cur));
            sym_expr_to_string(prev, buf_prv, sizeof(buf_prv));
            sym_expr_release(prev);
            prev = s2;
            if (strcmp(buf_cur, buf_prv) == 0) break;
        } else {
            prev = s2;
        }
        sym_expr_release(cur);
        sym_expr_retain(prev);
        cur = prev;
    }
    sym_expr_release(cur);
    return prev;
}
