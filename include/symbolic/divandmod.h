/*
 * symbolic/divandmod — algebraic simplification of division and modulo.
 *
 * Mirrors TinyGrad's tinygrad/uop/divandmod.py.
 *
 * Index arithmetic in kernel codegen frequently involves patterns like:
 *   (a * b) // b  → a
 *   (a + b * k) % b → a % b   (when 0 ≤ a < b)
 *   (x // n) * n + x % n → x
 *
 * This module extends the core SymExpr system with:
 *   - Floor division  (sym_floordiv):   ⌊a / b⌋
 *   - Ceiling division (sym_ceildiv):   ⌈a / b⌉
 *   - Positive modulo  (sym_mod_pos):   ((a % b) + b) % b
 *
 * And provides a simplification pass that rewrites expressions using the
 * algebraic identities above, reducing the complexity of generated index
 * expressions.
 */

#ifndef CML_SYMBOLIC_DIVANDMOD_H
#define CML_SYMBOLIC_DIVANDMOD_H

#include "symbolic/symbolic.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Extended division / modulo constructors ---- */

/*
 * Floor division: ⌊a / b⌋ (rounds toward −∞).
 * Applies algebraic rewrites immediately when possible.
 * Caller owns the returned expression (ref-counted).
 */
SymExpr* sym_floordiv(SymExpr* a, SymExpr* b);

/*
 * Floor division by an integer constant — common in index calculations.
 */
SymExpr* sym_floordiv_const(SymExpr* a, int64_t b);

/*
 * Ceiling division: ⌈a / b⌉  =  (a + b - 1) // b  for b > 0.
 */
SymExpr* sym_ceildiv(SymExpr* a, SymExpr* b);
SymExpr* sym_ceildiv_const(SymExpr* a, int64_t b);

/*
 * Positive (Python-style) modulo: result has the same sign as b.
 *   sym_mod_pos(a, b) = ((a % b) + b) % b   when b > 0.
 * Avoids negative remainders from C's truncating % operator.
 */
SymExpr* sym_mod_pos(SymExpr* a, SymExpr* b);
SymExpr* sym_mod_pos_const(SymExpr* a, int64_t b);

/* ---- Algebraic identities ---- */

/*
 * Try to prove that (expr // divisor) * divisor + (expr % divisor) == expr,
 * i.e. that divmod reconstruction is lossless.  Returns true when the
 * symbolic bounds guarantee this without a runtime check.
 */
bool sym_divmod_exact(const SymExpr* expr, int64_t divisor);

/*
 * Returns true if the value of expr is known to be divisible by divisor
 * (i.e. expr % divisor == 0) from bounds / structure alone.
 */
bool sym_divisible_by(const SymExpr* expr, int64_t divisor);

/* ---- Simplification pass ---- */

/*
 * Apply all div/mod algebraic rewrite rules to e.
 * Rewrites include:
 *   (a * b) // b → a             (when b is a positive constant)
 *   (a * b) % b  → 0             (same condition)
 *   (a + b*k) // b → a//b + k   (when 0 ≤ a < b, k constant)
 *   (a + b*k) % b  → a % b      (same condition)
 *   (a // b) // c → a // (b*c)   (associativity)
 *   (a % b) // b   → 0
 *   n // n         → 1            (trivial)
 *   0 // b         → 0
 *   a % 1          → 0
 *   a // 1         → a
 *
 * Returns a new ref-counted expression; caller must sym_expr_release() it.
 */
SymExpr* sym_simplify_divmod(SymExpr* e);

/*
 * Full simplification: apply sym_simplify() then sym_simplify_divmod()
 * repeatedly until convergence.  Returns a new ref-counted expression.
 */
SymExpr* sym_simplify_full(SymExpr* e);

#ifdef __cplusplus
}
#endif

#endif /* CML_SYMBOLIC_DIVANDMOD_H */
