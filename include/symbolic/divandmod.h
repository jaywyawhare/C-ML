

#ifndef CML_SYMBOLIC_DIVANDMOD_H
#define CML_SYMBOLIC_DIVANDMOD_H

#include "symbolic/symbolic.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

SymExpr* sym_floordiv(SymExpr* a, SymExpr* b);

SymExpr* sym_floordiv_const(SymExpr* a, int64_t b);

SymExpr* sym_ceildiv(SymExpr* a, SymExpr* b);
SymExpr* sym_ceildiv_const(SymExpr* a, int64_t b);

SymExpr* sym_mod_pos(SymExpr* a, SymExpr* b);
SymExpr* sym_mod_pos_const(SymExpr* a, int64_t b);

bool sym_divmod_exact(const SymExpr* expr, int64_t divisor);

bool sym_divisible_by(const SymExpr* expr, int64_t divisor);

SymExpr* sym_simplify_divmod(SymExpr* e);

SymExpr* sym_simplify_full(SymExpr* e);

#ifdef __cplusplus
}
#endif

#endif 
