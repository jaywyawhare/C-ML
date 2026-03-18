#ifndef CML_SYMBOLIC_H
#define CML_SYMBOLIC_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SYM_CONST,  // Constant integer value
    SYM_VAR,    // Named variable with bounds [vmin, vmax]
    SYM_ADD,    // left + right
    SYM_MUL,    // left * right
    SYM_DIV,    // left / right (integer division)
    SYM_MOD,    // left % right
    SYM_MIN,    // min(left, right)
    SYM_MAX     // max(left, right)
} SymExprType;

typedef struct SymExpr {
    SymExprType type;
    union {
        int64_t const_val;
        struct { char name[32]; int64_t vmin; int64_t vmax; int id; } var;
        struct { struct SymExpr* left; struct SymExpr* right; } binop;
    };
    int ref_count;
} SymExpr;

typedef struct SymDim {
    bool is_symbolic;
    union {
        int concrete;
        SymExpr* expr;
    };
} SymDim;

typedef struct SymShape {
    SymDim* dims;
    int ndim;
    int ref_count;
} SymShape;

SymExpr* sym_const(int64_t value);

SymExpr* sym_var(const char* name, int64_t vmin, int64_t vmax);

SymExpr* sym_add(SymExpr* a, SymExpr* b);
SymExpr* sym_mul(SymExpr* a, SymExpr* b);
SymExpr* sym_div(SymExpr* a, SymExpr* b);
SymExpr* sym_mod(SymExpr* a, SymExpr* b);
SymExpr* sym_min_expr(SymExpr* a, SymExpr* b);
SymExpr* sym_max_expr(SymExpr* a, SymExpr* b);

int64_t sym_expr_min(const SymExpr* e);

int64_t sym_expr_max(const SymExpr* e);

int64_t sym_eval(const SymExpr* e, const char** var_names, const int64_t* values, int num_vars);

/* Simplify: constant folding, identity removal.
 * Returns a new ref-counted expression (caller must release). */
SymExpr* sym_simplify(SymExpr* e);

void sym_expr_retain(SymExpr* e);
void sym_expr_release(SymExpr* e);

int sym_expr_to_string(const SymExpr* e, char* buf, int buf_size);

SymDim sym_dim_concrete(int value);

SymDim sym_dim_symbolic(SymExpr* expr);

void sym_dim_release(SymDim* dim);

SymShape* sym_shape_from_concrete(const int* dims, int ndim);

SymShape* sym_shape_broadcast(const SymShape* a, const SymShape* b);

int sym_shape_eval(const SymShape* shape, const char** var_names, const int64_t* values,
                   int num_vars, int* out_dims);

int sym_shape_to_string(const SymShape* shape, char* buf, int buf_size);

void sym_shape_retain(SymShape* shape);

void sym_shape_release(SymShape* shape);

#ifdef __cplusplus
}
#endif

#endif // CML_SYMBOLIC_H
