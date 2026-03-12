/**
 * @file symbolic.h
 * @brief Symbolic shapes for dynamic tensor dimensions with bounds
 *
 * Inspired by tinygrad's Variable system, this provides symbolic expressions
 * that represent tensor dimensions whose values are only known at runtime.
 * Supports arithmetic operations, bound inference, evaluation, and simplification.
 *
 * Key design: orthogonal to existing Tensor struct. Ref-counted for memory safety.
 */

#ifndef CML_SYMBOLIC_H
#define CML_SYMBOLIC_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Symbolic expression node types
 */
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

/**
 * @brief Symbolic expression node (ref-counted)
 */
typedef struct SymExpr {
    SymExprType type;
    union {
        int64_t const_val;
        struct { char name[32]; int64_t vmin; int64_t vmax; int id; } var;
        struct { struct SymExpr* left; struct SymExpr* right; } binop;
    };
    int ref_count;
} SymExpr;

/**
 * @brief A single dimension that is either concrete or symbolic
 */
typedef struct SymDim {
    bool is_symbolic;
    union {
        int concrete;
        SymExpr* expr;
    };
} SymDim;

/**
 * @brief A shape composed of SymDim entries (ref-counted)
 */
typedef struct SymShape {
    SymDim* dims;
    int ndim;
    int ref_count;
} SymShape;

// ============================================================================
// Expression Creation
// ============================================================================

/** Create a constant expression */
SymExpr* sym_const(int64_t value);

/** Create a named variable with bounds [vmin, vmax] */
SymExpr* sym_var(const char* name, int64_t vmin, int64_t vmax);

// ============================================================================
// Arithmetic (constant-folding at construction time)
// ============================================================================

SymExpr* sym_add(SymExpr* a, SymExpr* b);
SymExpr* sym_mul(SymExpr* a, SymExpr* b);
SymExpr* sym_div(SymExpr* a, SymExpr* b);
SymExpr* sym_mod(SymExpr* a, SymExpr* b);
SymExpr* sym_min_expr(SymExpr* a, SymExpr* b);
SymExpr* sym_max_expr(SymExpr* a, SymExpr* b);

// ============================================================================
// Bounds Inference
// ============================================================================

/** Compute the minimum possible value of the expression */
int64_t sym_expr_min(const SymExpr* e);

/** Compute the maximum possible value of the expression */
int64_t sym_expr_max(const SymExpr* e);

// ============================================================================
// Evaluation
// ============================================================================

/**
 * Substitute variable values and compute the result.
 * @param e Expression to evaluate
 * @param var_names Array of variable name strings
 * @param values Corresponding values
 * @param num_vars Number of variables
 * @return Evaluated integer value
 */
int64_t sym_eval(const SymExpr* e, const char** var_names, const int64_t* values, int num_vars);

// ============================================================================
// Simplification
// ============================================================================

/**
 * Simplify an expression: constant folding, identity removal
 * (x+0 -> x, x*1 -> x, x*0 -> 0, 0/x -> 0, x%1 -> 0)
 * Returns a new ref-counted expression (caller must release).
 */
SymExpr* sym_simplify(SymExpr* e);

// ============================================================================
// Memory Management (reference counting)
// ============================================================================

void sym_expr_retain(SymExpr* e);
void sym_expr_release(SymExpr* e);

// ============================================================================
// Debug / String Conversion
// ============================================================================

/**
 * Convert expression to human-readable string.
 * @param e Expression
 * @param buf Output buffer
 * @param buf_size Buffer size
 * @return Number of characters written (excluding NUL)
 */
int sym_expr_to_string(const SymExpr* e, char* buf, int buf_size);

// ============================================================================
// SymDim / SymShape
// ============================================================================

/** Create a concrete SymDim */
SymDim sym_dim_concrete(int value);

/** Create a symbolic SymDim (retains the expression) */
SymDim sym_dim_symbolic(SymExpr* expr);

/** Release a SymDim (releases expr if symbolic) */
void sym_dim_release(SymDim* dim);

/** Create a SymShape from concrete integer array */
SymShape* sym_shape_from_concrete(const int* dims, int ndim);

/** Broadcast two shapes. Returns new SymShape or NULL on incompatibility. */
SymShape* sym_shape_broadcast(const SymShape* a, const SymShape* b);

/**
 * Evaluate all symbolic dims in a shape.
 * @param shape Input shape
 * @param var_names Variable names
 * @param values Variable values
 * @param num_vars Number of variables
 * @param out_dims Output array (caller-allocated, size >= shape->ndim)
 * @return 0 on success, -1 on error
 */
int sym_shape_eval(const SymShape* shape, const char** var_names, const int64_t* values,
                   int num_vars, int* out_dims);

/**
 * Convert shape to string.
 * @param shape Input shape
 * @param buf Output buffer
 * @param buf_size Buffer size
 * @return Characters written
 */
int sym_shape_to_string(const SymShape* shape, char* buf, int buf_size);

/** Retain a SymShape */
void sym_shape_retain(SymShape* shape);

/** Release a SymShape (frees dims and any symbolic expressions) */
void sym_shape_release(SymShape* shape);

#ifdef __cplusplus
}
#endif

#endif // CML_SYMBOLIC_H
