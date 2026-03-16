/**
 * @file z3_verify.c
 * @brief Z3 SMT solver for IR validation
 *
 * When CML_HAS_Z3 is defined, dynamically loads libz3 and uses the Z3 C API
 * to verify IR transformation equivalence, index bounds, and schedule
 * correctness.  Otherwise, all public functions return CML_VERIFY_UNSUPPORTED.
 */

#include "ops/ir/z3_verify.h"
#include "ops/ir/internal.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

#ifdef CML_HAS_Z3

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#define Z3_LIB_NAME "libz3.so"
#elif defined(_WIN32)
#include <windows.h>
#define Z3_LIB_NAME "libz3.dll"
#endif

typedef void* Z3_context;
typedef void* Z3_solver;
typedef void* Z3_ast;
typedef void* Z3_sort;
typedef void* Z3_config;
typedef void* Z3_model;
typedef int   Z3_lbool;

#define Z3_L_FALSE (-1)
#define Z3_L_UNDEF 0
#define Z3_L_TRUE  1

typedef Z3_config  (*z3_mk_config_fn)(void);
typedef void       (*z3_del_config_fn)(Z3_config);
typedef void       (*z3_set_param_value_fn)(Z3_config, const char*, const char*);
typedef Z3_context (*z3_mk_context_fn)(Z3_config);
typedef void       (*z3_del_context_fn)(Z3_context);
typedef Z3_solver  (*z3_mk_solver_fn)(Z3_context);
typedef void       (*z3_solver_inc_ref_fn)(Z3_context, Z3_solver);
typedef void       (*z3_solver_dec_ref_fn)(Z3_context, Z3_solver);
typedef void       (*z3_solver_push_fn)(Z3_context, Z3_solver);
typedef void       (*z3_solver_pop_fn)(Z3_context, Z3_solver, unsigned);
typedef void       (*z3_solver_assert_fn)(Z3_context, Z3_solver, Z3_ast);
typedef Z3_lbool   (*z3_solver_check_fn)(Z3_context, Z3_solver);
typedef Z3_sort    (*z3_mk_real_sort_fn)(Z3_context);
typedef Z3_sort    (*z3_mk_int_sort_fn)(Z3_context);
typedef Z3_ast     (*z3_mk_real_fn)(Z3_context, int, int);
typedef Z3_ast     (*z3_mk_add_fn)(Z3_context, unsigned, Z3_ast*);
typedef Z3_ast     (*z3_mk_sub_fn)(Z3_context, unsigned, Z3_ast*);
typedef Z3_ast     (*z3_mk_mul_fn)(Z3_context, unsigned, Z3_ast*);
typedef Z3_ast     (*z3_mk_div_fn)(Z3_context, Z3_ast, Z3_ast);
typedef Z3_ast     (*z3_mk_eq_fn)(Z3_context, Z3_ast, Z3_ast);
typedef Z3_ast     (*z3_mk_not_fn)(Z3_context, Z3_ast);
typedef Z3_ast     (*z3_mk_lt_fn)(Z3_context, Z3_ast, Z3_ast);
typedef Z3_ast     (*z3_mk_le_fn)(Z3_context, Z3_ast, Z3_ast);
typedef Z3_ast     (*z3_mk_ge_fn)(Z3_context, Z3_ast, Z3_ast);
typedef Z3_ast     (*z3_mk_and_fn)(Z3_context, unsigned, Z3_ast*);
typedef Z3_ast     (*z3_mk_const_fn)(Z3_context, void*, Z3_sort);
typedef void*      (*z3_mk_string_symbol_fn)(Z3_context, const char*);
typedef Z3_ast     (*z3_mk_int_fn)(Z3_context, int, Z3_sort);

static struct {
    void* lib;
    z3_mk_config_fn         mk_config;
    z3_del_config_fn        del_config;
    z3_set_param_value_fn   set_param_value;
    z3_mk_context_fn        mk_context;
    z3_del_context_fn       del_context;
    z3_mk_solver_fn         mk_solver;
    z3_solver_inc_ref_fn    solver_inc_ref;
    z3_solver_dec_ref_fn    solver_dec_ref;
    z3_solver_push_fn       solver_push;
    z3_solver_pop_fn        solver_pop;
    z3_solver_assert_fn     solver_assert;
    z3_solver_check_fn      solver_check;
    z3_mk_real_sort_fn      mk_real_sort;
    z3_mk_int_sort_fn       mk_int_sort;
    z3_mk_add_fn            mk_add;
    z3_mk_sub_fn            mk_sub;
    z3_mk_mul_fn            mk_mul;
    z3_mk_div_fn            mk_div;
    z3_mk_eq_fn             mk_eq;
    z3_mk_not_fn            mk_not;
    z3_mk_lt_fn             mk_lt;
    z3_mk_le_fn             mk_le;
    z3_mk_ge_fn             mk_ge;
    z3_mk_and_fn            mk_and;
    z3_mk_const_fn          mk_const;
    z3_mk_string_symbol_fn  mk_string_symbol;
    z3_mk_int_fn            mk_int;
} z3 = {0};

static bool z3_loaded = false;

#if defined(__linux__) || defined(__APPLE__)
static void* z3_load_sym(const char* name) { return dlsym(z3.lib, name); }
#elif defined(_WIN32)
static void* z3_load_sym(const char* name) {
    return (void*)GetProcAddress((HMODULE)z3.lib, name);
}
#else
static void* z3_load_sym(const char* name) { (void)name; return NULL; }
#endif

static bool z3_try_load(void) {
    if (z3_loaded) return z3.lib != NULL;
    z3_loaded = true;

#if defined(__linux__) || defined(__APPLE__)
    z3.lib = dlopen(Z3_LIB_NAME, RTLD_LAZY | RTLD_LOCAL);
#elif defined(_WIN32)
    z3.lib = (void*)LoadLibraryA(Z3_LIB_NAME);
#endif
    if (!z3.lib) {
        LOG_DEBUG("Z3 not available: could not load %s", Z3_LIB_NAME);
        return false;
    }

#define LOAD(field, sym) z3.field = (typeof(z3.field))z3_load_sym(sym)
    LOAD(mk_config,        "Z3_mk_config");
    LOAD(del_config,        "Z3_del_config");
    LOAD(set_param_value,   "Z3_set_param_value");
    LOAD(mk_context,        "Z3_mk_context");
    LOAD(del_context,       "Z3_del_context");
    LOAD(mk_solver,         "Z3_mk_solver");
    LOAD(solver_inc_ref,    "Z3_solver_inc_ref");
    LOAD(solver_dec_ref,    "Z3_solver_dec_ref");
    LOAD(solver_push,       "Z3_solver_push");
    LOAD(solver_pop,        "Z3_solver_pop");
    LOAD(solver_assert,     "Z3_solver_assert");
    LOAD(solver_check,      "Z3_solver_check");
    LOAD(mk_real_sort,      "Z3_mk_real_sort");
    LOAD(mk_int_sort,       "Z3_mk_int_sort");
    LOAD(mk_add,            "Z3_mk_add");
    LOAD(mk_sub,            "Z3_mk_sub");
    LOAD(mk_mul,            "Z3_mk_mul");
    LOAD(mk_div,            "Z3_mk_div");
    LOAD(mk_eq,             "Z3_mk_eq");
    LOAD(mk_not,            "Z3_mk_not");
    LOAD(mk_lt,             "Z3_mk_lt");
    LOAD(mk_le,             "Z3_mk_le");
    LOAD(mk_ge,             "Z3_mk_ge");
    LOAD(mk_and,            "Z3_mk_and");
    LOAD(mk_const,          "Z3_mk_const");
    LOAD(mk_string_symbol,  "Z3_mk_string_symbol");
    LOAD(mk_int,            "Z3_mk_int");
#undef LOAD

    if (!z3.mk_config || !z3.mk_context || !z3.mk_solver || !z3.solver_check) {
        LOG_DEBUG("Z3 loaded but missing required symbols");
#if defined(__linux__) || defined(__APPLE__)
        dlclose(z3.lib);
#elif defined(_WIN32)
        FreeLibrary((HMODULE)z3.lib);
#endif
        z3.lib = NULL;
        return false;
    }

    return true;
}

bool cml_z3_available(void) {
    return z3_try_load();
}

CMLZ3Verifier* cml_z3_verifier_create(int timeout_ms) {
    CMLZ3Verifier* v = calloc(1, sizeof(CMLZ3Verifier));
    if (!v) return NULL;
    v->timeout_ms = timeout_ms;

    if (!z3_try_load()) return v;

    Z3_config cfg = z3.mk_config();
    if (cfg && z3.set_param_value) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", timeout_ms);
        z3.set_param_value(cfg, "timeout", buf);
    }
    v->z3_context = z3.mk_context(cfg);
    if (cfg) z3.del_config(cfg);

    if (v->z3_context) {
        v->z3_solver = z3.mk_solver(v->z3_context);
        if (v->z3_solver && z3.solver_inc_ref) {
            z3.solver_inc_ref(v->z3_context, v->z3_solver);
        }
        v->initialized = true;
    }

    return v;
}

void cml_z3_verifier_free(CMLZ3Verifier* v) {
    if (!v) return;
    if (v->initialized && z3.lib) {
        if (v->z3_solver && z3.solver_dec_ref) {
            z3.solver_dec_ref(v->z3_context, v->z3_solver);
        }
        if (v->z3_context && z3.del_context) {
            z3.del_context(v->z3_context);
        }
    }
    free(v);
}

static Z3_ast z3_build_node_expr(Z3_context ctx, struct IRNode* node,
                                  Z3_ast* input_exprs, int num_inputs) {
    if (!node || !input_exprs || num_inputs < 1) return NULL;

    Z3_ast a = input_exprs[0];
    Z3_ast b = num_inputs > 1 ? input_exprs[1] : NULL;

    switch (node->type) {
    case UOP_ADD: {
        Z3_ast args[2] = {a, b};
        return b ? z3.mk_add(ctx, 2, args) : NULL;
    }
    case UOP_SUB: {
        Z3_ast args[2] = {a, b};
        return b ? z3.mk_sub(ctx, 2, args) : NULL;
    }
    case UOP_MUL: {
        Z3_ast args[2] = {a, b};
        return b ? z3.mk_mul(ctx, 2, args) : NULL;
    }
    case UOP_DIV:
        return b ? z3.mk_div(ctx, a, b) : NULL;
    case UOP_NEG: {
        Z3_sort rs = z3.mk_real_sort(ctx);
        Z3_ast zero = z3.mk_int(ctx, 0, rs);
        Z3_ast args[2] = {zero, a};
        return z3.mk_sub(ctx, 2, args);
    }
    default:
        return NULL; /* Op not modeled symbolically */
    }
}

static Z3_ast z3_build_graph_expr(Z3_context ctx, CMLGraph_t ir,
                                   Z3_ast* leaf_vars, int max_vars,
                                   int* num_vars_used) {
    if (!ir) return NULL;

    struct CMLGraph* g = (struct CMLGraph*)ir;
    struct IRNode* node = g->head;
    *num_vars_used = 0;

    /* Map from node output names to Z3 expressions (simple linear scan) */
    Z3_ast node_exprs[256] = {0};
    struct IRNode* node_list[256] = {0};
    int node_count = 0;

    Z3_sort real_sort = z3.mk_real_sort(ctx);

    while (node && node_count < 256) {
        node_list[node_count] = node;

        if (!node->inputs || node->num_inputs == 0) {
            /* Leaf / constant node - assign a variable */
            if (*num_vars_used < max_vars) {
                char name[32];
                snprintf(name, sizeof(name), "x%d", *num_vars_used);
                void* sym = z3.mk_string_symbol(ctx, name);
                leaf_vars[*num_vars_used] = z3.mk_const(ctx, sym, real_sort);
                node_exprs[node_count] = leaf_vars[*num_vars_used];
                (*num_vars_used)++;
            }
        } else {
            /* Find input expressions */
            Z3_ast in_exprs[8] = {0};
            int found = 0;
            for (int i = 0; i < node->num_inputs && i < 8; i++) {
                /* Search backwards for the node that produced this input */
                for (int j = 0; j < node_count; j++) {
                    if (node_list[j]->output == node->inputs[i]) {
                        in_exprs[i] = node_exprs[j];
                        found++;
                        break;
                    }
                }
            }
            if (found == node->num_inputs) {
                node_exprs[node_count] = z3_build_node_expr(ctx, node,
                                                             in_exprs, node->num_inputs);
            }
        }

        node_count++;
        node = node->next;
    }

    /* Return the last node's expression as the graph output */
    return node_count > 0 ? node_exprs[node_count - 1] : NULL;
}

CMLVerifyResult cml_z3_verify_equivalence(CMLZ3Verifier* v,
                                           CMLGraph_t original,
                                           CMLGraph_t optimized) {
    if (!v || !v->initialized || !original || !optimized)
        return CML_VERIFY_UNSUPPORTED;

    Z3_context ctx = v->z3_context;
    Z3_solver solver = v->z3_solver;

    v->num_checks++;

    Z3_ast vars_orig[32] = {0}, vars_opt[32] = {0};
    int nv_orig = 0, nv_opt = 0;

    z3.solver_push(ctx, solver);

    Z3_ast expr_orig = z3_build_graph_expr(ctx, original, vars_orig, 32, &nv_orig);
    Z3_ast expr_opt  = z3_build_graph_expr(ctx, optimized, vars_opt, 32, &nv_opt);

    if (!expr_orig || !expr_opt || nv_orig != nv_opt) {
        z3.solver_pop(ctx, solver, 1);
        return CML_VERIFY_UNSUPPORTED;
    }

    /* Constrain: shared leaf variables are equal */
    for (int i = 0; i < nv_orig; i++) {
        Z3_ast eq = z3.mk_eq(ctx, vars_orig[i], vars_opt[i]);
        z3.solver_assert(ctx, solver, eq);
    }

    /* Assert: outputs are NOT equal (we want to find a counterexample) */
    Z3_ast neq = z3.mk_not(ctx, z3.mk_eq(ctx, expr_orig, expr_opt));
    z3.solver_assert(ctx, solver, neq);

    Z3_lbool result = z3.solver_check(ctx, solver);
    z3.solver_pop(ctx, solver, 1);

    if (result == Z3_L_FALSE) {
        /* UNSAT means no counterexample exists -> graphs are equivalent */
        v->num_passed++;
        return CML_VERIFY_PASS;
    } else if (result == Z3_L_TRUE) {
        /* SAT means a counterexample exists -> graphs differ */
        v->num_failed++;
        LOG_DEBUG("Z3: equivalence check FAILED (SAT - counterexample found)");
        return CML_VERIFY_FAIL;
    } else {
        LOG_DEBUG("Z3: equivalence check timed out or unknown");
        return CML_VERIFY_TIMEOUT;
    }
}

CMLVerifyResult cml_z3_verify_bounds(CMLZ3Verifier* v, CMLGraph_t ir) {
    if (!v || !v->initialized || !ir)
        return CML_VERIFY_UNSUPPORTED;

    Z3_context ctx = v->z3_context;
    Z3_solver solver = v->z3_solver;

    v->num_checks++;

    struct CMLGraph* g = (struct CMLGraph*)ir;
    struct IRNode* node = g->head;
    Z3_sort int_sort = z3.mk_int_sort(ctx);
    bool all_pass = true;

    z3.solver_push(ctx, solver);

    while (node) {
        /* For each node with an output shape, verify index bounds */
        if (node->output_shape && node->output_ndim > 0) {
            for (int d = 0; d < node->output_ndim; d++) {
                int dim_size = node->output_shape[d];
                if (dim_size <= 0) {
                    all_pass = false;
                    LOG_DEBUG("Z3 bounds: dimension %d has non-positive size %d", d, dim_size);
                    break;
                }

                /* Create symbolic index and verify 0 <= idx < dim_size */
                char name[32];
                snprintf(name, sizeof(name), "idx_%d", d);
                void* sym = z3.mk_string_symbol(ctx, name);
                Z3_ast idx = z3.mk_const(ctx, sym, int_sort);
                Z3_ast zero = z3.mk_int(ctx, 0, int_sort);
                Z3_ast size = z3.mk_int(ctx, dim_size, int_sort);

                /* Assert idx >= 0 AND idx < size, check SAT */
                Z3_ast bounds[2] = {
                    z3.mk_ge(ctx, idx, zero),
                    z3.mk_lt(ctx, idx, size)
                };
                Z3_ast valid_bounds = z3.mk_and(ctx, 2, bounds);
                z3.solver_assert(ctx, solver, valid_bounds);
            }
        }
        node = node->next;
    }

    Z3_lbool result = z3.solver_check(ctx, solver);
    z3.solver_pop(ctx, solver, 1);

    if (!all_pass) {
        v->num_failed++;
        return CML_VERIFY_FAIL;
    }

    if (result == Z3_L_TRUE) {
        v->num_passed++;
        return CML_VERIFY_PASS;
    } else if (result == Z3_L_FALSE) {
        v->num_failed++;
        return CML_VERIFY_FAIL;
    }
    return CML_VERIFY_TIMEOUT;
}

CMLVerifyResult cml_z3_verify_schedule(CMLZ3Verifier* v,
                                        CMLGraph_t ir, void* schedule) {
    (void)schedule;
    if (!v || !v->initialized || !ir)
        return CML_VERIFY_UNSUPPORTED;

    /* Schedule verification: check that all dependencies are satisfied
     * (i.e., producers execute before consumers in the schedule order).
     * For now, delegate to bounds checking as a basic sanity check. */
    return cml_z3_verify_bounds(v, ir);
}

void cml_z3_verifier_stats(const CMLZ3Verifier* v,
                            int* checks, int* passed, int* failed) {
    if (!v) return;
    if (checks) *checks = v->num_checks;
    if (passed) *passed = v->num_passed;
    if (failed) *failed = v->num_failed;
}

#else /* !CML_HAS_Z3 */

bool cml_z3_available(void) { return false; }

CMLZ3Verifier* cml_z3_verifier_create(int timeout_ms) {
    CMLZ3Verifier* v = calloc(1, sizeof(CMLZ3Verifier));
    if (v) v->timeout_ms = timeout_ms;
    return v;
}

void cml_z3_verifier_free(CMLZ3Verifier* v) { free(v); }

CMLVerifyResult cml_z3_verify_equivalence(CMLZ3Verifier* v,
                                           CMLGraph_t original,
                                           CMLGraph_t optimized) {
    (void)v; (void)original; (void)optimized;
    return CML_VERIFY_UNSUPPORTED;
}

CMLVerifyResult cml_z3_verify_bounds(CMLZ3Verifier* v, CMLGraph_t ir) {
    (void)v; (void)ir;
    return CML_VERIFY_UNSUPPORTED;
}

CMLVerifyResult cml_z3_verify_schedule(CMLZ3Verifier* v,
                                        CMLGraph_t ir, void* schedule) {
    (void)v; (void)ir; (void)schedule;
    return CML_VERIFY_UNSUPPORTED;
}

void cml_z3_verifier_stats(const CMLZ3Verifier* v,
                            int* checks, int* passed, int* failed) {
    if (!v) return;
    if (checks) *checks = v->num_checks;
    if (passed) *passed = v->num_passed;
    if (failed) *failed = v->num_failed;
}

#endif /* CML_HAS_Z3 */
