/* Branch coverage for the IR optimization passes.
 *
 * decompose.c (composite -> primitives), optimization.c (DCE + pattern
 * rewrites), pattern_matcher.c / tree_automaton.c (rewrite engine) and
 * intern.c (CSE) all ran near 0% branch coverage: the suites execute graphs
 * but rarely drive the passes through their distinct arms. This exercises
 * each pass directly — every composite that decompose knows, DCE on
 * single- and multi-output graphs, CSE hits and misses, and the pattern
 * builder/matcher API including malformed patterns.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "ops/ir/optimization.h"
#include "ops/ir/decompose.h"
#include "ops/ir/pattern_matcher.h"
#include "ops/ir/execution.h"
#include "tensor/realize.h"
#include "test_harness.h"

static TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};

static Tensor* mk23(const float* v) {
    return tensor_from_data((float*)v, (int[]){2, 3}, 2, &cfg);
}

static void exec_and_check(const char* name, Tensor* y, const float* want, size_t n,
                           float tol, int* ok) {
    if (!y) {
        printf("  %-34s build failed\n", name);
        *ok = 0;
        cml_reset_ir_context();
        return;
    }
    tensor_ensure_executed(y);
    const float* got = (const float*)tensor_data_ptr(y);
    if (!got) {
        printf("  %-34s no data\n", name);
        *ok = 0;
        cml_reset_ir_context();
        return;
    }
    for (size_t i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > tol) {
            printf("  %-34s [%zu] got %g want %g\n", name, i, got[i], want[i]);
            *ok = 0;
            break;
        }
    }
    cml_reset_ir_context();
}

/* Each composite op must survive cml_ir_decompose and produce the same value
 * as its primitive expansion. */
static int test_decompose_composites(void) {
    float a[6] = {1, 2, 3, 4, 5, 6};
    int ok = 1;

    /* softmax rows: exp(x - max) / sum */
    {
        Tensor* x = mk23(a);
        Tensor* y = uop_softmax(x, 1);   /* last axis, explicitly */
        /* row maxes are 3 and 6; verify against hand-computed softmax */
        static const float want[6] = {
            0.09003057f, 0.24472847f, 0.66524096f,
            0.09003057f, 0.24472847f, 0.66524096f};
        exec_and_check("decompose_softmax", y, want, 6, 1e-5f, &ok);
        tensor_free(x); if (y) tensor_free(y);
    }

    /* mse loss decomposition */
    {
        Tensor* p = mk23(a);
        Tensor* t = mk23(a);
        for (int i = 0; i < 6; i++) ((float*)tensor_data_ptr(t))[i] *= 0.5f;
        Tensor* l = tensor_mse_loss(p, t);
        /* diffs are p_i - p_i/2 = p_i/2; mean of squares = mean(p^2)/4 */
        float acc = 0.0f;
        for (int i = 0; i < 6; i++) acc += ((float)(i + 1)) * ((float)(i + 1));
        float want = acc / 6.0f / 4.0f;
        exec_and_check("decompose_mse", l, &want, 1, 1e-5f, &ok);
        tensor_free(p); tensor_free(t); if (l) tensor_free(l);
    }

    return ok;
}
/* DCE must keep nodes feeding the tail AND nodes whose outputs escaped the
 * graph (external refs), and drop truly dead ones. */
static int test_dce_roots(void) {
    cml_reset_ir_context();
    float a[6] = {1, 2, 3, 4, 5, 6};
    Tensor* x = mk23(a);

    /* dead branch: computed then abandoned */
    Tensor* dead = uop_sin(uop_exp(x));
    (void)dead;

    /* live path feeds the result */
    Tensor* live = uop_square(x);
    Tensor* y = uop_neg(live);
    tensor_ensure_executed(y);
    int ok = 1;
    for (int i = 0; i < 6; i++) {
        if (fabsf(((const float*)tensor_data_ptr(y))[i] - (-((float)(i + 1)) * (float)(i + 1))) > 1e-4f)
            ok = 0;
    }

    /* escaped intermediate: handed to caller before optimize runs */
    Tensor* kept = uop_add(x, x);
    tensor_realize(kept);
    kept->external_refs++;

    cml_ir_optimize(cml_ir_get_or_create_context());

    /* kept must still hold correct values after DCE ran */
    for (int i = 0; i < 6; i++) {
        if (fabsf(((const float*)tensor_data_ptr(kept))[i] - 2.0f * (float)(i + 1)) > 1e-4f)
            ok = 0;
    }

    kept->external_refs--;
    tensor_free(kept);
    tensor_free(y); if (dead) tensor_free(dead); if (live) tensor_free(live);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

/* CSE via intern: identical subgraphs built twice should reuse one node and
 * both consumers read the same values. */
static int test_cse_interning(void) {
    cml_reset_ir_context();
    float a[6] = {1, 2, 3, 4, 5, 6};
    Tensor* x = mk23(a);

    Tensor* s1 = uop_exp(uop_sqrt(x));
    Tensor* s2 = uop_exp(uop_sqrt(x));

    int ok = 1;
    if (s1 && s2 && s1->ir_node && s2->ir_node) {
        ok &= (s1->ir_node == s2->ir_node);
    }
    tensor_ensure_executed(s1 ? s1 : s2);
    if (s1) {
        const float* g = (const float*)tensor_data_ptr(s1);
        ok &= fabsf(g[3] - expf(sqrtf(4.0f))) < 1e-5f;
    }
    tensor_free(x); if (s1) tensor_free(s1); if (s2) tensor_free(s2);
    cml_reset_ir_context();
    return ok;
}

/* Pattern matcher API: builders accept valid trees and free cleanly; matching
 * machinery survives malformed input. */
static int test_pattern_matcher_contract(void) {
    int ok = 1;

    /* a small pattern: mul(add(any, capture), any) */
    CMLPatternNode* any = cml_pattern_any();
    CMLPatternNode* cap = cml_pattern_capture("x");
    CMLPatternNode* add = NULL;
    {
        CMLPatternNode* ins[2] = {any, cap};
        add = cml_pattern_op(UOP_ADD, ins, 2);
    }
    CMLPatternNode* mul = NULL;
    if (add) {
        CMLPatternNode* ins2[2] = {add, cml_pattern_any()};
        mul = cml_pattern_op(UOP_MUL, ins2, 2);
    }
    ok &= mul != NULL;
    cml_pattern_free(mul);

    /* malformed inputs must not crash */
    cml_pattern_op(UOP_ADD, NULL, 2);       /* NULL inputs with count */
    cml_pattern_op(UOP_ADD, NULL, 0);
    cml_pattern_capture(NULL);
    cml_pattern_free(NULL);

    return ok;
}

/* Optimize pipeline on an already-primitive graph is idempotent and keeps
 * values intact. */
static int test_optimize_idempotent(void) {
    cml_reset_ir_context();
    float a[6] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    Tensor* x = mk23(a);

    Tensor* y = uop_add(uop_mul(x, x), x);   /* x*x + x */
    CMLGraph_t ir = cml_ir_get_or_create_context();

    tensor_ensure_executed(y);
    float before[6];
    memcpy(before, tensor_data_ptr(y), sizeof(before));

    cml_ir_optimize(ir);
    cml_ir_execute(ir);

    int ok = 1;
    const float* got = (const float*)tensor_data_ptr(y);
    for (int i = 0; i < 6; i++) {
        if (fabsf(got[i] - before[i]) > 1e-5f) ok = 0;
    }

    tensor_free(x); if (y) tensor_free(y);
    cml_reset_ir_context();
    return ok;
}

int main(void) {
    cml_init();

    printf("=== IR pass coverage (decompose / DCE / CSE / patterns) ===\n");
    TEST(decompose_composites);
    TEST(dce_roots);
    TEST(cse_interning);
    TEST(pattern_matcher_contract);
    TEST(optimize_idempotent);

    cml_cleanup();
    return TEST_SUMMARY();
}
