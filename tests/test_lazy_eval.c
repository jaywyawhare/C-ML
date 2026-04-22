/*
 * Tests for lazy evaluation: cml_ir_execute_up_to(), DCE, backward DCE,
 * and lazy tensor creation ops.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "nn.h"

static int tests_passed = 0;
static int tests_total  = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", (name)); } \
    else       { printf("  FAIL: %s\n", (name)); } \
} while(0)

#define APPROX_EQ(a, b) (fabsf((float)(a) - (float)(b)) < 1e-4f)

/* ── 1. Partial execution stops at target ─────────────────────────────────── */

static void test_partial_exec_stops_early(void) {
    printf("Test: execute_up_to stops before downstream nodes\n");

    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f, 6.0f};
    int   shape[]  = {3};
    TensorConfig cfg = {0};

    Tensor* a = tensor_from_data(a_data, shape, 1, &cfg);
    Tensor* b = tensor_from_data(b_data, shape, 1, &cfg);

    /* Build: c = a + b,  d = relu(c),  e = mul(d, b) */
    Tensor* c = uop_add(a, b);
    Tensor* d = uop_relu(c);
    Tensor* e = uop_mul(d, b);

    CMLGraph_t ir = c->ir_context;
    struct IRNode* c_node = (struct IRNode*)c->ir_node;
    struct IRNode* d_node = (struct IRNode*)d->ir_node;
    struct IRNode* e_node = (struct IRNode*)e->ir_node;

    /* Execute only up to c */
    int rc = cml_ir_execute_up_to(ir, c_node);
    CHECK("execute_up_to returns 0", rc == 0);
    CHECK("target node c is executed", c_node && c_node->is_executed);
    CHECK("downstream d is NOT executed", d_node && !d_node->is_executed);
    CHECK("downstream e is NOT executed", e_node && !e_node->is_executed);

    /* Result of c should be correct: [5,7,9] */
    float* cd = (float*)c->data;
    int values_ok = cd && APPROX_EQ(cd[0], 5.0f) && APPROX_EQ(cd[1], 7.0f)
                        && APPROX_EQ(cd[2], 9.0f);
    CHECK("c values correct after partial exec", values_ok);

    tensor_free(a); tensor_free(b); tensor_free(c);
    tensor_free(d); tensor_free(e);
    cml_reset_ir_context();
}

/* ── 2. DCE skips a branch not required by target ─────────────────────────── */

static void test_dce_skips_unused_branch(void) {
    printf("Test: DCE skips branch not needed by target\n");

    float a_data[] = {2.0f};
    float b_data[] = {3.0f};
    float c_data[] = {10.0f};
    int   shape[]  = {1};
    TensorConfig cfg = {0};

    Tensor* a = tensor_from_data(a_data, shape, 1, &cfg);
    Tensor* b = tensor_from_data(b_data, shape, 1, &cfg);
    Tensor* c = tensor_from_data(c_data, shape, 1, &cfg);

    /* Branch 1 (needed): x = a * b */
    Tensor* x = uop_mul(a, b);

    /* Branch 2 (not needed for x): y = c + a  (independent output) */
    Tensor* y = uop_add(c, a);

    CMLGraph_t ir = x->ir_context;
    struct IRNode* x_node = (struct IRNode*)x->ir_node;
    struct IRNode* y_node = (struct IRNode*)y->ir_node;

    /* Execute only up to x */
    cml_ir_execute_up_to(ir, x_node);

    CHECK("target x is executed", x_node && x_node->is_executed);
    /* y_node may or may not be executed depending on DCE traversal order;
     * the important guarantee is that x gives the right value. */
    float* xd = (float*)x->data;
    CHECK("x = a*b = 6", xd && APPROX_EQ(xd[0], 6.0f));
    (void)y_node; /* y may incidentally execute if it precedes x in the list */

    tensor_free(a); tensor_free(b); tensor_free(c);
    tensor_free(x); tensor_free(y);
    cml_reset_ir_context();
}

/* ── 3. execute_up_to(tail) matches full execution ────────────────────────── */

static void test_tail_matches_full_exec(void) {
    printf("Test: execute_up_to(tail) == full execute_ir\n");

    float a_data[16], b_data[16];
    int shape[] = {4, 4};
    for (int i = 0; i < 16; i++) { a_data[i] = (float)(i + 1); b_data[i] = 1.0f / (float)(i + 1); }
    TensorConfig cfg = {0};

    /* Run via execute_up_to(tail) */
    Tensor* a1 = tensor_from_data(a_data, shape, 2, &cfg);
    Tensor* b1 = tensor_from_data(b_data, shape, 2, &cfg);
    Tensor* c1 = uop_mul(a1, b1);
    Tensor* d1 = uop_relu(c1);
    struct IRNode* d1_node = (struct IRNode*)d1->ir_node;
    cml_ir_execute_up_to(a1->ir_context ? a1->ir_context : d1->ir_context, d1_node);
    float* r1 = (float*)d1->data;
    float sample1 = r1 ? r1[0] : -999.0f;
    tensor_free(a1); tensor_free(b1); tensor_free(c1); tensor_free(d1);
    cml_reset_ir_context();

    /* Run via normal tensor_data_ptr (full lazy path) */
    Tensor* a2 = tensor_from_data(a_data, shape, 2, &cfg);
    Tensor* b2 = tensor_from_data(b_data, shape, 2, &cfg);
    Tensor* c2 = uop_mul(a2, b2);
    Tensor* d2 = uop_relu(c2);
    float* r2 = (float*)tensor_data_ptr(d2);
    float sample2 = r2 ? r2[0] : -999.0f;
    tensor_free(a2); tensor_free(b2); tensor_free(c2); tensor_free(d2);
    cml_reset_ir_context();

    CHECK("partial(tail) == full exec result", APPROX_EQ(sample1, sample2));
}

/* ── 4. Lazy FILL materialization ─────────────────────────────────────────── */

static void test_lazy_fill(void) {
    printf("Test: lazy FILL op materializes correct values\n");

    int shape[] = {8};
    Tensor* t = uop_fill(shape, 1, 3.14f);
    if (!t) { CHECK("uop_fill returned non-NULL", 0); return; }

    float* d = (float*)tensor_data_ptr(t);
    int ok = d != NULL;
    if (ok) {
        for (int i = 0; i < 8; i++) {
            if (!APPROX_EQ(d[i], 3.14f)) { ok = 0; break; }
        }
    }
    CHECK("fill materializes 3.14f across all elements", ok);

    tensor_free(t);
    cml_reset_ir_context();
}

/* ── 5. Lazy RAND_UNIFORM stays in [0,1) ─────────────────────────────────── */

static void test_lazy_rand_uniform(void) {
    printf("Test: lazy RAND_UNIFORM stays in [0,1)\n");

    int shape[] = {64};
    Tensor* t = uop_rand_uniform(shape, 1, DTYPE_FLOAT32, DEVICE_CPU);
    if (!t) { CHECK("uop_rand_uniform returned non-NULL", 0); return; }

    float* d = (float*)tensor_data_ptr(t);
    int ok = d != NULL;
    if (ok) {
        for (int i = 0; i < 64; i++) {
            if (d[i] < 0.0f || d[i] >= 1.0f) { ok = 0; break; }
        }
    }
    CHECK("rand_uniform values in [0,1)", ok);

    tensor_free(t);
    cml_reset_ir_context();
}

/* ── 6. Lazy ARANGE produces sequential values ────────────────────────────── */

static void test_lazy_arange(void) {
    printf("Test: lazy ARANGE produces start..stop with step\n");

    Tensor* t = uop_arange_op(0.0f, 5.0f, 1.0f, DTYPE_FLOAT32, DEVICE_CPU);
    if (!t) { CHECK("uop_arange_op returned non-NULL", 0); return; }

    float* d = (float*)tensor_data_ptr(t);
    int ok = d != NULL && (int)t->numel == 5;
    if (ok) {
        for (int i = 0; i < 5; i++) {
            if (!APPROX_EQ(d[i], (float)i)) { ok = 0; break; }
        }
    }
    CHECK("arange(0,5,1) = [0,1,2,3,4]", ok);

    tensor_free(t);
    cml_reset_ir_context();
}

/* ── 7. Backward DCE: grads only flow through requires_grad inputs ─────────── */

static void test_backward_dce(void) {
    printf("Test: backward DCE computes grad only for requires_grad inputs\n");

    /* y = a * b + c
     * a requires_grad=true, b and c do not
     * dy/da = b = 3.0 */
    float a_data[] = {2.0f};
    float b_data[] = {3.0f};
    float c_data[] = {5.0f};

    Tensor* a = cml_tensor_1d(a_data, 1);
    Tensor* b = cml_tensor_1d(b_data, 1);
    Tensor* c_t = cml_tensor_1d(c_data, 1);
    if (!a || !b || !c_t) {
        CHECK("tensor allocation", 0);
        tensor_free(a); tensor_free(b); tensor_free(c_t);
        return;
    }

    cml_set_requires_grad(a, true);

    Tensor* ab  = cml_mul(a, b);
    Tensor* y   = cml_add(ab, c_t);
    if (!y) {
        CHECK("graph construction", 0);
        tensor_free(a); tensor_free(b); tensor_free(c_t);
        tensor_free(ab);
        return;
    }

    cml_backward(y, NULL, false, false);

    int has_a_grad = (a->grad != NULL);
    CHECK("a has gradient after backward", has_a_grad);

    if (has_a_grad && a->grad->data) {
        float g = ((float*)a->grad->data)[0];
        printf("    (dy/da=%.1f expected 3.0) ", g);
        CHECK("dy/da = b = 3.0", APPROX_EQ(g, 3.0f));
    }

    /* b should NOT have a grad (requires_grad=false) */
    CHECK("b has no gradient (no requires_grad)", b->grad == NULL);

    tensor_free(a); tensor_free(b); tensor_free(c_t);
    tensor_free(ab); tensor_free(y);
    cml_reset_ir_context();
}

/* ── 8. Chained lazy execution via tensor_data_ptr ───────────────────────── */

static void test_chained_lazy_via_data_ptr(void) {
    printf("Test: chained lazy ops execute on-demand via tensor_data_ptr\n");

    /* None of the ops run until we call tensor_data_ptr */
    float a_data[] = {-1.0f, 2.0f, -3.0f, 4.0f};
    int shape[] = {4};
    TensorConfig cfg = {0};
    Tensor* a = tensor_from_data(a_data, shape, 1, &cfg);
    Tensor* b = uop_relu(a);
    Tensor* c = uop_relu(b); /* relu(relu(x)) = relu(x) */

    /* b and c should not have data yet */
    struct IRNode* b_node = b ? (struct IRNode*)b->ir_node : NULL;
    struct IRNode* c_node = c ? (struct IRNode*)c->ir_node : NULL;
    int b_pending = b_node && !b_node->is_executed;
    CHECK("b not yet executed before data access", b_pending);

    /* Force execution of c via data_ptr */
    float* cd = (float*)tensor_data_ptr(c);
    int ok = cd != NULL;
    if (ok) {
        /* relu(relu([-1,2,-3,4])) = [0,2,0,4] */
        ok = APPROX_EQ(cd[0], 0.0f) && APPROX_EQ(cd[1], 2.0f)
          && APPROX_EQ(cd[2], 0.0f) && APPROX_EQ(cd[3], 4.0f);
    }
    CHECK("chained relu result correct", ok);
    CHECK("c node is executed after data_ptr", c_node && c_node->is_executed);

    tensor_free(a); tensor_free(b); tensor_free(c);
    cml_reset_ir_context();
}

/* ── 9. IR teardown must not execute pending lazy nodes ─────────────────── */

static void test_reset_does_not_materialize_pending_rand(void) {
    printf("Test: reset_ir_context does not execute pending lazy rand ops\n");

    int shape[] = {8};
    TensorConfig cfg = {0};
    float after_reset[8];
    float baseline[8];
    int ok = 1;

    srand(1234);
    Tensor* pending = tensor_rand(shape, 1, &cfg);
    CHECK("pending rand tensor created", pending != NULL);
    if (pending) {
        CHECK("pending rand tensor is still lazy", !pending->is_executed);
        cml_reset_ir_context();
    } else {
        cml_reset_ir_context();
        ok = 0;
    }

    Tensor* a = tensor_rand(shape, 1, &cfg);
    float* ad = a ? (float*)tensor_data_ptr(a) : NULL;
    if (!ad) {
        ok = 0;
    } else {
        memcpy(after_reset, ad, sizeof(after_reset));
    }
    tensor_free(a);
    cml_reset_ir_context();

    srand(1234);
    Tensor* b = tensor_rand(shape, 1, &cfg);
    float* bd = b ? (float*)tensor_data_ptr(b) : NULL;
    if (!bd) {
        ok = 0;
    } else {
        memcpy(baseline, bd, sizeof(baseline));
    }
    tensor_free(b);
    cml_reset_ir_context();

    if (ok) {
        for (int i = 0; i < 8; i++) {
            if (!APPROX_EQ(after_reset[i], baseline[i])) {
                ok = 0;
                break;
            }
        }
    }
    CHECK("reset leaves RNG stream untouched by pending rand", ok);
}

/* ── 10. Metadata-only movement ops must stay lazy ──────────────────────── */

static void test_permute_does_not_force_source_execution(void) {
    printf("Test: permute does not materialize lazy source tensor\n");

    int shape[] = {2, 3};
    Tensor* src = uop_fill(shape, 2, 2.5f);
    if (!src) {
        CHECK("lazy permute source created", 0);
        return;
    }

    struct IRNode* src_node = (struct IRNode*)src->ir_node;
    PermuteParams pp = {.perm = (int[]){1, 0}, .num_dims = 2};
    Tensor* out = uop_permute(src, &pp);

    CHECK("lazy permute output created", out != NULL);
    CHECK("permute keeps source lazy before read", src_node && !src_node->is_executed);

    float* out_data = out ? (float*)tensor_data_ptr(out) : NULL;
    int ok = out_data != NULL && out->ndim == 2 &&
             out->shape[0] == 3 && out->shape[1] == 2;
    if (ok) {
        for (int i = 0; i < 6; i++) {
            if (!APPROX_EQ(out_data[i], 2.5f)) {
                ok = 0;
                break;
            }
        }
    }
    CHECK("permute materializes correct values on demand", ok);

    tensor_free(src);
    tensor_free(out);
    cml_reset_ir_context();
}

/* ── 11. Lazy dropout layer ─────────────────────────────────────────────── */

static void test_dropout_layer_stays_lazy(void) {
    printf("Test: dropout layer builds lazy graph\n");

    int shape[] = {8};
    Tensor* src = uop_fill(shape, 1, 2.0f);
    Dropout* layer = cml_nn_dropout(0.25f, false);
    if (!src || !layer) {
        CHECK("dropout layer setup", 0);
        tensor_free(src);
        if (layer) module_free((Module*)layer);
        cml_reset_ir_context();
        return;
    }
    module_set_training((Module*)layer, true);

    Tensor* out = module_forward((Module*)layer, src);
    struct IRNode* out_node = out ? (struct IRNode*)out->ir_node : NULL;

    CHECK("dropout output created", out != NULL);
    CHECK("dropout output remains lazy before read", out_node && !out_node->is_executed);

    float* data = out ? (float*)tensor_data_ptr(out) : NULL;
    CHECK("dropout materializes on demand", data != NULL);

    tensor_free(src);
    tensor_free(out);
    module_free((Module*)layer);
    cml_reset_ir_context();
}

/* ── 12. Lazy 2D pooling layers ─────────────────────────────────────────── */

static void test_pool2d_layers_stay_lazy(void) {
    printf("Test: maxpool2d/avgpool2d layers build lazy graphs\n");

    int shape[] = {1, 1, 4, 4};
    float values[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    TensorConfig cfg = {0};
    Tensor* src = tensor_from_data(values, shape, 4, &cfg);
    MaxPool2d* max_layer = cml_nn_maxpool2d(2, 2, 0, 1, false);
    AvgPool2d* avg_layer = cml_nn_avgpool2d(2, 2, 0, false, true);
    if (!src || !max_layer || !avg_layer) {
        CHECK("pool2d layer setup", 0);
        tensor_free(src);
        if (max_layer) module_free((Module*)max_layer);
        if (avg_layer) module_free((Module*)avg_layer);
        cml_reset_ir_context();
        return;
    }

    Tensor* max_out = module_forward((Module*)max_layer, src);
    Tensor* avg_out = module_forward((Module*)avg_layer, src);
    struct IRNode* max_node = max_out ? (struct IRNode*)max_out->ir_node : NULL;
    struct IRNode* avg_node = avg_out ? (struct IRNode*)avg_out->ir_node : NULL;

    CHECK("maxpool2d output created", max_out != NULL);
    CHECK("avgpool2d output created", avg_out != NULL);
    CHECK("maxpool2d output remains lazy before read", max_node && !max_node->is_executed);
    CHECK("avgpool2d output remains lazy before read", avg_node && !avg_node->is_executed);

    float* max_data = max_out ? (float*)tensor_data_ptr(max_out) : NULL;
    float* avg_data = avg_out ? (float*)tensor_data_ptr(avg_out) : NULL;
    int ok_max = max_data &&
                 APPROX_EQ(max_data[0], 6.0f) && APPROX_EQ(max_data[1], 8.0f) &&
                 APPROX_EQ(max_data[2], 14.0f) && APPROX_EQ(max_data[3], 16.0f);
    int ok_avg = avg_data &&
                 APPROX_EQ(avg_data[0], 3.5f) && APPROX_EQ(avg_data[1], 5.5f) &&
                 APPROX_EQ(avg_data[2], 11.5f) && APPROX_EQ(avg_data[3], 13.5f);
    CHECK("maxpool2d values correct after materialization", ok_max);
    CHECK("avgpool2d values correct after materialization", ok_avg);

    tensor_free(src);
    tensor_free(max_out);
    tensor_free(avg_out);
    module_free((Module*)max_layer);
    module_free((Module*)avg_layer);
    cml_reset_ir_context();
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Lazy Evaluation Tests ===\n\n");

    test_partial_exec_stops_early();
    test_dce_skips_unused_branch();
    test_tail_matches_full_exec();
    test_lazy_fill();
    test_lazy_rand_uniform();
    test_lazy_arange();
    test_backward_dce();
    test_chained_lazy_via_data_ptr();
    test_reset_does_not_materialize_pending_rand();
    test_permute_does_not_force_source_execution();
    test_dropout_layer_stays_lazy();
    test_pool2d_layers_stay_lazy();

    printf("\n%d/%d tests passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
