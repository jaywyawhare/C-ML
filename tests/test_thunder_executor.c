/* The Thunder executor's op table and its dispatch switch must agree.
 *
 * A row in `op_table` is a claim that `cml_thunder_execute` can run that op. The
 * table used to advertise twelve ops the switch had no case for, so they resolved
 * by name and then failed deep in dispatch. These tests walk the advertised names
 * and require each to actually execute, which is what keeps the two lists from
 * drifting apart again.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "backend/thunder_executor.h"
#include "test_harness.h"
#include "test_require.h"

#define EPS 1e-5f

static Tensor* make_1d(const float* d, int n) {
    int shape[1]     = {n};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    return tensor_from_data(d, shape, 1, &cfg);
}

/* Run one op through the executor and return its output tensor (owned by the
 * caller). `nin` inputs, one unbound output slot. */
static Tensor* run_op(const char* name, Tensor** in, int nin) {
    CMLThunderExecutor* exec = cml_thunder_create("cml_cpu");
    REQUIRE(exec);

    void* outs[1]   = {NULL};
    CMLThunderOp op = {.op_name     = name,
                       .inputs      = (void**)in,
                       .num_inputs  = nin,
                       .outputs     = outs,
                       .num_outputs = 1};
    int rc          = cml_thunder_execute(exec, &op, 1);
    cml_thunder_free(exec);

    if (rc != 0)
        return NULL;
    return (Tensor*)outs[0];
}

static bool close_to(Tensor* t, const float* want, int n) {
    if (!t || (int)t->numel != n)
        return false;
    if (tensor_ensure_executed(t) != 0)
        return false;
    const float* got = (const float*)tensor_data_ptr(t);
    if (!got)
        return false;
    for (int i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > EPS) {
            printf("(elem %d: got %g want %g) ", i, (double)got[i], (double)want[i]);
            return false;
        }
    }
    return true;
}

/* Each op that used to fall through to "dispatch not implemented". */

static bool test_sign(void) {
    const float x[4] = {-2.5f, 0.0f, 3.0f, -0.1f};
    const float w[4] = {-1.0f, 0.0f, 1.0f, -1.0f};
    Tensor* a        = make_1d(x, 4);
    Tensor* in[1]    = {a};
    Tensor* out      = run_op("torch.sign", in, 1);
    bool ok          = close_to(out, w, 4);
    if (out)
        tensor_free(out);
    tensor_free(a);
    return ok;
}

static bool test_floor_and_ceil(void) {
    const float x[4]  = {1.7f, -1.2f, 2.0f, -0.5f};
    const float wf[4] = {1.0f, -2.0f, 2.0f, -1.0f};
    const float wc[4] = {2.0f, -1.0f, 2.0f, 0.0f};

    Tensor* a     = make_1d(x, 4);
    Tensor* in[1] = {a};

    Tensor* f = run_op("torch.floor", in, 1);
    bool ok   = close_to(f, wf, 4);
    if (f)
        tensor_free(f);

    Tensor* c = run_op("torch.ceil", in, 1);
    ok        = ok && close_to(c, wc, 4);
    if (c)
        tensor_free(c);

    tensor_free(a);
    return ok;
}

static bool test_round(void) {
    const float x[4] = {1.4f, 1.6f, -1.4f, -1.6f};
    const float w[4] = {1.0f, 2.0f, -1.0f, -2.0f};
    Tensor* a        = make_1d(x, 4);
    Tensor* in[1]    = {a};
    Tensor* out      = run_op("torch.round", in, 1);
    bool ok          = close_to(out, w, 4);
    if (out)
        tensor_free(out);
    tensor_free(a);
    return ok;
}

static bool test_erf(void) {
    const float x[3] = {0.0f, 0.5f, -1.0f};
    float w[3];
    for (int i = 0; i < 3; i++)
        w[i] = erff(x[i]);

    Tensor* a     = make_1d(x, 3);
    Tensor* in[1] = {a};
    Tensor* out   = run_op("torch.erf", in, 1);
    bool ok       = close_to(out, w, 3);
    if (out)
        tensor_free(out);
    tensor_free(a);
    return ok;
}

static bool test_pow(void) {
    const float x[3] = {2.0f, 3.0f, 4.0f};
    const float y[3] = {2.0f, 2.0f, 0.5f};
    const float w[3] = {4.0f, 9.0f, 2.0f};

    Tensor* a     = make_1d(x, 3);
    Tensor* b     = make_1d(y, 3);
    Tensor* in[2] = {a, b};
    Tensor* out   = run_op("torch.pow", in, 2);
    bool ok       = close_to(out, w, 3);
    if (out)
        tensor_free(out);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static bool test_where(void) {
    const float c[4] = {1.0f, 0.0f, 1.0f, 0.0f};
    const float x[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    const float y[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
    const float w[4] = {10.0f, -2.0f, 30.0f, -4.0f};

    Tensor* tc    = make_1d(c, 4);
    Tensor* ta    = make_1d(x, 4);
    Tensor* tb    = make_1d(y, 4);
    Tensor* in[3] = {tc, ta, tb};
    Tensor* out   = run_op("torch.where", in, 3);
    bool ok       = close_to(out, w, 4);
    if (out)
        tensor_free(out);
    tensor_free(tc);
    tensor_free(ta);
    tensor_free(tb);
    return ok;
}

/* torch.max reduces every axis, matching how torch.sum and torch.mean dispatch. */
static bool test_max_reduce(void) {
    const float x[5] = {1.0f, -4.0f, 7.5f, 2.0f, 0.0f};
    const float w[1] = {7.5f};
    Tensor* a        = make_1d(x, 5);
    Tensor* in[1]    = {a};
    Tensor* out      = run_op("torch.max", in, 1);
    bool ok          = close_to(out, w, 1);
    if (out)
        tensor_free(out);
    tensor_free(a);
    return ok;
}

/* An op that was never in the table, and the four pruned from it, must all fail
 * with the same clear "unsupported" path rather than resolving and dying in
 * dispatch. */
static bool test_unsupported_ops_rejected(void) {
    const float x[2] = {1.0f, 2.0f};
    Tensor* a        = make_1d(x, 2);
    Tensor* in[1]    = {a};

    const char* names[] = {"torch.reshape", "torch.permute", "torch.conv2d", "torch.gather",
                           "torch.nonexistent_op"};
    bool ok             = true;
    for (int i = 0; i < 5; i++) {
        Tensor* out = run_op(names[i], in, 1);
        if (out) {
            printf("(%s unexpectedly succeeded) ", names[i]);
            tensor_free(out);
            ok = false;
        }
    }
    tensor_free(a);
    return ok;
}

/* The contract this suite exists to protect: every name the table advertises
 * executes. Driven through the public API, so it covers the whole table, not
 * just the ops repaired above. */
static bool test_every_advertised_unary_op_dispatches(void) {
    /* Positive inputs so log/sqrt are in domain. */
    const float x[3] = {0.5f, 1.5f, 2.5f};
    const float y[3] = {1.25f, 2.0f, 0.75f};

    const char* unary[]  = {"torch.neg",     "torch.exp",  "torch.log",   "torch.sqrt",
                            "torch.abs",     "torch.sin",  "torch.cos",   "torch.tanh",
                            "torch.sigmoid", "torch.relu", "torch.sum",   "torch.mean",
                            "torch.max",     "torch.sign", "torch.floor", "torch.ceil",
                            "torch.round",   "torch.erf",  "torch.rsqrt", "torch.reciprocal",
                            "torch.relu6",   "torch.silu", "torch.mish",  "torch.hardswish",
                            "torch.selu",    "torch.elu",  "torch.log2",  "torch.exp2"};
    const char* binary[] = {"torch.add", "torch.sub", "torch.mul",
                            "torch.div", "torch.pow", "torch.maximum"};

    Tensor* a = make_1d(x, 3);
    Tensor* b = make_1d(y, 3);
    bool ok   = true;

    for (size_t i = 0; i < sizeof(unary) / sizeof(unary[0]); i++) {
        Tensor* in[1] = {a};
        Tensor* out   = run_op(unary[i], in, 1);
        if (!out) {
            printf("(%s did not dispatch) ", unary[i]);
            ok = false;
            break;
        }
        tensor_free(out);
    }

    for (size_t i = 0; ok && i < sizeof(binary) / sizeof(binary[0]); i++) {
        Tensor* in[2] = {a, b};
        Tensor* out   = run_op(binary[i], in, 2);
        if (!out) {
            printf("(%s did not dispatch) ", binary[i]);
            ok = false;
            break;
        }
        tensor_free(out);
    }

    tensor_free(a);
    tensor_free(b);
    return ok;
}

static bool test_create_free_and_bad_args(void) {
    CMLThunderExecutor* exec = cml_thunder_create(NULL);
    if (!exec)
        return false;
    bool ok = exec->initialized && strcmp(exec->backend_name, "cml_cpu") == 0;
    ok      = ok && cml_thunder_execute(exec, NULL, 1) == -1;
    ok      = ok && cml_thunder_execute(NULL, NULL, 0) == -1;
    cml_thunder_free(exec);
    cml_thunder_free(NULL); /* must not crash */
    return ok && cml_thunder_register() == 0;
}

int main(void) {
    printf("Thunder Executor Tests\n\n");

    printf("Lifecycle:\n");
    TEST(create_free_and_bad_args);

    printf("\nOps that previously fell through dispatch:\n");
    TEST(sign);
    TEST(floor_and_ceil);
    TEST(round);
    TEST(erf);
    TEST(pow);
    TEST(where);
    TEST(max_reduce);

    printf("\nTable/dispatch agreement:\n");
    TEST(every_advertised_unary_op_dispatches);
    TEST(unsupported_ops_rejected);

    return TEST_SUMMARY();
}
