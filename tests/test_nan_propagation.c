/* NaN must survive every op that could quietly drop it.
 *
 * The selection ops are written with ordered comparisons -- `x > 0 ? x : 0`,
 * `a > b ? a : b`, `if (v > mx) mx = v`. An ordered compare is false when either
 * side is NaN, so all of these silently return the *other* value: relu(NaN) was
 * 0, maximum(NaN, 0) was 0, and a max-reduction over a tensor containing NaN
 * returned the largest finite element. That is the worst possible failure mode
 * for a training library -- a diverged tensor turns back into plausible finite
 * numbers instead of showing NaN, so the divergence is never noticed.
 *
 * The convention here is torch.maximum/np.maximum, not IEEE maxNum/fmax: NaN
 * propagates. This test pins it across every op, dtype and execution path,
 * because f32 and f64 previously disagreed with each other and the JIT
 * disagreed with the interpreter.
 */

#include "cml.h"
#include "test_require.h"
#include <math.h>
#include <stdio.h>

static int checks = 0, failures = 0;

static Tensor* mk(DType dt, int n, const double* v) {
    TensorConfig c = {.dtype = dt, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int shape[1]   = {n};
    Tensor* t      = tensor_zeros(shape, 1, &c);
    if (t && v)
        for (int i = 0; i < n; i++)
            tensor_set_float(t, i, (float)v[i]);
    return t;
}

static void expect_nan(const char* name, DType dt, Tensor* r, int idx) {
    checks++;
    const char* dn = (dt == DTYPE_FLOAT32) ? "f32" : "f64";
    if (!r) {
        printf("  %-24s %s  NULL\n", name, dn);
        failures++;
        return;
    }
    tensor_ensure_executed(r);
    float v = tensor_get_float(r, (size_t)idx);
    if (!isnan(v)) {
        printf("  %-24s %s  got %g, expected nan (NaN was swallowed)\n", name, dn, v);
        failures++;
    }
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("NaN propagation:\n");

    const double with_nan[3] = {NAN, 1.0, 2.0};
    const double zeros[3]    = {0.0, 0.0, 0.0};
    const DType dtypes[2]    = {DTYPE_FLOAT32, DTYPE_FLOAT64};

    for (int d = 0; d < 2; d++) {
        DType dt = dtypes[d];
        expect_nan("relu", dt, uop_relu(mk(dt, 3, with_nan)), 0);
        expect_nan("relu6", dt, uop_relu6(mk(dt, 3, with_nan)), 0);
        expect_nan("abs", dt, uop_abs(mk(dt, 3, with_nan)), 0);
        expect_nan("clamp", dt, uop_clamp(mk(dt, 3, with_nan), 0.0f, 10.0f), 0);
        expect_nan("maximum", dt, uop_max(mk(dt, 3, with_nan), mk(dt, 3, zeros)), 0);
        expect_nan("minimum", dt, uop_minimum(mk(dt, 3, with_nan), mk(dt, 3, zeros)), 0);
        expect_nan("max_reduce", dt, uop_max_reduce(mk(dt, 3, with_nan), NULL), 0);
        expect_nan("min_reduce", dt, uop_min_reduce(mk(dt, 3, with_nan), NULL), 0);
        expect_nan("sum", dt, uop_sum(mk(dt, 3, with_nan), NULL), 0);
        expect_nan("mean", dt, uop_mean(mk(dt, 3, with_nan), NULL), 0);

        /* Clamping activations: fminf/fmaxf are IEEE minNum/maxNum and return
         * the non-NaN operand, so every clamp built on them dropped NaN. */
        expect_nan("hard_tanh", dt, uop_hard_tanh(mk(dt, 3, with_nan)), 0);
        expect_nan("hard_sigmoid", dt, uop_hard_sigmoid(mk(dt, 3, with_nan)), 0);

        /* These reach NaN through a log/exp term rather than the clamp, so they
         * were already correct -- pinned so a "simplification" cannot break them. */
        expect_nan("softplus", dt, uop_softplus(mk(dt, 3, with_nan)), 0);
        expect_nan("logsigmoid", dt, uop_logsigmoid(mk(dt, 3, with_nan)), 0);
        expect_nan("mish", dt, uop_mish(mk(dt, 3, with_nan)), 0);
        expect_nan("silu", dt, uop_silu(mk(dt, 3, with_nan)), 0);
        expect_nan("elu", dt, uop_elu(mk(dt, 3, with_nan), 1.0f), 0);
        expect_nan("selu", dt, uop_selu(mk(dt, 3, with_nan)), 0);
        expect_nan("gelu", dt, uop_gelu(mk(dt, 3, with_nan)), 0);
        expect_nan("sigmoid", dt, uop_sigmoid(mk(dt, 3, with_nan)), 0);
        expect_nan("tanh", dt, uop_tanh(mk(dt, 3, with_nan)), 0);
    }

    /* relu has a same-size branch and a broadcasting branch; only the first was
     * fixed initially, so the broadcast path still returned 0 for NaN. */
    {
        const double one_nan[1] = {NAN};
        Tensor* scalar          = mk(DTYPE_FLOAT32, 1, one_nan);
        expect_nan("relu (broadcast)", DTYPE_FLOAT32, uop_relu(scalar), 0);
    }

    /* tensor_from_data builds a different graph than tensor_zeros + set_float,
     * and only the former reached the JIT's *fused-chain* emitter (fe_emit_op),
     * which carried its own copies of relu/max/minimum. Those still swallowed
     * NaN long after the standalone kernels were fixed, so the input must be
     * constructed both ways for the check to mean anything. */
    {
        float raw[4]   = {NAN, -2.0f, 3.0f, 1.0f};
        int shape[1]   = {4};
        TensorConfig c = {
            .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
        Tensor* t = tensor_from_data(raw, shape, 1, &c);
        Tensor* r = uop_relu(t);
        checks++;
        tensor_ensure_executed(r);
        const float* g = (const float*)tensor_data_ptr(r);
        if (!g || !isnan(g[0]) || g[1] != 0.0f || g[2] != 3.0f || g[3] != 1.0f) {
            printf("  relu (from_data) = [%s %g %g %g], expected [nan 0 3 1]\n",
                   (g && isnan(g[0])) ? "nan" : "FINITE", g ? g[1] : 0.0f, g ? g[2] : 0.0f,
                   g ? g[3] : 0.0f);
            failures++;
        }
        cml_reset_ir_context();
    }

    /* NaN in the second operand must propagate too -- a one-sided check passes
     * even when only the first operand is tested. */
    {
        const double nan_second[3] = {1.0, 2.0, 3.0};
        Tensor* a                  = mk(DTYPE_FLOAT32, 3, nan_second);
        Tensor* b                  = mk(DTYPE_FLOAT32, 3, with_nan);
        expect_nan("maximum(finite,nan)", DTYPE_FLOAT32, uop_max(a, b), 0);
    }

    /* Finite behaviour must be unchanged by the comparison flips. */
    {
        const double v[3] = {-2.0, 0.0, 3.0};
        Tensor* r         = uop_relu(mk(DTYPE_FLOAT32, 3, v));
        tensor_ensure_executed(r);
        checks++;
        if (tensor_get_float(r, 0) != 0.0f || tensor_get_float(r, 1) != 0.0f ||
            tensor_get_float(r, 2) != 3.0f) {
            printf("  relu finite values changed: [%g %g %g], expected [0 0 3]\n",
                   tensor_get_float(r, 0), tensor_get_float(r, 1), tensor_get_float(r, 2));
            failures++;
        }
        cml_reset_ir_context();
    }

    printf("\n%d checks, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
