/* Op results must match an independent reference, not merely match each other.
 *
 * test_path_equivalence proves the JIT, interpreter and fused kernels agree.
 * That is necessary but not sufficient: they can agree and all be wrong, which
 * is what a bad constant or the wrong convention looks like. This test pins the
 * actual numbers against values computed independently (numpy/libm), so a
 * mis-typed gelu coefficient or the wrong rounding mode fails here even when
 * every internal path is perfectly consistent.
 *
 * That is not hypothetical: round() used C's roundf, which rounds halves away
 * from zero (-2.5 -> -3), while numpy, PyTorch and the IEEE-754 default round
 * halves to even (-2.5 -> -2). Every internal path agreed with every other and
 * all of them disagreed with the rest of the world.
 *
 * Expected values below are computed by numpy/libm in double precision and
 * pasted in; the tolerance is loose enough for f32 evaluation but far tighter
 * than any formula or convention error.
 */

#include "cml.h"
#include "test_require.h"
#include <math.h>
#include <stdio.h>

static int checks = 0, failures = 0;

/* Includes exact halves, so ties-to-even vs away-from-zero is distinguishable. */
static const float X[8] = {-3.5f, -2.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.5f, 3.5f};

static Tensor* mk(const float* v, int n) {
    TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                      .has_dtype = true, .has_device = true};
    int shape[1] = {n};
    Tensor* t = tensor_zeros(shape, 1, &c);
    for (int i = 0; i < n; i++) tensor_set_float(t, i, v[i]);
    return t;
}

/* Pure relative comparison, no absolute floor.
 *
 * expect() adds "+ 2e-5" so that values near zero do not demand absurd
 * precision. That floor also makes every small-magnitude error invisible: a
 * kernel returning 0 for tanh(1e-8) is off by 100%% yet lands 1e-8 from the
 * reference, far inside the floor. Checks aimed at cancellation must compare
 * relatively or they prove nothing. */
static void expect_rel(const char* name, Tensor* r, const double* want, int n, double rel) {
    checks++;
    if (!r) {
        printf("  %-18s built NULL\n", name);
        failures++;
        return;
    }
    tensor_ensure_executed(r);
    for (int i = 0; i < n; i++) {
        double got = (double)tensor_get_float(r, (size_t)i);
        double err = (want[i] == 0.0) ? fabs(got) : fabs(got - want[i]) / fabs(want[i]);
        if (err > rel) {
            printf("  %-18s [%d] = %.9g, reference %.9g  (rel err %.3g)\n",
                   name, i, got, want[i], err);
            failures++;
            break;
        }
    }
    cml_reset_ir_context();
}

static void expect(const char* name, Tensor* r, const double* want, int n) {
    checks++;
    if (!r) {
        printf("  %-14s built NULL\n", name);
        failures++;
        return;
    }
    tensor_ensure_executed(r);
    if ((int)r->numel != n) {
        printf("  %-14s numel %zu, expected %d\n", name, r->numel, n);
        failures++;
        cml_reset_ir_context();
        return;
    }
    for (int i = 0; i < n; i++) {
        double got = (double)tensor_get_float(r, (size_t)i);
        double tol = fabs(want[i]) * 2e-5 + 2e-5;
        if (fabs(got - want[i]) > tol) {
            printf("  %-14s [%d] = %.9g, reference %.9g\n", name, i, got, want[i]);
            failures++;
            break;
        }
    }
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("Reference accuracy (independent values, not self-consistency):\n");

    /* @GENERATED-BEGIN */
    { const double want[8] = {-4, -2, -0, 0, 0, 2, 2, 4};
      expect("round", uop_round(mk(X, 8)), want, 8); }
    { const double want[8] = {-4, -3, -1, 0, 0, 1, 2, 3};
      expect("floor", uop_floor(mk(X, 8)), want, 8); }
    { const double want[8] = {-3, -2, -0, 0, 1, 2, 3, 4};
      expect("ceil", uop_ceil(mk(X, 8)), want, 8); }
    { const double want[8] = {-3, -2, -0, 0, 0, 1, 2, 3};
      expect("trunc", uop_trunc(mk(X, 8)), want, 8); }
    { const double want[8] = {-0.000616197655, -0.0150842661, -0.15428599, 0, 0.34571401, 1.39957158, 2.48491573, 3.4993838};
      expect("gelu", uop_gelu(mk(X, 8)), want, 8); }
    { const double want[8] = {-0.102592808, -0.18964545, -0.188770334, 0, 0.311229666, 1.22636171, 2.31035455, 3.39740719};
      expect("silu", uop_silu(mk(X, 8)), want, 8); }
    { const double want[8] = {0.0297504183, 0.0788897343, 0.474076984, 0.693147181, 0.974076984, 1.70141328, 2.57888973, 3.52975042};
      expect("softplus", uop_softplus(mk(X, 8)), want, 8); }
    { const double want[8] = {-1.70500934, -1.61378576, -0.691758188, 0, 0.525350494, 1.57605148, 2.62675247, 3.67745346};
      expect("selu", uop_selu(mk(X, 8)), want, 8); }
    { const double want[8] = {-0.999999257, -0.999593048, -0.520499878, 0, 0.520499878, 0.966105146, 0.999593048, 0.999999257};
      expect("erf", uop_erf(mk(X, 8)), want, 8); }
    { const double want[8] = {0, -0.208333333, -0.208333333, 0, 0.291666667, 1.125, 2.29166667, 3.5};
      expect("hardswish", uop_hardswish(mk(X, 8)), want, 8); }
    { const double want[8] = {0.0293122308, 0.07585818, 0.377540669, 0.5, 0.622459331, 0.817574476, 0.92414182, 0.970687769};
      expect("sigmoid", uop_sigmoid(mk(X, 8)), want, 8); }
    { const double want[8] = {-0.998177898, -0.986614298, -0.462117157, 0, 0.462117157, 0.905148254, 0.986614298, 0.998177898};
      expect("tanh", uop_tanh(mk(X, 8)), want, 8); }
    { const double want[8] = {-3.5, -6, -6.5, -6.5, -6, -4.5, -2, 1.5};
      expect("cumsum", uop_cumsum(mk(X, 8), 0), want, 8); }

    /* Small magnitudes.
     *
     * Every input above has |x| >= 0.5, where the identity 2*sigmoid(2x)-1 is a
     * perfectly good tanh. Near zero it is not: it evaluates 1 - 1 and loses
     * every significant bit. The LLVM JIT used that identity and returned
     * exactly 0 for tanh(1e-8) -- 100% relative error -- while the interpreter
     * called tanhf and was accurate to 2.7e-8. Both paths "agreed" with each
     * other on the coarse inputs above, so only a small-magnitude reference
     * check catches it. */
    {
        static const float XS[8] = {1e-08f, 1e-06f, 0.0001f, 0.001f, 0.01f, -1e-08f, -1e-06f, -0.01f};
        { const double want[8] = {1e-08, 1e-06, 9.999999967e-05, 0.0009999996667, 0.00999966668, -1e-08, -1e-06, -0.00999966668};
          expect_rel("tanh (small x)", uop_tanh(mk(XS, 8)), want, 8, 1e-5); }
        { const double want[8] = {0.5000000025, 0.50000025, 0.500025, 0.50025, 0.5024999792, 0.4999999975, 0.49999975, 0.4975000208};
          expect_rel("sigmoid (small x)", uop_sigmoid(mk(XS, 8)), want, 8, 1e-5); }
    }

    /* Saturating tails of the softplus family.
     *
     * These are computed as max(x,0) + log(1 + exp(-|x|)). Written with log()
     * instead of log1p(), the `1 +` destroys the significance of a small
     * exponential: softplus(-10) had 4.1e-4 relative error, four orders worse
     * than float epsilon. Only |x| large enough for exp(-|x|) to be tiny shows
     * it, so the moderate inputs above all passed. */
    {
        static const float XT[8] = {-10.0f, -5.0f, -1.0f, -0.01f, 0.01f, 1.0f, 5.0f, 10.0f};
        { const double want[8] = {4.539889922e-05, 0.006715348489, 0.3132616875, 0.6881596805, 0.6981596805, 1.313261688, 5.006715348, 10.0000454};
          expect_rel("softplus (tails)", uop_softplus(mk(XT, 8)), want, 8, 1e-5); }
        { const double want[8] = {-10.0000454, -5.006715348, -1.313261688, -0.6981596805, -0.6881596805, -0.3132616875, -0.006715348489, -4.539889922e-05};
          expect_rel("logsigmoid (tails)", uop_logsigmoid(mk(XT, 8)), want, 8, 1e-5); }
        { const double want[8] = {-0.0004539889919, -0.03357623773, -0.3034014614, -0.005967984459, 0.006031983541, 0.8650983883, 4.999552078, 9.999999959};
          expect_rel("mish (tails)", uop_mish(mk(XT, 8)), want, 8, 1e-5); }
    }

    /* Large-N summation.
     *
     * A single running float accumulator loses O(N * eps): summing 1e6 copies
     * of 0.1f gave 100958 instead of 100000, ~1%% error, which would show up
     * directly in a loss or a normalisation statistic. Pairwise summation keeps
     * it near eps and is also faster, because four independent accumulators
     * break the serial dependency chain. Small N cannot show this -- the error
     * only accumulates over many terms. */
    {
        const int N = 1000000;
        TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                          .has_dtype = true, .has_device = true};
        int shape[1] = {N};
        Tensor* t = tensor_zeros(shape, 1, &c);
        for (int i = 0; i < N; i++) tensor_set_float(t, i, 0.1f);
        Tensor* r = uop_sum(t, NULL);
        checks++;
        tensor_ensure_executed(r);
        /* 0.1f is not exactly 0.1, so the reference is N * (double)0.1f. */
        double want = (double)N * (double)0.1f;
        double got  = (double)tensor_get_float(r, 0);
        double rel  = fabs(got - want) / want;
        if (rel > 1e-5) {
            printf("  %-18s sum of %d x 0.1f = %.6f, reference %.6f (rel err %.3g)\n",
                   "sum (large N)", N, got, want, rel);
            failures++;
        }
        cml_reset_ir_context();
    }



    { const double want[1] = {1.5};
      expect("sum", uop_sum(mk(X, 8), NULL), want, 1); }
    { const double want[1] = {0.1875};
      expect("mean", uop_mean(mk(X, 8), NULL), want, 1); }
    { const double want[1] = {4.93359375};
      expect("var", uop_var(mk(X, 8), NULL), want, 1); }
    { const double want[1] = {2.22116946};
      expect("std", uop_std(mk(X, 8), NULL), want, 1); }
    { const double want[1] = {3.97306484};
      expect("logsumexp", uop_logsumexp(mk(X, 8), NULL), want, 1); }
    /* @GENERATED-END */

    printf("\n%d checks, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
