/* Reductions over an empty tensor must return the reduction's identity.
 *
 * An empty tensor has no data pointer, so every reduction kernel bailed on
 * `!in1_data`; that failure was swallowed and the zero-filled output buffer was
 * handed back as the result. Every reduction therefore answered 0 --
 * prod([]) was 0 rather than 1, all([]) was 0 rather than 1 (it is vacuously
 * true), logsumexp([]) was 0 rather than -inf, and max([]) claimed a maximum of
 * 0 for a tensor with no elements.
 *
 * Reductions with no identity (max/min/argmax/argmin) must fail rather than
 * invent a value -- inventing one is exactly what produced the bug.
 */

#include "cml.h"
#include "test_require.h"
#include "core/error_stack.h"
#include <math.h>
#include <stdio.h>

static int checks = 0, failures = 0;

static Tensor* empty_tensor(void) {
    TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                      .has_dtype = true, .has_device = true};
    int shape[1] = {0};
    return tensor_zeros(shape, 1, &c);
}

static void expect_value(const char* name, Tensor* r, double want) {
    checks++;
    if (!r) {
        printf("  %-12s built NULL, expected %g\n", name, want);
        failures++;
        return;
    }
    tensor_ensure_executed(r);
    double got = (double)tensor_get_float(r, 0);
    int ok = isnan(want) ? isnan(got)
           : isinf(want) ? (isinf(got) && ((got > 0) == (want > 0)))
                         : (got == want);
    if (!ok) {
        printf("  %-12s got %g, expected %g\n", name, got, want);
        failures++;
    }
    cml_reset_ir_context();
}

/* The op must not report success: a reduction with no identity has no answer. */
static void expect_rejected(const char* name, Tensor* r) {
    checks++;
    if (r) {
        error_stack_clear();
        tensor_ensure_executed(r);
        if (!error_stack_has_errors()) {
            printf("  %-12s reported success over an empty tensor\n", name);
            failures++;
        }
        error_stack_clear();
    }
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("Reductions over an empty tensor:\n");

    expect_value("sum",       uop_sum(empty_tensor(), NULL),       0.0);
    expect_value("prod",      uop_prod(empty_tensor(), NULL),      1.0);
    expect_value("mean",      uop_mean(empty_tensor(), NULL),      NAN);
    expect_value("any",       uop_any(empty_tensor(), NULL),       0.0);
    expect_value("all",       uop_all(empty_tensor(), NULL),       1.0);
    expect_value("logsumexp", uop_logsumexp(empty_tensor(), NULL), -INFINITY);

    expect_rejected("max_reduce", uop_max_reduce(empty_tensor(), NULL));
    expect_rejected("min_reduce", uop_min_reduce(empty_tensor(), NULL));

    /* Non-empty reductions must be unaffected. */
    {
        TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                          .has_dtype = true, .has_device = true};
        int shape[1] = {3};
        Tensor* t = tensor_zeros(shape, 1, &c);
        for (int i = 0; i < 3; i++) tensor_set_float(t, i, (float)(i + 2));  /* 2,3,4 */
        expect_value("prod(2,3,4)", uop_prod(t, NULL), 24.0);
    }

    printf("\n%d checks, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
