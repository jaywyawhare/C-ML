/**
 * The eager backward engine (GRAD_MODE=eager) reads every gradient buffer as
 * float*, so a half-precision grad node would be misread and overrun. The
 * engine must refuse such a graph up front (no grad produced, no crash) rather
 * than corrupt memory. f32 eager backward must still work. The default graph
 * autodiff path handles half natively and is covered elsewhere.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cml.h"
#include "autograd/autograd.h"
#include "ops/uops.h"

#define D 4

static TensorConfig F32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/* f32 eager backward still works: d/dx sum(x*x) = 2x. */
static int check_f32(void) {
    float xd[D]      = {0.5f, -1.0f, 2.0f, 0.25f};
    int s[]          = {1, D};
    Tensor* x        = cml_tensor(xd, s, 2, &F32);
    x->requires_grad = true;
    Tensor* y        = uop_mul(x, x);
    Tensor* loss     = uop_sum(y, &(ReduceParams){0});
    tensor_backward(loss, NULL, false, false);
    const float* g = x->grad ? (const float*)tensor_data_ptr(x->grad) : NULL;
    int ok         = g != NULL;
    float maxdiff  = 0.0f;
    for (int i = 0; ok && i < D; i++) {
        float d = fabsf(g[i] - 2.0f * xd[i]);
        if (d > maxdiff)
            maxdiff = d;
    }
    ok = ok && maxdiff < 1e-5f;
    printf("  f32 eager backward grad=2x maxdiff=%.3e %s\n", (double)maxdiff, ok ? "PASS" : "FAIL");
    cml_reset_ir_context();
    return ok;
}

/* A bf16 grad node must be refused: no grad produced, no crash/corruption. */
static int check_half_refused(void) {
    float xd[D]       = {0.5f, -1.0f, 2.0f, 0.25f};
    int s[]           = {1, D};
    Tensor* xf        = cml_tensor(xd, s, 2, &F32);
    xf->requires_grad = true;
    Tensor* xb        = cml_cast(xf, DTYPE_BFLOAT16); /* bf16, requires_grad */
    Tensor* y         = uop_mul(xb, xb);
    Tensor* loss      = uop_sum(y, &(ReduceParams){0});
    tensor_backward(loss, NULL, false, false);
    /* Guard refuses before running: no gradient written for the leaf. */
    int ok = xf->grad == NULL;
    printf("  bf16 eager backward refused (no grad) %s\n", ok ? "PASS" : "FAIL");
    cml_reset_ir_context();
    return ok;
}

int main(void) {
    setenv("GRAD_MODE", "eager", 1);
    cml_init();
    printf("Eager backward half-precision guard\n");
    int ok = 1;
    ok &= check_f32();
    ok &= check_half_refused();
    printf(ok ? "Eager backward guard OK.\n" : "FAILED.\n");
    return ok ? 0 : 1;
}
