/**
 * The fused softmax/log-softmax kernels are f32-only (they read ->data as
 * float*). uop_softmax must route non-f32 inputs to the dtype-safe reduce/expand
 * chain instead of the fused kernel, producing correct results (not garbage from
 * misreading half data). This checks a bf16 softmax matches the f32 reference.
 */
#include <math.h>
#include <stdio.h>

#include "cml.h"
#include "ops/uops.h"

#define D 4

int main(void) {
    cml_init();
    printf("Fused-op dtype safety (bf16 softmax falls back correctly)\n");

    float xf[D] = {1.0f, 2.0f, 3.0f, 0.5f};

    /* f32 reference (fused path). */
    TensorConfig f32 = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int s[]         = {1, D};
    Tensor* xf32    = cml_tensor(xf, s, 2, &f32);
    Tensor* yf32    = uop_softmax(xf32, -1);
    const float* rf = (const float*)tensor_data_ptr(yf32);
    float ref[D];
    for (int j = 0; j < D; j++)
        ref[j] = rf[j];
    cml_reset_ir_context();

    /* bf16 input (properly converted from f32): must not hit the f32 fused
     * kernel. cml_tensor interprets raw bytes as the config dtype, so build the
     * bf16 tensor by casting rather than relabeling. */
    Tensor* xf32b = cml_tensor(xf, s, 2, &f32);
    Tensor* xbf   = cml_cast(xf32b, DTYPE_BFLOAT16);
    Tensor* ybf   = uop_softmax(xbf, -1);
    if (!ybf) {
        printf("  bf16 softmax returned NULL FAIL\n");
        return 1;
    }
    const float* rb_any = (const float*)tensor_data_ptr(ybf);
    if (!rb_any) {
        printf("  bf16 softmax has no data FAIL\n");
        return 1;
    }
    float maxdiff = 0.0f;
    for (int j = 0; j < D; j++) {
        float got = tensor_get_float(ybf, j); /* dtype-aware read */
        float d   = fabsf(got - ref[j]);
        if (d > maxdiff)
            maxdiff = d;
    }
    cml_reset_ir_context();

    /* bf16 has ~3 decimal digits; tolerate its quantization. */
    int pass = maxdiff < 2e-2f;
    printf("  bf16 vs f32 softmax maxdiff=%.3e %s\n", (double)maxdiff, pass ? "PASS" : "FAIL");
    printf(pass ? "Fused-op dtype safety passed.\n" : "FAILED.\n");
    return pass ? 0 : 1;
}
