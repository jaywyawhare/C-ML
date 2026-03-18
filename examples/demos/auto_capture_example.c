#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    cml_init();

    CMLGraph_t ir = cml_ir_new(IR_TARGET_CUDA);
    if (!ir) {
        fprintf(stderr, "Failed to create IR context\n");
        cml_cleanup();
        return 1;
    }

    if (cml_ir_enable_auto_capture(ir) != 0) {
        fprintf(stderr, "Failed to enable auto-capture\n");
        cml_ir_free(ir);
        cml_cleanup();
        return 1;
    }

    int shape[]         = {4};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* a = tensor_empty(shape, 1, &config);
    Tensor* b = tensor_empty(shape, 1, &config);

    float* a_data = (float*)tensor_data_ptr(a);
    float* b_data = (float*)tensor_data_ptr(b);
    for (int i = 0; i < 4; i++) {
        a_data[i] = (float)(i + 1);
        b_data[i] = (float)(i + 2);
    }

    printf("Input A: [%.1f, %.1f, %.1f, %.1f]\n", (double)a_data[0], (double)a_data[1],
           (double)a_data[2], (double)a_data[3]);
    printf("Input B: [%.1f, %.1f, %.1f, %.1f]\n", (double)b_data[0], (double)b_data[1],
           (double)b_data[2], (double)b_data[3]);

    // e = exp((a + b) * a)
    Tensor* c = tensor_add(a, b);
    Tensor* d = tensor_mul(c, a);
    Tensor* e = tensor_exp(d);

    printf("\nExecuting graph on CPU to verify results...\n");
    float* result_data = (float*)tensor_data_ptr(e);

    if (result_data) {
        printf("Result (exp((a+b)*a)):\n");
        for (int i = 0; i < 4; i++) {
            float val_a    = (float)(i + 1);
            float val_b    = (float)(i + 2);
            float val_c    = val_a + val_b;
            float val_d    = val_c * val_a;
            float expected = expf(val_d);

            printf("  [%d]: %.4f (Expected: %.4f)\n", i, (double)result_data[i], (double)expected);
        }
    } else {
        printf("Failed to execute graph!\n");
    }

    cml_ir_optimize(ir);

    char* cuda_code = cml_ir_compile(ir, NULL);
    if (cuda_code) {
        printf("\nGenerated CUDA code (after optimization/fusion):\n%s\n", cuda_code);
        free(cuda_code);
    }

    if (e)
        tensor_free(e);
    if (d)
        tensor_free(d);
    if (c)
        tensor_free(c);
    tensor_free(b);
    tensor_free(a);

    cml_ir_disable_auto_capture();
    cml_ir_free(ir);
    cml_cleanup();

    return 0;
}
