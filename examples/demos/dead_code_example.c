#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    printf("=== Dead Code & IR Optimization Demo ===\n\n");

    cml_init();
    cml_seed(42);

    int shape[]         = {64, 128};
    TensorConfig config = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    Tensor* x    = tensor_full(shape, 2, &config, 0.5f);
    Tensor* w1   = tensor_full(shape, 2, &config, 0.3f);
    Tensor* w2   = tensor_full(shape, 2, &config, 0.7f);
    Tensor* bias = tensor_full(shape, 2, &config, 0.1f);

    printf("Input shapes: [%d, %d]\n\n", shape[0], shape[1]);

    printf("Creating dead code (unused operations)...\n");

    Tensor* dead1 = tensor_mul(x, w1);
    Tensor* dead2 = tensor_add(dead1, bias);
    Tensor* dead3 = tensor_relu(dead2);

    Tensor* dead4 = tensor_sub(w1, w2);
    Tensor* dead5 = tensor_mul(dead4, dead4);

    printf("  - Created 5 dead nodes (will be eliminated)\n");

    printf("\nCreating unoptimized element-wise chain...\n");

    Tensor* t1 = tensor_mul(x, w2);
    Tensor* t2 = tensor_add(t1, bias);
    Tensor* t3 = tensor_relu(t2);

    Tensor* scale = tensor_full(shape, 2, &config, 2.0f);
    Tensor* t4    = tensor_mul(t3, scale);

    Tensor* offset = tensor_full(shape, 2, &config, 1.0f);
    Tensor* t5     = tensor_add(t4, offset);

    printf("  - Created chain of 5 element-wise ops (can be fused)\n");

    printf("\nCreating redundant computations...\n");

    Tensor* r1 = tensor_mul(x, w2);
    Tensor* r2 = tensor_mul(x, w2);
    Tensor* r3 = tensor_add(r1, r2);

    printf("  - Created 2 redundant mul ops (CSE opportunity)\n");

    printf("\nComputing final result...\n");

    Tensor* result = tensor_add(t5, r3);
    Tensor* output = tensor_mean(result, -1, false);

    float value = tensor_get_float(output, 0);
    printf("\nFinal output value: %.6f\n", value);

    printf("\n=== Exporting IR Analysis ===\n");

    if (output->ir_context) {
        char* unopt = cml_ir_export_kernel_analysis(output->ir_context, false);

        printf("\nRunning optimization passes:\n");
        printf("  - Dead code elimination\n");
        printf("  - Common subexpression elimination\n");
        printf("  - Kernel fusion\n");
        cml_ir_optimize(output->ir_context);

        autograd_export_json(output, "graph.json");
        printf("Exported graph.json\n");

        char* opt = cml_ir_export_kernel_analysis(output->ir_context, true);

        if (unopt && opt) {
            FILE* f = fopen("kernels.json", "w");
            if (f) {
                fprintf(f, "{\"unoptimized\":%s,\"optimized\":%s}", unopt, opt);
                fclose(f);
                printf("Exported kernels.json with unoptimized/optimized comparison\n");
            }
        }

        if (unopt)
            free(unopt);
        if (opt)
            free(opt);
    } else {
        printf("Warning: No IR context available\n");
    }

    printf("\n=== Summary ===\n");
    printf("Dead nodes created:     5 (should be eliminated)\n");
    printf("Element-wise chain:     5 ops (should be fused)\n");
    printf("Redundant computations: 2 (CSE should eliminate 1)\n");
    printf("\nRun with VIZ=1 to see optimization in Kernel Studio!\n");

    tensor_free(x);
    tensor_free(w1);
    tensor_free(w2);
    tensor_free(bias);
    tensor_free(dead1);
    tensor_free(dead2);
    tensor_free(dead3);
    tensor_free(dead4);
    tensor_free(dead5);
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);
    tensor_free(scale);
    tensor_free(t4);
    tensor_free(offset);
    tensor_free(t5);
    tensor_free(r1);
    tensor_free(r2);
    tensor_free(r3);
    tensor_free(result);
    tensor_free(output);

    cml_cleanup();

    printf("\n=== Done ===\n");
    return 0;
}
