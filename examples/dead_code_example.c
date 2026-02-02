/**
 * @file dead_code_example.c
 * @brief Example demonstrating dead code elimination and IR optimization
 *
 * This example creates a computation graph with:
 * - Dead code (unused tensor operations)
 * - Redundant operations that can be fused
 * - Unoptimized element-wise chains
 *
 * Run with VIZ=1 to see the unoptimized vs optimized kernels in Kernel Studio.
 */

#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    printf("=== Dead Code & IR Optimization Demo ===\n\n");

    cml_init();
    cml_seed(42);

    // Create input tensors
    int shape[]         = {64, 128};
    TensorConfig config = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    // Create tensors with different values for interesting computation
    Tensor* x    = tensor_full(shape, 2, &config, 0.5f);
    Tensor* w1   = tensor_full(shape, 2, &config, 0.3f);
    Tensor* w2   = tensor_full(shape, 2, &config, 0.7f);
    Tensor* bias = tensor_full(shape, 2, &config, 0.1f);

    printf("Input shapes: [%d, %d]\n\n", shape[0], shape[1]);

    // ============================================
    // DEAD CODE: These operations are never used
    // ============================================
    printf("Creating dead code (unused operations)...\n");

    // Dead branch 1: computed but never used in final result
    Tensor* dead1 = tensor_mul(x, w1);       // Dead: x * w1
    Tensor* dead2 = tensor_add(dead1, bias); // Dead: (x * w1) + bias
    Tensor* dead3 = tensor_relu(dead2);      // Dead: relu((x * w1) + bias)

    // Dead branch 2: another unused computation
    Tensor* dead4 = tensor_sub(w1, w2);       // Dead: w1 - w2
    Tensor* dead5 = tensor_mul(dead4, dead4); // Dead: (w1 - w2)^2

    printf("  - Created 5 dead nodes (will be eliminated)\n");

    // ============================================
    // UNOPTIMIZED: Chain of element-wise ops
    // These can be fused into a single kernel
    // ============================================
    printf("\nCreating unoptimized element-wise chain...\n");

    // Unoptimized chain: x * w2 + bias -> relu -> * 2 -> + 1
    // Each of these is a separate kernel in unoptimized IR
    Tensor* t1 = tensor_mul(x, w2);    // Kernel 1: multiply
    Tensor* t2 = tensor_add(t1, bias); // Kernel 2: add bias
    Tensor* t3 = tensor_relu(t2);      // Kernel 3: relu

    // More operations that could be fused
    Tensor* scale = tensor_full(shape, 2, &config, 2.0f);
    Tensor* t4    = tensor_mul(t3, scale); // Kernel 4: scale by 2

    Tensor* offset = tensor_full(shape, 2, &config, 1.0f);
    Tensor* t5     = tensor_add(t4, offset); // Kernel 5: add offset

    printf("  - Created chain of 5 element-wise ops (can be fused)\n");

    // ============================================
    // REDUNDANT: Duplicate computations
    // ============================================
    printf("\nCreating redundant computations...\n");

    // These compute the same thing - CSE can eliminate
    Tensor* r1 = tensor_mul(x, w2);  // Same as t1
    Tensor* r2 = tensor_mul(x, w2);  // Same as t1 (redundant)
    Tensor* r3 = tensor_add(r1, r2); // Uses redundant computation

    printf("  - Created 2 redundant mul ops (CSE opportunity)\n");

    // ============================================
    // FINAL RESULT: Only this path matters
    // ============================================
    printf("\nComputing final result...\n");

    // Combine the live computation with redundant one
    Tensor* result = tensor_add(t5, r3);

    // Compute mean to get scalar output
    Tensor* output = tensor_mean(result, -1, false);

    // Force evaluation
    float value = tensor_get_float(output, 0);
    printf("\nFinal output value: %.6f\n", value);

    // ============================================
    // EXPORT IR FOR VISUALIZATION
    // ============================================
    printf("\n=== Exporting IR Analysis ===\n");

    if (output->ir_context) {
        // Export unoptimized kernels
        char* unopt = cml_ir_export_kernel_analysis(output->ir_context, false);

        // Run optimization passes
        printf("\nRunning optimization passes:\n");
        printf("  - Dead code elimination\n");
        printf("  - Common subexpression elimination\n");
        printf("  - Kernel fusion\n");
        cml_ir_optimize(output->ir_context);

        // Export optimized graph - has correct dead/fused counts matching Kernel Studio
        autograd_export_json(output, "graph.json");
        printf("Exported graph.json\n");

        // Export optimized kernels
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

    // Print summary
    printf("\n=== Summary ===\n");
    printf("Dead nodes created:     5 (should be eliminated)\n");
    printf("Element-wise chain:     5 ops (should be fused)\n");
    printf("Redundant computations: 2 (CSE should eliminate 1)\n");
    printf("\nRun with VIZ=1 to see optimization in Kernel Studio!\n");

    // Cleanup
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
