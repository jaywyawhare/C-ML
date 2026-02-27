/**
 * @file comprehensive_fusion_example.c
 * @brief Comprehensive example demonstrating all fusion cases, dead code elimination, and
 * optimizations
 *
 * This example shows:
 * 1. FUSION_FMA (MUL + ADD -> FMA)
 * 2. FUSION_CHAIN_ELEMENTWISE (long chains of elementwise operations)
 * 3. Dead code elimination (unused operations)
 * 4. Multiple operation types (ADD, SUB, MUL, DIV, EXP, LOG, SQRT, NEG, etc.)
 * 5. Generic code generation for any chain
 */

#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    cml_init();
    // Enable debug logging to see optimization details
    // cml_set_log_level(LOG_LEVEL_DEBUG);

    printf("=== Comprehensive Fusion Example ===\n\n");

    // Create IR context for CUDA generation
    CMLIR_t ir = cml_ir_new(IR_TARGET_CUDA);
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

    // Create input tensors
    int shape[]         = {8};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* a = tensor_empty(shape, 1, &config);
    Tensor* b = tensor_empty(shape, 1, &config);
    Tensor* c = tensor_empty(shape, 1, &config);

    // Initialize with some values
    float* a_data = (float*)tensor_data_ptr(a);
    float* b_data = (float*)tensor_data_ptr(b);
    float* c_data = (float*)tensor_data_ptr(c);
    for (int i = 0; i < 8; i++) {
        a_data[i] = (float)(i + 1) * 0.5f;
        b_data[i] = (float)(i + 2) * 0.3f;
        c_data[i] = (float)(i + 3) * 0.2f;
    }

    printf("Input tensors initialized\n");
    printf("A: [%.2f, %.2f, %.2f, %.2f, ...]\n", (double)a_data[0], (double)a_data[1],
           (double)a_data[2], (double)a_data[3]);
    printf("B: [%.2f, %.2f, %.2f, %.2f, ...]\n", (double)b_data[0], (double)b_data[1],
           (double)b_data[2], (double)b_data[3]);
    printf("C: [%.2f, %.2f, %.2f, %.2f, ...]\n", (double)c_data[0], (double)c_data[1],
           (double)c_data[2], (double)c_data[3]);

    printf("\n=== Creating computation graph ===\n");

    // Test Case 1: FUSION_FMA - MUL + ADD -> FMA
    // result1 = a * b + c (should fuse to fmaf)
    printf("\n1. FUSION_FMA test: result1 = a * b + c\n");
    Tensor* mul_result = tensor_mul(a, b);
    Tensor* result1    = tensor_add(mul_result, c);

    // Test Case 2: Long elementwise chain (should fuse into one kernel)
    // result2 = exp(log(sqrt((a + b) * (a - b))))
    printf("2. Long chain test: result2 = exp(log(sqrt((a+b)*(a-b))))\n");
    Tensor* add_ab     = tensor_add(a, b);
    Tensor* sub_ab     = tensor_sub(a, b);
    Tensor* mul_chain  = tensor_mul(add_ab, sub_ab);
    Tensor* sqrt_chain = tensor_sqrt(mul_chain);
    Tensor* log_chain  = tensor_log(sqrt_chain);
    Tensor* result2    = tensor_exp(log_chain);

    // Test Case 3: Complex chain with multiple operations
    // result3 = exp((a * b) / (c + a)) - sqrt(a)
    printf("3. Complex chain: result3 = exp((a*b)/(c+a)) - sqrt(a)\n");
    Tensor* mul_ab     = tensor_mul(a, b);
    Tensor* add_ca     = tensor_add(c, a);
    Tensor* div_result = tensor_div(mul_ab, add_ca);
    Tensor* exp_result = tensor_exp(div_result);
    Tensor* sqrt_a     = tensor_sqrt(a);
    Tensor* result3    = tensor_sub(exp_result, sqrt_a);

    // Test Case 4: Dead code - operations that won't be used
    // These should be eliminated by dead code elimination
    printf("4. Dead code test: Creating unused operations\n");
    Tensor* dead_mul = tensor_mul(b, c);        // Not used in any output
    Tensor* dead_add = tensor_add(dead_mul, a); // Not used
    Tensor* dead_exp = tensor_exp(dead_add);    // Not used

    // Test Case 5: NEG + operations (should fuse)
    // result4 = exp(-a) + log(-b)
    // Note: Using 0 - a instead of tensor_neg if not available
    printf("5. NEG fusion test: result4 = exp(-a) + log(-b)\n");
    Tensor* zeros_a   = tensor_ones(shape, 1, &config);
    float* zeros_data = (float*)tensor_data_ptr(zeros_a);
    for (int i = 0; i < 8; i++)
        zeros_data[i] = 0.0f;
    Tensor* neg_a       = tensor_sub(zeros_a, a);
    Tensor* exp_neg_a   = tensor_exp(neg_a);
    Tensor* zeros_b     = tensor_ones(shape, 1, &config);
    float* zeros_b_data = (float*)tensor_data_ptr(zeros_b);
    for (int i = 0; i < 8; i++)
        zeros_b_data[i] = 0.0f;
    Tensor* neg_b     = tensor_sub(zeros_b, b);
    Tensor* log_neg_b = tensor_log(neg_b);
    Tensor* result4   = tensor_add(exp_neg_a, log_neg_b);

    // Test Case 6: Very long chain (testing max fusion)
    // result5 = sqrt(exp(log(exp(sqrt(a + b)))))
    printf("6. Very long chain: result5 = sqrt(exp(log(exp(sqrt(a+b)))))\n");
    Tensor* add_long   = tensor_add(a, b);
    Tensor* sqrt_long1 = tensor_sqrt(add_long);
    Tensor* exp_long1  = tensor_exp(sqrt_long1);
    Tensor* log_long   = tensor_log(exp_long1);
    Tensor* exp_long2  = tensor_exp(log_long);
    Tensor* result5    = tensor_sqrt(exp_long2);

    // Test Case 7: Multiple outputs (some may be dead code)
    // result6 = a * b (simple, should fuse)
    printf("7. Simple fusion: result6 = a * b\n");
    Tensor* result6 = tensor_mul(a, b);

    // Combine ALL results into one final output so they are all "live"
    // final = result1 + result2 + result3 + result4 + result5 + result6
    printf("\n8. Combining all results so they are reachable...\n");
    Tensor* sum1         = tensor_add(result1, result2);
    Tensor* sum2         = tensor_add(sum1, result3);
    Tensor* sum3         = tensor_add(sum2, result4);
    Tensor* sum4         = tensor_add(sum3, result5);
    Tensor* final_result = tensor_add(sum4, result6);

    // Execute all graphs to verify correctness
    printf("\n=== Executing computation graphs ===\n");

    float* r1 = (float*)tensor_data_ptr(result1);
    float* r2 = (float*)tensor_data_ptr(result2);
    float* r3 = (float*)tensor_data_ptr(result3);
    float* r4 = (float*)tensor_data_ptr(result4);
    float* r5 = (float*)tensor_data_ptr(result5);
    float* r6 = (float*)tensor_data_ptr(result6);

    printf("\nResults (first 4 elements):\n");
    printf("result1[0-3]: [%.4f, %.4f, %.4f, %.4f]\n", (double)r1[0], (double)r1[1], (double)r1[2],
           (double)r1[3]);
    printf("result2[0-3]: [%.4f, %.4f, %.4f, %.4f]\n", (double)r2[0], (double)r2[1], (double)r2[2],
           (double)r2[3]);
    printf("result3[0-3]: [%.4f, %.4f, %.4f, %.4f]\n", (double)r3[0], (double)r3[1], (double)r3[2],
           (double)r3[3]);
    printf("result4[0-3]: [%.4f, %.4f, %.4f, %.4f]\n", (double)r4[0], (double)r4[1], (double)r4[2],
           (double)r4[3]);
    printf("result5[0-3]: [%.4f, %.4f, %.4f, %.4f]\n", (double)r5[0], (double)r5[1], (double)r5[2],
           (double)r5[3]);
    printf("result6[0-3]: [%.4f, %.4f, %.4f, %.4f]\n", (double)r6[0], (double)r6[1], (double)r6[2],
           (double)r6[3]);

    // Export raw kernel analysis (before optimization)
    printf("\n=== Exporting raw kernel analysis ===\n");
    char* raw_json = cml_ir_export_kernel_analysis(ir, false);

    // Now optimize IR to trigger fusion, dead code elimination, etc.
    printf("\n=== Optimizing IR (fusion, dead code elimination, etc.) ===\n");
    cml_ir_optimize(ir);

    // Export optimized kernel analysis (after optimization)
    printf("\n=== Exporting optimized kernel analysis ===\n");
    char* opt_json = cml_ir_export_kernel_analysis(ir, true);

    // Write to kernels.json
    if (raw_json && opt_json) {
        FILE* f = fopen("kernels.json", "w");
        if (f) {
            fprintf(f, "{\"unoptimized\":%s,\"optimized\":%s}", raw_json, opt_json);
            fclose(f);
            printf("Exported kernel analysis to kernels.json\n");
        } else {
            printf("Failed to open kernels.json for writing\n");
        }
    }
    if (raw_json)
        free(raw_json);
    if (opt_json)
        free(opt_json);

    // Export graph topology to graph.json
    // We use the optimized IR view for the graph
    printf("\n=== Exporting graph topology ===\n");
    char* graph_json = cml_ir_export_graph_json(ir);
    if (graph_json) {
        FILE* f = fopen("graph.json", "w");
        if (f) {
            fprintf(f, "%s", graph_json);
            fclose(f);
            printf("Exported graph topology to graph.json\n");
        } else {
            printf("Failed to open graph.json for writing\n");
        }
        free(graph_json);
    }

    // Generate optimized CUDA code
    printf("\n=== Generated CUDA code (after optimization) ===\n");
    char* cuda_code = cml_ir_compile(ir, NULL);
    if (cuda_code) {
        printf("%s\n", cuda_code);
        free(cuda_code);
    } else {
        printf("Failed to generate CUDA code\n");
    }

    // Print IR summary
    char* ir_str = cml_ir_to_string(ir);
    if (ir_str) {
        printf("\n=== IR Summary ===\n%s\n", ir_str);
        free(ir_str);
    }

    // Clean up
    printf("\n=== Cleaning up ===\n");
    tensor_free(final_result);
    tensor_free(sum4);
    tensor_free(sum3);
    tensor_free(sum2);
    tensor_free(sum1);
    tensor_free(result6);
    tensor_free(result5);
    tensor_free(result4);
    tensor_free(log_neg_b);
    tensor_free(neg_b);
    tensor_free(zeros_b);
    tensor_free(exp_neg_a);
    tensor_free(neg_a);
    tensor_free(zeros_a);
    tensor_free(dead_exp);
    tensor_free(dead_add);
    tensor_free(dead_mul);
    tensor_free(result3);
    tensor_free(sqrt_a);
    tensor_free(exp_result);
    tensor_free(div_result);
    tensor_free(add_ca);
    tensor_free(mul_ab);
    tensor_free(result2);
    tensor_free(log_chain);
    tensor_free(sqrt_chain);
    tensor_free(mul_chain);
    tensor_free(sub_ab);
    tensor_free(add_ab);
    tensor_free(result1);
    tensor_free(mul_result);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);

    cml_ir_disable_auto_capture();
    cml_ir_free(ir);
    cml_cleanup();

    printf("\n=== Example completed successfully ===\n");
    return 0;
}
