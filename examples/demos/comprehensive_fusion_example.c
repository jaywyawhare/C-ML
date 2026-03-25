#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    cml_init();

    printf("Comprehensive Fusion Example\n\n");

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

    int shape[]         = {8};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* a = tensor_empty(shape, 1, &config);
    Tensor* b = tensor_empty(shape, 1, &config);
    Tensor* c = tensor_empty(shape, 1, &config);

    float* a_data = (float*)tensor_data_ptr(a);
    float* b_data = (float*)tensor_data_ptr(b);
    float* c_data = (float*)tensor_data_ptr(c);
    for (int i = 0; i < 8; i++) {
        a_data[i] = (float)(i + 1) * 0.5f;  /* 0.5 .. 4.0 */
        b_data[i] = (float)(i + 2) * 0.3f;  /* 0.6 .. 2.7 */
        c_data[i] = (float)(i + 3) * 0.2f;  /* 0.6 .. 2.0 */
    }

    printf("Input tensors initialized\n");
    printf("A: [%.2f, %.2f, %.2f, %.2f, ...]\n", (double)a_data[0], (double)a_data[1],
           (double)a_data[2], (double)a_data[3]);
    printf("B: [%.2f, %.2f, %.2f, %.2f, ...]\n", (double)b_data[0], (double)b_data[1],
           (double)b_data[2], (double)b_data[3]);
    printf("C: [%.2f, %.2f, %.2f, %.2f, ...]\n", (double)c_data[0], (double)c_data[1],
           (double)c_data[2], (double)c_data[3]);

    printf("\nCreating computation graph\n");

    /* Test 1: FUSION_FMA - MUL + ADD -> FMA
       result1 = a * b + c */
    printf("\n1. FUSION_FMA test: result1 = a * b + c\n");
    Tensor* mul_ab  = tensor_mul(a, b);
    Tensor* result1 = tensor_add(mul_ab, c);

    /* Test 2: Long elementwise chain (should fuse into one kernel)
       result2 = sqrt(exp(log(a + c)))
       (using a+c to avoid duplicate a+b; values always positive) */
    printf("2. Long chain test: result2 = sqrt(exp(log(a + c)))\n");
    Tensor* add_ac     = tensor_add(a, c);
    Tensor* log_chain  = tensor_log(add_ac);
    Tensor* exp_chain  = tensor_exp(log_chain);
    Tensor* result2    = tensor_sqrt(exp_chain);

    /* Test 3: Complex chain with multiple operations
       result3 = exp((a * c) / (b + c)) - sqrt(b) */
    printf("3. Complex chain: result3 = exp((a*c)/(b+c)) - sqrt(b)\n");
    Tensor* mul_ac     = tensor_mul(a, c);
    Tensor* add_bc     = tensor_add(b, c);
    Tensor* div_result = tensor_div(mul_ac, add_bc);
    Tensor* exp_result = tensor_exp(div_result);
    Tensor* sqrt_b     = tensor_sqrt(b);
    Tensor* result3    = tensor_sub(exp_result, sqrt_b);

    /* Test 4: Dead code - operations that won't be used
       These should be eliminated by dead code elimination */
    printf("4. Dead code test: Creating unused operations\n");
    Tensor* dead_mul = tensor_mul(b, c);
    Tensor* dead_add = tensor_add(dead_mul, a);
    Tensor* dead_exp = tensor_exp(dead_add);

    /* Test 5: NEG + operations (should fuse)
       result4 = exp(-a) + log(b) */
    printf("5. NEG fusion test: result4 = exp(-a) + log(b)\n");
    Tensor* neg_a     = tensor_neg(a);
    Tensor* exp_neg_a = tensor_exp(neg_a);
    Tensor* log_b     = tensor_log(b);
    Tensor* result4   = tensor_add(exp_neg_a, log_b);

    /* Test 6: Very long chain (testing max fusion)
       result5 = sqrt(exp(log(sqrt(a + b)))) */
    printf("6. Very long chain: result5 = sqrt(exp(log(sqrt(a+b))))\n");
    Tensor* add_ab     = tensor_add(a, b);
    Tensor* sqrt_long1 = tensor_sqrt(add_ab);
    Tensor* log_long   = tensor_log(sqrt_long1);
    Tensor* exp_long   = tensor_exp(log_long);
    Tensor* result5    = tensor_sqrt(exp_long);

    /* Test 7: Simple elementwise
       result6 = a + (b * c) */
    printf("7. Simple fusion: result6 = a + (b * c)\n");
    Tensor* mul_bc  = tensor_mul(b, c);
    Tensor* result6 = tensor_add(a, mul_bc);

    /* Combine ALL results into one final output so they are all "live" */
    printf("\n8. Combining all results so they are reachable...\n");
    Tensor* sum1         = tensor_add(result1, result2);
    Tensor* sum2         = tensor_add(sum1, result3);
    Tensor* sum3         = tensor_add(sum2, result4);
    Tensor* sum4         = tensor_add(sum3, result5);
    Tensor* final_result = tensor_add(sum4, result6);

    printf("\nExecuting computation graphs\n");

    float* r1 = (float*)tensor_data_ptr(result1);
    float* r2 = (float*)tensor_data_ptr(result2);
    float* r3 = (float*)tensor_data_ptr(result3);
    float* r4 = (float*)tensor_data_ptr(result4);
    float* r5 = (float*)tensor_data_ptr(result5);
    float* r6 = (float*)tensor_data_ptr(result6);
    float* rf = (float*)tensor_data_ptr(final_result);

    int ok = 1;
    if (!r1 || !r2 || !r3 || !r4 || !r5 || !r6 || !rf) {
        printf("ERROR: One or more results returned NULL\n");
        if (!r1) printf("  result1 is NULL\n");
        if (!r2) printf("  result2 is NULL\n");
        if (!r3) printf("  result3 is NULL\n");
        if (!r4) printf("  result4 is NULL\n");
        if (!r5) printf("  result5 is NULL\n");
        if (!r6) printf("  result6 is NULL\n");
        if (!rf) printf("  final_result is NULL\n");
        ok = 0;
    }

    if (ok) {
        printf("\nResults (first 4 elements):\n");
        printf("result1[0-3]: [%.4f, %.4f, %.4f, %.4f]\n",
               (double)r1[0], (double)r1[1], (double)r1[2], (double)r1[3]);
        printf("result2[0-3]: [%.4f, %.4f, %.4f, %.4f]\n",
               (double)r2[0], (double)r2[1], (double)r2[2], (double)r2[3]);
        printf("result3[0-3]: [%.4f, %.4f, %.4f, %.4f]\n",
               (double)r3[0], (double)r3[1], (double)r3[2], (double)r3[3]);
        printf("result4[0-3]: [%.4f, %.4f, %.4f, %.4f]\n",
               (double)r4[0], (double)r4[1], (double)r4[2], (double)r4[3]);
        printf("result5[0-3]: [%.4f, %.4f, %.4f, %.4f]\n",
               (double)r5[0], (double)r5[1], (double)r5[2], (double)r5[3]);
        printf("result6[0-3]: [%.4f, %.4f, %.4f, %.4f]\n",
               (double)r6[0], (double)r6[1], (double)r6[2], (double)r6[3]);

        /* Verify results manually */
        printf("\nVerification:\n");
        int pass = 1;
        for (int i = 0; i < 4; i++) {
            float ai = a_data[i], bi = b_data[i], ci = c_data[i];

            /* result1 = a*b + c */
            float e1 = ai * bi + ci;
            int ok1 = fabsf(r1[i] - e1) < 1e-3f;
            if (!ok1) { printf("  FAIL result1[%d]: got=%.4f expected=%.4f\n", i, (double)r1[i], (double)e1); pass = 0; }

            /* result2 = sqrt(exp(log(a+c))) ~ sqrt(a+c), with float precision loss */
            float e2 = sqrtf(ai + ci);
            int ok2 = fabsf(r2[i] - e2) < 0.01f;
            if (!ok2) { printf("  FAIL result2[%d]: got=%.4f expected=%.4f\n", i, (double)r2[i], (double)e2); pass = 0; }

            /* result6 = a + b*c */
            float e6 = ai + bi * ci;
            int ok6 = fabsf(r6[i] - e6) < 1e-3f;
            if (!ok6) { printf("  FAIL result6[%d]: got=%.4f expected=%.4f\n", i, (double)r6[i], (double)e6); pass = 0; }
        }
        if (pass) printf("  All verifications passed!\n");
    }

    /* Export kernel analysis */
    printf("\nExporting kernel analysis\n");
    char* raw_json = cml_ir_export_kernel_analysis(ir, false);

    printf("Optimizing IR (fusion, dead code elimination)\n");
    cml_ir_optimize(ir);

    char* opt_json = cml_ir_export_kernel_analysis(ir, true);

    if (raw_json && opt_json) {
        FILE* f = fopen("kernels.json", "w");
        if (f) {
            fprintf(f, "{\"unoptimized\":%s,\"optimized\":%s}", raw_json, opt_json);
            fclose(f);
            printf("Exported kernel analysis to kernels.json\n");
        }
    }
    free(raw_json);
    free(opt_json);

    char* graph_json = cml_ir_export_graph_json(ir);
    if (graph_json) {
        FILE* f = fopen("graph.json", "w");
        if (f) {
            fprintf(f, "%s", graph_json);
            fclose(f);
            printf("Exported graph topology to graph.json\n");
        }
        free(graph_json);
    }

    /* Print IR summary */
    char* ir_str = cml_ir_to_string(ir);
    if (ir_str) {
        printf("\nIR Summary:\n%s\n", ir_str);
        free(ir_str);
    }

    /* Clean up */
    tensor_free(final_result);
    tensor_free(sum4);
    tensor_free(sum3);
    tensor_free(sum2);
    tensor_free(sum1);
    tensor_free(result6);
    tensor_free(mul_bc);
    tensor_free(result5);
    tensor_free(exp_long);
    tensor_free(log_long);
    tensor_free(sqrt_long1);
    tensor_free(add_ab);
    tensor_free(result4);
    tensor_free(log_b);
    tensor_free(exp_neg_a);
    tensor_free(neg_a);
    tensor_free(dead_exp);
    tensor_free(dead_add);
    tensor_free(dead_mul);
    tensor_free(result3);
    tensor_free(sqrt_b);
    tensor_free(exp_result);
    tensor_free(div_result);
    tensor_free(add_bc);
    tensor_free(mul_ac);
    tensor_free(result2);
    tensor_free(exp_chain);
    tensor_free(log_chain);
    tensor_free(add_ac);
    tensor_free(result1);
    tensor_free(mul_ab);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);

    cml_ir_disable_auto_capture();
    cml_ir_free(ir);
    cml_cleanup();

    printf("\nExecution completed successfully\n");
    return 0;
}
