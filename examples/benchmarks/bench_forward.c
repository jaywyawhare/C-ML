/**
 * @file bench_forward.c
 * @brief Benchmark forward pass performance
 */

#include "cml.h"
#include "nn/layers/sequential.h"
#include "ops/ir/execution.h"
#include "ops/ir/graph_cache.h"
#include "ops/ir/context.h"
#include "ops/simd_math.h"
#include "backend/blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Wall-clock time (clock() measures CPU time which inflates with multi-threading)
static double get_time_sec(void) {
#if defined(__linux__) || defined(__APPLE__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

int main(void) {
    cml_init();
    cml_seed(42);

    // Print SIMD capabilities
    cml_print_simd_caps();

    // Check BLAS availability
    CMLBlasContext* blas = cml_blas_get_context();
    if (blas && blas->initialized) {
        const char* lib_name = cml_blas_get_library_name(blas);
        printf("BLAS: %s\n", lib_name);
        // Tips based on library
        if (strstr(lib_name, "openblas")) {
            printf("  TIP: For small models, try: OPENBLAS_NUM_THREADS=1\n");
            printf("       Or use reference BLAS: CML_BLAS_LIB=libcblas.so.3\n");
        }
    } else {
        printf("BLAS: NOT AVAILABLE (using slow fallback!)\n");
        printf("  TIP: Install a BLAS library for faster matmul\n");
    }

    int batch_size     = 64;
    int input_size     = 784;
    int hidden_size    = 128;
    int output_size    = 10;
    int num_iterations = 100;

    printf("=== Forward Pass Benchmark ===\n");
    printf("Batch size: %d, Input: %d, Hidden: %d, Output: %d\n", batch_size, input_size,
           hidden_size, output_size);
    printf("Iterations: %d\n\n", num_iterations);

    // Create input tensor
    int x_shape[]       = {batch_size, input_size};
    TensorConfig config = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    Tensor* X = tensor_ones(x_shape, 2, &config); // Just need any input

    // Build simple model: Linear -> ReLU -> Linear
    Sequential* model = cml_nn_sequential();
    DeviceType device = cml_get_default_device();
    DType dtype       = cml_get_default_dtype();

    model = cml_nn_sequential_add(
        model, (Module*)cml_nn_linear(input_size, hidden_size, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(
        model, (Module*)cml_nn_linear(hidden_size, output_size, dtype, device, true));

    // Graph caching is available but currently slower than direct execution
    // due to BLAS matmul dominating execution time. Useful when:

    cml_summary((Module*)model);

    // Warmup
    printf("\nWarmup...\n");
    for (int i = 0; i < 5; i++) {
        Tensor* out = cml_nn_module_forward((Module*)model, X);
        // Force execution by accessing data
        float* data = (float*)tensor_data_ptr(out);
        (void)data;
    }

    // Reset stats before benchmark
    cml_reset_exec_stats();

    // Benchmark forward pass
    printf("Benchmarking forward pass...\n");
    double start = get_time_sec();

    for (int i = 0; i < num_iterations; i++) {
        // IR context is now automatically reset in Sequential forward
        Tensor* out = cml_nn_module_forward((Module*)model, X);
        float* data = (float*)tensor_data_ptr(out);
        // Print first output to verify computation is happening
        if (i == 0 || i == num_iterations - 1) {
            printf("  Iter %d output[0]: %.6f\n", i, (double)data[0]);
        }
    }

    double elapsed = get_time_sec() - start;

    printf("\nResults:\n");
    printf("  Total time: %.3f s\n", elapsed);
    printf("  Time per forward: %.3f ms\n", (elapsed / num_iterations) * 1000);
    printf("  Throughput: %.1f samples/sec\n", (num_iterations * batch_size) / elapsed);

    // Print cache stats
    printf("\n");
    cml_graph_cache_print_stats(NULL); // Print global cache stats

    // Print execution stats
    printf("\n");
    cml_print_exec_stats();

    // Print buffer cache stats
    printf("\n");
    cml_print_buffer_cache_stats();

    tensor_free(X);
    cml_cleanup();
    return 0;
}
