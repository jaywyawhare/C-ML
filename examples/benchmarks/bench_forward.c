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

    cml_print_simd_caps();

    CMLBlasContext* blas = cml_blas_get_context();
    if (blas && blas->initialized)
        printf("BLAS: %s\n", cml_blas_get_library_name(blas));
    else
        printf("BLAS: NOT AVAILABLE (using slow fallback)\n");

    int batch_size     = 64;
    int input_size     = 784;
    int hidden_size    = 128;
    int output_size    = 10;
    int num_iterations = 100;

    printf("Forward Pass Benchmark\n");
    printf("Batch size: %d, Input: %d, Hidden: %d, Output: %d\n", batch_size, input_size,
           hidden_size, output_size);
    printf("Iterations: %d\n\n", num_iterations);

    int x_shape[]       = {batch_size, input_size};
    TensorConfig config = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    Tensor* X = tensor_ones(x_shape, 2, &config);

    Sequential* model = cml_nn_sequential();
    DeviceType device = cml_get_default_device();
    DType dtype       = cml_get_default_dtype();

    model = cml_nn_sequential_add(
        model, (Module*)cml_nn_linear(input_size, hidden_size, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(
        model, (Module*)cml_nn_linear(hidden_size, output_size, dtype, device, true));

    cml_summary((Module*)model);

    printf("\nWarmup...\n");
    for (int i = 0; i < 5; i++) {
        Tensor* out = cml_nn_module_forward((Module*)model, X);
        (void)tensor_data_ptr(out);
    }

    cml_reset_exec_stats();

    printf("Benchmarking forward pass...\n");
    double start = get_time_sec();

    for (int i = 0; i < num_iterations; i++) {
        Tensor* out = cml_nn_module_forward((Module*)model, X);
        float* data = (float*)tensor_data_ptr(out);
        if (i == 0 || i == num_iterations - 1) {
            printf("  Iter %d output[0]: %.6f\n", i, (double)data[0]);
        }
    }

    double elapsed = get_time_sec() - start;

    printf("\nResults:\n");
    printf("  Total time: %.3f s\n", elapsed);
    printf("  Time per forward: %.3f ms\n", (elapsed / num_iterations) * 1000);
    printf("  Throughput: %.1f samples/sec\n", (num_iterations * batch_size) / elapsed);

    printf("\n");
    cml_graph_cache_print_stats(NULL);
    printf("\n");
    cml_print_exec_stats();
    printf("\n");
    cml_print_buffer_cache_stats();

    tensor_free(X);
    cml_cleanup();
    return 0;
}
