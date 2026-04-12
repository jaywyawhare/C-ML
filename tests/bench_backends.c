#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "cml.h"
#include "ops/ir/dispatch.h"
#include "ops/ir/kernel_cache.h"
#include "ops/ir/ir.h"
#include "ops/ir/context.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "backend/blas.h"
#include "core/logging.h"

typedef struct {
    struct timespec start;
    struct timespec end;
} BenchTimer;

static void bench_timer_start(BenchTimer* t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

static double bench_timer_stop(BenchTimer* t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
    double elapsed = (t->end.tv_sec - t->start.tv_sec) * 1000.0;
    elapsed += (t->end.tv_nsec - t->start.tv_nsec) / 1000000.0;
    return elapsed;  // milliseconds
}

static void bench_matmul(int size, int iterations) {
    printf("\nMatrix Multiplication (size=%dx%d, iterations=%d)\n\n",
           size, size, iterations);

    int mat_shape[] = {size, size};
    TensorConfig cpu_cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* A = tensor_empty(mat_shape, 2, &cpu_cfg);
    Tensor* B = tensor_empty(mat_shape, 2, &cpu_cfg);

    if (!A || !B) {
        printf("Failed to allocate tensors\n");
        return;
    }

    float* a_data = (float*)A->data;
    float* b_data = (float*)B->data;
    for (int i = 0; i < size * size; i++) {
        a_data[i] = (float)(rand() % 100) / 100.0f;
        b_data[i] = (float)(rand() % 100) / 100.0f;
    }

    BenchTimer timer;

    {
        CMLDispatchContext* ctx = cml_dispatch_create();
        cml_dispatch_init(ctx);
        cml_dispatch_set_preferred(ctx, CML_BACKEND_CPU_FALLBACK);

        double total_time = 0;
        for (int i = 0; i < iterations; i++) {
            CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
            cml_ir_set_global_context(ir);
            Tensor* C = tensor_matmul(A, B);
            Tensor* inputs[] = {A, B};
            Tensor* outputs[] = {C};
            bench_timer_start(&timer);
            cml_dispatch_execute(ctx, ir, inputs, 2, outputs, 1);
            total_time += bench_timer_stop(&timer);
            tensor_free(C);
            cml_ir_free(ir);
        }

        printf("  CPU Fallback:    %8.2f ms avg (%.2f ms total)\n",
               total_time / iterations, total_time);
        cml_dispatch_free(ctx);
    }

    {
        CMLBlasContext* blas = cml_blas_init();
        if (blas) {
            float* C = (float*)calloc(size * size, sizeof(float));

            double total_time = 0;
            for (int i = 0; i < iterations; i++) {
                memset(C, 0, size * size * sizeof(float));
                bench_timer_start(&timer);
                cml_blas_sgemm(blas, a_data, b_data, C, size, size, size, 1.0f, 0.0f);
                total_time += bench_timer_stop(&timer);
            }

            printf("  BLAS (%s): %8.2f ms avg (%.2f ms total)\n",
                   cml_blas_get_library_name(blas), total_time / iterations, total_time);

            free(C);
            cml_blas_free(blas);
        } else {
            printf("  BLAS:            Not available\n");
        }
    }

    {
        CMLDispatchContext* ctx = cml_dispatch_create();
        cml_dispatch_init(ctx);

        if (cml_dispatch_backend_available(ctx, CML_BACKEND_CPU_LLVM)) {
            cml_dispatch_set_preferred(ctx, CML_BACKEND_CPU_LLVM);
            cml_dispatch_enable_cache(ctx, 100);

            CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
            cml_ir_set_global_context(ir);

            Tensor* C = tensor_matmul(A, B);
            if (!C) {
                printf("  CPU LLVM JIT:    Failed to create matmul operation\n");
                cml_ir_free(ir);
                cml_dispatch_free(ctx);
                tensor_free(A);
                tensor_free(B);
                return;
            }

            Tensor* inputs[] = {A, B};
            Tensor* outputs[] = {C};

            double total_time = 0;
            for (int i = 0; i < iterations; i++) {
                bench_timer_start(&timer);
                cml_dispatch_execute(ctx, ir, inputs, 2, outputs, 1);
                total_time += bench_timer_stop(&timer);
            }

            size_t cache_hits = 0, cache_misses = 0, cache_size = 0;
            cml_dispatch_cache_stats(ctx, &cache_hits, &cache_misses, &cache_size);

            printf("  CPU LLVM JIT:    %8.2f ms avg (%.2f ms total) [cache: %zu hits, %zu misses]\n",
                   total_time / iterations, total_time, cache_hits, cache_misses);

            tensor_free(C);
            cml_ir_free(ir);
        } else {
            printf("  CPU LLVM JIT:    Not available\n");
        }

        cml_dispatch_free(ctx);
    }

    {
        CMLDispatchContext* ctx = cml_dispatch_create();
        cml_dispatch_init(ctx);

        if (cml_dispatch_backend_available(ctx, CML_BACKEND_CUDA)) {
            cml_dispatch_set_preferred(ctx, CML_BACKEND_CUDA);
            cml_dispatch_enable_cache(ctx, 100);

            CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
            cml_ir_set_global_context(ir);

            Tensor* C = tensor_matmul(A, B);
            if (!C) {
                printf("  CUDA:            Failed to create matmul operation\n");
                cml_ir_free(ir);
                cml_dispatch_free(ctx);
                tensor_free(A);
                tensor_free(B);
                return;
            }

            Tensor* inputs[] = {A, B};
            Tensor* outputs[] = {C};

            double total_time = 0;
            for (int i = 0; i < iterations; i++) {
                bench_timer_start(&timer);
                cml_dispatch_execute(ctx, ir, inputs, 2, outputs, 1);
                total_time += bench_timer_stop(&timer);
            }

            size_t cache_hits = 0, cache_misses = 0, cache_size = 0;
            cml_dispatch_cache_stats(ctx, &cache_hits, &cache_misses, &cache_size);

            printf("  CUDA:            %8.2f ms avg (%.2f ms total) [cache: %zu hits, %zu misses]\n",
                   total_time / iterations, total_time, cache_hits, cache_misses);

            tensor_free(C);
            cml_ir_free(ir);
        } else {
            printf("  CUDA:            Not available\n");
        }

        cml_dispatch_free(ctx);
    }

    tensor_free(A);
    tensor_free(B);
}

static void bench_kernel_cache(int iterations) {
    printf("\nKernel Cache (iterations=%d)\n\n", iterations);

    CMLKernelCache* cache = cml_kernel_cache_create(1000);
    if (!cache) {
        printf("Failed to create cache\n");
        return;
    }

    BenchTimer timer;

    bench_timer_start(&timer);
    for (int i = 0; i < iterations; i++) {
        uint64_t hash = (uint64_t)i * 0x1234567890ABCDEFULL;
        cml_kernel_cache_insert(cache, hash, CML_KERNEL_CPU_LLVM, (void*)(uintptr_t)(i + 1), 1024);
    }
    double insert_time = bench_timer_stop(&timer);

    bench_timer_start(&timer);
    int hits = 0;
    for (int i = 0; i < iterations; i++) {
        uint64_t hash = (uint64_t)i * 0x1234567890ABCDEFULL;
        if (cml_kernel_cache_lookup(cache, hash)) hits++;
    }
    double hit_time = bench_timer_stop(&timer);

    bench_timer_start(&timer);
    int misses = 0;
    for (int i = iterations; i < iterations * 2; i++) {
        uint64_t hash = (uint64_t)i * 0x1234567890ABCDEFULL;
        if (!cml_kernel_cache_lookup(cache, hash)) misses++;
    }
    double miss_time = bench_timer_stop(&timer);

    printf("  Insert:     %8.2f ms for %d entries (%.2f us/entry)\n",
           insert_time, iterations, insert_time * 1000.0 / iterations);
    printf("  Lookup hit: %8.2f ms for %d lookups (%.2f us/lookup) [%d hits]\n",
           hit_time, iterations, hit_time * 1000.0 / iterations, hits);
    printf("  Lookup miss:%8.2f ms for %d lookups (%.2f us/lookup) [%d misses]\n",
           miss_time, iterations, miss_time * 1000.0 / iterations, misses);

    cml_kernel_cache_free(cache);
}

static void bench_elementwise(int size, int iterations) {
    printf("\nElement-wise Operations (size=%d, iterations=%d)\n\n",
           size, iterations);

    int shape[] = {size};
    TensorConfig cpu_cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* A = tensor_empty(shape, 1, &cpu_cfg);
    Tensor* B = tensor_empty(shape, 1, &cpu_cfg);

    if (!A || !B) {
        printf("Failed to allocate tensors\n");
        return;
    }

    float* a_data = (float*)A->data;
    float* b_data = (float*)B->data;
    for (int i = 0; i < size; i++) {
        a_data[i] = (float)(rand() % 100) / 100.0f;
        b_data[i] = (float)(rand() % 100) / 100.0f;
    }

    BenchTimer timer;

    {
        CMLDispatchContext* ctx = cml_dispatch_create();
        cml_dispatch_init(ctx);

        double total_time = 0;
        for (int i = 0; i < iterations; i++) {
            CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
            cml_ir_set_global_context(ir);
            Tensor* C = tensor_add(A, B);
            Tensor* inputs[] = {A, B};
            Tensor* outputs[] = {C};
            bench_timer_start(&timer);
            cml_dispatch_execute(ctx, ir, inputs, 2, outputs, 1);
            total_time += bench_timer_stop(&timer);
            tensor_free(C);
            cml_ir_free(ir);
        }

        printf("  Add:    %8.2f ms avg\n", total_time / iterations);
        cml_dispatch_free(ctx);
    }

    {
        CMLDispatchContext* ctx = cml_dispatch_create();
        cml_dispatch_init(ctx);

        double total_time = 0;
        for (int i = 0; i < iterations; i++) {
            CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
            cml_ir_set_global_context(ir);
            Tensor* C = tensor_mul(A, B);
            Tensor* inputs[] = {A, B};
            Tensor* outputs[] = {C};
            bench_timer_start(&timer);
            cml_dispatch_execute(ctx, ir, inputs, 2, outputs, 1);
            total_time += bench_timer_stop(&timer);
            tensor_free(C);
            cml_ir_free(ir);
        }

        printf("  Mul:    %8.2f ms avg\n", total_time / iterations);
        cml_dispatch_free(ctx);
    }

    {
        CMLDispatchContext* ctx = cml_dispatch_create();
        cml_dispatch_init(ctx);

        double total_time = 0;
        for (int i = 0; i < iterations; i++) {
            CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
            cml_ir_set_global_context(ir);
            Tensor* C = tensor_exp(A);
            Tensor* inputs[] = {A};
            Tensor* outputs[] = {C};
            bench_timer_start(&timer);
            cml_dispatch_execute(ctx, ir, inputs, 1, outputs, 1);
            total_time += bench_timer_stop(&timer);
            tensor_free(C);
            cml_ir_free(ir);
        }

        printf("  Exp:    %8.2f ms avg\n", total_time / iterations);
        cml_dispatch_free(ctx);
    }

    tensor_free(A);
    tensor_free(B);
}

static void bench_dispatch_overhead(int iterations) {
    printf("\nDispatch Overhead (iterations=%d)\n\n", iterations);

    BenchTimer timer;

    bench_timer_start(&timer);
    for (int i = 0; i < iterations; i++) {
        CMLDispatchContext* ctx = cml_dispatch_create();
        cml_dispatch_free(ctx);
    }
    double create_time = bench_timer_stop(&timer);

    bench_timer_start(&timer);
    for (int i = 0; i < iterations; i++) {
        CMLDispatchContext* ctx = cml_dispatch_create();
        cml_dispatch_init(ctx);
        cml_dispatch_free(ctx);
    }
    double init_time = bench_timer_stop(&timer);

    CMLDispatchContext* ctx = cml_dispatch_create();
    bench_timer_start(&timer);
    for (int i = 0; i < iterations; i++) {
        cml_dispatch_detect_backends(ctx);
    }
    double detect_time = bench_timer_stop(&timer);
    cml_dispatch_free(ctx);

    printf("  Context create/free: %8.2f ms (%.2f us/op)\n",
           create_time, create_time * 1000.0 / iterations);
    printf("  Context init:        %8.2f ms (%.2f us/op)\n",
           init_time - create_time, (init_time - create_time) * 1000.0 / iterations);
    printf("  Backend detection:   %8.2f ms (%.2f us/op)\n",
           detect_time, detect_time * 1000.0 / iterations);
}

static void print_system_info(void) {
    printf("\nSystem Information\n\n");

    CMLDispatchContext* ctx = cml_dispatch_create();
    cml_dispatch_init(ctx);

    printf("Available backends:\n");
    for (int i = 0; i < CML_BACKEND_COUNT; i++) {
        const CMLBackendInfo* info = cml_dispatch_get_backend_info(ctx, i);
        printf("  [%d] %-20s: %s\n", i, info->name,
               info->status != CML_BACKEND_STATUS_UNAVAILABLE ? "Available" : "Not available");
    }

    printf("\nBest backend: %s\n", cml_dispatch_backend_name(cml_dispatch_get_best_backend(ctx)));

    CMLBlasContext* blas = cml_blas_init();
    if (blas) {
        printf("BLAS library: %s\n", cml_blas_get_library_name(blas));
        cml_blas_free(blas);
    } else {
        printf("BLAS library: Not available\n");
    }

    cml_dispatch_free(ctx);
}

int main(int argc, char* argv[]) {
    int matmul_size = 256;
    int iterations = 10;
    int cache_iterations = 10000;
    int elementwise_size = 100000;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--size=", 7) == 0) {
            matmul_size = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--iterations=", 13) == 0) {
            iterations = atoi(argv[i] + 13);
        } else if (strncmp(argv[i], "--cache-iterations=", 19) == 0) {
            cache_iterations = atoi(argv[i] + 19);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --size=N            Matrix size for matmul benchmark (default: 256)\n");
            printf("  --iterations=N      Number of iterations (default: 10)\n");
            printf("  --cache-iterations=N Cache benchmark iterations (default: 10000)\n");
            printf("  --help, -h          Show this help\n");
            return 0;
        }
    }

    printf("\nCML Backend Performance Benchmark\n");

    srand(42);

    print_system_info();
    bench_dispatch_overhead(1000);
    bench_kernel_cache(cache_iterations);
    bench_matmul(matmul_size, iterations);
    bench_elementwise(elementwise_size, iterations);

    printf("\nBenchmark Complete\n");

    return 0;
}
