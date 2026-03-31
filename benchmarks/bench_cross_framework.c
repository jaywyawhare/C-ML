/**
 * Cross-framework benchmark for CML.
 * Outputs JSON so the Python driver can parse and compare.
 *
 * Benchmarks (all float32):
 *   1. GEMM: NxN matmul (N=512, 1024, 2048)
 *   2. Fused: matmul + bias + relu (same sizes)
 *   3. MLP forward: batch=64, 784->128->ReLU->10, 100 iters
 *   4. MLP training step: forward + MSE loss + backward + SGD step
 *   5. Conv2d forward: batch=8, 3x32x32 -> 16x30x30
 *
 * Set CML_BACKEND=opencl to benchmark OpenCL GPU path.
 */
#define _POSIX_C_SOURCE 199309L
#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static DeviceType g_device = DEVICE_CPU;

static void cooldown_ms(int ms) {
    struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_random(float* buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static double median(double* arr, int n) {
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    return (n % 2) ? arr[n/2] : (arr[n/2 - 1] + arr[n/2]) / 2.0;
}

static double bench_gemm(int N) {
    int shape[] = {N, N};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = g_device,
                        .has_dtype = true, .has_device = true};

    float* a_data = malloc(sizeof(float) * N * N);
    float* b_data = malloc(sizeof(float) * N * N);
    fill_random(a_data, N * N);
    fill_random(b_data, N * N);

    Tensor* A = cml_tensor(a_data, shape, 2, &cfg);
    Tensor* B = cml_tensor(b_data, shape, 2, &cfg);

    for (int i = 0; i < 3; i++) {
        Tensor* C = cml_matmul(A, B);
        (void)tensor_data_ptr(C);
        cml_reset_ir_context();
    }

    /* Adaptive: fewer iterations for large sizes to reduce thermal throttling */
    int iters = N >= 2048 ? 3 : 5;
    int rounds = N >= 2048 ? 5 : 5;
    double times[5];
    for (int r = 0; r < rounds; r++) {
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* C = cml_matmul(A, B);
            (void)tensor_data_ptr(C);
            cml_reset_ir_context();
        }
        times[r] = (now() - t0) / iters * 1e3;
    }

    free(a_data);
    free(b_data);
    return median(times, rounds);
}

static double bench_fused(int N) {
    int mat_shape[] = {N, N};
    int bias_shape[] = {1, N};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = g_device,
                        .has_dtype = true, .has_device = true};

    float* a_data = malloc(sizeof(float) * N * N);
    float* b_data = malloc(sizeof(float) * N * N);
    float* bias_data = malloc(sizeof(float) * N);
    fill_random(a_data, N * N);
    fill_random(b_data, N * N);
    fill_random(bias_data, N);

    Tensor* A = cml_tensor(a_data, mat_shape, 2, &cfg);
    Tensor* B = cml_tensor(b_data, mat_shape, 2, &cfg);
    Tensor* bias = cml_tensor(bias_data, bias_shape, 2, &cfg);

    for (int i = 0; i < 3; i++) {
        Tensor* C = cml_relu(cml_add(cml_matmul(A, B), bias));
        (void)tensor_data_ptr(C);
        cml_reset_ir_context();
    }

    int iters = N >= 2048 ? 3 : 5;
    int rounds = N >= 2048 ? 5 : 5;
    double times[5];
    for (int r = 0; r < rounds; r++) {
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* C = cml_relu(cml_add(cml_matmul(A, B), bias));
            (void)tensor_data_ptr(C);
            cml_reset_ir_context();
        }
        times[r] = (now() - t0) / iters * 1e3;
    }

    free(a_data);
    free(b_data);
    free(bias_data);
    return median(times, rounds);
}

static double bench_mlp_forward(void) {
    int batch = 64, in_f = 784, hid = 128, out_f = 10;
    int x_shape[] = {batch, in_f};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = g_device,
                        .has_dtype = true, .has_device = true};

    float* x_data = malloc(sizeof(float) * batch * in_f);
    fill_random(x_data, batch * in_f);
    Tensor* X = cml_tensor(x_data, x_shape, 2, &cfg);

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(in_f, hid, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(hid, out_f, DTYPE_FLOAT32, DEVICE_CPU, true));
    module_set_training((Module*)model, false);

    for (int i = 0; i < 5; i++) {
        Tensor* out = cml_nn_sequential_forward(model, X);
        (void)tensor_data_ptr(out);
        cml_reset_ir_context();
    }

    int iters = 100;
    double times[5];
    for (int r = 0; r < 5; r++) {
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = cml_nn_sequential_forward(model, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }
        times[r] = (now() - t0) / iters * 1e3;
    }

    free(x_data);
    module_free((Module*)model);
    return median(times, 5);
}

static double bench_mlp_train(void) {
    int batch = 64, in_f = 784, hid = 128, out_f = 10;
    int x_shape[] = {batch, in_f};
    int y_shape[] = {batch, out_f};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = g_device,
                        .has_dtype = true, .has_device = true};

    float* x_data = malloc(sizeof(float) * batch * in_f);
    float* y_data = malloc(sizeof(float) * batch * out_f);
    fill_random(x_data, batch * in_f);
    fill_random(y_data, batch * out_f);

    Tensor* X = cml_tensor(x_data, x_shape, 2, &cfg);
    Tensor* Y = cml_tensor(y_data, y_shape, 2, &cfg);

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(in_f, hid, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(hid, out_f, DTYPE_FLOAT32, DEVICE_CPU, true));
    module_set_training((Module*)model, true);

    Optimizer* opt = cml_optim_sgd_for_model((Module*)model, 0.01f, 0.0f, 0.0f);

    for (int i = 0; i < 5; i++) {
        Tensor* out = cml_nn_sequential_forward(model, X);
        Tensor* loss = cml_nn_mse_loss(out, Y);
        cml_optim_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);
        cml_reset_ir_context();
    }

    int iters = 50;
    double times[5];
    for (int r = 0; r < 5; r++) {
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = cml_nn_sequential_forward(model, X);
            Tensor* loss = cml_nn_mse_loss(out, Y);
            cml_optim_zero_grad(opt);
            cml_backward(loss, NULL, false, false);
            cml_optim_step(opt);
            cml_reset_ir_context();
        }
        times[r] = (now() - t0) / iters * 1e3;
    }

    free(x_data);
    free(y_data);
    optimizer_free(opt);
    module_free((Module*)model);
    return median(times, 5);
}

static double bench_conv2d(void) {
    int batch = 8, ic = 3, h = 32, w = 32, oc = 16, ksize = 3;
    int x_shape[] = {batch, ic, h, w};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = g_device,
                        .has_dtype = true, .has_device = true};

    int numel = batch * ic * h * w;
    float* x_data = malloc(sizeof(float) * numel);
    fill_random(x_data, numel);
    Tensor* X = cml_tensor(x_data, x_shape, 4, &cfg);

    Conv2d* conv_layer = cml_nn_conv2d(ic, oc, ksize, 1, 0, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
    Module* conv = (Module*)conv_layer;
    module_set_training(conv, false);

    for (int i = 0; i < 5; i++) {
        Tensor* out = module_forward(conv, X);
        (void)tensor_data_ptr(out);
        cml_reset_ir_context();
    }

    int iters = 100;
    double times[5];
    for (int r = 0; r < 5; r++) {
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = module_forward(conv, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }
        times[r] = (now() - t0) / iters * 1e3;
    }

    free(x_data);
    module_free(conv);
    return median(times, 5);
}

int main(void) {
    const char* backend = getenv("CML_BACKEND");
    if (backend && strcmp(backend, "opencl") == 0) {
        g_device = DEVICE_OPENCL;
        fprintf(stderr, "bench: using OpenCL GPU backend\n");
    }

    cml_init();
    srand(42);

    double gemm_512  = bench_gemm(512);
    double fused_512  = bench_fused(512);
    double gemm_1024 = bench_gemm(1024);
    double fused_1024 = bench_fused(1024);
    cooldown_ms(200);
    double gemm_2048 = bench_gemm(2048);
    cooldown_ms(200);
    double fused_2048 = bench_fused(2048);
    cooldown_ms(100);
    double mlp_fwd   = bench_mlp_forward();
    double mlp_train  = bench_mlp_train();
    double conv2d_fwd = bench_conv2d();

    printf("{\n");
    printf("  \"gemm_512\": %.3f,\n", gemm_512);
    printf("  \"gemm_1024\": %.3f,\n", gemm_1024);
    printf("  \"gemm_2048\": %.3f,\n", gemm_2048);
    printf("  \"fused_512\": %.3f,\n", fused_512);
    printf("  \"fused_1024\": %.3f,\n", fused_1024);
    printf("  \"fused_2048\": %.3f,\n", fused_2048);
    printf("  \"mlp_forward\": %.3f,\n", mlp_fwd);
    printf("  \"mlp_train_step\": %.3f,\n", mlp_train);
    printf("  \"conv2d_forward\": %.3f\n", conv2d_fwd);
    printf("}\n");

    cml_cleanup();
    return 0;
}
