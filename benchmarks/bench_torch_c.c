/**
 * Cross-framework benchmark for the PyTorch-like torch_c C API.
 * Outputs JSON so the Python driver can parse and compare against PyTorch.
 *
 * Benchmarks (all float32):
 *   1. GEMM: NxN matmul (N=512, 1024, 2048)
 *   2. Fused: matmul + bias + relu (same sizes)
 *   3. MLP forward: batch=64, 784->128->ReLU->10, 100 iters
 *   4. MLP training step: forward + MSE loss + backward + SGD step
 *   5. Conv2d forward: batch=8, 3x32x32 -> 16x30x30
 *
 * Set CML_BACKEND=opencl to benchmark OpenCL GPU path.
 * Set CML_BACKEND=metal to benchmark Metal GPU path (macOS).
 */
#define _POSIX_C_SOURCE 199309L
#include "torch/torch_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define BENCH_FAIL(msg)                                                                   \
    do {                                                                                  \
        fprintf(stderr, "bench_torch_c: %s\n", msg);                                      \
        exit(1);                                                                          \
    } while (0)

#define BENCH_REQUIRE_ALLOC(p)                                                            \
    do {                                                                                  \
        if (!(p))                                                                         \
            BENCH_FAIL("allocation failed");                                              \
    } while (0)

#define BENCH_REQUIRE_TENSOR(t)                                                           \
    do {                                                                                  \
        if (!(t))                                                                         \
            BENCH_FAIL("tensor creation failed");                                         \
    } while (0)

static DeviceType g_device = DEVICE_CPU;

static TorchTensorOptions torch_opts(void) {
    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts = torch_options_device(opts, g_device);
    return opts;
}

static void cooldown_ms(int ms) {
    struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000L};
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
    return (n % 2) ? arr[n / 2] : (arr[n / 2 - 1] + arr[n / 2]) / 2.0;
}

static double bench_gemm(int N) {
    int shape[] = {N, N};
    TorchTensorOptions opts = torch_opts();

    float* a_data = malloc(sizeof(float) * N * N);
    float* b_data = malloc(sizeof(float) * N * N);
    BENCH_REQUIRE_ALLOC(a_data);
    BENCH_REQUIRE_ALLOC(b_data);
    fill_random(a_data, N * N);
    fill_random(b_data, N * N);

    Tensor* A = torch_from_blob(a_data, shape, 2, &opts);
    Tensor* B = torch_from_blob(b_data, shape, 2, &opts);
    BENCH_REQUIRE_TENSOR(A);
    BENCH_REQUIRE_TENSOR(B);

    for (int i = 0; i < 3; i++) {
        Tensor* C = torch_matmul(A, B);
        (void)torch_tensor_data_ptr(C);
        torch_tensor_free(C);
        torch_reset_ir();
    }

    int iters  = N >= 2048 ? 3 : 5;
    int rounds = 5;
    double times[5];
    for (int r = 0; r < rounds; r++) {
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* C = torch_matmul(A, B);
            (void)torch_tensor_data_ptr(C);
            torch_tensor_free(C);
            torch_reset_ir();
        }
        times[r] = (now() - t0) / iters * 1e3;
    }

    torch_tensor_free(A);
    torch_tensor_free(B);
    free(a_data);
    free(b_data);
    return median(times, rounds);
}

static double bench_fused(int N) {
    int mat_shape[]  = {N, N};
    int bias_shape[] = {1, N};
    TorchTensorOptions opts = torch_opts();

    float* a_data    = malloc(sizeof(float) * N * N);
    float* b_data    = malloc(sizeof(float) * N * N);
    float* bias_data = malloc(sizeof(float) * N);
    BENCH_REQUIRE_ALLOC(a_data);
    BENCH_REQUIRE_ALLOC(b_data);
    BENCH_REQUIRE_ALLOC(bias_data);
    fill_random(a_data, N * N);
    fill_random(b_data, N * N);
    fill_random(bias_data, N);

    Tensor* A    = torch_from_blob(a_data, mat_shape, 2, &opts);
    Tensor* B    = torch_from_blob(b_data, mat_shape, 2, &opts);
    Tensor* bias = torch_from_blob(bias_data, bias_shape, 2, &opts);
    BENCH_REQUIRE_TENSOR(A);
    BENCH_REQUIRE_TENSOR(B);
    BENCH_REQUIRE_TENSOR(bias);

    for (int i = 0; i < 3; i++) {
        Tensor* C = torch_relu(torch_add(torch_matmul(A, B), bias));
        (void)torch_tensor_data_ptr(C);
        torch_tensor_free(C);
        torch_reset_ir();
    }

    int iters  = N >= 2048 ? 3 : 5;
    int rounds = 5;
    double times[5];
    for (int r = 0; r < rounds; r++) {
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* C = torch_relu(torch_add(torch_matmul(A, B), bias));
            (void)torch_tensor_data_ptr(C);
            torch_tensor_free(C);
            torch_reset_ir();
        }
        times[r] = (now() - t0) / iters * 1e3;
    }

    torch_tensor_free(A);
    torch_tensor_free(B);
    torch_tensor_free(bias);
    free(a_data);
    free(b_data);
    free(bias_data);
    return median(times, rounds);
}

static double bench_mlp_forward(void) {
    int batch = 64, in_f = 784, hid = 128, out_f = 10;
    int x_shape[] = {batch, in_f};
    TorchTensorOptions opts = torch_opts();

    float* x_data = malloc(sizeof(float) * batch * in_f);
    BENCH_REQUIRE_ALLOC(x_data);
    fill_random(x_data, batch * in_f);
    Tensor* X = torch_from_blob(x_data, x_shape, 2, &opts);
    BENCH_REQUIRE_TENSOR(X);

    Sequential* model = torch_nn_sequential();
    torch_nn_sequential_add(model, (Module*)torch_nn_linear(in_f, hid, true));
    torch_nn_sequential_add(model, (Module*)torch_nn_relu());
    torch_nn_sequential_add(model, (Module*)torch_nn_linear(hid, out_f, true));
    torch_module_eval((Module*)model);

    for (int i = 0; i < 5; i++) {
        Tensor* out = torch_nn_sequential_forward(model, X);
        (void)torch_tensor_data_ptr(out);
        torch_tensor_free(out);
        torch_reset_ir_soft();
    }

    int iters = 100;
    double times[5];
    for (int r = 0; r < 5; r++) {
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = torch_nn_sequential_forward(model, X);
            (void)torch_tensor_data_ptr(out);
            torch_tensor_free(out);
            torch_reset_ir_soft();
        }
        times[r] = (now() - t0) / iters * 1e3;
    }
    torch_reset_ir();

    torch_tensor_free(X);
    free(x_data);
    module_free((Module*)model);
    return median(times, 5);
}

static double bench_mlp_train(void) {
    int batch = 64, in_f = 784, hid = 128, out_f = 10;
    int x_shape[] = {batch, in_f};
    int y_shape[] = {batch, out_f};
    TorchTensorOptions opts = torch_opts();

    float* x_data = malloc(sizeof(float) * batch * in_f);
    float* y_data = malloc(sizeof(float) * batch * out_f);
    BENCH_REQUIRE_ALLOC(x_data);
    BENCH_REQUIRE_ALLOC(y_data);
    fill_random(x_data, batch * in_f);
    fill_random(y_data, batch * out_f);

    Tensor* X = torch_from_blob(x_data, x_shape, 2, &opts);
    Tensor* Y = torch_from_blob(y_data, y_shape, 2, &opts);
    BENCH_REQUIRE_TENSOR(X);
    BENCH_REQUIRE_TENSOR(Y);

    Sequential* model = torch_nn_sequential();
    torch_nn_sequential_add(model, (Module*)torch_nn_linear(in_f, hid, true));
    torch_nn_sequential_add(model, (Module*)torch_nn_relu());
    torch_nn_sequential_add(model, (Module*)torch_nn_linear(hid, out_f, true));
    torch_module_train((Module*)model);

    Optimizer* opt = torch_optim_sgd((Module*)model, 0.01f, 0.0f, 0.0f);

    for (int i = 0; i < 5; i++) {
        Tensor* out  = torch_nn_sequential_forward(model, X);
        Tensor* loss = torch_nn_mse_loss(out, Y);
        torch_optim_zero_grad(opt);
        torch_backward(loss, NULL, false, false);
        torch_optim_step(opt);
        torch_tensor_free(out);
        torch_tensor_free(loss);
        torch_reset_ir();
    }

    int iters = 50;
    double times[5];
    for (int r = 0; r < 5; r++) {
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out  = torch_nn_sequential_forward(model, X);
            Tensor* loss = torch_nn_mse_loss(out, Y);
            torch_optim_zero_grad(opt);
            torch_backward(loss, NULL, false, false);
            torch_optim_step(opt);
            torch_tensor_free(out);
            torch_tensor_free(loss);
            torch_reset_ir_soft();
        }
        times[r] = (now() - t0) / iters * 1e3;
        torch_reset_ir();
    }

    torch_tensor_free(X);
    torch_tensor_free(Y);
    free(x_data);
    free(y_data);
    torch_optim_free(opt);
    module_free((Module*)model);
    return median(times, 5);
}

static double bench_conv2d(void) {
    int batch = 8, ic = 3, h = 32, w = 32, oc = 16, ksize = 3;
    int x_shape[] = {batch, ic, h, w};
    TorchTensorOptions opts = torch_opts();

    int numel     = batch * ic * h * w;
    float* x_data = malloc(sizeof(float) * numel);
    BENCH_REQUIRE_ALLOC(x_data);
    fill_random(x_data, numel);
    Tensor* X = torch_from_blob(x_data, x_shape, 4, &opts);
    BENCH_REQUIRE_TENSOR(X);

    Conv2d* conv_layer =
        nn_conv2d(ic, oc, ksize, 1, 0, 1, true, DTYPE_FLOAT32, g_device);
    Module* conv = (Module*)conv_layer;
    torch_module_eval(conv);

    for (int i = 0; i < 5; i++) {
        Tensor* out = torch_module_forward(conv, X);
        (void)torch_tensor_data_ptr(out);
        torch_tensor_free(out);
        torch_reset_ir_soft();
    }

    int iters = 100;
    double times[5];
    for (int r = 0; r < 5; r++) {
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = torch_module_forward(conv, X);
            (void)torch_tensor_data_ptr(out);
            torch_tensor_free(out);
            torch_reset_ir_soft();
        }
        times[r] = (now() - t0) / iters * 1e3;
    }
    torch_reset_ir();

    torch_tensor_free(X);
    free(x_data);
    module_free(conv);
    return median(times, 5);
}

int main(void) {
    const char* backend = getenv("CML_BACKEND");
    if (backend && strcmp(backend, "metal") == 0) {
        g_device = DEVICE_METAL;
        fprintf(stderr, "bench_torch_c: using Metal GPU backend\n");
    } else if (backend && strcmp(backend, "opencl") == 0) {
        g_device = DEVICE_OPENCL;
        fprintf(stderr, "bench_torch_c: using OpenCL GPU backend\n");
    }

    torch_init();
    torch_set_default_device(g_device);
    torch_set_default_dtype(DTYPE_FLOAT32);
    torch_manual_seed(42);
    srand(42);

    double gemm_512   = bench_gemm(512);
    double fused_512  = bench_fused(512);
    double gemm_1024  = bench_gemm(1024);
    double fused_1024 = bench_fused(1024);
    cooldown_ms(200);
    double gemm_2048 = bench_gemm(2048);
    cooldown_ms(200);
    double fused_2048 = bench_fused(2048);
    cooldown_ms(100);
    double mlp_fwd    = bench_mlp_forward();
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

    torch_cleanup();
    return 0;
}
