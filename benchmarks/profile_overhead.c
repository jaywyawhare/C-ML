/**
 * Profile CML overhead: isolate IR graph creation vs actual computation.
 */
#define _POSIX_C_SOURCE 199309L
#include "cml.h"
#include "backend/blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_random(float* buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

int main(void) {
    cml_init();
    srand(42);

    int N = 1024;
    int iters = 20;

    printf("=== Profiling CML Overhead (N=%d, %d iters) ===\n\n", N, iters);

    /* --- 1. Raw BLAS matmul (zero overhead baseline) --- */
    {
        CMLBlasContext* blas = cml_blas_get_context();
        float* A = malloc(sizeof(float) * N * N);
        float* B = malloc(sizeof(float) * N * N);
        float* C = malloc(sizeof(float) * N * N);
        fill_random(A, N * N);
        fill_random(B, N * N);

        if (blas && blas->initialized) {
            cml_blas_sgemm(blas, A, B, C, N, N, N, 1.0f, 0.0f); /* warmup */
            double t0 = now();
            for (int i = 0; i < iters; i++)
                cml_blas_sgemm(blas, A, B, C, N, N, N, 1.0f, 0.0f);
            double ms = (now() - t0) / iters * 1e3;
            printf("1. Raw BLAS sgemm:              %8.3f ms\n", ms);
        } else {
            printf("1. Raw BLAS: NOT AVAILABLE\n");
        }
        free(A); free(B); free(C);
    }

    /* --- 2. cml_matmul through IR (includes graph overhead) --- */
    {
        int shape[] = {N, N};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* a_data = malloc(sizeof(float) * N * N);
        float* b_data = malloc(sizeof(float) * N * N);
        fill_random(a_data, N * N);
        fill_random(b_data, N * N);
        Tensor* A = cml_tensor(a_data, shape, 2, &cfg);
        Tensor* B = cml_tensor(b_data, shape, 2, &cfg);

        /* warmup */
        Tensor* w = cml_matmul(A, B);
        (void)tensor_data_ptr(w);
        tensor_free(w);

        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* C = cml_matmul(A, B);
            (void)tensor_data_ptr(C);
            tensor_free(C);
        }
        double ms = (now() - t0) / iters * 1e3;
        printf("2. cml_matmul (IR path):        %8.3f ms\n", ms);

        tensor_free(A); tensor_free(B);
        free(a_data); free(b_data);
    }

    /* --- 3. Fused: cml_matmul + cml_add + cml_relu --- */
    {
        int mat_shape[] = {N, N};
        int bias_shape[] = {1, N};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
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

        /* warmup */
        Tensor* w = cml_relu(cml_add(cml_matmul(A, B), bias));
        (void)tensor_data_ptr(w);

        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* C = cml_relu(cml_add(cml_matmul(A, B), bias));
            (void)tensor_data_ptr(C);
        }
        double ms = (now() - t0) / iters * 1e3;
        printf("3. cml fused (mm+add+relu):     %8.3f ms\n", ms);

        tensor_free(A); tensor_free(B); tensor_free(bias);
        free(a_data); free(b_data); free(bias_data);
    }

    /* --- 4. IR graph creation only (no computation) --- */
    {
        int shape[] = {N, N};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* a_data = malloc(sizeof(float) * N * N);
        float* b_data = malloc(sizeof(float) * N * N);
        fill_random(a_data, N * N);
        fill_random(b_data, N * N);
        Tensor* A = cml_tensor(a_data, shape, 2, &cfg);
        Tensor* B = cml_tensor(b_data, shape, 2, &cfg);

        double t0 = now();
        for (int i = 0; i < iters * 100; i++) {
            Tensor* C = cml_matmul(A, B);
            (void)C; /* don't execute */
        }
        double ms = (now() - t0) / (iters * 100) * 1e3;
        printf("4. IR graph creation only (mm): %8.3f ms  (x%d iters)\n", ms, iters * 100);

        tensor_free(A); tensor_free(B);
        free(a_data); free(b_data);
    }

    /* --- 5. cml_reset_ir_context cost --- */
    {
        int shape[] = {64, 784};
        int shape2[] = {784, 128};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* a_data = malloc(sizeof(float) * 64 * 784);
        float* b_data = malloc(sizeof(float) * 784 * 128);
        fill_random(a_data, 64 * 784);
        fill_random(b_data, 784 * 128);

        double total_reset = 0;
        for (int i = 0; i < iters * 10; i++) {
            Tensor* A = cml_tensor(a_data, shape, 2, &cfg);
            Tensor* B = cml_tensor(b_data, shape2, 2, &cfg);
            Tensor* C = cml_relu(cml_add(cml_matmul(A, B),
                         cml_matmul(A, B)));
            (void)tensor_data_ptr(C);

            double t0 = now();
            cml_reset_ir_context();
            total_reset += now() - t0;
        }
        double ms = total_reset / (iters * 10) * 1e3;
        printf("5. cml_reset_ir_context cost:   %8.3f ms  (per reset, %d resets)\n", ms, iters * 10);
        free(a_data); free(b_data);
    }

    /* --- 6. MLP forward: breakdown --- */
    {
        int batch = 64, in_f = 784, hid = 128, out_f = 10;
        int x_shape[] = {batch, in_f};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* x_data = malloc(sizeof(float) * batch * in_f);
        fill_random(x_data, batch * in_f);
        Tensor* X = cml_tensor(x_data, x_shape, 2, &cfg);

        Sequential* model = cml_nn_sequential();
        cml_nn_sequential_add(model, (Module*)cml_nn_linear(in_f, hid, DTYPE_FLOAT32, DEVICE_CPU, true));
        cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
        cml_nn_sequential_add(model, (Module*)cml_nn_linear(hid, out_f, DTYPE_FLOAT32, DEVICE_CPU, true));

        /* warmup */
        for (int i = 0; i < 3; i++) {
            Tensor* out = cml_nn_sequential_forward(model, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }

        /* Forward only */
        double t0 = now();
        for (int i = 0; i < iters * 10; i++) {
            Tensor* out = cml_nn_sequential_forward(model, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }
        double fwd_ms = (now() - t0) / (iters * 10) * 1e3;

        /* Forward + backward */
        int y_shape[] = {batch, out_f};
        float* y_data = malloc(sizeof(float) * batch * out_f);
        fill_random(y_data, batch * out_f);
        Tensor* Y = cml_tensor(y_data, y_shape, 2, &cfg);
        Optimizer* opt = cml_optim_sgd_for_model((Module*)model, 0.01f, 0.0f, 0.0f);

        for (int i = 0; i < 3; i++) {
            Tensor* out = cml_nn_sequential_forward(model, X);
            Tensor* loss = cml_nn_mse_loss(out, Y);
            cml_optim_zero_grad(opt);
            cml_backward(loss, NULL, false, false);
            cml_optim_step(opt);
            cml_reset_ir_context();
        }

        double t1 = now();
        for (int i = 0; i < iters * 5; i++) {
            Tensor* out = cml_nn_sequential_forward(model, X);
            Tensor* loss = cml_nn_mse_loss(out, Y);
            cml_optim_zero_grad(opt);
            cml_backward(loss, NULL, false, false);
            cml_optim_step(opt);
            cml_reset_ir_context();
        }
        double train_ms = (now() - t1) / (iters * 5) * 1e3;

        printf("\n6. MLP breakdown:\n");
        printf("   Forward only:                %8.3f ms\n", fwd_ms);
        printf("   Full train step:             %8.3f ms\n", train_ms);
        printf("   Backward+optim overhead:     %8.3f ms\n", train_ms - fwd_ms);

        free(x_data); free(y_data);
        optimizer_free(opt);
        module_free((Module*)model);
    }

    /* --- 7. Conv2d --- */
    {
        int batch = 8, ic = 3, h = 32, w = 32, oc = 16, ksize = 3;
        int x_shape[] = {batch, ic, h, w};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* x_data = malloc(sizeof(float) * batch * ic * h * w);
        fill_random(x_data, batch * ic * h * w);
        Tensor* X = cml_tensor(x_data, x_shape, 4, &cfg);

        Conv2d* conv = cml_nn_conv2d(ic, oc, ksize, 1, 0, 1, true, DTYPE_FLOAT32, DEVICE_CPU);

        for (int i = 0; i < 3; i++) {
            Tensor* out = module_forward((Module*)conv, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }

        double t0 = now();
        for (int i = 0; i < iters * 5; i++) {
            Tensor* out = module_forward((Module*)conv, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }
        double ms = (now() - t0) / (iters * 5) * 1e3;
        printf("\n7. Conv2d forward (8x3x32x32): %8.3f ms\n", ms);

        free(x_data);
        module_free((Module*)conv);
    }

    printf("\n=== Summary ===\n");
    printf("Overhead = (CML IR path) - (Raw BLAS) for matmul\n");
    printf("If fused > unfused: IR overhead dominates fusion savings\n");

    cml_cleanup();
    return 0;
}
