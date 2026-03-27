/**
 * Targeted profiling: break down MLP and Conv2d into sub-operations.
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

    printf("=== MLP Forward Breakdown ===\n");
    printf("Model: Linear(784,128) -> ReLU -> Linear(128,10)\n");
    printf("Input: [64, 784]\n\n");

    int batch = 64, in_f = 784, hid = 128, out_f = 10;
    int iters = 500;

    /* --- Raw BLAS equivalent of MLP forward --- */
    {
        CMLBlasContext* blas = cml_blas_get_context();
        if (blas && blas->initialized) {
            float* X   = malloc(sizeof(float) * batch * in_f);
            float* W1  = malloc(sizeof(float) * hid * in_f);   /* [128,784] stored */
            float* B1  = malloc(sizeof(float) * hid);
            float* H   = malloc(sizeof(float) * batch * hid);
            float* W2  = malloc(sizeof(float) * out_f * hid);  /* [10,128] stored */
            float* B2  = malloc(sizeof(float) * out_f);
            float* OUT = malloc(sizeof(float) * batch * out_f);
            fill_random(X, batch * in_f);
            fill_random(W1, hid * in_f);
            fill_random(B1, hid);
            fill_random(W2, out_f * hid);
            fill_random(B2, out_f);

            /* warmup */
            for (int i = 0; i < 5; i++) {
                /* H = X @ W1^T + B1 */
                cml_blas_sgemm_ex(blas, X, W1, H, batch, hid, in_f, 1.0f, 0.0f, false, true);
                for (int r = 0; r < batch; r++)
                    for (int c = 0; c < hid; c++) {
                        H[r*hid+c] += B1[c];
                        if (H[r*hid+c] < 0) H[r*hid+c] = 0; /* relu */
                    }
                /* OUT = H @ W2^T + B2 */
                cml_blas_sgemm_ex(blas, H, W2, OUT, batch, out_f, hid, 1.0f, 0.0f, false, true);
                for (int r = 0; r < batch; r++)
                    for (int c = 0; c < out_f; c++)
                        OUT[r*out_f+c] += B2[c];
            }

            double t0 = now();
            for (int i = 0; i < iters; i++) {
                cml_blas_sgemm_ex(blas, X, W1, H, batch, hid, in_f, 1.0f, 0.0f, false, true);
                for (int r = 0; r < batch; r++)
                    for (int c = 0; c < hid; c++) {
                        H[r*hid+c] += B1[c];
                        if (H[r*hid+c] < 0) H[r*hid+c] = 0;
                    }
                cml_blas_sgemm_ex(blas, H, W2, OUT, batch, out_f, hid, 1.0f, 0.0f, false, true);
                for (int r = 0; r < batch; r++)
                    for (int c = 0; c < out_f; c++)
                        OUT[r*out_f+c] += B2[c];
            }
            double ms = (now() - t0) / iters * 1e3;
            printf("Raw BLAS MLP forward:    %8.3f ms\n", ms);

            free(X); free(W1); free(B1); free(H); free(W2); free(B2); free(OUT);
        }
    }

    /* --- CML MLP forward --- */
    {
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

        for (int i = 0; i < 5; i++) {
            Tensor* out = cml_nn_sequential_forward(model, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }

        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = cml_nn_sequential_forward(model, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }
        double ms = (now() - t0) / iters * 1e3;
        printf("CML Sequential forward:  %8.3f ms\n", ms);

        /* Now test individual ops */
        double t_transpose = 0, t_matmul1 = 0, t_biasadd1 = 0, t_relu = 0;
        double t_matmul2 = 0, t_biasadd2 = 0, t_ir_reset = 0;

        /* Get weight tensors */
        Parameter* params_arr[10];
        int np = 0;
        module_collect_parameters((Module*)model, (Parameter***)&params_arr, &np, true);

        for (int i = 0; i < iters; i++) {
            double t;

            /* Linear1: transpose + matmul + bias_add */
            t = now();
            Tensor* out = cml_nn_sequential_forward(model, X);
            (void)tensor_data_ptr(out);
            t_matmul1 += now() - t;

            t = now();
            cml_reset_ir_context();
            t_ir_reset += now() - t;
        }
        printf("CML forward+execute+reset: fwd=%.3f reset=%.3f total=%.3f ms\n",
               t_matmul1 / iters * 1e3, t_ir_reset / iters * 1e3,
               (t_matmul1 + t_ir_reset) / iters * 1e3);

        free(x_data);
        module_free((Module*)model);
    }

    printf("\n=== Conv2d Breakdown ===\n");
    printf("Conv2d(3, 16, 3) input [8, 3, 32, 32]\n\n");

    /* --- Raw BLAS im2col+matmul conv --- */
    {
        CMLBlasContext* blas = cml_blas_get_context();
        if (blas && blas->initialized) {
            int cb = 8, ic = 3, ih = 32, iw = 32, oc = 16, kh = 3, kw = 3;
            int oh = ih - kh + 1, ow = iw - kw + 1;
            int col_h = ic * kh * kw;  /* 27 */
            int col_w = oh * ow;       /* 900 */

            float* input = malloc(sizeof(float) * cb * ic * ih * iw);
            float* weight = malloc(sizeof(float) * oc * col_h);
            float* col = malloc(sizeof(float) * col_h * col_w);
            float* output = malloc(sizeof(float) * cb * oc * oh * ow);
            fill_random(input, cb * ic * ih * iw);
            fill_random(weight, oc * col_h);

            for (int i = 0; i < 5; i++) {
                for (int b = 0; b < cb; b++) {
                    /* im2col */
                    for (int c = 0; c < ic; c++)
                        for (int kr = 0; kr < kh; kr++)
                            for (int kc = 0; kc < kw; kc++) {
                                int row = (c * kh + kr) * kw + kc;
                                for (int r = 0; r < oh; r++)
                                    for (int cc = 0; cc < ow; cc++)
                                        col[row * col_w + r * ow + cc] =
                                            input[((b * ic + c) * ih + r + kr) * iw + cc + kc];
                            }
                    cml_blas_sgemm(blas, weight, col,
                                   output + (size_t)b * oc * oh * ow,
                                   oc, col_w, col_h, 1.0f, 0.0f);
                }
            }

            double t0 = now();
            for (int i = 0; i < iters; i++) {
                for (int b = 0; b < cb; b++) {
                    for (int c = 0; c < ic; c++)
                        for (int kr = 0; kr < kh; kr++)
                            for (int kc = 0; kc < kw; kc++) {
                                int row = (c * kh + kr) * kw + kc;
                                for (int r = 0; r < oh; r++)
                                    for (int cc = 0; cc < ow; cc++)
                                        col[row * col_w + r * ow + cc] =
                                            input[((b * ic + c) * ih + r + kr) * iw + cc + kc];
                            }
                    cml_blas_sgemm(blas, weight, col,
                                   output + (size_t)b * oc * oh * ow,
                                   oc, col_w, col_h, 1.0f, 0.0f);
                }
            }
            double ms = (now() - t0) / iters * 1e3;
            printf("Raw BLAS im2col+mm conv: %8.3f ms\n", ms);

            free(input); free(weight); free(col); free(output);
        }
    }

    /* --- CML Conv2d --- */
    {
        int cb = 8, ic = 3, h = 32, w = 32, oc = 16, ksize = 3;
        int x_shape[] = {cb, ic, h, w};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* x_data = malloc(sizeof(float) * cb * ic * h * w);
        fill_random(x_data, cb * ic * h * w);
        Tensor* X = cml_tensor(x_data, x_shape, 4, &cfg);

        Conv2d* conv = cml_nn_conv2d(ic, oc, ksize, 1, 0, 1, true, DTYPE_FLOAT32, DEVICE_CPU);

        for (int i = 0; i < 5; i++) {
            Tensor* out = module_forward((Module*)conv, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }

        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = module_forward((Module*)conv, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }
        double ms = (now() - t0) / iters * 1e3;
        printf("CML Conv2d forward:      %8.3f ms\n", ms);

        free(x_data);
        module_free((Module*)conv);
    }

    cml_cleanup();
    return 0;
}
