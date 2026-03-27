/**
 * Detailed overhead profiler: isolate every source of CML overhead vs raw BLAS.
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

    CMLBlasContext* blas = cml_blas_get_context();
    if (!blas || !blas->initialized) {
        printf("BLAS not available!\n");
        return 1;
    }

    int iters = 200;

    /* ═══ 1. GEMM 512 overhead breakdown ═══ */
    printf("=== GEMM 512x512 Overhead Breakdown (%d iters) ===\n\n", iters);
    {
        int N = 512;
        float* A = malloc(sizeof(float) * N * N);
        float* B = malloc(sizeof(float) * N * N);
        float* C = malloc(sizeof(float) * N * N);
        fill_random(A, N * N);
        fill_random(B, N * N);

        /* Warmup */
        for (int i = 0; i < 5; i++)
            cml_blas_sgemm(blas, A, B, C, N, N, N, 1.0f, 0.0f);

        /* 1a. Raw BLAS only */
        double t0 = now();
        for (int i = 0; i < iters; i++)
            cml_blas_sgemm(blas, A, B, C, N, N, N, 1.0f, 0.0f);
        double raw_blas = (now() - t0) / iters * 1e3;

        /* 1b. CML tensor_matmul (IR path) with reset */
        int shape[] = {N, N};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        Tensor* tA = cml_tensor(A, shape, 2, &cfg);
        Tensor* tB = cml_tensor(B, shape, 2, &cfg);
        for (int i = 0; i < 5; i++) {
            Tensor* tC = cml_matmul(tA, tB);
            (void)tensor_data_ptr(tC);
            cml_reset_ir_context();
        }

        t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* tC = cml_matmul(tA, tB);
            (void)tensor_data_ptr(tC);
            cml_reset_ir_context();
        }
        double cml_ir = (now() - t0) / iters * 1e3;

        /* 1c. Isolate: IR create only (no execute) */
        t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* tC = cml_matmul(tA, tB);
            (void)tC;
            cml_reset_ir_context();
        }
        double ir_create = (now() - t0) / iters * 1e3;

        /* 1d. CML without reset (accumulating graph) — just 20 iters to show effect */
        for (int i = 0; i < 5; i++) {
            Tensor* tC = cml_matmul(tA, tB);
            (void)tensor_data_ptr(tC);
        }
        cml_reset_ir_context();

        t0 = now();
        for (int i = 0; i < 20; i++) {
            Tensor* tC = cml_matmul(tA, tB);
            (void)tensor_data_ptr(tC);
        }
        double cml_no_reset = (now() - t0) / 20 * 1e3;
        cml_reset_ir_context();

        printf("  Raw BLAS sgemm:         %8.3f ms\n", raw_blas);
        printf("  CML IR (with reset):    %8.3f ms  (%.1fx BLAS)\n", cml_ir, cml_ir / raw_blas);
        printf("  CML IR (no reset):      %8.3f ms  (%.1fx BLAS)\n", cml_no_reset, cml_no_reset / raw_blas);
        printf("  IR create+reset only:   %8.3f ms\n", ir_create);
        printf("  Overhead (IR - BLAS):   %8.3f ms\n\n", cml_ir - raw_blas);

        free(A); free(B); free(C);
    }

    /* ═══ 2. Fused 512: mm+bias+relu ═══ */
    printf("=== Fused 512 (mm+bias+relu) Overhead Breakdown ===\n\n");
    {
        int N = 512;
        float* A = malloc(sizeof(float) * N * N);
        float* B = malloc(sizeof(float) * N * N);
        float* C = malloc(sizeof(float) * N * N);
        float* bias = malloc(sizeof(float) * N);
        fill_random(A, N * N);
        fill_random(B, N * N);
        fill_random(bias, N);

        for (int i = 0; i < 5; i++)
            cml_blas_sgemm(blas, A, B, C, N, N, N, 1.0f, 0.0f);

        /* 2a. Raw BLAS + manual bias + relu */
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            cml_blas_sgemm(blas, A, B, C, N, N, N, 1.0f, 0.0f);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++) {
                    C[r * N + c] += bias[c];
                    if (C[r * N + c] < 0) C[r * N + c] = 0;
                }
        }
        double raw = (now() - t0) / iters * 1e3;

        /* 2b. CML fused with reset */
        int mat_shape[] = {N, N};
        int bias_shape[] = {1, N};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* a_cpy = malloc(sizeof(float) * N * N);
        float* b_cpy = malloc(sizeof(float) * N * N);
        float* bi_cpy = malloc(sizeof(float) * N);
        memcpy(a_cpy, A, sizeof(float) * N * N);
        memcpy(b_cpy, B, sizeof(float) * N * N);
        memcpy(bi_cpy, bias, sizeof(float) * N);
        Tensor* tA = cml_tensor(a_cpy, mat_shape, 2, &cfg);
        Tensor* tB = cml_tensor(b_cpy, mat_shape, 2, &cfg);
        Tensor* tBias = cml_tensor(bi_cpy, bias_shape, 2, &cfg);
        for (int i = 0; i < 5; i++) {
            Tensor* tC = cml_relu(cml_add(cml_matmul(tA, tB), tBias));
            (void)tensor_data_ptr(tC);
            cml_reset_ir_context();
        }

        t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* tC = cml_relu(cml_add(cml_matmul(tA, tB), tBias));
            (void)tensor_data_ptr(tC);
            cml_reset_ir_context();
        }
        double cml_fused = (now() - t0) / iters * 1e3;

        printf("  Raw BLAS+bias+relu:     %8.3f ms\n", raw);
        printf("  CML fused (with reset): %8.3f ms  (%.1fx raw)\n", cml_fused, cml_fused / raw);
        printf("  Overhead:               %8.3f ms\n\n", cml_fused - raw);

        free(A); free(B); free(C); free(bias);
        free(a_cpy); free(b_cpy); free(bi_cpy);
    }

    /* ═══ 3. MLP forward ═══ */
    printf("=== MLP Forward (64x784->128->10) Overhead Breakdown ===\n\n");
    {
        int batch = 64, in_f = 784, hid = 128, out_f = 10;

        /* 3a. Raw BLAS MLP */
        float* X   = malloc(sizeof(float) * batch * in_f);
        float* W1  = malloc(sizeof(float) * hid * in_f);
        float* B1  = malloc(sizeof(float) * hid);
        float* H   = malloc(sizeof(float) * batch * hid);
        float* W2  = malloc(sizeof(float) * out_f * hid);
        float* B2  = malloc(sizeof(float) * out_f);
        float* OUT = malloc(sizeof(float) * batch * out_f);
        fill_random(X, batch * in_f);
        fill_random(W1, hid * in_f);
        fill_random(B1, hid);
        fill_random(W2, out_f * hid);
        fill_random(B2, out_f);

        for (int i = 0; i < 10; i++) {
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
        double raw_mlp = (now() - t0) / iters * 1e3;

        /* 3b. CML MLP forward */
        int x_shape[] = {batch, in_f};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* x_data = malloc(sizeof(float) * batch * in_f);
        memcpy(x_data, X, sizeof(float) * batch * in_f);
        Tensor* tX = cml_tensor(x_data, x_shape, 2, &cfg);
        Sequential* model = cml_nn_sequential();
        cml_nn_sequential_add(model, (Module*)cml_nn_linear(in_f, hid, DTYPE_FLOAT32, DEVICE_CPU, true));
        cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
        cml_nn_sequential_add(model, (Module*)cml_nn_linear(hid, out_f, DTYPE_FLOAT32, DEVICE_CPU, true));

        for (int i = 0; i < 10; i++) {
            Tensor* out = cml_nn_sequential_forward(model, tX);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }

        /* Time total */
        t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = cml_nn_sequential_forward(model, tX);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }
        double cml_mlp = (now() - t0) / iters * 1e3;

        /* Time IR creation only (no execution) */
        t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = cml_nn_sequential_forward(model, tX);
            (void)out;
            cml_reset_ir_context();
        }
        double ir_only = (now() - t0) / iters * 1e3;

        /* Time reset only */
        {
            Tensor* out = cml_nn_sequential_forward(model, tX);
            (void)tensor_data_ptr(out);
            t0 = now();
            for (int i = 0; i < iters; i++) {
                cml_reset_ir_context();
            }
            double reset_only = (now() - t0) / iters * 1e3;

            printf("  Raw BLAS MLP:           %8.3f ms\n", raw_mlp);
            printf("  CML MLP (total):        %8.3f ms  (%.1fx BLAS)\n", cml_mlp, cml_mlp / raw_mlp);
            printf("  IR creation only:       %8.3f ms\n", ir_only);
            printf("  IR reset only:          %8.3f ms\n", reset_only);
            printf("  Execution only:         %8.3f ms  (total - create - reset)\n", cml_mlp - ir_only - reset_only);
            printf("  Overhead vs BLAS:       %8.3f ms\n\n", cml_mlp - raw_mlp);
        }

        free(X); free(W1); free(B1); free(H); free(W2); free(B2); free(OUT);
        free(x_data);
        module_free((Module*)model);
    }

    /* ═══ 4. Memory allocation overhead test ═══ */
    printf("=== Memory Allocation Overhead ===\n\n");
    {
        int N = 512;
        int alloc_iters = 10000;

        /* calloc + free */
        double t0 = now();
        for (int i = 0; i < alloc_iters; i++) {
            float* p = calloc(N * N, sizeof(float));
            free(p);
        }
        double alloc_time = (now() - t0) / alloc_iters * 1e3;

        /* malloc + memset + free */
        t0 = now();
        for (int i = 0; i < alloc_iters; i++) {
            float* p = malloc(N * N * sizeof(float));
            memset(p, 0, N * N * sizeof(float));
            free(p);
        }
        double malloc_time = (now() - t0) / alloc_iters * 1e3;

        printf("  calloc(%d) + free:      %8.4f ms\n", N*N, alloc_time);
        printf("  malloc+memset+free:     %8.4f ms\n\n", malloc_time);
    }

    /* ═══ 5. Conv2d breakdown ═══ */
    printf("=== Conv2d (8x3x32x32 -> 16x30x30) Breakdown ===\n\n");
    {
        int cb = 8, ic = 3, ih = 32, iw = 32, oc = 16, kh = 3, kw = 3;
        int oh = ih - kh + 1, ow = iw - kw + 1;
        int col_h = ic * kh * kw;
        int col_w = oh * ow;

        float* input  = malloc(sizeof(float) * cb * ic * ih * iw);
        float* weight = malloc(sizeof(float) * oc * col_h);
        float* col    = malloc(sizeof(float) * col_h * col_w);
        float* output = malloc(sizeof(float) * cb * oc * oh * ow);
        fill_random(input, cb * ic * ih * iw);
        fill_random(weight, oc * col_h);

        /* Warmup */
        for (int i = 0; i < 5; i++) {
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
                cml_blas_sgemm(blas, weight, col, output + (size_t)b * oc * oh * ow,
                               oc, col_w, col_h, 1.0f, 0.0f);
            }
        }

        /* Time im2col only */
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
            }
        }
        double im2col_time = (now() - t0) / iters * 1e3;

        /* Time sgemm only */
        t0 = now();
        for (int i = 0; i < iters; i++) {
            for (int b = 0; b < cb; b++) {
                cml_blas_sgemm(blas, weight, col, output + (size_t)b * oc * oh * ow,
                               oc, col_w, col_h, 1.0f, 0.0f);
            }
        }
        double sgemm_time = (now() - t0) / iters * 1e3;

        /* Time full raw conv */
        t0 = now();
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
                cml_blas_sgemm(blas, weight, col, output + (size_t)b * oc * oh * ow,
                               oc, col_w, col_h, 1.0f, 0.0f);
            }
        }
        double raw_conv = (now() - t0) / iters * 1e3;

        /* CML Conv2d */
        int x_shape[] = {cb, ic, ih, iw};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* x_data = malloc(sizeof(float) * cb * ic * ih * iw);
        memcpy(x_data, input, sizeof(float) * cb * ic * ih * iw);
        Tensor* X = cml_tensor(x_data, x_shape, 4, &cfg);
        Conv2d* conv = cml_nn_conv2d(ic, oc, kh, 1, 0, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
        for (int i = 0; i < 5; i++) {
            Tensor* out = module_forward((Module*)conv, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }
        t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = module_forward((Module*)conv, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }
        double cml_conv = (now() - t0) / iters * 1e3;

        printf("  im2col only:            %8.3f ms\n", im2col_time);
        printf("  sgemm only:             %8.3f ms\n", sgemm_time);
        printf("  Raw im2col+sgemm:       %8.3f ms\n", raw_conv);
        printf("  CML Conv2d:             %8.3f ms  (%.1fx raw)\n", cml_conv, cml_conv / raw_conv);
        printf("  CML overhead:           %8.3f ms\n\n", cml_conv - raw_conv);

        free(input); free(weight); free(col); free(output); free(x_data);
        module_free((Module*)conv);
    }

    /* ═══ 6. Thread count check ═══ */
    printf("=== OpenBLAS Thread Info ===\n");
    {
        /* Try to detect thread count via library name */
        const char* name = cml_blas_get_library_name(blas);
        printf("  BLAS library: %s\n", name ? name : "unknown");
        const char* omp = getenv("OMP_NUM_THREADS");
        const char* obl = getenv("OPENBLAS_NUM_THREADS");
        printf("  OMP_NUM_THREADS: %s\n", omp ? omp : "unset");
        printf("  OPENBLAS_NUM_THREADS: %s\n", obl ? obl : "unset");
    }

    cml_cleanup();
    return 0;
}
