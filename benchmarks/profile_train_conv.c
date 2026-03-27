/**
 * Targeted profiling of MLP training step and Conv2d overhead.
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
    int iters = 200;

    /* ═══ MLP Training Step Breakdown ═══ */
    printf("=== MLP Training Step Breakdown (%d iters) ===\n\n", iters);
    {
        int batch = 64, in_f = 784, hid = 128, out_f = 10;
        int x_shape[] = {batch, in_f};
        int y_shape[] = {batch, out_f};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
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

        /* warmup */
        for (int i = 0; i < 10; i++) {
            Tensor* out = cml_nn_sequential_forward(model, X);
            Tensor* loss = cml_nn_mse_loss(out, Y);
            cml_optim_zero_grad(opt);
            cml_backward(loss, NULL, false, false);
            cml_optim_step(opt);
            cml_reset_ir_context();
        }

        /* Time each phase */
        double t_forward = 0, t_loss = 0, t_zero = 0, t_backward = 0, t_step = 0, t_reset = 0;
        double t_total = 0;

        double t_total_start = now();
        for (int i = 0; i < iters; i++) {
            double t;

            t = now();
            Tensor* out = cml_nn_sequential_forward(model, X);
            (void)tensor_data_ptr(out);
            t_forward += now() - t;

            t = now();
            Tensor* loss = cml_nn_mse_loss(out, Y);
            (void)tensor_data_ptr(loss);
            t_loss += now() - t;

            t = now();
            cml_optim_zero_grad(opt);
            t_zero += now() - t;

            t = now();
            cml_backward(loss, NULL, false, false);
            t_backward += now() - t;

            t = now();
            cml_optim_step(opt);
            t_step += now() - t;

            t = now();
            cml_reset_ir_context();
            t_reset += now() - t;
        }
        t_total = now() - t_total_start;

        printf("  Forward:         %8.3f ms  (%.0f%%)\n", t_forward / iters * 1e3, t_forward / t_total * 100);
        printf("  Loss:            %8.3f ms  (%.0f%%)\n", t_loss / iters * 1e3, t_loss / t_total * 100);
        printf("  Zero grad:       %8.3f ms  (%.0f%%)\n", t_zero / iters * 1e3, t_zero / t_total * 100);
        printf("  Backward:        %8.3f ms  (%.0f%%)\n", t_backward / iters * 1e3, t_backward / t_total * 100);
        printf("  Optimizer step:  %8.3f ms  (%.0f%%)\n", t_step / iters * 1e3, t_step / t_total * 100);
        printf("  IR reset:        %8.3f ms  (%.0f%%)\n", t_reset / iters * 1e3, t_reset / t_total * 100);
        printf("  ─────────────────────────────\n");
        printf("  Total:           %8.3f ms\n\n", t_total / iters * 1e3);

        free(x_data); free(y_data);
        optimizer_free(opt);
        module_free((Module*)model);
    }

    /* ═══ Conv2d Forward Detailed ═══ */
    printf("=== Conv2d Forward Detailed (%d iters) ===\n\n", iters);
    {
        int cb = 8, ic = 3, ih = 32, iw = 32, oc = 16, kh = 3, kw = 3;
        int oh = ih - kh + 1, ow = iw - kw + 1;

        /* Time raw components */
        CMLBlasContext* blas = cml_blas_get_context();
        int col_h = ic * kh * kw;
        int col_w = oh * ow;
        float* input  = malloc(sizeof(float) * cb * ic * ih * iw);
        float* weight = malloc(sizeof(float) * oc * col_h);
        float* col    = malloc(sizeof(float) * col_h * col_w);
        float* output = malloc(sizeof(float) * cb * oc * oh * ow);
        float* bias   = malloc(sizeof(float) * oc);
        fill_random(input, cb * ic * ih * iw);
        fill_random(weight, oc * col_h);
        fill_random(bias, oc);

        /* Warmup */
        for (int i = 0; i < 10; i++) {
            for (int b = 0; b < cb; b++) {
                for (int c = 0; c < ic; c++)
                    for (int kr = 0; kr < kh; kr++)
                        for (int kc = 0; kc < kw; kc++) {
                            int row = (c * kh + kr) * kw + kc;
                            const float* src = input + ((b * ic + c) * ih + kr) * iw + kc;
                            float* dst = col + row * col_w;
                            for (int r = 0; r < oh; r++)
                                memcpy(dst + r * ow, src + r * iw, ow * sizeof(float));
                        }
                cml_blas_sgemm(blas, weight, col, output + (size_t)b * oc * oh * ow,
                               oc, col_w, col_h, 1.0f, 0.0f);
                for (int o = 0; o < oc; o++) {
                    float bv = bias[o];
                    float* row = output + ((size_t)b * oc + o) * (oh * ow);
                    for (int j = 0; j < oh * ow; j++) row[j] += bv;
                }
            }
        }

        /* Time raw im2col+sgemm+bias (matching CML's work exactly) */
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            for (int b = 0; b < cb; b++) {
                for (int c = 0; c < ic; c++)
                    for (int kr = 0; kr < kh; kr++)
                        for (int kc = 0; kc < kw; kc++) {
                            int row = (c * kh + kr) * kw + kc;
                            const float* src = input + ((b * ic + c) * ih + kr) * iw + kc;
                            float* dst = col + row * col_w;
                            for (int r = 0; r < oh; r++)
                                memcpy(dst + r * ow, src + r * iw, ow * sizeof(float));
                        }
                cml_blas_sgemm(blas, weight, col, output + (size_t)b * oc * oh * ow,
                               oc, col_w, col_h, 1.0f, 0.0f);
                for (int o = 0; o < oc; o++) {
                    float bv = bias[o];
                    float* row = output + ((size_t)b * oc + o) * (oh * ow);
                    for (int j = 0; j < oh * ow; j++) row[j] += bv;
                }
            }
        }
        double raw_full = (now() - t0) / iters * 1e3;

        /* CML Conv2d */
        int x_shape[] = {cb, ic, ih, iw};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* x_data = malloc(sizeof(float) * cb * ic * ih * iw);
        memcpy(x_data, input, sizeof(float) * cb * ic * ih * iw);
        Tensor* X = cml_tensor(x_data, x_shape, 4, &cfg);
        Conv2d* conv = cml_nn_conv2d(ic, oc, kh, 1, 0, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
        for (int i = 0; i < 10; i++) {
            Tensor* out = module_forward((Module*)conv, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }

        /* Time CML total */
        t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = module_forward((Module*)conv, X);
            (void)tensor_data_ptr(out);
            cml_reset_ir_context();
        }
        double cml_total = (now() - t0) / iters * 1e3;

        /* Time IR creation only */
        t0 = now();
        for (int i = 0; i < iters; i++) {
            Tensor* out = module_forward((Module*)conv, X);
            (void)out;
            cml_reset_ir_context();
        }
        double ir_create = (now() - t0) / iters * 1e3;

        printf("  Raw im2col+sgemm+bias:  %8.3f ms\n", raw_full);
        printf("  CML Conv2d total:       %8.3f ms  (%.1fx raw)\n", cml_total, cml_total / raw_full);
        printf("  IR creation only:       %8.3f ms\n", ir_create);
        printf("  Execution only:         %8.3f ms\n", cml_total - ir_create);
        printf("  Overhead:               %8.3f ms\n\n", cml_total - raw_full);

        free(input); free(weight); free(col); free(output); free(bias); free(x_data);
        module_free((Module*)conv);
    }

    cml_cleanup();
    return 0;
}
