/**
 * Targeted profiling: break down MLP and Conv2d into sub-operations.
 */
#include "profile_common.h"

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
            float* X   = cml_malloc(sizeof(float) * batch * in_f);
            float* W1  = cml_malloc(sizeof(float) * hid * in_f);   /* [128,784] stored */
            float* B1  = cml_malloc(sizeof(float) * hid);
            float* H   = cml_malloc(sizeof(float) * batch * hid);
            float* W2  = cml_malloc(sizeof(float) * out_f * hid);  /* [10,128] stored */
            float* B2  = cml_malloc(sizeof(float) * out_f);
            float* OUT = cml_malloc(sizeof(float) * batch * out_f);
            fill_random(X, batch * in_f);
            fill_random(W1, hid * in_f);
            fill_random(B1, hid);
            fill_random(W2, out_f * hid);
            fill_random(B2, out_f);

            /* warmup */
            for (int i = 0; i < 5; i++)
                raw_blas_mlp_forward(blas, X, W1, B1, H, W2, B2, OUT,
                                     batch, in_f, hid, out_f);

            double t0 = now();
            for (int i = 0; i < iters; i++) {
                raw_blas_mlp_forward(blas, X, W1, B1, H, W2, B2, OUT,
                                     batch, in_f, hid, out_f);
            }
            double ms = (now() - t0) / iters * 1e3;
            printf("Raw BLAS MLP forward:    %8.3f ms\n", ms);

            cml_free(X); cml_free(W1); cml_free(B1); cml_free(H); cml_free(W2); cml_free(B2); cml_free(OUT);
        }
    }

    /* --- CML MLP forward --- */
    {
        int x_shape[] = {batch, in_f};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* x_data = cml_malloc(sizeof(float) * batch * in_f);
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
        double t_matmul1 = 0, t_ir_reset = 0;

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

        cml_free(x_data);
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

            float* input = cml_malloc(sizeof(float) * cb * ic * ih * iw);
            float* weight = cml_malloc(sizeof(float) * oc * col_h);
            float* col = cml_malloc(sizeof(float) * col_h * col_w);
            float* output = cml_malloc(sizeof(float) * cb * oc * oh * ow);
            fill_random(input, cb * ic * ih * iw);
            fill_random(weight, oc * col_h);

            for (int i = 0; i < 5; i++)
                raw_blas_conv2d_forward(blas, input, weight, NULL, col, output,
                                        cb, ic, ih, iw, oc, oh, ow, kh, kw);

            double t0 = now();
            for (int i = 0; i < iters; i++)
                raw_blas_conv2d_forward(blas, input, weight, NULL, col, output,
                                        cb, ic, ih, iw, oc, oh, ow, kh, kw);
            double ms = (now() - t0) / iters * 1e3;
            printf("Raw BLAS im2col+mm conv: %8.3f ms\n", ms);

            cml_free(input); cml_free(weight); cml_free(col); cml_free(output);
        }
    }

    /* --- CML Conv2d --- */
    {
        int cb = 8, ic = 3, h = 32, w = 32, oc = 16, ksize = 3;
        int x_shape[] = {cb, ic, h, w};
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        float* x_data = cml_malloc(sizeof(float) * cb * ic * h * w);
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

        cml_free(x_data);
        module_free((Module*)conv);
    }

    cml_cleanup();
    return 0;
}
