/* Verify the no_grad direct-conv kernel matches the grad-enabled decompose
 * (im2col+matmul) path bit-for-bit-ish, across several shapes incl. padding &
 * stride & shallow/deep channels. */
#define _POSIX_C_SOURCE 199309L
#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "alloc/cml_allocator.h"

static void fill_random(float* b, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++) b[i] = (float)rand() / RAND_MAX - 0.5f;
}

static int check_one(int N, int IC, int OC, int H, int W, int K, int stride, int pad) {
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    int xn = N * IC * H * W;
    float* xd = cml_malloc(sizeof(float) * xn);
    fill_random(xd, xn, 7);
    int xs[4] = {N, IC, H, W};

    Conv2d* conv = cml_nn_conv2d(IC, OC, K, stride, pad, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
    module_set_training((Module*)conv, false);

    /* Reference: grad enabled -> decompose (im2col + matmul). */
    cml_enable_grad();
    Tensor* Xr = cml_tensor(xd, xs, 4, &cfg);
    Tensor* ref = module_forward((Module*)conv, Xr);
    float* rp = tensor_data_ptr(ref);
    size_t no = ref->numel;
    float* refbuf = cml_malloc(sizeof(float) * no);
    for (size_t i = 0; i < no; i++) refbuf[i] = rp[i];
    cml_reset_ir_context();

    /* Test: no_grad -> direct conv kernel. */
    cml_no_grad();
    Tensor* Xt = cml_tensor(xd, xs, 4, &cfg);
    Tensor* got = module_forward((Module*)conv, Xt);
    float* gp = tensor_data_ptr(got);

    double maxdiff = 0;
    for (size_t i = 0; i < no; i++) {
        double d = fabs((double)gp[i] - (double)refbuf[i]);
        if (d > maxdiff) maxdiff = d;
    }
    cml_reset_ir_context();
    cml_enable_grad();

    int ok = maxdiff < 1e-4;
    printf("  N%d IC%d OC%d %dx%d k%d s%d p%d : maxdiff=%.2e  %s\n",
           N, IC, OC, H, W, K, stride, pad, maxdiff, ok ? "PASS" : "FAIL");
    cml_free(xd);
    cml_free(refbuf);
    module_free((Module*)conv);
    return ok;
}

int main(void) {
    cml_init();
    int ok = 1;
    ok &= check_one(8, 3, 16, 32, 32, 3, 1, 0);  /* the benchmark shape (shallow) */
    ok &= check_one(2, 3, 8, 16, 16, 3, 1, 1);   /* shallow + padding */
    ok &= check_one(2, 3, 8, 20, 20, 3, 2, 1);   /* shallow + stride 2 + pad */
    ok &= check_one(1, 8, 4, 12, 12, 5, 1, 2);   /* 5x5 kernel */
    ok &= check_one(2, 32, 16, 16, 16, 3, 1, 1); /* deep (>=16) -> im2col/wino path */
    printf("%s\n", ok ? "ALL CONV VERIFY PASS" : "CONV VERIFY FAILED");
    return ok ? 0 : 1;
}
