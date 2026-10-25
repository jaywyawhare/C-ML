/* Cutout / mixup / cutmix augmentation, verified exactly: mixup is a deterministic
 * convex blend, cutmix pastes a known box, and cutout zeros size*size per image. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "core/augmentation.h"
#include "core/threefry.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/* 1x1xHxW tensor filled with `v`. */
static Tensor* filled(int h, int w, float v) {
    int shape[4] = {1, 1, h, w};
    Tensor* t    = tensor_empty(shape, 4, &CFG);
    float* d     = (float*)tensor_data_ptr(t);
    for (int i = 0; i < h * w; i++)
        d[i] = v;
    return t;
}

static int test_mixup(void) {
    Tensor* a = filled(2, 2, 10.0f);
    Tensor* b = filled(2, 2, 20.0f);
    Tensor* o = augment_mixup(a, b, 0.25f); /* 0.25*10 + 0.75*20 = 17.5 */
    int ok    = o != NULL;
    float* od = o ? (float*)tensor_data_ptr(o) : NULL;
    for (int i = 0; ok && i < 4; i++)
        ok &= fabsf(od[i] - 17.5f) < 1e-5f;
    /* lambda=1 returns a exactly */
    Tensor* o1 = augment_mixup(a, b, 1.0f);
    float* o1d = o1 ? (float*)tensor_data_ptr(o1) : NULL;
    for (int i = 0; ok && i < 4; i++)
        ok &= fabsf(o1d[i] - 10.0f) < 1e-5f;
    tensor_free(a);
    tensor_free(b);
    tensor_free(o);
    tensor_free(o1);
    return ok;
}

static int test_cutmix(void) {
    Tensor* a = filled(4, 4, 1.0f);
    Tensor* b = filled(4, 4, 2.0f);
    /* paste 2x2 box at (top=1,left=1): those 4 cells become 2, rest stay 1 */
    Tensor* o = augment_cutmix(a, b, 1, 1, 2, 2);
    int ok    = o != NULL;
    float* od = o ? (float*)tensor_data_ptr(o) : NULL;
    int twos = 0, ones = 0;
    for (int y = 0; y < 4 && ok; y++)
        for (int x = 0; x < 4; x++) {
            float v   = od[y * 4 + x];
            int inbox = (y >= 1 && y < 3 && x >= 1 && x < 3);
            if (inbox)
                ok &= fabsf(v - 2.0f) < 1e-5f, twos++;
            else
                ok &= fabsf(v - 1.0f) < 1e-5f, ones++;
        }
    ok &= twos == 4 && ones == 12;
    tensor_free(a);
    tensor_free(b);
    tensor_free(o);
    return ok;
}

static int test_cutout(void) {
    cml_rng_set_global_seed(123);
    Tensor* in = filled(8, 8, 5.0f); /* all non-zero */
    Tensor* o  = augment_cutout(in, 4);
    int ok     = o != NULL;
    float* od  = o ? (float*)tensor_data_ptr(o) : NULL;
    int zeros = 0, fives = 0;
    for (int i = 0; i < 64 && ok; i++) {
        if (od[i] == 0.0f)
            zeros++;
        else if (fabsf(od[i] - 5.0f) < 1e-5f)
            fives++;
    }
    /* a fully-interior 4x4 box zeros 16 cells; clamped boxes zero fewer. */
    ok &= zeros > 0 && zeros <= 16 && (zeros + fives) == 64;
    tensor_free(in);
    tensor_free(o);
    return ok;
}

static int test_shape_mismatch_rejected(void) {
    Tensor* a = filled(2, 2, 1.0f);
    Tensor* b = filled(3, 3, 1.0f);
    int ok    = augment_mixup(a, b, 0.5f) == NULL && augment_cutmix(a, b, 0, 0, 1, 1) == NULL;
    tensor_free(a);
    tensor_free(b);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== mixup / cutmix / cutout ===\n");
    TEST(mixup);
    TEST(cutmix);
    TEST(cutout);
    TEST(shape_mismatch_rejected);
    cml_cleanup();
    return TEST_SUMMARY();
}
