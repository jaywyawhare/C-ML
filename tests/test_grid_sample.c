/*
 * affine_grid / grid_sample validation. Oracles: identity affine warp must
 * reproduce the image exactly (round-trip); bilinear/nearest sample values are
 * cross-checked against a numpy reference of torch's align_corners semantics
 * (anchor points hand-verified: center->5, corners->1/9); affine_grid gradient
 * is finite-differenced.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "nn/grid_sample.h"
#include "test_harness.h"

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static Tensor* mk(const float* data, int* shape, int ndim, bool grad) {
    Tensor* t = tensor_zeros(shape, ndim, &cfg);
    memcpy(tensor_data_ptr(t), data, tensor_numel(shape, ndim) * sizeof(float));
    t->requires_grad = grad;
    return t;
}

static int approx(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > tol)
            return 0;
    return 1;
}

static int test_affine_grid_identity(void) {
    float td[6] = {1, 0, 0, 0, 1, 0}; /* identity */
    int tsh[3]  = {1, 2, 3};
    cml_reset_ir_context();
    Tensor* th = mk(td, tsh, 3, false);
    Tensor* g  = cml_affine_grid(th, 1, 3, 3, true);
    if (!g || tensor_ensure_executed(g) != 0) {
        cml_reset_ir_context();
        return 0;
    }
    float* gd = (float*)tensor_data_ptr(g);
    /* top-left (-1,-1), center (0,0), bottom-right (1,1). */
    int ok = fabsf(gd[0] + 1) < 1e-5f && fabsf(gd[1] + 1) < 1e-5f && /* (0,0) */
             fabsf(gd[8] - 0) < 1e-5f && fabsf(gd[9] - 0) < 1e-5f && /* center idx 4 -> 8,9 */
             fabsf(gd[16] - 1) < 1e-5f && fabsf(gd[17] - 1) < 1e-5f; /* (2,2) */
    cml_reset_ir_context();
    return ok;
}

static int test_identity_warp_roundtrip(void) {
    float img[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float td[6]  = {1, 0, 0, 0, 1, 0};
    int ish[4]   = {1, 1, 3, 3};
    int tsh[3]   = {1, 2, 3};
    cml_reset_ir_context();
    Tensor* in = mk(img, ish, 4, false);
    Tensor* th = mk(td, tsh, 3, false);
    Tensor* g  = cml_affine_grid(th, 1, 3, 3, true);
    Tensor* o  = cml_grid_sample(in, g, CML_GRID_BILINEAR, CML_GRID_PAD_BORDER, true);
    if (!o || tensor_ensure_executed(o) != 0) {
        cml_reset_ir_context();
        return 0;
    }
    int ok = approx((float*)tensor_data_ptr(o), img, 9, 1e-4f);
    cml_reset_ir_context();
    return ok;
}

/* 5 probe coords as a [1,5,1,2] grid; compare to the numpy reference vectors. */
static int sample_probes(CMLGridSampleMode mode, CMLGridPadMode pad, bool ac, const float* want) {
    float img[9]   = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float grid[10] = {0, 0, -1, -1, 1, 1, -0.5f, 0.25f, 2, 2};
    int ish[4]     = {1, 1, 3, 3};
    int gsh[4]     = {1, 5, 1, 2};
    cml_reset_ir_context();
    Tensor* in = mk(img, ish, 4, false);
    Tensor* g  = mk(grid, gsh, 4, false);
    Tensor* o  = cml_grid_sample(in, g, mode, pad, ac);
    if (!o || tensor_ensure_executed(o) != 0) {
        cml_reset_ir_context();
        return 0;
    }
    int ok = approx((float*)tensor_data_ptr(o), want, 5, 1e-4f);
    cml_reset_ir_context();
    return ok;
}

static int test_grid_sample_bilinear_ac_zeros(void) {
    float want[5] = {5.0f, 1.0f, 9.0f, 5.25f, 0.0f};
    return sample_probes(CML_GRID_BILINEAR, CML_GRID_PAD_ZEROS, true, want);
}

static int test_grid_sample_bilinear_noac_zeros(void) {
    float want[5] = {5.0f, 0.25f, 2.25f, 5.375f, 0.0f};
    return sample_probes(CML_GRID_BILINEAR, CML_GRID_PAD_ZEROS, false, want);
}

static int test_grid_sample_bilinear_ac_border(void) {
    float want[5] = {5.0f, 1.0f, 9.0f, 5.25f, 9.0f};
    return sample_probes(CML_GRID_BILINEAR, CML_GRID_PAD_BORDER, true, want);
}

static int test_grid_sample_nearest_ac_border(void) {
    float want[5] = {5.0f, 1.0f, 9.0f, 4.0f, 9.0f};
    return sample_probes(CML_GRID_NEAREST, CML_GRID_PAD_BORDER, true, want);
}

static float grid_sum(const float* td) {
    int tsh[3] = {1, 2, 3};
    cml_reset_ir_context();
    Tensor* th   = mk(td, tsh, 3, false);
    Tensor* g    = cml_affine_grid(th, 1, 3, 4, false);
    Tensor* loss = cml_sum(g, -1, false);
    tensor_ensure_executed(loss);
    float r = ((float*)tensor_data_ptr(loss))[0];
    cml_reset_ir_context();
    return r;
}

static int test_affine_grid_grad(void) {
    float td[6] = {0.9f, 0.1f, 0.2f, -0.1f, 1.1f, -0.2f};
    int tsh[3]  = {1, 2, 3};
    cml_reset_ir_context();
    Tensor* th   = mk(td, tsh, 3, true);
    Tensor* g    = cml_affine_grid(th, 1, 3, 4, false);
    Tensor* loss = cml_sum(g, -1, false);
    tensor_backward(loss, NULL, false, false);
    if (!th->grad) {
        cml_reset_ir_context();
        return 0;
    }
    tensor_ensure_executed(th->grad);
    float ana[6];
    memcpy(ana, tensor_data_ptr(th->grad), sizeof(ana));
    cml_reset_ir_context();

    const float eps = 1e-3f;
    for (int i = 0; i < 6; i++) {
        float s   = td[i];
        td[i]     = s + eps;
        float fp  = grid_sum(td);
        td[i]     = s - eps;
        float fm  = grid_sum(td);
        td[i]     = s;
        float num = (fp - fm) / (2 * eps);
        if (fabsf(num - ana[i]) > 1e-2f * fmaxf(1.0f, fabsf(num)))
            return 0;
    }
    return 1;
}

int main(void) {
    printf("test_grid_sample\n\n");
    TEST(affine_grid_identity);
    TEST(identity_warp_roundtrip);
    TEST(grid_sample_bilinear_ac_zeros);
    TEST(grid_sample_bilinear_noac_zeros);
    TEST(grid_sample_bilinear_ac_border);
    TEST(grid_sample_nearest_ac_border);
    TEST(affine_grid_grad);
    cml_reset_ir_context();
    return TEST_SUMMARY();
}
