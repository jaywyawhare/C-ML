#include "nn/grid_sample.h"

#include <math.h>
#include <string.h>

#include "cml.h"

Tensor* cml_affine_grid(Tensor* theta, int N, int H, int W, bool align_corners) {
    if (!theta || theta->ndim != 3 || theta->shape[0] != N || theta->shape[1] != 2 ||
        theta->shape[2] != 3 || H <= 0 || W <= 0)
        return NULL;

    int hw           = H * W;
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = theta->device, .has_dtype = true, .has_device = true};
    int bt_sh[2] = {3, hw};
    Tensor* bt   = cml_zeros(bt_sh, 2, &cfg); /* [x;y;1] per output location */
    if (!bt)
        return NULL;
    float* b = (float*)tensor_data_ptr(bt);
    for (int y = 0; y < H; y++) {
        float gy = (H == 1)        ? 0.0f
                   : align_corners ? (2.0f * y / (H - 1) - 1.0f)
                                   : ((2.0f * y + 1.0f) / H - 1.0f);
        for (int x = 0; x < W; x++) {
            float gx      = (W == 1)        ? 0.0f
                            : align_corners ? (2.0f * x / (W - 1) - 1.0f)
                                            : ((2.0f * x + 1.0f) / W - 1.0f);
            int p         = y * W + x;
            b[0 * hw + p] = gx;
            b[1 * hw + p] = gy;
            b[2 * hw + p] = 1.0f;
        }
    }

    int tf_sh[2] = {N * 2, 3};
    Tensor* tf   = cml_reshape(theta, tf_sh, 2); /* [N*2, 3] */
    Tensor* mm   = cml_matmul(tf, bt);           /* [N*2, HW] */
    int m_sh[3]  = {N, 2, hw};
    Tensor* m3   = cml_reshape(mm, m_sh, 3);
    Tensor* mt   = cml_transpose(m3, 1, 2); /* [N, HW, 2] */
    int g_sh[4]  = {N, H, W, 2};
    return cml_reshape(mt, g_sh, 4);
}

/**
 * Map a normalized coordinate in [-1,1] to a source pixel index.
 * @param c (float)
 * @param size (int)
 * @param align_corners (bool)
 * @return (float)
 */
static float unnormalize(float c, int size, bool align_corners) {
    return align_corners ? (c + 1.0f) * 0.5f * (size - 1) : ((c + 1.0f) * size - 1.0f) * 0.5f;
}

/**
 * Fetch a source pixel with the selected out-of-bounds policy.
 * @return (float)
 */
static float fetch(const float* img, int H, int W, int y, int x, CMLGridPadMode padding) {
    if (padding == CML_GRID_PAD_BORDER) {
        if (y < 0)
            y = 0;
        if (y > H - 1)
            y = H - 1;
        if (x < 0)
            x = 0;
        if (x > W - 1)
            x = W - 1;
    } else if (x < 0 || x > W - 1 || y < 0 || y > H - 1) {
        return 0.0f;
    }
    return img[(size_t)y * W + x];
}

Tensor* cml_grid_sample(Tensor* input, Tensor* grid, CMLGridSampleMode mode, CMLGridPadMode padding,
                        bool align_corners) {
    if (!input || !grid || input->ndim != 4 || grid->ndim != 4)
        return NULL;
    int N = input->shape[0], C = input->shape[1], Hin = input->shape[2], Win = input->shape[3];
    int Hout = grid->shape[1], Wout = grid->shape[2];
    if (grid->shape[0] != N || grid->shape[3] != 2)
        return NULL;

    if (tensor_ensure_executed(input) != 0 || tensor_ensure_executed(grid) != 0)
        return NULL;
    const float* in = (const float*)tensor_data_ptr(input);
    const float* gr = (const float*)tensor_data_ptr(grid);
    if (!in || !gr)
        return NULL;

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = input->device, .has_dtype = true, .has_device = true};
    int out_sh[4] = {N, C, Hout, Wout};
    Tensor* out   = cml_zeros(out_sh, 4, &cfg);
    if (!out)
        return NULL;
    float* od = (float*)tensor_data_ptr(out);

    for (int n = 0; n < N; n++) {
        for (int oy = 0; oy < Hout; oy++) {
            for (int ox = 0; ox < Wout; ox++) {
                size_t gidx = (((size_t)n * Hout + oy) * Wout + ox) * 2;
                float fx    = unnormalize(gr[gidx + 0], Win, align_corners);
                float fy    = unnormalize(gr[gidx + 1], Hin, align_corners);

                for (int c = 0; c < C; c++) {
                    const float* img = in + ((size_t)n * C + c) * Hin * Win;
                    float val;
                    if (mode == CML_GRID_NEAREST) {
                        int xr = (int)nearbyintf(fx);
                        int yr = (int)nearbyintf(fy);
                        val    = fetch(img, Hin, Win, yr, xr, padding);
                    } else {
                        int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
                        int x1 = x0 + 1, y1 = y0 + 1;
                        float ax = fx - x0, ay = fy - y0;
                        float v00 = fetch(img, Hin, Win, y0, x0, padding);
                        float v01 = fetch(img, Hin, Win, y0, x1, padding);
                        float v10 = fetch(img, Hin, Win, y1, x0, padding);
                        float v11 = fetch(img, Hin, Win, y1, x1, padding);
                        val       = v00 * (1 - ax) * (1 - ay) + v01 * ax * (1 - ay) +
                              v10 * (1 - ax) * ay + v11 * ax * ay;
                    }
                    od[(((size_t)n * C + c) * Hout + oy) * Wout + ox] = val;
                }
            }
        }
    }
    return out;
}
