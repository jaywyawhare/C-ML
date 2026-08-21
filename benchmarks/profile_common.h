/**
 * Shared scaffolding for the profile_*.c harnesses.
 *
 * Each harness isolates a different source of overhead, but they all measure
 * against the same raw-BLAS reference implementations -- and they must use the
 * *same* ones, or their numbers cannot be read side by side.
 */
#ifndef CML_PROFILE_COMMON_H
#define CML_PROFILE_COMMON_H

#include "bench_timing.h"

#include "cml.h"
#include "backend/blas.h"
#include "alloc/cml_allocator.h"

#include <stdio.h>

/* One MLP forward pass -- Linear -> ReLU -> Linear -- written straight against
 * BLAS with no tensors, graph or dispatch. This is the floor every framework
 * path in these harnesses is measured against, so it has to be the identical
 * arithmetic in each of them; that is why it lives here rather than being
 * spelled out at each timing site. */
static inline void raw_blas_mlp_forward(CMLBlasContext* blas, const float* X, const float* W1,
                                        const float* B1, float* H, const float* W2,
                                        const float* B2, float* OUT, int batch, int in_f, int hid,
                                        int out_f) {
    cml_blas_sgemm_ex(blas, X, W1, H, batch, hid, in_f, 1.0f, 0.0f, false, true);
    for (int r = 0; r < batch; r++)
        for (int c = 0; c < hid; c++) {
            H[r * hid + c] += B1[c];
            if (H[r * hid + c] < 0) H[r * hid + c] = 0;
        }
    cml_blas_sgemm_ex(blas, H, W2, OUT, batch, out_f, hid, 1.0f, 0.0f, false, true);
    for (int r = 0; r < batch; r++)
        for (int c = 0; c < out_f; c++)
            OUT[r * out_f + c] += B2[c];
}

/* Lower image `b` of the batch into `col` so a plain sgemm computes the
 * convolution. Split out from the full pass below because the detailed
 * profiler times this half on its own.
 *
 * Each (channel, kernel-row, kernel-col) patch row is a strided run of whole
 * output rows, so it copies row-at-a-time rather than element-at-a-time. */
static inline void raw_im2col(const float* input, float* col, int b, int ic, int ih, int iw, int oh,
                              int ow, int kh, int kw) {
    const int col_w = oh * ow;
    for (int c = 0; c < ic; c++)
        for (int kr = 0; kr < kh; kr++)
            for (int kc = 0; kc < kw; kc++) {
                int row           = (c * kh + kr) * kw + kc;
                const float* src  = input + ((b * ic + c) * ih + kr) * iw + kc;
                float* dst        = col + row * col_w;
                for (int r = 0; r < oh; r++)
                    memcpy(dst + r * ow, src + r * iw, ow * sizeof(float));
            }
}

/* One conv2d forward pass over the batch as im2col + sgemm -- the raw-BLAS
 * counterpart to raw_blas_mlp_forward, and for the same reason. Pass `bias`
 * NULL to time the convolution alone. */
static inline void raw_blas_conv2d_forward(CMLBlasContext* blas, const float* input,
                                           const float* weight, const float* bias, float* col,
                                           float* output, int cb, int ic, int ih, int iw, int oc,
                                           int oh, int ow, int kh, int kw) {
    const int col_h = ic * kh * kw;
    const int col_w = oh * ow;
    for (int b = 0; b < cb; b++) {
        raw_im2col(input, col, b, ic, ih, iw, oh, ow, kh, kw);
        cml_blas_sgemm(blas, weight, col, output + (size_t)b * oc * oh * ow, oc, col_w, col_h, 1.0f,
                       0.0f);
        if (!bias) continue;
        for (int o = 0; o < oc; o++) {
            float bv   = bias[o];
            float* row = output + ((size_t)b * oc + o) * col_w;
            for (int j = 0; j < col_w; j++)
                row[j] += bv;
        }
    }
}

#endif /* CML_PROFILE_COMMON_H */
