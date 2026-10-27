#include "nn/quantization.h"
#include <math.h>
#include <stddef.h>

/** Round-half-away-from-zero and clamp to [lo, hi]. */
static int quant_round_clamp(float x, int lo, int hi) {
    int v = (int)lrintf(x);
    if (v < lo)
        v = lo;
    if (v > hi)
        v = hi;
    return v;
}

void cml_quantize_int8_row(const float* in, int rows, int cols, int8_t* q, float* scales) {
    if (!in || !q || !scales || rows <= 0 || cols <= 0)
        return;
    for (int r = 0; r < rows; r++) {
        const float* row = in + (size_t)r * cols;
        float absmax     = 0.0f;
        for (int c = 0; c < cols; c++) {
            float a = fabsf(row[c]);
            if (a > absmax)
                absmax = a;
        }
        float scale = (absmax > 0.0f) ? absmax / 127.0f : 1.0f;
        scales[r]   = scale;
        int8_t* qr  = q + (size_t)r * cols;
        float inv   = 1.0f / scale;
        for (int c = 0; c < cols; c++)
            qr[c] = (int8_t)quant_round_clamp(row[c] * inv, -127, 127);
    }
}

void cml_dequantize_int8_row(const int8_t* q, const float* scales, int rows, int cols, float* out) {
    if (!q || !scales || !out || rows <= 0 || cols <= 0)
        return;
    for (int r = 0; r < rows; r++) {
        float scale = scales[r];
        for (int c = 0; c < cols; c++)
            out[(size_t)r * cols + c] = (float)q[(size_t)r * cols + c] * scale;
    }
}

void cml_quantize_int4_group(const float* in, int n, int group_size, int8_t* q, float* scales) {
    if (!in || !q || !scales || n <= 0 || group_size <= 0)
        return;
    int g = 0;
    for (int start = 0; start < n; start += group_size, g++) {
        int end = start + group_size;
        if (end > n)
            end = n;
        float absmax = 0.0f;
        for (int i = start; i < end; i++) {
            float a = fabsf(in[i]);
            if (a > absmax)
                absmax = a;
        }
        float scale = (absmax > 0.0f) ? absmax / 7.0f : 1.0f;
        scales[g]   = scale;
        float inv   = 1.0f / scale;
        for (int i = start; i < end; i++)
            q[i] = (int8_t)quant_round_clamp(in[i] * inv, -8, 7);
    }
}

void cml_dequantize_int4_group(const int8_t* q, const float* scales, int n, int group_size,
                               float* out) {
    if (!q || !scales || !out || n <= 0 || group_size <= 0)
        return;
    for (int i = 0; i < n; i++)
        out[i] = (float)q[i] * scales[i / group_size];
}

int cml_matmul_int8(const int8_t* A, const float* scales_a, int M, int K, const int8_t* Bt,
                    const float* scales_b, int N, float* C) {
    if (!A || !scales_a || !Bt || !scales_b || !C || M <= 0 || K <= 0 || N <= 0)
        return -1;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc         = 0;
            const int8_t* a_row = A + (size_t)m * K;
            const int8_t* b_row = Bt + (size_t)n * K;
            for (int k = 0; k < K; k++)
                acc += (int32_t)a_row[k] * (int32_t)b_row[k];
            C[(size_t)m * N + n] = (float)acc * scales_a[m] * scales_b[n];
        }
    }
    return 0;
}
