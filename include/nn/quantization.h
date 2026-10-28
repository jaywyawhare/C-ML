#ifndef CML_NN_QUANTIZATION_H
#define CML_NN_QUANTIZATION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Symmetric per-row int8 quantization: scale[r] = max|row|/127, q = round(x/scale) in [-127,127].
 * @param in (const float*)
 * @param rows (int)
 * @param cols (int)
 * @param q (int8_t*)
 * @param scales (float*)
 * @return (void)
 */
void cml_quantize_int8_row(const float* in, int rows, int cols, int8_t* q, float* scales);

/**
 * Inverse of cml_quantize_int8_row: out[r][c] = q[r][c] * scales[r].
 * @param q (const int8_t*)
 * @param scales (const float*)
 * @param rows (int)
 * @param cols (int)
 * @param out (float*)
 * @return (void)
 */
void cml_dequantize_int8_row(const int8_t* q, const float* scales, int rows, int cols, float* out);

/**
 * Symmetric groupwise int4 quantization: per contiguous group of group_size values, scale =
 * max|group|/7 and q = round(x/scale) in [-8,7] (stored one value per int8). The last group may
 * be short. scales holds ceil(n/group_size) entries.
 * @param in (const float*)
 * @param n (int)
 * @param group_size (int)
 * @param q (int8_t*)
 * @param scales (float*)
 * @return (void)
 */
void cml_quantize_int4_group(const float* in, int n, int group_size, int8_t* q, float* scales);

/**
 * Inverse of cml_quantize_int4_group: out[i] = q[i] * scales[i / group_size].
 * @param q (const int8_t*)
 * @param scales (const float*)
 * @param n (int)
 * @param group_size (int)
 * @param out (float*)
 * @return (void)
 */
void cml_dequantize_int4_group(const int8_t* q, const float* scales, int n, int group_size,
                               float* out);

/**
 * int8 matmul C[M,N] = A @ Bt^T with int32 accumulation rescaled to f32, where A[M,K] and the
 * weight Bt[N,K] (nn.Linear layout) are both per-row int8 (scales_a[M], scales_b[N]). Returns 0
 * on success, -1 on bad args.
 * @param A (const int8_t*)
 * @param scales_a (const float*)
 * @param M (int)
 * @param K (int)
 * @param Bt (const int8_t*)
 * @param scales_b (const float*)
 * @param N (int)
 * @param C (float*)
 * @return (int)
 */
int cml_matmul_int8(const int8_t* A, const float* scales_a, int M, int K, const int8_t* Bt,
                    const float* scales_b, int N, float* C);

/**
 * int8 linear forward y[M,N] = x[M,K] @ W^T with pre-quantized weights: quantizes the activations
 * x per row, then runs the int8 matmul against Wq[N,K] (nn.Linear layout, quantize once with
 * cml_quantize_int8_row). Returns 0 on success, -1 on bad args or allocation failure.
 * @param x (const float*)
 * @param M (int)
 * @param K (int)
 * @param Wq (const int8_t*)
 * @param W_scales (const float*)
 * @param N (int)
 * @param y (float*)
 * @return (int)
 */
int cml_linear_int8_forward(const float* x, int M, int K, const int8_t* Wq, const float* W_scales,
                            int N, float* y);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_QUANTIZATION_H */
