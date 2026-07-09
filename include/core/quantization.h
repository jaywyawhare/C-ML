/*
 * Per-tensor affine quantization:
 *   quantized = clamp(round(value / scale) + zero_point, -128, 127)
 *   dequantized = (quantized - zero_point) * scale
 */

#ifndef CML_CORE_QUANTIZATION_H
#define CML_CORE_QUANTIZATION_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QuantParams {
    float scale;       // Scale factor
    int32_t zero_point; // Zero point offset
} QuantParams;

QuantParams cml_quantize_compute_params(Tensor* tensor, bool symmetric);
Tensor* cml_quantize_int8(Tensor* tensor, const QuantParams* params, QuantParams* out_params);
Tensor* cml_dequantize_int8(Tensor* tensor, const QuantParams* params);
Tensor* cml_quantize_uint8(Tensor* tensor, const QuantParams* params, QuantParams* out_params);
Tensor* cml_dequantize_uint8(Tensor* tensor, const QuantParams* params);

/*
 * Weight-only affine-int8 quantization for matmul.
 *
 * cml_quantize_weight_int8 quantizes a 2D [K,N] float weight into an int8
 * tensor that carries its affine params (quant_type = CML_QUANT_AFFINE_INT8,
 * quant_scale/quant_zero_point set, int8 data in ->data).  When such a tensor is
 * the right-hand operand of a matmul, the executor dispatches to the fast
 * integer-weight path below instead of dequantizing to f32 first.
 */
Tensor* cml_quantize_weight_int8(Tensor* weight, bool symmetric);

/*
 * y[M,N] = scale * ( x[M,K] @ w[K,N] - zero_point * rowsum(x) )
 * where w is int8 and x/y are float32.  Returns 0 on success, -1 on bad args.
 */
int cml_qmatmul_affine_int8(const float* x, const int8_t* w, float scale,
                            int32_t zero_point, float* y, int M, int K, int N);

/* NF4 (Normal Float 4-bit) lookup table - 16 values optimal for normal distribution */
extern const float CML_NF4_TABLE[16];

/*
 * Each uint8 stores two NF4 values (high nibble + low nibble).
 * Block size determines granularity of scale factors.
 */
Tensor* cml_quantize_nf4(Tensor* tensor, int block_size,
                          float** out_scales, int* out_num_scales);

Tensor* cml_dequantize_nf4(Tensor* nf4_tensor, const float* scales,
                            int num_scales, int block_size, size_t original_numel);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_QUANTIZATION_H
