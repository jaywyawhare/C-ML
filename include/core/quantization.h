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
