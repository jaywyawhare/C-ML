/**
 * @file quantization.h
 * @brief Int8 quantization and dequantization support
 *
 * Provides per-tensor affine quantization:
 *   quantized = clamp(round(value / scale) + zero_point, -128, 127)
 *   dequantized = (quantized - zero_point) * scale
 */

#ifndef CML_CORE_QUANTIZATION_H
#define CML_CORE_QUANTIZATION_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Quantization parameters for a tensor
 */
typedef struct QuantParams {
    float scale;       // Scale factor
    int32_t zero_point; // Zero point offset
} QuantParams;

/**
 * @brief Compute quantization parameters from a float32 tensor
 *
 * Determines scale and zero_point for symmetric or asymmetric quantization.
 *
 * @param tensor Input float32 tensor
 * @param symmetric If true, zero_point = 0 (symmetric around 0)
 * @return Quantization parameters
 */
QuantParams cml_quantize_compute_params(Tensor* tensor, bool symmetric);

/**
 * @brief Quantize a float32 tensor to int8
 *
 * @param tensor Input float32 tensor
 * @param params Quantization parameters (if NULL, computed automatically)
 * @param out_params Output: computed quantization parameters
 * @return New int8 tensor, or NULL on failure
 */
Tensor* cml_quantize_int8(Tensor* tensor, const QuantParams* params, QuantParams* out_params);

/**
 * @brief Dequantize an int8 tensor back to float32
 *
 * @param tensor Input int8 tensor
 * @param params Quantization parameters used during quantization
 * @return New float32 tensor, or NULL on failure
 */
Tensor* cml_dequantize_int8(Tensor* tensor, const QuantParams* params);

/**
 * @brief Quantize a float32 tensor to uint8
 *
 * @param tensor Input float32 tensor
 * @param params Quantization parameters (if NULL, computed automatically)
 * @param out_params Output: computed quantization parameters
 * @return New uint8 tensor, or NULL on failure
 */
Tensor* cml_quantize_uint8(Tensor* tensor, const QuantParams* params, QuantParams* out_params);

/**
 * @brief Dequantize a uint8 tensor back to float32
 *
 * @param tensor Input uint8 tensor
 * @param params Quantization parameters used during quantization
 * @return New float32 tensor, or NULL on failure
 */
Tensor* cml_dequantize_uint8(Tensor* tensor, const QuantParams* params);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_QUANTIZATION_H
