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

/* ===== NF4 Quantization ===== */

/** NF4 (Normal Float 4-bit) lookup table - 16 values optimal for normal distribution */
extern const float CML_NF4_TABLE[16];

/** Quantize a float32 tensor to NF4 (4-bit, packed into uint8)
 * Each uint8 stores two NF4 values (high nibble + low nibble).
 * Block size determines granularity of scale factors.
 * @param tensor Input float32 tensor
 * @param block_size Number of elements per quantization block (default: 64)
 * @param out_scales Output: scale factors per block (caller frees)
 * @param out_num_scales Output: number of scale factors
 * @return Packed NF4 data as uint8 tensor (numel/2 elements), or NULL
 */
Tensor* cml_quantize_nf4(Tensor* tensor, int block_size,
                          float** out_scales, int* out_num_scales);

/** Dequantize NF4 back to float32
 * @param nf4_tensor Packed NF4 tensor (uint8, numel/2)
 * @param scales Block scale factors
 * @param num_scales Number of scale factors
 * @param block_size Elements per block
 * @param original_numel Original number of float elements
 * @return Float32 tensor, or NULL
 */
Tensor* cml_dequantize_nf4(Tensor* nf4_tensor, const float* scales,
                            int num_scales, int block_size, size_t original_numel);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_QUANTIZATION_H
