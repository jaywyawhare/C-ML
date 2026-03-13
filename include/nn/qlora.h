/**
 * @file qlora.h
 * @brief QLoRA (Quantized Low-Rank Adaptation) for memory-efficient fine-tuning
 *
 * Implements QLoRA as described in "QLoRA: Efficient Finetuning of Quantized
 * LLMs" (Dettmers et al., 2023). QLoRA uses NF4 (Normal Float 4-bit)
 * quantization for the frozen base weights and adds trainable low-rank
 * adapters (LoRA) in full precision.
 *
 * For a linear layer y = xW^T, QLoRA computes:
 *   y = x @ dequant(W_nf4)^T + (alpha/rank) * x @ A^T @ B^T
 *
 * The base weight is stored in NF4 (4-bit), reducing memory by ~8x compared
 * to float32. LoRA matrices A and B remain in float32 for training.
 */

#ifndef CML_NN_QLORA_H
#define CML_NN_QLORA_H

#include <stdbool.h>
#include <stddef.h>
#include "tensor/tensor.h"
#include "core/quantization.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NF4-quantized tensor representation
 *
 * Stores a float32 tensor in NF4 format: 4-bit indices packed into uint8,
 * plus per-block scale factors for dequantization.
 */
typedef struct CMLNF4Tensor {
    Tensor* packed_data;    /* uint8 tensor, numel = original_numel / 2 */
    float* scales;          /* Per-block scale factors */
    int num_scales;
    int block_size;
    size_t original_numel;
    int* original_shape;
    int original_ndim;
} CMLNF4Tensor;

/**
 * @brief QLoRA-adapted linear layer
 *
 * Combines an NF4-quantized base weight with trainable LoRA adapters.
 * The forward pass dequantizes the base weight on-the-fly and adds
 * the low-rank LoRA contribution.
 */
typedef struct CMLQLoRALinear {
    int in_features;
    int out_features;
    int rank;
    float alpha;
    float scaling;          /* alpha / rank */

    CMLNF4Tensor* base_weight_nf4;  /* NF4-quantized base weight */
    Tensor* lora_A;         /* [rank, in_features] float32 */
    Tensor* lora_B;         /* [out_features, rank] float32 */

    bool enable_double_quant;  /* Quantize the scales too */
} CMLQLoRALinear;

/**
 * @brief Create an NF4-quantized tensor from a float32 tensor
 *
 * @param float_tensor Input float32 tensor (any shape)
 * @param block_size Elements per quantization block (typically 64)
 * @return NF4 tensor, or NULL on failure
 */
CMLNF4Tensor* cml_nf4_tensor_create(Tensor* float_tensor, int block_size);

/**
 * @brief Free an NF4 tensor and all its resources
 *
 * @param nf4 NF4 tensor to free
 */
void cml_nf4_tensor_free(CMLNF4Tensor* nf4);

/**
 * @brief Dequantize an NF4 tensor back to float32
 *
 * Reconstructs a float32 tensor with the original shape.
 *
 * @param nf4 NF4 tensor to dequantize
 * @return Float32 tensor with original shape, or NULL on failure
 */
Tensor* cml_nf4_tensor_dequantize(const CMLNF4Tensor* nf4);

/**
 * @brief Create a QLoRA linear layer
 *
 * Quantizes the base weight to NF4 and initializes LoRA matrices:
 * - lora_A: [rank, in_features] initialized with small random values (Xavier)
 * - lora_B: [out_features, rank] initialized to zeros
 *
 * @param base_weight Base weight tensor [out_features, in_features] (consumed/quantized)
 * @param rank LoRA rank (typically 4, 8, or 16)
 * @param alpha LoRA scaling factor (typically equal to rank)
 * @param block_size NF4 quantization block size (typically 64)
 * @return QLoRA linear layer, or NULL on failure
 */
CMLQLoRALinear* cml_qlora_linear_create(Tensor* base_weight, int rank,
                                         float alpha, int block_size);

/**
 * @brief Free a QLoRA linear layer and all its resources
 *
 * Frees the NF4 base weight and LoRA A/B matrices.
 *
 * @param qlora QLoRA linear layer to free
 */
void cml_qlora_linear_free(CMLQLoRALinear* qlora);

/**
 * @brief Forward pass through QLoRA linear layer
 *
 * Computes: output = input @ dequant(W_nf4)^T + scaling * input @ A^T @ B^T
 *
 * The base weight is dequantized to a float32 temporary for the matmul,
 * then freed. LoRA contribution is computed separately and added.
 *
 * @param qlora QLoRA linear layer
 * @param input Input tensor [batch, in_features]
 * @return Output tensor [batch, out_features], or NULL on failure
 */
Tensor* cml_qlora_linear_forward(CMLQLoRALinear* qlora, Tensor* input);

/**
 * @brief Compute memory usage of the QLoRA layer
 *
 * Returns the total bytes used by:
 * - NF4-packed base weight (numel/2 bytes)
 * - Per-block scale factors
 * - LoRA A and B matrices in float32
 *
 * @param qlora QLoRA linear layer
 * @return Memory usage in bytes
 */
size_t cml_qlora_memory_usage(const CMLQLoRALinear* qlora);

/**
 * @brief Compute memory that a full float32 weight would use
 *
 * Returns in_features * out_features * sizeof(float) for comparison
 * with cml_qlora_memory_usage.
 *
 * @param in_features Number of input features
 * @param out_features Number of output features
 * @return Full float32 memory usage in bytes
 */
size_t cml_qlora_full_memory_usage(int in_features, int out_features);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_QLORA_H */
