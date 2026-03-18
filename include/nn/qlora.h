/*
 * QLoRA: NF4-quantized base weights + trainable LoRA adapters in float32.
 * y = x @ dequant(W_nf4)^T + (alpha/rank) * x @ A^T @ B^T
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

typedef struct CMLNF4Tensor {
    Tensor* packed_data;    /* uint8 tensor, numel = original_numel / 2 */
    float* scales;          /* Per-block scale factors */
    int num_scales;
    int block_size;
    size_t original_numel;
    int* original_shape;
    int original_ndim;
} CMLNF4Tensor;

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

CMLNF4Tensor* cml_nf4_tensor_create(Tensor* float_tensor, int block_size);

void cml_nf4_tensor_free(CMLNF4Tensor* nf4);

Tensor* cml_nf4_tensor_dequantize(const CMLNF4Tensor* nf4);

/* Quantizes base_weight to NF4, initializes lora_A (Xavier) and lora_B (zeros). */
CMLQLoRALinear* cml_qlora_linear_create(Tensor* base_weight, int rank,
                                         float alpha, int block_size);

void cml_qlora_linear_free(CMLQLoRALinear* qlora);

/* output = input @ dequant(W_nf4)^T + scaling * input @ A^T @ B^T
 * Base weight is dequantized to a temporary for the matmul, then freed. */
Tensor* cml_qlora_linear_forward(CMLQLoRALinear* qlora, Tensor* input);

size_t cml_qlora_memory_usage(const CMLQLoRALinear* qlora);

size_t cml_qlora_full_memory_usage(int in_features, int out_features);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_QLORA_H */
