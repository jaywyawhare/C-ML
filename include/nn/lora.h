/**
 * @file lora.h
 * @brief LoRA (Low-Rank Adaptation) for efficient fine-tuning
 *
 * Implements LoRA as described in "LoRA: Low-Rank Adaptation of Large
 * Language Models" (Hu et al., 2021). LoRA freezes the pre-trained model
 * weights and injects trainable rank decomposition matrices into each
 * layer, greatly reducing the number of trainable parameters.
 *
 * For a linear layer y = xW^T, LoRA modifies it to:
 *   y = xW^T + (alpha/rank) * x A^T B^T
 *
 * where A is [rank, in_features] and B is [out_features, rank].
 * B is initialized to zero so the initial LoRA output is zero,
 * meaning the model starts from the pre-trained weights.
 */

#ifndef CML_NN_LORA_H
#define CML_NN_LORA_H

#include <stdbool.h>
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A single LoRA-adapted linear layer
 *
 * Wraps a base weight matrix and adds low-rank A and B matrices.
 * The forward pass computes: x @ W^T + scaling * x @ A^T @ B^T
 */
typedef struct CMLLoRALinear {
    int in_features;
    int out_features;
    int rank;
    float alpha;
    float scaling;           /* alpha / rank */
    Tensor* base_weight;     /* [out_features, in_features] - not owned */
    Tensor* lora_A;          /* [rank, in_features] */
    Tensor* lora_B;          /* [out_features, rank] */
    Tensor* frozen_base;     /* Copy of base_weight before merge (for unmerge) */
    bool merged;
} CMLLoRALinear;

/**
 * @brief Container for multiple LoRA layers forming an adapter
 *
 * Groups LoRA layers together so they can be merged/unmerged as a unit.
 */
typedef struct CMLLoRAAdapter {
    char name[64];
    int rank;
    float alpha;
    float scaling;           /* alpha / rank */
    int num_layers;
    struct CMLLoRALinear** layers;
    bool merged;
} CMLLoRAAdapter;

/**
 * @brief Create a LoRA linear layer wrapping a base weight matrix
 *
 * Initializes lora_A with small random values (Xavier) and lora_B with zeros,
 * so the initial LoRA contribution is zero.
 *
 * @param base_weight Base weight tensor [out_features, in_features] (not owned, must outlive LoRA)
 * @param rank LoRA rank (typically 4, 8, or 16)
 * @param alpha LoRA scaling factor (typically equal to rank)
 * @return New LoRA linear layer, or NULL on failure
 */
CMLLoRALinear* cml_lora_linear_create(Tensor* base_weight, int rank, float alpha);

/**
 * @brief Free a LoRA linear layer
 *
 * Frees lora_A, lora_B, and frozen_base tensors. Does NOT free base_weight.
 *
 * @param lora LoRA linear layer to free
 */
void cml_lora_linear_free(CMLLoRALinear* lora);

/**
 * @brief Forward pass through LoRA linear layer
 *
 * Computes: output = input @ base_weight^T + scaling * input @ A^T @ B^T
 *
 * @param lora LoRA linear layer
 * @param input Input tensor [batch, in_features]
 * @return Output tensor [batch, out_features], or NULL on failure
 */
Tensor* cml_lora_linear_forward(CMLLoRALinear* lora, Tensor* input);

/**
 * @brief Merge LoRA weights into base weight
 *
 * Modifies base_weight in-place: W += scaling * B @ A
 * Saves a frozen copy of the original weights for later unmerge.
 *
 * @param lora LoRA linear layer
 * @return 0 on success, -1 on failure
 */
int cml_lora_linear_merge(CMLLoRALinear* lora);

/**
 * @brief Unmerge LoRA weights from base weight
 *
 * Restores base_weight from the frozen copy saved during merge.
 *
 * @param lora LoRA linear layer
 * @return 0 on success, -1 on failure
 */
int cml_lora_linear_unmerge(CMLLoRALinear* lora);

/**
 * @brief Create a LoRA adapter container
 *
 * @param name Adapter name (max 63 characters)
 * @param rank LoRA rank for this adapter
 * @param alpha LoRA alpha scaling factor
 * @return New adapter, or NULL on failure
 */
CMLLoRAAdapter* cml_lora_adapter_create(const char* name, int rank, float alpha);

/**
 * @brief Free a LoRA adapter and all its layers
 *
 * @param adapter Adapter to free
 */
void cml_lora_adapter_free(CMLLoRAAdapter* adapter);

/**
 * @brief Add a LoRA linear layer to an adapter
 *
 * The adapter takes ownership of the layer.
 *
 * @param adapter Target adapter
 * @param layer LoRA linear layer to add
 * @return 0 on success, -1 on failure
 */
int cml_lora_adapter_add_layer(CMLLoRAAdapter* adapter, CMLLoRALinear* layer);

/**
 * @brief Merge all LoRA layers in the adapter into their base weights
 *
 * @param adapter Adapter to merge
 * @return 0 on success, -1 on failure
 */
int cml_lora_adapter_merge_all(CMLLoRAAdapter* adapter);

/**
 * @brief Unmerge all LoRA layers in the adapter from their base weights
 *
 * @param adapter Adapter to unmerge
 * @return 0 on success, -1 on failure
 */
int cml_lora_adapter_unmerge_all(CMLLoRAAdapter* adapter);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_LORA_H */
