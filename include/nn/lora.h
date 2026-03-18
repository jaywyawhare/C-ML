/*
 * LoRA: y = xW^T + (alpha/rank) * x A^T B^T
 * B is initialized to zero so the model starts from pre-trained weights.
 */

#ifndef CML_NN_LORA_H
#define CML_NN_LORA_H

#include <stdbool.h>
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

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

typedef struct CMLLoRAAdapter {
    char name[64];
    int rank;
    float alpha;
    float scaling;           /* alpha / rank */
    int num_layers;
    struct CMLLoRALinear** layers;
    bool merged;
} CMLLoRAAdapter;

/* base_weight is not owned and must outlive the LoRA layer.
 * lora_A initialized with Xavier, lora_B initialized to zeros. */
CMLLoRALinear* cml_lora_linear_create(Tensor* base_weight, int rank, float alpha);

/* Frees lora_A, lora_B, frozen_base. Does NOT free base_weight. */
void cml_lora_linear_free(CMLLoRALinear* lora);

Tensor* cml_lora_linear_forward(CMLLoRALinear* lora, Tensor* input);

/* Modifies base_weight in-place: W += scaling * B @ A.
 * Saves a frozen copy for later unmerge. */
int cml_lora_linear_merge(CMLLoRALinear* lora);

int cml_lora_linear_unmerge(CMLLoRALinear* lora);

CMLLoRAAdapter* cml_lora_adapter_create(const char* name, int rank, float alpha);

void cml_lora_adapter_free(CMLLoRAAdapter* adapter);

/* The adapter takes ownership of the layer. */
int cml_lora_adapter_add_layer(CMLLoRAAdapter* adapter, CMLLoRALinear* layer);

int cml_lora_adapter_merge_all(CMLLoRAAdapter* adapter);

int cml_lora_adapter_unmerge_all(CMLLoRAAdapter* adapter);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_LORA_H */
