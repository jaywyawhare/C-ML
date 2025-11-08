/**
 * @file cleanup.c
 * @brief Centralized cleanup helper implementation
 */

#include "Core/cleanup.h"
#include "cml.h"
#include "nn/module.h"
#include "optim/optimizer.h"
#include "tensor/tensor.h"
#include "Core/dataset.h"
#include "Core/memory_management.h"
#include "Core/logging.h"
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 16

CleanupContext* cleanup_context_create(void) {
    CleanupContext* ctx = CM_MALLOC(sizeof(CleanupContext));
    if (!ctx)
        return NULL;

    ctx->model     = NULL;
    ctx->params    = NULL;
    ctx->optimizer = NULL;

    ctx->tensors         = NULL;
    ctx->num_tensors     = 0;
    ctx->tensor_capacity = 0;

    ctx->datasets         = NULL;
    ctx->num_datasets     = 0;
    ctx->dataset_capacity = 0;

    ctx->memory_ptrs     = NULL;
    ctx->num_memory_ptrs = 0;
    ctx->memory_capacity = 0;

    // Automatically register with global cleanup system
    extern void cml_register_cleanup_context(CleanupContext*);
    cml_register_cleanup_context(ctx);

    return ctx;
}

static int ensure_tensor_capacity(CleanupContext* ctx, size_t needed) {
    if (ctx->num_tensors + needed <= ctx->tensor_capacity) {
        return 0;
    }

    size_t new_capacity = ctx->tensor_capacity == 0 ? INITIAL_CAPACITY : ctx->tensor_capacity * 2;
    while (new_capacity < ctx->num_tensors + needed) {
        new_capacity *= 2;
    }

    Tensor** new_tensors = CM_REALLOC(ctx->tensors, new_capacity * sizeof(Tensor*));
    if (!new_tensors)
        return -1;

    ctx->tensors         = new_tensors;
    ctx->tensor_capacity = new_capacity;
    return 0;
}

static int ensure_dataset_capacity(CleanupContext* ctx, size_t needed) {
    if (ctx->num_datasets + needed <= ctx->dataset_capacity) {
        return 0;
    }

    size_t new_capacity = ctx->dataset_capacity == 0 ? INITIAL_CAPACITY : ctx->dataset_capacity * 2;
    while (new_capacity < ctx->num_datasets + needed) {
        new_capacity *= 2;
    }

    Dataset** new_datasets = CM_REALLOC(ctx->datasets, new_capacity * sizeof(Dataset*));
    if (!new_datasets)
        return -1;

    ctx->datasets         = new_datasets;
    ctx->dataset_capacity = new_capacity;
    return 0;
}

static int ensure_memory_capacity(CleanupContext* ctx, size_t needed) {
    if (ctx->num_memory_ptrs + needed <= ctx->memory_capacity) {
        return 0;
    }

    size_t new_capacity = ctx->memory_capacity == 0 ? INITIAL_CAPACITY : ctx->memory_capacity * 2;
    while (new_capacity < ctx->num_memory_ptrs + needed) {
        new_capacity *= 2;
    }

    void** new_ptrs = CM_REALLOC(ctx->memory_ptrs, new_capacity * sizeof(void*));
    if (!new_ptrs)
        return -1;

    ctx->memory_ptrs     = new_ptrs;
    ctx->memory_capacity = new_capacity;
    return 0;
}

int cleanup_register_model(CleanupContext* ctx, Module* model) {
    if (!ctx || !model)
        return -1;
    ctx->model = model;
    return 0;
}

int cleanup_register_params(CleanupContext* ctx, Parameter** params) {
    if (!ctx || !params)
        return -1;
    ctx->params = params;
    return 0;
}

int cleanup_register_optimizer(CleanupContext* ctx, Optimizer* optimizer) {
    if (!ctx || !optimizer)
        return -1;
    ctx->optimizer = optimizer;
    return 0;
}

int cleanup_register_tensor(CleanupContext* ctx, Tensor* tensor) {
    if (!ctx || !tensor)
        return -1;

    if (ensure_tensor_capacity(ctx, 1) != 0) {
        return -1;
    }

    ctx->tensors[ctx->num_tensors++] = tensor;
    return 0;
}

int cleanup_register_dataset(CleanupContext* ctx, Dataset* dataset) {
    if (!ctx || !dataset)
        return -1;

    if (ensure_dataset_capacity(ctx, 1) != 0) {
        return -1;
    }

    ctx->datasets[ctx->num_datasets++] = dataset;
    return 0;
}

int cleanup_register_memory(CleanupContext* ctx, void* ptr) {
    if (!ctx || !ptr)
        return -1;

    if (ensure_memory_capacity(ctx, 1) != 0) {
        return -1;
    }

    ctx->memory_ptrs[ctx->num_memory_ptrs++] = ptr;
    return 0;
}

void cleanup_clear_all(CleanupContext* ctx) {
    if (!ctx)
        return;

    if (ctx->model) {
        module_free(ctx->model);
        ctx->model = NULL;
    }

    if (ctx->params) {
        CM_FREE(ctx->params);
        ctx->params = NULL;
    }

    if (ctx->optimizer) {
        optimizer_free(ctx->optimizer);
        ctx->optimizer = NULL;
    }

    // Free tensors
    for (size_t i = 0; i < ctx->num_tensors; i++) {
        if (ctx->tensors[i]) {
            tensor_free(ctx->tensors[i]);
        }
    }
    if (ctx->tensors) {
        CM_FREE(ctx->tensors);
        ctx->tensors = NULL;
    }
    ctx->num_tensors     = 0;
    ctx->tensor_capacity = 0;

    // Free datasets
    for (size_t i = 0; i < ctx->num_datasets; i++) {
        if (ctx->datasets[i]) {
            dataset_free(ctx->datasets[i]);
        }
    }
    if (ctx->datasets) {
        CM_FREE(ctx->datasets);
        ctx->datasets = NULL;
    }
    ctx->num_datasets     = 0;
    ctx->dataset_capacity = 0;

    // Free memory pointers
    for (size_t i = 0; i < ctx->num_memory_ptrs; i++) {
        if (ctx->memory_ptrs[i]) {
            CM_FREE(ctx->memory_ptrs[i]);
        }
    }
    if (ctx->memory_ptrs) {
        CM_FREE(ctx->memory_ptrs);
        ctx->memory_ptrs = NULL;
    }
    ctx->num_memory_ptrs = 0;
    ctx->memory_capacity = 0;
}

void cleanup_context_free(CleanupContext* ctx) {
    if (!ctx)
        return;

    cleanup_clear_all(ctx);
    CM_FREE(ctx);
}
