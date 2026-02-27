/**
 * @file cleanup.c
 * @brief Centralized cleanup helper implementation
 */

#include "core/cleanup.h"
#include "cml.h"
#include "nn.h"
#include "optim.h"
#include "tensor/tensor.h"
#include "core/dataset.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 16

CleanupContext* cleanup_context_create(void) {
    CleanupContext* ctx = malloc(sizeof(CleanupContext));
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

    Tensor** new_tensors = realloc(ctx->tensors, (size_t)new_capacity * sizeof(Tensor*));
    if (!new_tensors)
        return -1;

    ctx->tensors         = new_tensors;
    ctx->tensor_capacity = new_capacity;
    return 0;
}

/* ensure_dataset_capacity and ensure_memory_capacity will be added
   when cleanup_register_dataset/cleanup_register_memory are implemented. */

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

void cleanup_clear_all(CleanupContext* ctx) {
    if (!ctx)
        return;

    if (ctx->model) {
        module_free(ctx->model);
        ctx->model = NULL;
    }

    if (ctx->params) {
        free(ctx->params);
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
        free(ctx->tensors);
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
        free(ctx->datasets);
        ctx->datasets = NULL;
    }
    ctx->num_datasets     = 0;
    ctx->dataset_capacity = 0;

    // Free memory pointers
    for (size_t i = 0; i < ctx->num_memory_ptrs; i++) {
        if (ctx->memory_ptrs[i]) {
            free(ctx->memory_ptrs[i]);
        }
    }
    if (ctx->memory_ptrs) {
        free(ctx->memory_ptrs);
        ctx->memory_ptrs = NULL;
    }
    ctx->num_memory_ptrs = 0;
    ctx->memory_capacity = 0;
}

void cleanup_context_free(CleanupContext* ctx) {
    if (!ctx)
        return;

    cleanup_clear_all(ctx);
    free(ctx);
}
