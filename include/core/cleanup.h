#ifndef CML_CORE_CLEANUP_H
#define CML_CORE_CLEANUP_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Module Module;
typedef struct Optimizer Optimizer;
typedef struct Tensor Tensor;
typedef struct Dataset Dataset;
typedef struct Parameter Parameter;

typedef struct {
    Module* model;
    Parameter** params;
    Optimizer* optimizer;

    Tensor** tensors;
    size_t num_tensors;
    size_t tensor_capacity;

    Dataset** datasets;
    size_t num_datasets;
    size_t dataset_capacity;

    void** memory_ptrs;
    size_t num_memory_ptrs;
    size_t memory_capacity;
} CleanupContext;

CleanupContext* cleanup_context_create(void);
void cleanup_context_free(CleanupContext* ctx);
int cleanup_register_model(CleanupContext* ctx, Module* model);
int cleanup_register_params(CleanupContext* ctx, Parameter** params);
int cleanup_register_optimizer(CleanupContext* ctx, Optimizer* optimizer);
int cleanup_register_tensor(CleanupContext* ctx, Tensor* tensor);
void cleanup_clear_all(CleanupContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_CLEANUP_H
