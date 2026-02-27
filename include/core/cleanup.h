/**
 * @file cleanup.h
 * @brief Centralized cleanup helper for resource management
 */

#ifndef CML_CORE_CLEANUP_H
#define CML_CORE_CLEANUP_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct Module Module;
typedef struct Optimizer Optimizer;
typedef struct Tensor Tensor;
typedef struct Dataset Dataset;
typedef struct Parameter Parameter;

/**
 * @brief Cleanup context structure
 *
 * This structure holds pointers to resources that need cleanup.
 * All pointers are initialized to NULL and checked before freeing.
 */
typedef struct {
    // Model resources
    Module* model;
    Parameter** params;
    Optimizer* optimizer;

    // Tensor resources
    Tensor** tensors;
    size_t num_tensors;
    size_t tensor_capacity;

    // Dataset resources
    Dataset** datasets;
    size_t num_datasets;
    size_t dataset_capacity;

    // Raw memory pointers
    void** memory_ptrs;
    size_t num_memory_ptrs;
    size_t memory_capacity;
} CleanupContext;

/**
 * @brief Create a new cleanup context
 * @return New cleanup context, or NULL on failure
 */
CleanupContext* cleanup_context_create(void);

/**
 * @brief Free cleanup context and all registered resources
 * @param ctx Cleanup context to free
 */
void cleanup_context_free(CleanupContext* ctx);

/**
 * @brief Register a model for cleanup
 * @param ctx Cleanup context
 * @param model Model to register
 * @return 0 on success, negative value on failure
 */
int cleanup_register_model(CleanupContext* ctx, Module* model);

/**
 * @brief Register parameters for cleanup
 * @param ctx Cleanup context
 * @param params Parameters array to register
 * @return 0 on success, negative value on failure
 */
int cleanup_register_params(CleanupContext* ctx, Parameter** params);

/**
 * @brief Register an optimizer for cleanup
 * @param ctx Cleanup context
 * @param optimizer Optimizer to register
 * @return 0 on success, negative value on failure
 */
int cleanup_register_optimizer(CleanupContext* ctx, Optimizer* optimizer);

/**
 * @brief Register a tensor for cleanup
 * @param ctx Cleanup context
 * @param tensor Tensor to register
 * @return 0 on success, negative value on failure
 */
int cleanup_register_tensor(CleanupContext* ctx, Tensor* tensor);

/**
 * @brief Clear all registered resources (free them)
 * @param ctx Cleanup context
 */
void cleanup_clear_all(CleanupContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_CLEANUP_H
