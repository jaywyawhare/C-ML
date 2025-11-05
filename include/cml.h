/**
 * @file cml.h
 * @brief C-ML: C Machine Learning Library - Main Header
 * @version 0.0.2
 * @author Arinjay
 * @date 2025
 *
 * This is the main header file for the C-ML library. Include this file
 * to access all the major components.
 *
 * @example
 * ```c
 * #include "cml.h"
 *
 * // Create model
 * Sequential *model = nn_sequential();
 * sequential_add(model, (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
 * sequential_add(model, (Module*)nn_relu(false));
 *
 * // Create optimizer
 * Parameter **params; int num_params;
 * module_collect_parameters((Module*)model, &params, &num_params, true);
 * Optimizer *optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
 *
 * // Training loop
 * for (int epoch = 0; epoch < epochs; epoch++) {
 *     optimizer_zero_grad(optimizer);
 *     Tensor *outputs = module_forward((Module*)model, X);
 *     Tensor *loss = tensor_mse_loss(outputs, y);
 *     tensor_backward(loss, NULL, false, false);
 *     optimizer_step(optimizer);
 * }
 * ```
 */

#ifndef CML_H
#define CML_H

// Version information
#define CML_VERSION_MAJOR 0
#define CML_VERSION_MINOR 0
#define CML_VERSION_PATCH 2
#define CML_VERSION_STRING "0.0.2"

// High-Level API
// All neural network components are available via nn/module.h and nn/layers.h
// All dataset functions are available via Core/dataset.h

// Core Components (automatically included)

// Core utilities
#include "Core/logging.h"
#include "Core/memory_management.h"
#include "Core/error_codes.h"
#include "Core/dataset.h"

// Tensor operations
#include "tensor/tensor.h"
#include "tensor/ops.h"

// Automatic differentiation
#include "autograd/autograd.h"
#include "autograd/loss_functions.h"

// Neural network components
#include "nn/module.h"
#include "nn/layers.h"

// Optimizers
#include "optim/optimizer.h"

// Global utility functions
void summary(struct Module* module);

// Library Management

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the C-ML library
 *
 * This function should be called before using any C-ML functionality.
 * It initializes internal systems, sets up logging, and prepares
 * the library for use.
 *
 * @return 0 on success, negative value on failure
 */
int cml_init(void);

/**
 * @brief Cleanup the C-ML library
 *
 * This function should be called when the library is no longer needed.
 * It cleans up internal resources and ensures proper shutdown.
 *
 * @return 0 on success, negative value on failure
 */
int cml_cleanup(void);

/**
 * @brief Get library version information
 *
 * @param major Pointer to store major version
 * @param minor Pointer to store minor version
 * @param patch Pointer to store patch version
 * @param version_string Pointer to store version string
 */
void cml_get_version(int* major, int* minor, int* patch, const char** version_string);

/**
 * @brief Get library build information
 *
 * @return String containing build information (compiler, flags, etc.)
 */
const char* cml_get_build_info(void);

/**
 * @brief Check if C-ML library is initialized
 *
 * @return true if initialized, false otherwise
 */
bool cml_is_initialized(void);

/**
 * @brief Get C-ML library initialization count
 *
 * @return Current initialization reference count
 */
int cml_get_init_count(void);

/**
 * @brief Force cleanup of C-ML library (ignores reference count)
 *
 * This function should be used with caution as it forces cleanup
 * regardless of the reference count.
 *
 * @return 0 on success, negative value on failure
 */
int cml_force_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // CML_H
