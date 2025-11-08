/**
 * @file module.h
 * @brief Base Module class for neural network components
 *
 * This header defines the Module base class that all neural network
 * layers, loss functions, and containers inherit from. It provides
 * a unified interface for forward passes, parameter management,
 * and module lifecycle.
 */

#ifndef CML_NN_MODULE_H
#define CML_NN_MODULE_H

#include "tensor/tensor.h"
#include "Core/logging.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct Module;
struct Parameter;

// Module function signatures
typedef Tensor* (*ForwardFn)(struct Module* module, Tensor* input);
typedef void (*FreeFn)(struct Module* module);

/**
 * @brief Parameter structure for trainable weights
 */
typedef struct Parameter {
    Tensor* tensor;     // The parameter tensor
    bool requires_grad; // Whether to compute gradients
    const char* name;   // Parameter name for identification
} Parameter;

/**
 * @brief Base Module structure for neural network components
 *
 * All neural network layers, loss functions, and containers
 * inherit from this base structure. It provides:
 * - Forward pass functionality
 * - Parameter management
 * - Module lifecycle management
 * - Chainable module composition
 */
typedef struct Module {
    const char* name;  // Module name for identification
    ForwardFn forward; // Forward pass function
    FreeFn free;       // Cleanup function

    Parameter** parameters;  // Array of trainable parameters
    int num_parameters;      // Number of parameters
    int parameters_capacity; // Capacity of parameters array

    struct Module* next; // Next module in sequence (for containers)

    // Module state
    bool training;   // Training mode flag
    void* user_data; // User-defined data

    // Module metadata
    const char* version;     // Module version
    const char* description; // Module description
} Module;

// Module Creation and Management

/**
 * @brief Initialize a module with basic settings
 *
 * @param module Module to initialize
 * @param name Module name
 * @param forward Forward pass function
 * @param free Cleanup function
 * @return 0 on success, negative value on failure
 */
int module_init(Module* module, const char* name, ForwardFn forward, FreeFn free);

/**
 * @brief Create a new module
 *
 * @param name Module name
 * @param forward Forward pass function
 * @param free Cleanup function
 * @return New module, or NULL on failure
 */
Module* module_create(const char* name, ForwardFn forward, FreeFn free);

/**
 * @brief Free a module and all its resources
 *
 * @param module Module to free
 */
void module_free(Module* module);

// Parameter Management

/**
 * @brief Add a parameter to a module
 *
 * @param module Target module
 * @param tensor Parameter tensor
 * @param name Parameter name
 * @param requires_grad Whether parameter requires gradients
 * @return 0 on success, negative value on failure
 */
int module_add_parameter(Module* module, Tensor* tensor, const char* name, bool requires_grad);

/**
 * @brief Get module parameters
 *
 * @param module Target module
 * @param params Array to store parameters (can be NULL to just get count)
 * @param num_parameters Pointer to store parameter count
 * @return 0 on success, negative value on failure
 */
int module_get_parameters(Module* module, Parameter** params, int* num_parameters);

/**
 * @brief Get parameter by name
 *
 * @param module Target module
 * @param name Parameter name
 * @return Parameter if found, NULL otherwise
 */
Parameter* module_get_parameter(Module* module, const char* name);

/**
 * @brief Set parameter value
 *
 * @param module Target module
 * @param name Parameter name
 * @param tensor New parameter tensor
 * @return 0 on success, negative value on failure
 */
int module_set_parameter(Module* module, const char* name, Tensor* tensor);

// Module Operations

/**
 * @brief Forward pass through a module
 *
 * @param module Target module
 * @param input Input tensor
 * @return Output tensor, or NULL on failure
 */
Tensor* module_forward(Module* module, Tensor* input);

/**
 * @brief Set module training mode
 *
 * @param module Target module
 * @param training Training mode flag
 */
void module_set_training(Module* module, bool training);

/**
 * @brief Check if module is in training mode
 *
 * @param module Target module
 * @return true if in training mode, false otherwise
 */
bool module_is_training(Module* module);

/**
 * @brief Zero all parameter gradients
 *
 * @param module Target module
 */
void module_zero_grad(Module* module);

// Module Information

/**
 * @brief Get module name
 *
 * @param module Target module
 * @return Module name
 */
const char* module_get_name(Module* module);

/**
 * @brief Get module parameter count
 *
 * @param module Target module
 * @return Number of parameters
 */
int module_get_parameter_count(Module* module);

/**
 * @brief Print module summary
 *
 * @param module Target module
 * @param indent Indentation level
 */
void module_print_summary(Module* module, int indent);

/**
 * @brief Get module total parameter count (recursive)
 *
 * @param module Target module
 * @return Total number of parameters including submodules
 */
int module_get_total_parameters(Module* module);

/**
 * @brief Collect all parameters from a module
 *
 * This function collects all parameters from a module and optionally
 * from chained modules. It allocates a new array of Parameter pointers.
 *
 * @param module Target module
 * @param params_out Output array of parameters (caller must free)
 * @param num_params_out Output number of parameters
 * @param recursive Whether to collect from chained modules
 * @return 0 on success, negative value on failure
 */
int module_collect_parameters(Module* module, Parameter*** params_out, int* num_params_out,
                              bool recursive);

// Module Composition

/**
 * @brief Chain modules together
 *
 * @param first First module in chain
 * @param second Second module in chain
 * @return 0 on success, negative value on failure
 */
int module_chain(Module* first, Module* second);

/**
 * @brief Get next module in chain
 *
 * @param module Target module
 * @return Next module, or NULL if none
 */
Module* module_get_next(Module* module);

/**
 * @brief Set next module in chain
 *
 * @param module Target module
 * @param next Next module
 */
void module_set_next(Module* module, Module* next);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_MODULE_H
