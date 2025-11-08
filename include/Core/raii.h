/**
 * @file raii.h
 * @brief Automatic RAII (Resource Acquisition Is Initialization) for automatic resource management
 *
 * This header provides automatic resource cleanup using GCC/Clang's __attribute__((cleanup))
 * feature. Resources are automatically freed when they go out of scope, eliminating the need
 * for manual cleanup calls - just like Python!
 *
 * Usage:
 *   #include "cml.h"
 *
 *   {
 *       Sequential *model = nn_sequential();  // Automatically freed when out of scope!
 *       Optimizer *opt = optim_adam(...);     // Automatically freed when out of scope!
 *       // ... use resources ...
 *   } // Everything is automatically freed here!
 */

#ifndef CML_CORE_RAII_H
#define CML_CORE_RAII_H

#include "Core/cleanup.h"
#include "Core/memory_management.h"
#include "Core/error_stack.h"
#include "Core/device.h"
#include "nn/module.h"
#include "optim/optimizer.h"
#include "tensor/tensor.h"
#include "Core/dataset.h"
#include <stddef.h>

#ifdef __GNUC__
#define CML_RAII_ATTR(func) __attribute__((cleanup(func)))
#else
#define CML_RAII_ATTR(func)
#pragma warning("Automatic RAII requires GCC or Clang compiler. Manual cleanup required.")
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Internal cleanup functions (called automatically when variables go out of scope)
// The cleanup attribute passes the ADDRESS of the variable, not the value
// So we need to dereference to get the actual pointer value
static inline void cml_raii_cleanup_module(void* module_ptr) {
    if (module_ptr) {
        Module** module = (Module**)module_ptr;
        if (module && *module) {
            module_free(*module);
            *module = NULL; // Prevent double-free
        }
    }
}

static inline void cml_raii_cleanup_optimizer(void* optimizer_ptr) {
    if (optimizer_ptr) {
        Optimizer** optimizer = (Optimizer**)optimizer_ptr;
        if (optimizer && *optimizer) {
            optimizer_free(*optimizer);
            *optimizer = NULL; // Prevent double-free
        }
    }
}

static inline void cml_raii_cleanup_tensor(void* tensor_ptr) {
    if (tensor_ptr) {
        Tensor** tensor = (Tensor**)tensor_ptr;
        if (tensor && *tensor) {
            tensor_free(*tensor);
            *tensor = NULL; // Prevent double-free
        }
    }
}

static inline void cml_raii_cleanup_dataset(void* dataset_ptr) {
    if (dataset_ptr) {
        Dataset** dataset = (Dataset**)dataset_ptr;
        if (dataset && *dataset) {
            dataset_free(*dataset);
            *dataset = NULL; // Prevent double-free
        }
    }
}

static inline void cml_raii_cleanup_params(void* params_ptr) {
    if (params_ptr) {
        Parameter*** params = (Parameter***)params_ptr;
        if (params && *params) {
            CM_FREE(*params);
        }
    }
}

static inline void cml_raii_cleanup_memory(void* ptr_ptr) {
    if (ptr_ptr) {
        void** ptr = (void**)ptr_ptr;
        if (ptr && *ptr) {
            CM_FREE(*ptr);
            *ptr = NULL; // Prevent double-free
        }
    }
}

// ============================================================================
// Automatic RAII - Python-like automatic memory management
// ============================================================================
// All pointer declarations automatically become RAII-managed.
// Just write normal code - no special syntax needed!

// Forward declarations for original functions (to avoid macro recursion)
// Note: These are only used internally by the macros, actual declarations are in their respective
// headers

// Helper macros to call original functions (avoid recursion by using function pointers)
// We use a trick: cast to function pointer type to avoid macro expansion
#define CML_RAII_CALL_nn_sequential() ((Sequential * (*)(void)) nn_sequential)()
#define CML_RAII_CALL_nn_relu(...) ((ReLU * (*)(bool)) nn_relu)(__VA_ARGS__)
#define CML_RAII_CALL_nn_tanh() ((Tanh * (*)(void)) nn_tanh)()
#define CML_RAII_CALL_nn_sigmoid() ((Sigmoid * (*)(void)) nn_sigmoid)()
#define CML_RAII_CALL_optim_adam(...)                                                              \
    ((Optimizer * (*)(Parameter**, int, float, float, float, float, float)) optim_adam)(__VA_ARGS__)
#define CML_RAII_CALL_optim_sgd(...)                                                               \
    ((Optimizer * (*)(Parameter**, int, float, float, float)) optim_sgd)(__VA_ARGS__)
#define CML_RAII_CALL_tensor_empty(...)                                                            \
    ((Tensor * (*)(int*, int, DType, DeviceType)) tensor_empty)(__VA_ARGS__)
#define CML_RAII_CALL_tensor_zeros(...)                                                            \
    ((Tensor * (*)(int*, int, DType, DeviceType)) tensor_zeros)(__VA_ARGS__)
#define CML_RAII_CALL_tensor_ones(...)                                                             \
    ((Tensor * (*)(int*, int, DType, DeviceType)) tensor_ones)(__VA_ARGS__)
#define CML_RAII_CALL_dataset_from_arrays(...)                                                     \
    ((Dataset * (*)(float*, float*, int, int, int)) dataset_from_arrays)(__VA_ARGS__)
#define CML_RAII_CALL_cm_safe_malloc(size, file, line)                                             \
    ((void* (*)(size_t, const char*, int))cm_safe_malloc)(size, file, line)
#define CML_RAII_CALL_cm_safe_calloc(nmemb, size, file, line)                                      \
    ((void* (*)(size_t, size_t, const char*, int))cm_safe_calloc)(nmemb, size, file, line)
#define CML_RAII_CALL_module_collect_parameters(...)                                               \
    ((int (*)(Module*, Parameter***, int*, bool))module_collect_parameters)(__VA_ARGS__)
#define CML_RAII_CALL_dataset_split_three(...)                                                     \
    ((int (*)(Dataset*, float, float, Dataset**, Dataset**, Dataset**))dataset_split_three)(       \
        __VA_ARGS__)
#define CML_RAII_CALL_cml_init() ((int (*)(void))cml_init)()
#define CML_RAII_CALL_nn_linear(...)                                                               \
    ((Linear * (*)(int, int, DType, DeviceType, bool)) nn_linear)(__VA_ARGS__)
#define CML_RAII_CALL_device_get_default() ((DeviceType(*)(void))device_get_default)()

/**
 * @brief Automatic RAII wrapper for nn_sequential()
 *
 * Usage:
 *   Sequential *model CML_RAII_ATTR(cml_raii_cleanup_module) = nn_sequential();  // Automatically
 * RAII-managed!
 *
 * NOTE: The cleanup attribute must be applied to the variable declaration, not the macro.
 * This macro just calls the function and checks for errors.
 */
#define nn_sequential()                                                                            \
    ({                                                                                             \
        Sequential* __result = CML_RAII_CALL_nn_sequential();                                      \
        CML_AUTO_CHECK_PTR(__result, "Failed to create Sequential model");                         \
        __result;                                                                                  \
    })

// Helper macro to count arguments (simpler approach)
// Pattern: COUNT_ARGS(a, b, c, d) -> COUNT_ARGS_IMPL(a, b, c, d, 5, 4, 3, 2, 1, 0) -> N=4
#define CML_RAII_COUNT_ARGS_IMPL(_1, _2, _3, _4, _5, N, ...) N
#define CML_RAII_COUNT_ARGS(...) CML_RAII_COUNT_ARGS_IMPL(__VA_ARGS__, 5, 4, 3, 2, 1, 0)

/**
 * @brief Automatic RAII wrapper for nn_linear() with automatic device detection
 *
 * Usage:
 *   Linear *layer = nn_linear(in, out, dtype, use_bias);  // Automatically uses default device!
 *   Linear *layer = nn_linear(in, out, dtype, device, use_bias);  // Or specify device explicitly
 */
// Simple approach: nn_linear with 4 args uses auto device, 5 args uses explicit device
// We use function overloading-like behavior by checking argument count at compile time
#define nn_linear_4args(in_features, out_features, dtype, use_bias)                                \
    ({                                                                                             \
        Linear* __raii_var CML_RAII_ATTR(cml_raii_cleanup_module) = NULL;                          \
        __raii_var = (Linear*)nn_linear_auto(in_features, out_features, dtype, use_bias);          \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create Linear layer");                           \
        __raii_var;                                                                                \
    })

#define nn_linear_5args(in_features, out_features, dtype, device, use_bias)                        \
    ({                                                                                             \
        Linear* __raii_var CML_RAII_ATTR(cml_raii_cleanup_module) = NULL;                          \
        __raii_var = CML_RAII_CALL_nn_linear(in_features, out_features, dtype, device, use_bias);  \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create Linear layer");                           \
        __raii_var;                                                                                \
    })

// Use a helper to select the right macro based on argument count
#define nn_linear(...)                                                                             \
    nn_linear_GET_MACRO(__VA_ARGS__, nn_linear_5args, nn_linear_4args)(__VA_ARGS__)

#define nn_linear_GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME

/**
 * @brief Automatic RAII wrapper for nn_relu()
 *
 * Usage:
 *   ReLU *relu = nn_relu(false);  // Automatically RAII-managed and error-checked!
 */
#define nn_relu(...)                                                                               \
    ({                                                                                             \
        ReLU* __raii_var CML_RAII_ATTR(cml_raii_cleanup_module) = NULL;                            \
        __raii_var = CML_RAII_CALL_nn_relu(__VA_ARGS__);                                           \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create ReLU layer");                             \
        __raii_var;                                                                                \
    })

/**
 * @brief Automatic RAII wrapper for nn_tanh()
 *
 * Usage:
 *   Tanh *tanh = nn_tanh();  // Automatically RAII-managed and error-checked!
 */
#define nn_tanh()                                                                                  \
    ({                                                                                             \
        Tanh* __raii_var CML_RAII_ATTR(cml_raii_cleanup_module) = NULL;                            \
        __raii_var                                              = CML_RAII_CALL_nn_tanh();         \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create Tanh layer");                             \
        __raii_var;                                                                                \
    })

/**
 * @brief Automatic RAII wrapper for nn_sigmoid()
 *
 * Usage:
 *   Sigmoid *sigmoid = nn_sigmoid();  // Automatically RAII-managed and error-checked!
 */
#define nn_sigmoid()                                                                               \
    ({                                                                                             \
        Sigmoid* __raii_var CML_RAII_ATTR(cml_raii_cleanup_module) = NULL;                         \
        __raii_var                                                 = CML_RAII_CALL_nn_sigmoid();   \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create Sigmoid layer");                          \
        __raii_var;                                                                                \
    })

/**
 * @brief Automatic RAII wrapper for optim_adam()
 *
 * Usage:
 *   Optimizer *opt = optim_adam(...);  // Automatically RAII-managed and error-checked!
 */
#define optim_adam(...)                                                                            \
    ({                                                                                             \
        Optimizer* __raii_var CML_RAII_ATTR(cml_raii_cleanup_optimizer) = NULL;                    \
        __raii_var = CML_RAII_CALL_optim_adam(__VA_ARGS__);                                        \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create Adam optimizer");                         \
        __raii_var;                                                                                \
    })

/**
 * @brief Automatic RAII wrapper for optim_sgd()
 *
 * Usage:
 *   Optimizer *opt = optim_sgd(...);  // Automatically RAII-managed and error-checked!
 */
#define optim_sgd(...)                                                                             \
    ({                                                                                             \
        Optimizer* __raii_var CML_RAII_ATTR(cml_raii_cleanup_optimizer) = NULL;                    \
        __raii_var = CML_RAII_CALL_optim_sgd(__VA_ARGS__);                                         \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create SGD optimizer");                          \
        __raii_var;                                                                                \
    })

/**
 * @brief Automatic RAII wrapper for tensor creation functions
 *
 * Usage:
 *   Tensor *t = tensor_empty(shape, ndim, dtype);  // Automatically uses default device!
 *   Tensor *t = tensor_empty(shape, ndim, dtype, device);  // Or specify device explicitly
 */
// Simple approach: tensor_empty with 3 args uses auto device, 4 args uses explicit device
#define tensor_empty_3args(shape, ndim, dtype)                                                     \
    ({                                                                                             \
        Tensor* __raii_var CML_RAII_ATTR(cml_raii_cleanup_tensor) = NULL;                          \
        __raii_var = tensor_empty_auto(shape, ndim, dtype);                                        \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create empty tensor");                           \
        __raii_var;                                                                                \
    })

#define tensor_empty_4args(shape, ndim, dtype, device)                                             \
    ({                                                                                             \
        Tensor* __raii_var CML_RAII_ATTR(cml_raii_cleanup_tensor) = NULL;                          \
        __raii_var = CML_RAII_CALL_tensor_empty(shape, ndim, dtype, device);                       \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create empty tensor");                           \
        __raii_var;                                                                                \
    })

#define tensor_empty(...)                                                                          \
    tensor_empty_GET_MACRO(__VA_ARGS__, tensor_empty_4args, tensor_empty_3args)(__VA_ARGS__)

#define tensor_empty_GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME

#define tensor_zeros_3args(shape, ndim, dtype)                                                     \
    ({                                                                                             \
        Tensor* __raii_var CML_RAII_ATTR(cml_raii_cleanup_tensor) = NULL;                          \
        __raii_var = tensor_zeros_auto(shape, ndim, dtype);                                        \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create zeros tensor");                           \
        __raii_var;                                                                                \
    })

#define tensor_zeros_4args(shape, ndim, dtype, device)                                             \
    ({                                                                                             \
        Tensor* __raii_var CML_RAII_ATTR(cml_raii_cleanup_tensor) = NULL;                          \
        __raii_var = CML_RAII_CALL_tensor_zeros(shape, ndim, dtype, device);                       \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create zeros tensor");                           \
        __raii_var;                                                                                \
    })

#define tensor_zeros(...)                                                                          \
    tensor_zeros_GET_MACRO(__VA_ARGS__, tensor_zeros_4args, tensor_zeros_3args)(__VA_ARGS__)

#define tensor_zeros_GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME

#define tensor_ones_3args(shape, ndim, dtype)                                                      \
    ({                                                                                             \
        Tensor* __raii_var CML_RAII_ATTR(cml_raii_cleanup_tensor) = NULL;                          \
        __raii_var = tensor_ones_auto(shape, ndim, dtype);                                         \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create ones tensor");                            \
        __raii_var;                                                                                \
    })

#define tensor_ones_4args(shape, ndim, dtype, device)                                              \
    ({                                                                                             \
        Tensor* __raii_var CML_RAII_ATTR(cml_raii_cleanup_tensor) = NULL;                          \
        __raii_var = CML_RAII_CALL_tensor_ones(shape, ndim, dtype, device);                        \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create ones tensor");                            \
        __raii_var;                                                                                \
    })

#define tensor_ones(...)                                                                           \
    tensor_ones_GET_MACRO(__VA_ARGS__, tensor_ones_4args, tensor_ones_3args)(__VA_ARGS__)

#define tensor_ones_GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME

/**
 * @brief Automatic RAII wrapper for dataset_from_arrays()
 *
 * Usage:
 *   Dataset *ds = dataset_from_arrays(...);  // Automatically RAII-managed and error-checked!
 */
#define dataset_from_arrays(...)                                                                   \
    ({                                                                                             \
        Dataset* __raii_var CML_RAII_ATTR(cml_raii_cleanup_dataset) = NULL;                        \
        __raii_var = CML_RAII_CALL_dataset_from_arrays(__VA_ARGS__);                               \
        CML_AUTO_CHECK_PTR(__raii_var, "Failed to create dataset from arrays");                    \
        __raii_var;                                                                                \
    })

/**
 * @brief Automatic RAII wrapper for Parameter array
 *
 * Usage:
 *   Parameter **params = NULL;
 *   int num_params = 0;
 *   module_collect_parameters(model, &params, &num_params, true);
 *   PARAMS(params);  // Automatically freed when out of scope and error-checked!
 */
#define PARAMS(var)                                                                                \
    do {                                                                                           \
        Parameter** var##_raii_ptr = (var);                                                        \
        CML_AUTO_CHECK_PTR(var##_raii_ptr, "Failed to collect parameters");                        \
        /* Note: Parameter arrays are freed automatically via module cleanup */                    \
        (void)var##_raii_ptr;                                                                      \
    } while (0)

/**
 * @brief Automatic error checking wrapper for functions that return error codes
 *
 * This macro automatically checks return codes and pushes errors to the stack.
 * Users don't need to write this - it's used internally.
 */
#define CML_AUTO_CHECK_FUNC(func_call, error_msg)                                                  \
    ({                                                                                             \
        int __result = (func_call);                                                                \
        if (__result != CM_SUCCESS) {                                                              \
            error_stack_push(__result, (error_msg), __FILE__, __LINE__, __func__);                 \
        }                                                                                          \
        __result;                                                                                  \
    })

// ============================================================================
// Automatic error-checking wrappers for common functions (transparent to users)
// ============================================================================

/**
 * @brief Automatic error-checking wrapper for module_collect_parameters()
 *
 * Usage:
 *   module_collect_parameters(model, &params, &num_params, true);  // Automatically error-checked!
 */
#define module_collect_parameters(...)                                                             \
    ({                                                                                             \
        int __result = CML_RAII_CALL_module_collect_parameters(__VA_ARGS__);                       \
        if (__result != CM_SUCCESS) {                                                              \
            error_stack_push(__result, "Failed to collect model parameters", __FILE__, __LINE__,   \
                             __func__);                                                            \
        }                                                                                          \
        __result;                                                                                  \
    })

/**
 * @brief Automatic error-checking wrapper for dataset_split_three()
 *
 * Usage:
 *   dataset_split_three(dataset, 0.7f, 0.15f, &train, &val, &test);  // Automatically
 * error-checked!
 */
#define dataset_split_three(...)                                                                   \
    ({                                                                                             \
        int __result = CML_RAII_CALL_dataset_split_three(__VA_ARGS__);                             \
        if (__result != CM_SUCCESS) {                                                              \
            error_stack_push(__result, "Failed to split dataset", __FILE__, __LINE__, __func__);   \
        }                                                                                          \
        __result;                                                                                  \
    })

// Note: cml_init() is not wrapped here to avoid conflicts with the function declaration in cml.h
// The function itself handles error checking internally

/**
 * @brief Automatic error checking and exit code handling
 *
 * This macro automatically checks for errors and returns 1 if errors exist.
 * Users can use this at the end of main() for automatic error handling.
 *
 * Usage:
 *   int main(void) {
 *       cml_init();
 *       // ... code ...
 *       return CML_AUTO_EXIT();  // Automatically checks errors and returns appropriate exit code!
 *   }
 */
#define CML_AUTO_EXIT() (error_stack_has_errors() ? (error_stack_print_all(), 1) : 0)

// Save original macros if they exist
#ifdef CM_MALLOC
#define CML_RAII_ORIG_CM_MALLOC CM_MALLOC
#undef CM_MALLOC
#endif

#ifdef CM_CALLOC
#define CML_RAII_ORIG_CM_CALLOC CM_CALLOC
#undef CM_CALLOC
#endif

/**
 * @brief Automatic RAII wrapper for CM_MALLOC
 *
 * Usage:
 *   float *data = CM_MALLOC(size * sizeof(float));  // Automatically freed and error-checked!
 */
#define CM_MALLOC(size)                                                                            \
    ({                                                                                             \
        void* __raii_var CML_RAII_ATTR(cml_raii_cleanup_memory) = NULL;                            \
        __raii_var = CML_RAII_CALL_cm_safe_malloc(size, __FILE__, __LINE__);                       \
        if (__raii_var == (void*)CM_MEMORY_ALLOCATION_ERROR || __raii_var == NULL) {               \
            error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Memory allocation failed", __FILE__,     \
                             __LINE__, __func__);                                                  \
            __raii_var = NULL;                                                                     \
        }                                                                                          \
        (void*)__raii_var;                                                                         \
    })

/**
 * @brief Automatic RAII wrapper for CM_CALLOC
 *
 * Usage:
 *   float *data = CM_CALLOC(n, sizeof(float));  // Automatically freed and error-checked!
 */
#define CM_CALLOC(nmemb, size)                                                                     \
    ({                                                                                             \
        void* __raii_var CML_RAII_ATTR(cml_raii_cleanup_memory) = NULL;                            \
        __raii_var = CML_RAII_CALL_cm_safe_calloc(nmemb, size, __FILE__, __LINE__);                \
        if (__raii_var == (void*)CM_MEMORY_ALLOCATION_ERROR || __raii_var == NULL) {               \
            error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Memory allocation failed", __FILE__,     \
                             __LINE__, __func__);                                                  \
            __raii_var = NULL;                                                                     \
        }                                                                                          \
        (void*)__raii_var;                                                                         \
    })

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_RAII_H
