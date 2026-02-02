/**
 * @file metal_backend.h
 * @brief Apple Metal backend for GPU kernel execution
 */

#ifndef CML_MLIR_BACKENDS_METAL_BACKEND_H
#define CML_MLIR_BACKENDS_METAL_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct Tensor;
typedef struct Tensor Tensor;

/**
 * @brief Metal backend context (opaque - implementation in Objective-C)
 */
typedef struct CMLMetalBackend CMLMetalBackend;

/**
 * @brief Metal compiled kernel
 */
typedef struct CMLMetalKernel {
    void* pipeline; // MTLComputePipelineState*
    void* function; // MTLFunction*
    char* kernel_name;
    int grid_dim[3];
    int threads_per_group[3];
} CMLMetalKernel;

// Backend lifecycle
bool cml_metal_available(void);
CMLMetalBackend* cml_metal_backend_create(void);
int cml_metal_backend_init(CMLMetalBackend* backend);
void cml_metal_backend_free(CMLMetalBackend* backend);

// Kernel operations
CMLMetalKernel* cml_metal_compile_source(CMLMetalBackend* backend, const char* msl_code,
                                         const char* kernel_name);
void cml_metal_kernel_free(CMLMetalBackend* backend, CMLMetalKernel* kernel);
int cml_metal_launch_kernel(CMLMetalBackend* backend, CMLMetalKernel* kernel, void** args,
                            int num_args);
int cml_metal_synchronize(CMLMetalBackend* backend);

// Memory operations (Metal uses unified memory on Apple Silicon)
void* cml_metal_malloc(CMLMetalBackend* backend, size_t size);
void cml_metal_free(CMLMetalBackend* backend, void* ptr);

// Tensor operations
int cml_metal_upload_tensor(CMLMetalBackend* backend, Tensor* tensor);
int cml_metal_download_tensor(CMLMetalBackend* backend, Tensor* tensor);

#ifdef __cplusplus
}
#endif

#endif // CML_MLIR_BACKENDS_METAL_BACKEND_H
