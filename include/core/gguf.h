/**
 * @file gguf.h
 * @brief GGUF (GPT-Generated Unified Format) serialization
 *
 * Read/write tensors and metadata in GGUF format, compatible with llama.cpp.
 */

#ifndef CML_CORE_GGUF_H
#define CML_CORE_GGUF_H

#include "tensor/tensor.h"
#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGUF_MAGIC 0x46475547  // "GGUF" in little-endian

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GGUFMetadataType;

typedef enum {
    GGUF_TENSOR_F32  = 0,
    GGUF_TENSOR_F16  = 1,
    GGUF_TENSOR_I8   = 16,
    GGUF_TENSOR_I16  = 17,
    GGUF_TENSOR_I32  = 18,
} GGUFTensorType;

/** @brief GGUF file context for reading/writing */
typedef struct GGUFContext GGUFContext;

/** @brief Open a GGUF file for reading */
GGUFContext* gguf_open_read(const char* filepath);

/** @brief Create a new GGUF file for writing */
GGUFContext* gguf_open_write(const char* filepath);

/** @brief Close and free GGUF context */
void gguf_close(GGUFContext* ctx);

/** @brief Get number of tensors in GGUF file */
int gguf_get_num_tensors(GGUFContext* ctx);

/** @brief Get tensor name by index */
const char* gguf_get_tensor_name(GGUFContext* ctx, int index);

/** @brief Read a tensor from GGUF file by name */
Tensor* gguf_read_tensor(GGUFContext* ctx, const char* name);

/** @brief Write a tensor to GGUF file */
int gguf_write_tensor(GGUFContext* ctx, const char* name, Tensor* tensor);

/** @brief Save a module's parameters to GGUF format */
int module_save_gguf(Module* module, const char* filepath);

/** @brief Load a module's parameters from GGUF format */
int module_load_gguf(Module* module, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_GGUF_H
