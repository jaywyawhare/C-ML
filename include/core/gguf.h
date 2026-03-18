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
    GGUF_TENSOR_Q4_0 = 2,
    GGUF_TENSOR_Q4_1 = 3,
    GGUF_TENSOR_Q8_0 = 7,
    GGUF_TENSOR_Q4_K = 12,
    GGUF_TENSOR_Q5_K = 13,
    GGUF_TENSOR_Q6_K = 14,
    GGUF_TENSOR_I8   = 16,
    GGUF_TENSOR_I16  = 17,
    GGUF_TENSOR_I32  = 18,
} GGUFTensorType;

typedef struct GGUFContext GGUFContext;

GGUFContext* gguf_open_read(const char* filepath);
GGUFContext* gguf_open_write(const char* filepath);
void gguf_close(GGUFContext* ctx);
int gguf_get_num_tensors(GGUFContext* ctx);
const char* gguf_get_tensor_name(GGUFContext* ctx, int index);
Tensor* gguf_read_tensor(GGUFContext* ctx, const char* name);
int gguf_write_tensor(GGUFContext* ctx, const char* name, Tensor* tensor);
int module_save_gguf(Module* module, const char* filepath);
int module_load_gguf(Module* module, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_GGUF_H
