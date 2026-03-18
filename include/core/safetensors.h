#ifndef CML_CORE_SAFETENSORS_H
#define CML_CORE_SAFETENSORS_H

#include "tensor/tensor.h"
#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SafeTensorsContext SafeTensorsContext;

SafeTensorsContext* safetensors_open_read(const char* filepath);
SafeTensorsContext* safetensors_open_write(const char* filepath);
void safetensors_close(SafeTensorsContext* ctx);
int safetensors_get_num_tensors(SafeTensorsContext* ctx);
const char* safetensors_get_tensor_name(SafeTensorsContext* ctx, int index);
Tensor* safetensors_read_tensor(SafeTensorsContext* ctx, const char* name);
int safetensors_write_tensor(SafeTensorsContext* ctx, const char* name, Tensor* tensor);
int module_save_safetensors(Module* module, const char* filepath);
int module_load_safetensors(Module* module, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_SAFETENSORS_H
