/**
 * @file safetensors.h
 * @brief SafeTensors serialization format
 *
 * Read/write tensors in SafeTensors format (JSON header + raw data).
 * Compatible with Hugging Face safetensors Python library.
 */

#ifndef CML_CORE_SAFETENSORS_H
#define CML_CORE_SAFETENSORS_H

#include "tensor/tensor.h"
#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief SafeTensors file context */
typedef struct SafeTensorsContext SafeTensorsContext;

/** @brief Open SafeTensors file for reading */
SafeTensorsContext* safetensors_open_read(const char* filepath);

/** @brief Create SafeTensors file for writing */
SafeTensorsContext* safetensors_open_write(const char* filepath);

/** @brief Close and free context */
void safetensors_close(SafeTensorsContext* ctx);

/** @brief Get number of tensors */
int safetensors_get_num_tensors(SafeTensorsContext* ctx);

/** @brief Get tensor name by index */
const char* safetensors_get_tensor_name(SafeTensorsContext* ctx, int index);

/** @brief Read tensor by name */
Tensor* safetensors_read_tensor(SafeTensorsContext* ctx, const char* name);

/** @brief Write tensor with name */
int safetensors_write_tensor(SafeTensorsContext* ctx, const char* name, Tensor* tensor);

/** @brief Save module parameters to SafeTensors format */
int module_save_safetensors(Module* module, const char* filepath);

/** @brief Load module parameters from SafeTensors format */
int module_load_safetensors(Module* module, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_SAFETENSORS_H
