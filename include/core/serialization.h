#ifndef CML_CORE_SERIALIZATION_H
#define CML_CORE_SERIALIZATION_H

#include "nn.h"
#include "tensor/tensor.h"
#include "optim.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int module_save(Module* module, const char* filepath);
int module_load(Module* module, const char* filepath);
int module_save_stream(Module* module, FILE* file);
int module_load_stream(Module* module, FILE* file);

int tensor_write_file(Tensor* tensor, const char* filepath);
Tensor* tensor_read_file(const char* filepath);
int tensor_write_stream(Tensor* tensor, FILE* file);
Tensor* tensor_read_stream(FILE* file);

int optimizer_save(Optimizer* optimizer, const char* filepath);
int optimizer_load(Optimizer* optimizer, const char* filepath);
int optimizer_save_stream(Optimizer* optimizer, FILE* file);
int optimizer_load_stream(Optimizer* optimizer, FILE* file);

typedef struct NamedParameter {
    char* name; // Non-const so we can allocate and free
    Parameter* parameter;
} NamedParameter;

int module_named_parameters(Module* module, NamedParameter** named_params, int* num_params);
void module_named_parameters_free(NamedParameter* named_params, int num_params);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_SERIALIZATION_H
