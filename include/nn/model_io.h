/**
 * @file model_io.h
 * @brief Model save and load API
 */

#ifndef CML_NN_MODEL_IO_H
#define CML_NN_MODEL_IO_H

#include "nn.h"
#include "optim.h"

#ifdef __cplusplus
extern "C" {
#endif

int model_save(Module* model, const char* filepath);
int model_load(Module* model, const char* filepath);

int model_save_checkpoint(Module* model, Optimizer* optimizer, int epoch, float loss,
                          const char* filepath);
int model_load_checkpoint(Module* model, Optimizer* optimizer, int* epoch, float* loss,
                          const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_MODEL_IO_H
