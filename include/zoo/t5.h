#ifndef CML_ZOO_T5_H
#define CML_ZOO_T5_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    int n_layer;
    int n_head;
    int d_model;
    int d_ff;
    int max_position;
    int num_buckets;
} T5Config;

Module* cml_zoo_t5_create(T5Config* config, DType dtype, DeviceType device);

Tensor* t5_encode(Module* module, Tensor* input);
Tensor* t5_decode(Module* module, Tensor* tgt, Tensor* memory);

T5Config cml_zoo_t5_config_small(void);
T5Config cml_zoo_t5_config_base(void);
T5Config cml_zoo_t5_config_large(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_T5_H */
