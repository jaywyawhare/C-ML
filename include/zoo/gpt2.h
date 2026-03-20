#ifndef CML_ZOO_GPT2_H
#define CML_ZOO_GPT2_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    int n_layer;
    int n_head;
    int n_embd;
    int block_size;
} GPT2Config;

Module* cml_zoo_gpt2_create(GPT2Config* config, DType dtype, DeviceType device);

GPT2Config cml_zoo_gpt2_config_small(void);
GPT2Config cml_zoo_gpt2_config_medium(void);
GPT2Config cml_zoo_gpt2_config_large(void);
GPT2Config cml_zoo_gpt2_config_xl(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_GPT2_H */
