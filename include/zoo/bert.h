#ifndef CML_ZOO_BERT_H
#define CML_ZOO_BERT_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    int n_layer;
    int n_head;
    int hidden_size;
    int intermediate_size;
    int max_position;
} BERTConfig;

Module* cml_zoo_bert_create(BERTConfig* config, DType dtype, DeviceType device);

BERTConfig cml_zoo_bert_config_tiny(void);
BERTConfig cml_zoo_bert_config_mini(void);
BERTConfig cml_zoo_bert_config_small(void);
BERTConfig cml_zoo_bert_config_base(void);
BERTConfig cml_zoo_bert_config_large(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_BERT_H */
