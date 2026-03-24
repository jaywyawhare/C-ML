#ifndef CML_NN_STATE_H
#define CML_NN_STATE_H

#include "nn.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct StateDictEntry {
    char*   key;    
    Tensor* value;  
} StateDictEntry;

typedef struct StateDict {
    StateDictEntry* entries;
    int             count;
    int             capacity;
} StateDict;

StateDict* nn_state_dict_create(void);
void       nn_state_dict_free(StateDict* sd);

int nn_state_dict_set(StateDict* sd, const char* key, Tensor* value);

Tensor* nn_state_dict_get(const StateDict* sd, const char* key);

int nn_state_dict_remove(StateDict* sd, const char* key);

StateDict* nn_get_state_dict(const Module* module, const char* prefix);

int nn_load_state_dict(Module* module, const StateDict* sd, bool strict);

int nn_save(const StateDict* sd, const char* path);

StateDict* nn_load(const char* path);

size_t nn_state_dict_num_params(const StateDict* sd);

size_t nn_state_dict_bytes(const StateDict* sd);

void nn_state_dict_print(const StateDict* sd);

int nn_state_dict_lerp(StateDict* dst, const StateDict* src, float alpha);

#ifdef __cplusplus
}
#endif

#endif 
