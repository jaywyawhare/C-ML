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
    char* key;
    Tensor* value;
} StateDictEntry;

typedef struct StateDict {
    StateDictEntry* entries;
    int count;
    int capacity;
} StateDict;

StateDict* nn_state_dict_create(void);
void nn_state_dict_free(StateDict* sd);

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

/* Running average of a model's weights (EMA or SWA): owns cloned shadow tensors kept
 * separate from the live parameters, so training is unaffected until they are copied back. */
typedef struct CMLWeightAvg {
    StateDict* avg; /* owned clones of each parameter */
    float decay;    /* EMA decay in (0,1) */
    long count;     /* number of SWA snapshots averaged in */
} CMLWeightAvg;

/**
 * Create a weight-average tracker seeded with a clone of @p model's current weights.
 * @param model (const Module*)
 * @param decay (float)
 * @return (CMLWeightAvg*)
 */
CMLWeightAvg* cml_weight_avg_create(const Module* model, float decay);

/**
 * EMA update: avg = decay*avg + (1-decay)*model.
 * @param wa (CMLWeightAvg*)
 * @param model (const Module*)
 * @return (void)
 */
void cml_weight_avg_update_ema(CMLWeightAvg* wa, const Module* model);

/**
 * SWA update: fold @p model into the equal running average of snapshots.
 * @param wa (CMLWeightAvg*)
 * @param model (const Module*)
 * @return (void)
 */
void cml_weight_avg_update_swa(CMLWeightAvg* wa, const Module* model);

/**
 * Copy the averaged weights into @p model (e.g. before evaluation). Returns 0 on success.
 * @param wa (CMLWeightAvg*)
 * @param model (Module*)
 * @return (int)
 */
int cml_weight_avg_copy_to(CMLWeightAvg* wa, Module* model);

/**
 * Free a weight-average tracker and its cloned tensors.
 * @param wa (CMLWeightAvg*)
 * @return (void)
 */
void cml_weight_avg_free(CMLWeightAvg* wa);

#ifdef __cplusplus
}
#endif

#endif
