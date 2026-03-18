#ifndef CML_PTH_LOADER_H
#define CML_PTH_LOADER_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

struct Module; /* forward declaration */

#ifdef __cplusplus
extern "C" {
#endif

#define CML_PTH_MAX_KEY_LEN 256

typedef struct CMLPthEntry {
    char key[CML_PTH_MAX_KEY_LEN];
    Tensor* tensor;
    DType original_dtype;
    size_t storage_offset;
    size_t num_elements;
} CMLPthEntry;

typedef struct CMLPthStateDict {
    CMLPthEntry* entries;
    int num_entries;
    int entry_capacity;

    /* Metadata */
    char* model_name;          /* If available from pickle data */
    int pytorch_version;       /* Detected PyTorch version */
    bool is_half_precision;    /* Model uses fp16 */
} CMLPthStateDict;

CMLPthStateDict* cml_pth_load(const char* path);
void cml_pth_free(CMLPthStateDict* sd);
Tensor* cml_pth_get_tensor(const CMLPthStateDict* sd, const char* key);
int cml_pth_num_entries(const CMLPthStateDict* sd);
const char* cml_pth_get_key(const CMLPthStateDict* sd, int index);
bool cml_pth_has_key(const CMLPthStateDict* sd, const char* key);
const char** cml_pth_list_keys(const CMLPthStateDict* sd, int* count);
int cml_pth_load_into_module(const CMLPthStateDict* sd, struct Module* module);
void cml_pth_print(const CMLPthStateDict* sd);
size_t cml_pth_total_params(const CMLPthStateDict* sd);
size_t cml_pth_total_bytes(const CMLPthStateDict* sd);

#ifdef __cplusplus
}
#endif

#endif /* CML_PTH_LOADER_H */
