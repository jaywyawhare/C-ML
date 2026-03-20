#ifndef CML_OPS_IR_INTERN_H
#define CML_OPS_IR_INTERN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct IRNode;

typedef struct CMLInternEntry {
    struct IRNode* node;
    uint64_t hash;
} CMLInternEntry;

typedef struct CMLInternTable {
    CMLInternEntry* entries;
    size_t capacity;
    size_t count;
} CMLInternTable;

CMLInternTable* cml_intern_table_create(void);
void cml_intern_table_free(CMLInternTable* table);

struct Tensor;

uint64_t cml_intern_hash_node(int op_type, int dtype, struct IRNode** inputs,
                              int num_inputs, const void* arg_bytes, size_t arg_len);

struct IRNode* cml_intern_lookup(CMLInternTable* table, uint64_t hash, int op_type,
                                 int dtype, struct IRNode** inputs, int num_inputs,
                                 const void* arg_bytes, size_t arg_len);

/* Variants that also distinguish leaf tensors by pointer identity */
uint64_t cml_intern_hash_node_ex(int op_type, int dtype, struct IRNode** inputs,
                                 struct Tensor** raw_inputs, int num_inputs,
                                 const void* arg_bytes, size_t arg_len);

struct IRNode* cml_intern_lookup_ex(CMLInternTable* table, uint64_t hash, int op_type,
                                    int dtype, struct IRNode** inputs,
                                    struct Tensor** raw_inputs, int num_inputs,
                                    const void* arg_bytes, size_t arg_len);

int cml_intern_insert(CMLInternTable* table, struct IRNode* node);

void cml_intern_remove(CMLInternTable* table, struct IRNode* node);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_INTERN_H
