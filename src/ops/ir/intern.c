
#include "ops/ir/intern.h"
#include "ops/ir/internal.h"
#include <stdlib.h>
#include <string.h>

#define INTERN_INITIAL_CAPACITY 64
#define INTERN_LOAD_FACTOR_NUM  3
#define INTERN_LOAD_FACTOR_DEN  4

#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME        0x100000001b3ULL

static uint64_t fnv1a_bytes(uint64_t h, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

static uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    return fnv1a_bytes(h, &v, sizeof(v));
}

static uint64_t fnv1a_i32(uint64_t h, int v) {
    return fnv1a_bytes(h, &v, sizeof(v));
}

uint64_t cml_intern_hash_node(int op_type, int dtype, struct IRNode** inputs,
                              int num_inputs, const void* arg_bytes, size_t arg_len) {
    uint64_t h = FNV_OFFSET_BASIS;
    h = fnv1a_i32(h, op_type);
    h = fnv1a_i32(h, dtype);
    h = fnv1a_i32(h, num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        uint64_t input_hash = inputs[i] ? inputs[i]->hash : 0;
        h = fnv1a_u64(h, input_hash);
    }
    if (arg_bytes && arg_len > 0)
        h = fnv1a_bytes(h, arg_bytes, arg_len);
    return h;
}

#include "tensor/tensor.h"

uint64_t cml_intern_hash_node_ex(int op_type, int dtype, struct IRNode** inputs,
                                 Tensor** raw_inputs, int num_inputs,
                                 const void* arg_bytes, size_t arg_len) {
    uint64_t h = FNV_OFFSET_BASIS;
    h = fnv1a_i32(h, op_type);
    h = fnv1a_i32(h, dtype);
    h = fnv1a_i32(h, num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i]) {
            h = fnv1a_u64(h, inputs[i]->hash);
        } else if (raw_inputs && raw_inputs[i]) {
            /* Leaf tensor: use its pointer address to distinguish different leaves */
            h = fnv1a_u64(h, (uint64_t)(uintptr_t)raw_inputs[i]);
        } else {
            h = fnv1a_u64(h, 0);
        }
    }
    if (arg_bytes && arg_len > 0)
        h = fnv1a_bytes(h, arg_bytes, arg_len);
    return h;
}

static int entries_match(struct IRNode* node, uint64_t hash, int op_type, int dtype,
                         struct IRNode** inputs, int num_inputs,
                         const void* arg_bytes, size_t arg_len) {
    if (node->hash != hash)
        return 0;
    if ((int)node->type != op_type)
        return 0;
    if (node->num_inputs != num_inputs)
        return 0;

    for (int i = 0; i < num_inputs; i++) {
        struct IRNode* a = (node->inputs && node->inputs[i]) ? node->inputs[i]->ir_node : NULL;
        struct IRNode* b = inputs ? inputs[i] : NULL;
        if (a != b)
            return 0;
    }

    (void)dtype;
    (void)arg_bytes;
    (void)arg_len;
    return 1;
}

static int entries_match_ex(struct IRNode* node, uint64_t hash, int op_type, int dtype,
                            struct IRNode** inputs, Tensor** raw_inputs,
                            int num_inputs,
                            const void* arg_bytes, size_t arg_len) {
    if (node->hash != hash)
        return 0;
    if ((int)node->type != op_type)
        return 0;
    if (node->num_inputs != num_inputs)
        return 0;

    for (int i = 0; i < num_inputs; i++) {
        struct IRNode* a = (node->inputs && node->inputs[i]) ? node->inputs[i]->ir_node : NULL;
        struct IRNode* b = inputs ? inputs[i] : NULL;
        if (a != b)
            return 0;
        /* For leaf tensors (ir_node == NULL), also compare tensor identity */
        if (!a && !b) {
            Tensor* ta = (node->inputs && node->inputs[i]) ? node->inputs[i] : NULL;
            Tensor* tb = (raw_inputs) ? raw_inputs[i] : NULL;
            if (ta != tb)
                return 0;
        }
    }

    (void)dtype;
    (void)arg_bytes;
    (void)arg_len;
    return 1;
}

CMLInternTable* cml_intern_table_create(void) {
    CMLInternTable* table = calloc(1, sizeof(CMLInternTable));
    if (!table)
        return NULL;

    table->capacity = INTERN_INITIAL_CAPACITY;
    table->entries  = calloc(table->capacity, sizeof(CMLInternEntry));
    if (!table->entries) {
        free(table);
        return NULL;
    }
    table->count = 0;
    return table;
}

void cml_intern_table_free(CMLInternTable* table) {
    if (!table)
        return;
    free(table->entries);
    free(table);
}

static size_t probe_index(uint64_t hash, size_t capacity) {
    return (size_t)(hash & (uint64_t)(capacity - 1));
}

static int intern_resize(CMLInternTable* table) {
    size_t new_cap = table->capacity * 2;
    CMLInternEntry* new_entries = calloc(new_cap, sizeof(CMLInternEntry));
    if (!new_entries)
        return -1;

    for (size_t i = 0; i < table->capacity; i++) {
        if (!table->entries[i].node)
            continue;
        size_t idx = probe_index(table->entries[i].hash, new_cap);
        while (new_entries[idx].node)
            idx = (idx + 1) & (new_cap - 1);
        new_entries[idx] = table->entries[i];
    }

    free(table->entries);
    table->entries  = new_entries;
    table->capacity = new_cap;
    return 0;
}

struct IRNode* cml_intern_lookup(CMLInternTable* table, uint64_t hash, int op_type,
                                 int dtype, struct IRNode** inputs, int num_inputs,
                                 const void* arg_bytes, size_t arg_len) {
    if (!table || table->count == 0)
        return NULL;

    size_t idx = probe_index(hash, table->capacity);
    for (size_t probes = 0; probes < table->capacity; probes++) {
        CMLInternEntry* e = &table->entries[idx];
        if (!e->node)
            return NULL;
        if (e->hash == hash && entries_match(e->node, hash, op_type, dtype,
                                             inputs, num_inputs, arg_bytes, arg_len))
            return e->node;
        idx = (idx + 1) & (table->capacity - 1);
    }
    return NULL;
}

struct IRNode* cml_intern_lookup_ex(CMLInternTable* table, uint64_t hash, int op_type,
                                    int dtype, struct IRNode** inputs,
                                    Tensor** raw_inputs, int num_inputs,
                                    const void* arg_bytes, size_t arg_len) {
    if (!table || table->count == 0)
        return NULL;

    size_t idx = probe_index(hash, table->capacity);
    for (size_t probes = 0; probes < table->capacity; probes++) {
        CMLInternEntry* e = &table->entries[idx];
        if (!e->node)
            return NULL;
        if (e->hash == hash && entries_match_ex(e->node, hash, op_type, dtype,
                                                inputs, raw_inputs, num_inputs,
                                                arg_bytes, arg_len))
            return e->node;
        idx = (idx + 1) & (table->capacity - 1);
    }
    return NULL;
}

int cml_intern_insert(CMLInternTable* table, struct IRNode* node) {
    if (!table || !node)
        return -1;

    if (table->count * INTERN_LOAD_FACTOR_DEN >= table->capacity * INTERN_LOAD_FACTOR_NUM) {
        if (intern_resize(table) != 0)
            return -1;
    }

    uint64_t hash = node->hash;
    size_t idx = probe_index(hash, table->capacity);
    while (table->entries[idx].node)
        idx = (idx + 1) & (table->capacity - 1);

    table->entries[idx].node = node;
    table->entries[idx].hash = hash;
    table->count++;
    return 0;
}

void cml_intern_remove(CMLInternTable* table, struct IRNode* node) {
    if (!table || !node || table->count == 0)
        return;

    uint64_t hash = node->hash;
    size_t idx = probe_index(hash, table->capacity);

    for (size_t probes = 0; probes < table->capacity; probes++) {
        CMLInternEntry* e = &table->entries[idx];
        if (!e->node)
            return;
        if (e->node == node) {
            e->node = NULL;
            e->hash = 0;
            table->count--;

            size_t next = (idx + 1) & (table->capacity - 1);
            while (table->entries[next].node) {
                CMLInternEntry displaced = table->entries[next];
                table->entries[next].node = NULL;
                table->entries[next].hash = 0;
                table->count--;

                size_t target = probe_index(displaced.hash, table->capacity);
                while (table->entries[target].node)
                    target = (target + 1) & (table->capacity - 1);
                table->entries[target] = displaced;
                table->count++;

                next = (next + 1) & (table->capacity - 1);
            }
            return;
        }
        idx = (idx + 1) & (table->capacity - 1);
    }
}
