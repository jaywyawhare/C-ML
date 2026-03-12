/**
 * @file tinyfs.h
 * @brief TinyFS distributed tensor storage
 *
 * A lightweight filesystem for distributing tensor data across multiple
 * nodes/disks. Supports sharding, replication, and lazy loading.
 */

#ifndef CML_TINYFS_H
#define CML_TINYFS_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum number of shards */
#define CML_TINYFS_MAX_SHARDS 64

/** Shard descriptor */
typedef struct CMLTinyFSShard {
    char path[256];         /* Path to shard file */
    size_t offset;          /* Offset within shard */
    size_t size;            /* Size of data in this shard */
    int shard_id;
    bool is_remote;         /* Whether shard is on remote node */
    char host[128];         /* Remote host (if is_remote) */
    int port;               /* Remote port (if is_remote) */
} CMLTinyFSShard;

/** TinyFS context */
typedef struct CMLTinyFS {
    char base_path[256];
    CMLTinyFSShard shards[CML_TINYFS_MAX_SHARDS];
    int num_shards;
    int replication_factor;
    size_t shard_size;       /* Target shard size in bytes */
    bool initialized;
} CMLTinyFS;

/** Create TinyFS context */
CMLTinyFS* cml_tinyfs_create(const char* base_path, int num_shards, size_t shard_size);

/** Free TinyFS context */
void cml_tinyfs_free(CMLTinyFS* fs);

/** Store tensor (automatically shards across nodes) */
int cml_tinyfs_store(CMLTinyFS* fs, const char* name, Tensor* tensor);

/** Load tensor (reassembles from shards) */
Tensor* cml_tinyfs_load(CMLTinyFS* fs, const char* name);

/** Check if tensor exists in filesystem */
bool cml_tinyfs_exists(CMLTinyFS* fs, const char* name);

/** Delete tensor from filesystem */
int cml_tinyfs_delete(CMLTinyFS* fs, const char* name);

/** List all stored tensors */
char** cml_tinyfs_list(CMLTinyFS* fs, int* count);

/** Get total storage used */
size_t cml_tinyfs_used_bytes(const CMLTinyFS* fs);

/** Print filesystem info */
void cml_tinyfs_print(const CMLTinyFS* fs);

#ifdef __cplusplus
}
#endif

#endif /* CML_TINYFS_H */
