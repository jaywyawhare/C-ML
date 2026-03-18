/*
 * Memory-maps tensor data from disk, enabling datasets larger than RAM.
 * Supports synchronous I/O with optional io_uring async I/O on Linux.
 */

#ifndef CML_DISK_BACKEND_H
#define CML_DISK_BACKEND_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CML_DISK_SYNC = 0,     /* Synchronous read/write */
    CML_DISK_MMAP,          /* Memory-mapped I/O */
    CML_DISK_ASYNC,         /* Async I/O (io_uring on Linux, fallback sync) */
} CMLDiskIOMode;

typedef struct CMLDiskBackend {
    char* base_path;          /* Base directory for tensor storage */
    CMLDiskIOMode io_mode;
    bool read_only;

    /* io_uring state (Linux only) */
    void* ring;               /* struct io_uring* */
    bool has_io_uring;

    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t num_reads;
    uint64_t num_writes;
    uint64_t num_mmaps;
} CMLDiskBackend;

typedef struct CMLDiskTensor {
    char* file_path;
    size_t file_offset;        /* Offset within file */
    size_t data_size;          /* Size of tensor data in bytes */
    int shape[8];              /* Tensor shape (max 8 dims) */
    int ndim;
    DType dtype;

    void* mmap_addr;
    size_t mmap_len;
    bool is_mapped;
} CMLDiskTensor;

CMLDiskBackend* cml_disk_backend_create(const char* base_path, CMLDiskIOMode mode);
void cml_disk_backend_free(CMLDiskBackend* backend);
int cml_disk_save_tensor(CMLDiskBackend* backend, const char* name, Tensor* tensor);
Tensor* cml_disk_load_tensor(CMLDiskBackend* backend, const char* name);

/* Memory-map tensor from disk (lazy, zero-copy) */
CMLDiskTensor* cml_disk_mmap_tensor(CMLDiskBackend* backend, const char* name);
int cml_disk_tensor_read(CMLDiskTensor* dt, void* buffer, size_t offset, size_t size);
void cml_disk_tensor_unmap(CMLDiskTensor* dt);
void cml_disk_tensor_free(CMLDiskTensor* dt);
Tensor* cml_disk_tensor_to_tensor(CMLDiskTensor* dt);

/* Returns immediately, use cml_disk_wait to complete */
int cml_disk_async_read(CMLDiskBackend* backend, const char* name,
                         void* buffer, size_t size);
int cml_disk_wait(CMLDiskBackend* backend);

void cml_disk_backend_stats(const CMLDiskBackend* backend,
                             uint64_t* bytes_read, uint64_t* bytes_written,
                             uint64_t* num_reads, uint64_t* num_writes);
void cml_disk_backend_print(const CMLDiskBackend* backend);

#ifdef __cplusplus
}
#endif

#endif /* CML_DISK_BACKEND_H */
