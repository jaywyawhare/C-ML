#include "backend/disk_backend.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define HAS_MMAP 1
#else
#define HAS_MMAP 0
#endif

/* Helper: construct file path for a tensor */
static char* make_tensor_path(const CMLDiskBackend* backend, const char* name) {
    size_t len = strlen(backend->base_path) + strlen(name) + 16;
    char* path = (char*)malloc(len);
    if (!path) return NULL;
    snprintf(path, len, "%s/%s.cml_tensor", backend->base_path, name);
    return path;
}

/* Tensor file header for disk serialization */
typedef struct {
    char magic[8];     /* "CMLTENS\0" */
    int32_t ndim;
    int32_t dtype;
    int32_t shape[8];
    uint64_t data_size;
} DiskTensorHeader;

CMLDiskBackend* cml_disk_backend_create(const char* base_path, CMLDiskIOMode mode) {
    if (!base_path) return NULL;

    CMLDiskBackend* b = (CMLDiskBackend*)calloc(1, sizeof(CMLDiskBackend));
    if (!b) return NULL;

    b->base_path = strdup(base_path);
    if (!b->base_path) { free(b); return NULL; }

    b->io_mode = mode;
    b->read_only = false;
    b->has_io_uring = false;

    return b;
}

void cml_disk_backend_free(CMLDiskBackend* backend) {
    if (!backend) return;
    free(backend->base_path);
    free(backend);
}

int cml_disk_save_tensor(CMLDiskBackend* backend, const char* name, Tensor* tensor) {
    if (!backend || !name || !tensor || backend->read_only) return -1;

    char* path = make_tensor_path(backend, name);
    if (!path) return -1;

    FILE* f = fopen(path, "wb");
    free(path);
    if (!f) return -1;

    /* Write header */
    DiskTensorHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "CMLTENS", 8);
    hdr.ndim = tensor->ndim;
    hdr.dtype = (int32_t)tensor->dtype;
    for (int i = 0; i < tensor->ndim && i < 8; i++)
        hdr.shape[i] = tensor->shape[i];
    hdr.data_size = tensor->numel * sizeof(float);

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }

    /* Write tensor data */
    if (tensor->data && tensor->numel > 0) {
        if (fwrite(tensor->data, sizeof(float), tensor->numel, f) != tensor->numel) {
            fclose(f);
            return -1;
        }
    }

    fclose(f);

    backend->bytes_written += sizeof(hdr) + tensor->numel * sizeof(float);
    backend->num_writes++;
    return 0;
}

Tensor* cml_disk_load_tensor(CMLDiskBackend* backend, const char* name) {
    if (!backend || !name) return NULL;

    char* path = make_tensor_path(backend, name);
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    free(path);
    if (!f) return NULL;

    /* Read header */
    DiskTensorHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NULL; }

    if (memcmp(hdr.magic, "CMLTENS", 8) != 0) { fclose(f); return NULL; }

    /* Create tensor */
    int shape[8];
    for (int i = 0; i < hdr.ndim && i < 8; i++)
        shape[i] = hdr.shape[i];

    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, hdr.ndim, &tc);
    if (!t) { fclose(f); return NULL; }

    /* Read data */
    size_t elements = hdr.data_size / sizeof(float);
    if (t->data && elements > 0) {
        size_t read = fread(t->data, sizeof(float), elements, f);
        (void)read;
    }

    fclose(f);

    backend->bytes_read += sizeof(hdr) + hdr.data_size;
    backend->num_reads++;
    return t;
}

CMLDiskTensor* cml_disk_mmap_tensor(CMLDiskBackend* backend, const char* name) {
    if (!backend || !name) return NULL;

    char* path = make_tensor_path(backend, name);
    if (!path) return NULL;

    CMLDiskTensor* dt = (CMLDiskTensor*)calloc(1, sizeof(CMLDiskTensor));
    if (!dt) { free(path); return NULL; }
    dt->file_path = path;

#if HAS_MMAP
    int fd = open(path, O_RDONLY);
    if (fd < 0) { free(dt->file_path); free(dt); return NULL; }

    /* Read header first */
    DiskTensorHeader hdr;
    if (read(fd, &hdr, sizeof(hdr)) != sizeof(hdr)) {
        close(fd);
        free(dt->file_path); free(dt);
        return NULL;
    }

    if (memcmp(hdr.magic, "CMLTENS", 8) != 0) {
        close(fd);
        free(dt->file_path); free(dt);
        return NULL;
    }

    dt->ndim = hdr.ndim;
    dt->dtype = (DType)hdr.dtype;
    dt->data_size = hdr.data_size;
    dt->file_offset = sizeof(hdr);
    for (int i = 0; i < hdr.ndim && i < 8; i++)
        dt->shape[i] = hdr.shape[i];

    /* Memory map the file */
    size_t total_size = sizeof(hdr) + hdr.data_size;
    dt->mmap_addr = mmap(NULL, total_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (dt->mmap_addr == MAP_FAILED) {
        dt->mmap_addr = NULL;
        dt->is_mapped = false;
    } else {
        dt->mmap_len = total_size;
        dt->is_mapped = true;
        backend->num_mmaps++;
    }
#else
    /* No mmap support - fall back to regular loading */
    dt->is_mapped = false;
#endif

    return dt;
}

int cml_disk_tensor_read(CMLDiskTensor* dt, void* buffer, size_t offset, size_t size) {
    if (!dt || !buffer) return -1;

    if (dt->is_mapped && dt->mmap_addr) {
        char* data_start = (char*)dt->mmap_addr + dt->file_offset;
        if (offset + size > dt->data_size) return -1;
        memcpy(buffer, data_start + offset, size);
        return 0;
    }

    /* Fallback: read from file */
    FILE* f = fopen(dt->file_path, "rb");
    if (!f) return -1;
    fseek(f, (long)(dt->file_offset + offset), SEEK_SET);
    size_t read_count = fread(buffer, 1, size, f);
    fclose(f);
    return read_count == size ? 0 : -1;
}

void cml_disk_tensor_unmap(CMLDiskTensor* dt) {
    if (!dt) return;
#if HAS_MMAP
    if (dt->is_mapped && dt->mmap_addr) {
        munmap(dt->mmap_addr, dt->mmap_len);
        dt->mmap_addr = NULL;
        dt->is_mapped = false;
    }
#endif
}

void cml_disk_tensor_free(CMLDiskTensor* dt) {
    if (!dt) return;
    cml_disk_tensor_unmap(dt);
    free(dt->file_path);
    free(dt);
}

Tensor* cml_disk_tensor_to_tensor(CMLDiskTensor* dt) {
    if (!dt) return NULL;

    TensorConfig dtc = {0};
    Tensor* t = tensor_empty(dt->shape, dt->ndim, &dtc);
    if (!t) return NULL;

    if (t->data && dt->data_size > 0) {
        cml_disk_tensor_read(dt, t->data, 0, dt->data_size);
    }
    return t;
}

int cml_disk_async_read(CMLDiskBackend* backend, const char* name,
                         void* buffer, size_t size) {
    /* Fall back to synchronous read */
    if (!backend || !name || !buffer) return -1;

    char* path = make_tensor_path(backend, name);
    if (!path) return -1;

    FILE* f = fopen(path, "rb");
    free(path);
    if (!f) return -1;

    /* Skip header */
    fseek(f, (long)sizeof(DiskTensorHeader), SEEK_SET);
    size_t read_count = fread(buffer, 1, size, f);
    fclose(f);

    backend->bytes_read += read_count;
    backend->num_reads++;
    return read_count == size ? 0 : -1;
}

int cml_disk_wait(CMLDiskBackend* backend) {
    (void)backend;
    /* Synchronous fallback: nothing to wait for */
    return 0;
}

void cml_disk_backend_stats(const CMLDiskBackend* backend,
                             uint64_t* bytes_read, uint64_t* bytes_written,
                             uint64_t* num_reads, uint64_t* num_writes) {
    if (!backend) return;
    if (bytes_read) *bytes_read = backend->bytes_read;
    if (bytes_written) *bytes_written = backend->bytes_written;
    if (num_reads) *num_reads = backend->num_reads;
    if (num_writes) *num_writes = backend->num_writes;
}

void cml_disk_backend_print(const CMLDiskBackend* backend) {
    if (!backend) {
        printf("DiskBackend: NULL\n");
        return;
    }

    const char* mode_str;
    switch (backend->io_mode) {
    case CML_DISK_SYNC:  mode_str = "sync"; break;
    case CML_DISK_MMAP:  mode_str = "mmap"; break;
    case CML_DISK_ASYNC: mode_str = "async"; break;
    default:             mode_str = "unknown"; break;
    }

    printf("Disk Backend\n");
    printf("Path: %s\n", backend->base_path);
    printf("Mode: %s\n", mode_str);
    printf("Read-only: %s\n", backend->read_only ? "yes" : "no");
    printf("io_uring: %s\n", backend->has_io_uring ? "yes" : "no");
    printf("Statistics:\n");
    printf("  Reads: %lu (%.1f MB)\n",
           (unsigned long)backend->num_reads,
           (double)backend->bytes_read / (1024.0 * 1024.0));
    printf("  Writes: %lu (%.1f MB)\n",
           (unsigned long)backend->num_writes,
           (double)backend->bytes_written / (1024.0 * 1024.0));
    printf("  Mmaps: %lu\n", (unsigned long)backend->num_mmaps);
    printf("\n");
}
