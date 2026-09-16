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
#include "alloc/cml_allocator.h"
#define HAS_MMAP 1
#else
#define HAS_MMAP 0
#endif

/* Helper: construct file path for a tensor */
static char* make_tensor_path(const CMLDiskBackend* backend, const char* name) {
    size_t len = strlen(backend->base_path) + strlen(name) + 16;
    char* path = (char*)cml_malloc(len);
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

#ifdef CML_HAS_IO_URING
#include <liburing.h>

#define CML_URING_DEPTH 32

/* Ring plus per-request state cml_disk_wait needs: each in-flight read keeps
 * its fd open until completion and its expected size for short-read detection. */
typedef struct {
    struct io_uring ring;
    int    pending_fds[CML_URING_DEPTH];
    size_t pending_size[CML_URING_DEPTH];
    int    num_pending;
} DiskUring;
#endif

CMLDiskBackend* cml_disk_backend_create(const char* base_path, CMLDiskIOMode mode) {
    if (!base_path) return NULL;

    CMLDiskBackend* b = (CMLDiskBackend*)cml_calloc(1, sizeof(CMLDiskBackend));
    if (!b) return NULL;

    b->base_path = cml_strdup(base_path);
    if (!b->base_path) { cml_free(b); return NULL; }

    b->io_mode = mode;
    b->read_only = false;
    b->has_io_uring = false;
    b->ring = NULL;

    if (mode == CML_DISK_ASYNC) {
#ifdef CML_HAS_IO_URING
        DiskUring* u = (DiskUring*)cml_calloc(1, sizeof(DiskUring));
        if (u && io_uring_queue_init(CML_URING_DEPTH, &u->ring, 0) == 0) {
            b->ring = u;
            b->has_io_uring = true;
        } else {
            cml_free(u);
            LOG_WARNING("cml_disk_backend_create: io_uring_queue_init failed; "
                        "reads run synchronously");
        }
#else
        LOG_WARNING("cml_disk_backend_create: CML_DISK_ASYNC requested but "
                    "io_uring support was not compiled in; reads run synchronously");
#endif
    }

    return b;
}

void cml_disk_backend_free(CMLDiskBackend* backend) {
    if (!backend) return;
#ifdef CML_HAS_IO_URING
    if (backend->has_io_uring && backend->ring) {
        DiskUring* u = (DiskUring*)backend->ring;
        for (int i = 0; i < u->num_pending; i++)
            if (u->pending_fds[i] >= 0) close(u->pending_fds[i]);
        io_uring_queue_exit(&u->ring);
        cml_free(u);
    }
#endif
    cml_free(backend->base_path);
    cml_free(backend);
}

int cml_disk_save_tensor(CMLDiskBackend* backend, const char* name, Tensor* tensor) {
    if (!backend || !name || !tensor || backend->read_only) return -1;

    char* path = make_tensor_path(backend, name);
    if (!path) return -1;

    FILE* f = fopen(path, "wb");
    cml_free(path);
    if (!f) return -1;

    /* Write header */
    DiskTensorHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "CMLTENS", 8);
    hdr.ndim = tensor->ndim;
    hdr.dtype = (int32_t)tensor->dtype;
    for (int i = 0; i < tensor->ndim && i < 8; i++)
        hdr.shape[i] = tensor->shape[i];
    size_t elem_size = cml_dtype_size(tensor->dtype);
    hdr.data_size = tensor->numel * elem_size;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }

    /* Write tensor data — raw bytes in the tensor's own dtype (was f32-only,
     * silently truncating every other dtype to its first quarter). */
    if (tensor->data && tensor->numel > 0) {
        if (fwrite(tensor->data, elem_size, tensor->numel, f) != tensor->numel) {
            fclose(f);
            return -1;
        }
    }

    fclose(f);

    backend->bytes_written += sizeof(hdr) + hdr.data_size;
    backend->num_writes++;
    return 0;
}

Tensor* cml_disk_load_tensor(CMLDiskBackend* backend, const char* name) {
    if (!backend || !name) return NULL;

    char* path = make_tensor_path(backend, name);
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    cml_free(path);
    if (!f) return NULL;

    /* Read header */
    DiskTensorHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NULL; }

    if (memcmp(hdr.magic, "CMLTENS", 8) != 0) { fclose(f); return NULL; }
    if (hdr.ndim <= 0 || hdr.ndim > 8) { fclose(f); return NULL; }

    /* Honor the dtype recorded in the header (the loader used to zero the
     * config and read everything as f32). */
    DType dtype = (DType)hdr.dtype;
    if ((int)dtype < 0 || (int)dtype >= 32) { fclose(f); return NULL; }
    size_t elem_size = cml_dtype_size(dtype);
    if (elem_size == 0 || hdr.data_size % elem_size != 0) { fclose(f); return NULL; }

    /* Create tensor */
    int shape[8];
    for (int i = 0; i < hdr.ndim && i < 8; i++)
        shape[i] = hdr.shape[i];

    TensorConfig tc = {.dtype = dtype, .device = DEVICE_CPU,
                       .has_dtype = true, .has_device = true};
    Tensor* t = tensor_empty(shape, hdr.ndim, &tc);
    if (!t) { fclose(f); return NULL; }

    size_t elements = hdr.data_size / elem_size;
    if (t->data && elements > 0) {
        size_t read = fread(t->data, elem_size, elements, f);
        if (read != elements) {
            tensor_free(t);
            fclose(f);
            return NULL;
        }
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

    CMLDiskTensor* dt = (CMLDiskTensor*)cml_calloc(1, sizeof(CMLDiskTensor));
    if (!dt) { cml_free(path); return NULL; }
    dt->file_path = path;

#if HAS_MMAP
    int fd = open(path, O_RDONLY);
    if (fd < 0) { cml_free(dt->file_path); cml_free(dt); return NULL; }

    /* Read header first */
    DiskTensorHeader hdr;
    if (read(fd, &hdr, sizeof(hdr)) != sizeof(hdr)) {
        close(fd);
        cml_free(dt->file_path); cml_free(dt);
        return NULL;
    }

    if (memcmp(hdr.magic, "CMLTENS", 8) != 0) {
        close(fd);
        cml_free(dt->file_path); cml_free(dt);
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
    cml_free(dt->file_path);
    cml_free(dt);
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
    if (!backend || !name || !buffer) return -1;

#ifdef CML_HAS_IO_URING
    if (backend->has_io_uring && backend->ring) {
        DiskUring* u = (DiskUring*)backend->ring;
        /* Drain first if the ring is full so we never overflow the SQ. */
        if (u->num_pending >= CML_URING_DEPTH && cml_disk_wait(backend) != 0)
            return -1;

        char* path = make_tensor_path(backend, name);
        if (!path) return -1;
        int fd = open(path, O_RDONLY);
        cml_free(path);
        if (fd < 0) return -1;

        struct io_uring_sqe* sqe = io_uring_get_sqe(&u->ring);
        if (!sqe) {
            /* SQ momentarily full: flush and retry once. */
            io_uring_submit(&u->ring);
            sqe = io_uring_get_sqe(&u->ring);
            if (!sqe) { close(fd); return -1; }
        }

        int slot = u->num_pending++;
        u->pending_fds[slot]  = fd;
        u->pending_size[slot] = size;
        io_uring_prep_read(sqe, fd, buffer, (unsigned)size,
                           (unsigned long long)sizeof(DiskTensorHeader));
        io_uring_sqe_set_data64(sqe, (unsigned long long)slot);

        if (io_uring_submit(&u->ring) < 0) {
            close(fd);
            u->num_pending--;
            return -1;
        }
        return 0;
    }
#endif

    /* Synchronous fallback: the read completes before returning. */
    char* path = make_tensor_path(backend, name);
    if (!path) return -1;

    FILE* f = fopen(path, "rb");
    cml_free(path);
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
    if (!backend) return -1;

#ifdef CML_HAS_IO_URING
    if (backend->has_io_uring && backend->ring) {
        DiskUring* u = (DiskUring*)backend->ring;
        int rc = 0;
        int outstanding = u->num_pending;
        for (int done = 0; done < outstanding; done++) {
            struct io_uring_cqe* cqe = NULL;
            if (io_uring_wait_cqe(&u->ring, &cqe) < 0 || !cqe) { rc = -1; break; }
            unsigned long long slot = io_uring_cqe_get_data64(cqe);
            int res = cqe->res;
            io_uring_cqe_seen(&u->ring, cqe);

            if (slot < (unsigned long long)CML_URING_DEPTH) {
                if (u->pending_fds[slot] >= 0) {
                    close(u->pending_fds[slot]);
                    u->pending_fds[slot] = -1;
                }
                if (res < 0 || (size_t)res != u->pending_size[slot]) rc = -1;
                else {
                    backend->bytes_read += (uint64_t)res;
                    backend->num_reads++;
                }
            }
        }
        u->num_pending = 0;
        return rc;
    }
#endif

    /* Synchronous fallback: reads already completed in cml_disk_async_read. */
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
