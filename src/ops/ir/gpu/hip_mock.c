/* Mock HIP runtime for validating ROCm backend / HCQ adapter code without
 * an AMD GPU.
 *
 * The real backend discovers HIP through dlopen("libamdhip64.so") and fills
 * its function-pointer table. This mock provides implementations for every
 * entry point that table references, with host-backed semantics:
 *
 *  - device memory is ordinary host memory (memcpy works in both directions)
 *  - allocations are tracked so tests can assert zero outstanding buffers
 *  - streams/events complete immediately but are journaled
 *  - every operation is recorded in an ordered journal that tests inspect
 *    to validate submission order (H2D -> launch -> D2H), buffer mapping,
 *    and event lifecycle.
 *
 * Enable by calling cml_rocm_backend_init_mock(), or automatically when
 * CML_HIP_MOCK=1 is set before cml_rocm_backend_init().
 */
#include "ops/ir/gpu/hip_mock.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define HIP_SUCCESS 0
#define HIP_ERROR_NOT_INITIALIZED 1
#define HIP_ERROR_OUT_OF_MEMORY 2
#define HIP_ERROR_INVALID_VALUE 3

/* mirrors rocm_backend.c's local constants */
#define MOCK_MEMCPY_H2D 1
#define MOCK_MEMCPY_D2H 2

#define MOCK_MAX_ALLOC_TRACKED 4096
#define MOCK_JOURNAL_CAP 1024
#define MOCK_DEVICE_NAME "Mock-RDNA3"

typedef enum {
    MOCK_OP_INIT = 0,
    MOCK_OP_STREAM_CREATE,
    MOCK_OP_STREAM_SYNC,
    MOCK_OP_MALLOC,
    MOCK_OP_FREE,
    MOCK_OP_MEMCPY_H2D,
    MOCK_OP_MEMCPY_D2H,
    MOCK_OP_MODULE_LOAD,
    MOCK_OP_MODULE_GETFN,
    MOCK_OP_LAUNCH,
    MOCK_OP_EVENT_CREATE,
    MOCK_OP_EVENT_RECORD,
    MOCK_OP_EVENT_SYNC,
} MockOpKind;

typedef struct {
    MockOpKind kind;
    size_t bytes;     /* memcpy/malloc payload */
    uint32_t grid[3]; /* launch geometry */
    char name[32];    /* kernel name / label */
} MockJournalEntry;

typedef struct {
    bool initialized;
    void* allocs[MOCK_MAX_ALLOC_TRACKED];
    int num_allocs;
    MockJournalEntry journal[MOCK_JOURNAL_CAP];
    int journal_len;
    bool stream_created;
    uint64_t launches;
} HipMockState;

static HipMockState g_mock;

static void journal(MockOpKind kind, size_t bytes, const char* name) {
    if (g_mock.journal_len >= MOCK_JOURNAL_CAP)
        return;
    MockJournalEntry* e = &g_mock.journal[g_mock.journal_len++];
    e->kind             = kind;
    e->bytes            = bytes;
    if (name) {
        strncpy(e->name, name, sizeof(e->name) - 1);
        e->name[sizeof(e->name) - 1] = '\0';
    } else {
        e->name[0] = '\0';
    }
}

static void track_alloc(void* p) {
    if (p && g_mock.num_allocs < MOCK_MAX_ALLOC_TRACKED)
        g_mock.allocs[g_mock.num_allocs++] = p;
}

static void untrack_alloc(void* p) {
    for (int i = 0; i < g_mock.num_allocs; i++) {
        if (g_mock.allocs[i] == p) {
            g_mock.allocs[i] = g_mock.allocs[--g_mock.num_allocs];
            return;
        }
    }
}

/* --- implemented HIP surface -------------------------------------------- */

static hipError_t m_hipInit(unsigned int flags) {
    (void)flags;
    journal(MOCK_OP_INIT, 0, NULL);
    g_mock.initialized = true;
    return HIP_SUCCESS;
}

static hipError_t m_hipGetDeviceCount(int* count) {
    if (!count || !g_mock.initialized)
        return HIP_ERROR_NOT_INITIALIZED;
    *count = 1;
    return HIP_SUCCESS;
}

static hipError_t m_hipSetDevice(int id) {
    (void)id;
    return g_mock.initialized ? HIP_SUCCESS : HIP_ERROR_NOT_INITIALIZED;
}

/* rocm_backend passes a pointer to its own properties struct layout; fill
 * conservatively through the documented leading fields only. */
static hipError_t m_hipGetDeviceProperties(void* prop, int deviceId) {
    (void)prop; /* layout is opaque here; backend re-reads via its own copy */
    (void)deviceId;
    return HIP_SUCCESS;
}

static hipError_t m_hipMalloc(void** ptr, size_t size) {
    if (!ptr || !size)
        return HIP_ERROR_INVALID_VALUE;
    void* p = cml_malloc(size);
    if (!p)
        return HIP_ERROR_OUT_OF_MEMORY;
    track_alloc(p);
    *ptr = p;
    journal(MOCK_OP_MALLOC, size, NULL);
    return HIP_SUCCESS;
}

static hipError_t m_hipFree(void* ptr) {
    if (!ptr)
        return HIP_ERROR_INVALID_VALUE;
    untrack_alloc(ptr);
    cml_free(ptr);
    journal(MOCK_OP_FREE, 0, NULL);
    return HIP_SUCCESS;
}

/* Device memory is host-backed, so both directions are plain copies. */
static hipError_t m_hipMemcpy(void* dst, const void* src, size_t bytes, int kind) {
    if (!dst || !src || !bytes)
        return HIP_ERROR_INVALID_VALUE;
    if (kind == MOCK_MEMCPY_H2D) {
        memcpy(dst, src, bytes);
        journal(MOCK_OP_MEMCPY_H2D, bytes, NULL);
    } else if (kind == MOCK_MEMCPY_D2H) {
        memcpy(dst, src, bytes);
        journal(MOCK_OP_MEMCPY_D2H, bytes, NULL);
    } else {
        return HIP_ERROR_INVALID_VALUE;
    }
    return HIP_SUCCESS;
}

static hipError_t m_hipStreamCreate(hipStream_t* s) {
    if (!s || !g_mock.initialized)
        return HIP_ERROR_NOT_INITIALIZED;
    static int dummy_stream;
    *s                    = &dummy_stream;
    g_mock.stream_created = true;
    journal(MOCK_OP_STREAM_CREATE, 0, NULL);
    return HIP_SUCCESS;
}

static hipError_t m_hipStreamDestroy(hipStream_t s) {
    (void)s;
    g_mock.stream_created = false;
    return HIP_SUCCESS;
}

static hipError_t m_hipStreamSynchronize(hipStream_t s) {
    (void)s;
    journal(MOCK_OP_STREAM_SYNC, 0, NULL);
    return HIP_SUCCESS;
}

static hipError_t m_hipDeviceSynchronize(void) {
    journal(MOCK_OP_STREAM_SYNC, 0, "device");
    return HIP_SUCCESS;
}

/* Code objects: HSACO blobs cannot execute on a CPU, so load succeeds
 * (validating the mapping/lifetime path) while launches are journaled with
 * their geometry instead of executed. */
static hipError_t m_hipModuleLoadData(hipModule_t* module, const void* image) {
    if (!module || !image)
        return HIP_ERROR_INVALID_VALUE;
    static int dummy_module;
    *module = &dummy_module;
    journal(MOCK_OP_MODULE_LOAD, 0, NULL);
    return HIP_SUCCESS;
}

static hipError_t m_hipModuleLoad(hipModule_t* module, const char* fname) {
    if (!module || !fname)
        return HIP_ERROR_INVALID_VALUE;
    return m_hipModuleLoadData(module, fname);
}

static hipError_t m_hipModuleUnload(hipModule_t module) {
    (void)module;
    return HIP_SUCCESS;
}

static hipError_t m_hipModuleGetFunction(hipFunction_t* fn, hipModule_t module, const char* kname) {
    if (!fn || !kname)
        return HIP_ERROR_INVALID_VALUE;
    (void)module;
    static char fn_storage[64];
    memset(fn_storage, 0, sizeof(fn_storage));
    snprintf(fn_storage, sizeof(fn_storage), "%s", kname);
    *fn = fn_storage; /* keep the name so launch can journal it */
    journal(MOCK_OP_MODULE_GETFN, 0, kname);
    return HIP_SUCCESS;
}

static hipError_t m_hipModuleLaunchKernel(hipFunction_t f, unsigned gx, unsigned gy, unsigned gz,
                                          unsigned bx, unsigned by, unsigned bz, unsigned shared,
                                          hipStream_t stream, void** kernelParams, void** extra) {
    (void)f;
    (void)stream;
    (void)shared;
    (void)extra;
    if (!kernelParams)
        return HIP_ERROR_INVALID_VALUE;
    for (int i = 0; kernelParams[i]; i++) {
        if (!kernelParams[i])
            return HIP_ERROR_INVALID_VALUE;
    }
    MockJournalEntry* e =
        &g_mock.journal[g_mock.journal_len < MOCK_JOURNAL_CAP ? g_mock.journal_len
                                                              : MOCK_JOURNAL_CAP - 1];
    (void)e;
    journal(MOCK_OP_LAUNCH, 0, f ? (const char*)f : "kernel");
    if (g_mock.journal_len > 0 && g_mock.journal[g_mock.journal_len - 1].kind == MOCK_OP_LAUNCH) {
        MockJournalEntry* last = &g_mock.journal[g_mock.journal_len - 1];
        last->grid[0]          = gx;
        last->grid[1]          = gy;
        last->grid[2]          = gz;
        last->bytes            = ((size_t)bx << 16) | ((size_t)by << 8) | bz; /* stash block dims */
    }
    g_mock.launches++;
    return HIP_SUCCESS;
}

static hipError_t m_hipEventCreate(void** event) {
    if (!event)
        return HIP_ERROR_INVALID_VALUE;
    static char event_storage[16];
    static int event_counter;
    if (event_counter >= 16)
        event_counter = 0;
    *event = &event_storage[event_counter++ * 4];
    journal(MOCK_OP_EVENT_CREATE, 0, NULL);
    return HIP_SUCCESS;
}

static hipError_t m_hipEventDestroy(void* event) {
    (void)event;
    return HIP_SUCCESS;
}

static hipError_t m_hipEventRecord(void* event, hipStream_t stream) {
    (void)event;
    (void)stream;
    journal(MOCK_OP_EVENT_RECORD, 0, NULL);
    return HIP_SUCCESS;
}

static hipError_t m_hipEventSynchronize(void* event) {
    (void)event;
    journal(MOCK_OP_EVENT_SYNC, 0, NULL);
    return HIP_SUCCESS;
}

/* --- public mock API ----------------------------------------------------- */

void cml_hip_mock_reset(void) { memset(&g_mock, 0, sizeof(g_mock)); }

int cml_rocm_backend_init_mock(CMLROCmBackend* backend) {
    if (!backend)
        return -1;
    if (backend->initialized)
        return 0;

    cml_hip_mock_reset();

    backend->hipInit                = m_hipInit;
    backend->hipGetDeviceCount      = m_hipGetDeviceCount;
    backend->hipSetDevice           = m_hipSetDevice;
    backend->hipGetDeviceProperties = m_hipGetDeviceProperties;
    backend->hipModuleLoad          = m_hipModuleLoad;
    backend->hipModuleLoadData      = m_hipModuleLoadData;
    backend->hipModuleUnload        = m_hipModuleUnload;
    backend->hipModuleGetFunction   = m_hipModuleGetFunction;
    backend->hipModuleLaunchKernel  = m_hipModuleLaunchKernel;
    backend->hipMalloc              = m_hipMalloc;
    backend->hipFree                = m_hipFree;
    backend->hipMemcpy              = m_hipMemcpy;
    backend->hipStreamCreate        = m_hipStreamCreate;
    backend->hipStreamDestroy       = m_hipStreamDestroy;
    backend->hipStreamSynchronize   = m_hipStreamSynchronize;
    backend->hipDeviceSynchronize   = m_hipDeviceSynchronize;
    backend->hipEventCreate         = m_hipEventCreate;
    backend->hipEventDestroy        = m_hipEventDestroy;
    backend->hipEventRecord         = m_hipEventRecord;
    backend->hipEventSynchronize    = m_hipEventSynchronize;
    backend->hip_lib                = NULL; /* no dlopen */
    snprintf(backend->device_name, sizeof(backend->device_name), "%s", MOCK_DEVICE_NAME);
    backend->total_memory          = 1ull << 30;
    backend->multiprocessor_count  = 1;
    backend->max_threads_per_block = 1024;

    hipError_t err = backend->hipInit(0);
    if (err != HIP_SUCCESS)
        return -1;
    int count = 0;
    backend->hipGetDeviceCount(&count);
    if (count < 1)
        return -1;
    backend->device = 0;
    backend->hipSetDevice(0);
    backend->hipStreamCreate(&backend->stream);
    backend->initialized = true;

    LOG_INFO("ROCm backend initialized against MOCK HIP driver");
    return 0;
}

bool cml_hip_mock_enabled_from_env(void) {
    const char* v = getenv("CML_HIP_MOCK");
    return v && v[0] == '1';
}

int cml_hip_mock_journal_len(void) { return g_mock.journal_len; }

const CMLHIPMockEntry* cml_hip_mock_journal_at(int i) {
    if (i < 0 || i >= g_mock.journal_len || i >= MOCK_JOURNAL_CAP)
        return NULL;
    static CMLHIPMockEntry out;
    out.kind            = (CMLHIPMockOpKind)g_mock.journal[i].kind;
    out.bytes           = g_mock.journal[i].bytes;
    out.launches_grid_x = g_mock.journal[i].grid[0];
    snprintf(out.name, sizeof(out.name), "%s", g_mock.journal[i].name);
    return &out;
}

int cml_hip_mock_outstanding_allocs(void) { return g_mock.num_allocs; }

uint64_t cml_hip_mock_launches(void) { return g_mock.launches; }
