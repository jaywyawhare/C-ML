/*
 * cml_allocator.c
 *
 * Extremely fast general-purpose allocator for C-ML.
 * - Thread-cached size-class segregated freelists (rpmalloc / tcmalloc inspiration, simplified).
 * - Slabs carved from backing (malloc for portability + simplicity; hot path is pure freelist).
 * - Low contention: hot alloc/free usually no locks.
 * - 16B alignment base. Larger requests get better alignment opportunistically.
 * - Direct path (mmap style via libc or aligned) for huge allocations.
 * Goal: beat or match system malloc in throughput + much lower latency variance for ML workloads.
 */

#include "alloc/cml_allocator.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>   /* only for optional stats print, remove if want zero dep */

/* System backing allocator for internal slab/large acquisition only.
 * We must NEVER call our own cml_* here for bootstrapping the allocator itself. */
static inline void* system_malloc(size_t sz) { return malloc(sz); }
static inline void  system_free(void* p)     { free(p); }
static inline int   system_posix_memalign(void** memptr, size_t alignment, size_t size) {
    return posix_memalign(memptr, alignment, size);
}

/* Tunables for "fast as fuck" */
#define CML_SLAB_SIZE          (256 * 1024)   /* 256 KiB slabs - sweet spot for cache + TLB */
#define CML_MAX_LOCAL_CACHE    64             /* max free objects kept per class in TLS before flushing batch */
#define CML_LARGE_THRESHOLD    (128 * 1024)   /* >=128KiB: direct path */
#define CML_MIN_ALIGN          16
#define CML_HEADER_SIZE        16             /* 16B header => good default alignment for returned ptrs */

_Static_assert(CML_HEADER_SIZE >= 16 && (CML_HEADER_SIZE % 16) == 0, "header must preserve alignment");

/* Per-allocation header. Lives immediately before user pointer.
 * size is size_t so large tensors / buffers well beyond 4 GiB are representable. */
typedef struct {
    size_t   size;       /* requested user bytes */
    uint16_t class_idx;  /* which size class (0xffff for large/direct) */
    uint16_t magic;      /* 0xC4A1 for sanity */
} AllocHeader;

_Static_assert(sizeof(AllocHeader) <= CML_HEADER_SIZE,
               "AllocHeader must fit in CML_HEADER_SIZE");

#define ALLOC_MAGIC 0xC4A1

static inline AllocHeader* header_from_user(void* user) {
    if (!user) return NULL;
    return (AllocHeader*)((char*)user - CML_HEADER_SIZE);
}

static inline void* user_from_header(AllocHeader* h) {
    return (char*)h + CML_HEADER_SIZE;
}

/* ---------------- Size classes ----------------
 * We use a compact table of increasing sizes. Good balance of internal fragmentation vs #classes.
 * Classes chosen so small objects (common for structs, nodes, small temps) have tight fits.
 */
static const size_t SIZE_CLASSES[] = {
    /* 0-15: 16B granularity for tiny */
    16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
    /* 16-23: 32B steps */
    288, 320, 352, 384, 416, 448, 480, 512,
    /* 24-29: 64B steps */
    576, 640, 704, 768, 832, 896,
    /* 30-33: 128B steps */
    1024, 1152, 1280, 1408,
    /* 34-37: 256B steps */
    1536, 1792, 2048, 2304,
    /* 38-41: 512B steps */
    2560, 3072, 3584, 4096,
    /* 42-45: 1KiB steps up to 8KiB */
    4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192,
    /* 50-53: 2KiB steps */
    9216, 10240, 11264, 12288,
    /* 54-57: 4KiB steps to 24KiB */
    14336, 16384, 18432, 20480, 22528, 24576,
    /* 60-62: bigger for "medium" */
    28672, 32768, 40960,
    /* last few before large threshold */
    49152, 65536, 98304
};
#define NUM_SIZE_CLASSES (sizeof(SIZE_CLASSES) / sizeof(SIZE_CLASSES[0]))

static inline int size_to_class(size_t size) {
    if (size <= SIZE_CLASSES[0]) return 0;
    /* Linear scan is fine: NUM_SIZE_CLASSES ~ 65, called only on slow paths or first alloc per size. */
    for (int i = 0; i < NUM_SIZE_CLASSES; ++i) {
        if (size <= SIZE_CLASSES[i]) return i;
    }
    return -1; /* large */
}

static inline size_t class_to_size(int cls) {
    if (cls < 0 || cls >= NUM_SIZE_CLASSES) return 0;
    return SIZE_CLASSES[cls];
}

/* ---------------- Slab + block structures ---------------- */

typedef struct FreeNode {
    struct FreeNode* next;
} FreeNode;

typedef struct Slab {
    struct Slab*   next;
    void*          system_base; /* original pointer from system_malloc; used for system_free */
    uint32_t       class_idx;
    uint32_t       num_blocks;
    uint32_t       used_blocks;
    char           data[];   /* flexible: the carved blocks start here */
} Slab;

/* Track live slabs so we can eventually system_free their original base.
 * Currently slabs are process-lifetime (standard for this style of allocator),
 * but we must not lose the system_malloc pointer after alignment. */
static pthread_mutex_t g_slab_list_lock = PTHREAD_MUTEX_INITIALIZER;
static Slab* g_slab_list = NULL;

/* One central freelist + mutex per size class */
typedef struct {
    pthread_mutex_t lock;
    FreeNode*       head;
    size_t          total_slabs;   /* approx */
} CentralBin;

/* Thread-local cache */
typedef struct {
    FreeNode* heads[NUM_SIZE_CLASSES];
    uint16_t  counts[NUM_SIZE_CLASSES];
    bool      initialized;
} ThreadCache;

static CentralBin   g_central[NUM_SIZE_CLASSES];
static pthread_mutex_t g_init_lock = PTHREAD_MUTEX_INITIALIZER;
static bool         g_initialized = false;

/* Thread local */
static __thread ThreadCache tl_cache = {0};

/* Stats (best effort, not perfectly accurate under races) */
static size_t g_total_allocated_bytes = 0;
static size_t g_peak_allocated_bytes  = 0;
static size_t g_alloc_count           = 0;

/* Forward */
static void*  alloc_from_class(int cls, size_t user_size);
static void   free_to_class(int cls, void* user_ptr, size_t user_size);
static void*  alloc_large(size_t size);
static void   free_large(void* ptr);
static void   init_once(void);
static Slab*  carve_new_slab(int cls);
static void   refill_from_central(int cls);
static void   flush_local_to_central(int cls, int keep);

/* ---------------- Initialization ---------------- */

static void init_central_bins(void) {
    for (int i = 0; i < NUM_SIZE_CLASSES; ++i) {
        pthread_mutex_init(&g_central[i].lock, NULL);
        g_central[i].head = NULL;
        g_central[i].total_slabs = 0;
    }
}

static void init_once(void) {
    pthread_mutex_lock(&g_init_lock);
    if (!g_initialized) {
        init_central_bins();
        g_initialized = true;
    }
    pthread_mutex_unlock(&g_init_lock);
}

static inline void ensure_init(void) {
    if (!g_initialized) {
        init_once();
    }
    if (!tl_cache.initialized) {
        memset(&tl_cache, 0, sizeof(tl_cache));
        tl_cache.initialized = true;
    }
}

/* ---------------- Slab carving ---------------- */

static Slab* carve_new_slab(int cls) {
    size_t bin_size = class_to_size(cls);
    if (bin_size == 0) return NULL;

    /* We allocate the slab header + data region with libc malloc.
     * This cost is amortized over hundreds of objects per slab.
     */
    size_t usable = CML_SLAB_SIZE - sizeof(Slab);
    size_t num_blocks = usable / (CML_HEADER_SIZE + bin_size);
    if (num_blocks < 1) {
        /* Extremely large class inside "small" path - fall back */
        num_blocks = 1;
    }

    size_t alloc_sz = sizeof(Slab) + num_blocks * (CML_HEADER_SIZE + bin_size);
    /* Align the slab allocation a bit; keep the original system pointer so we can free it. */
    void* raw = system_malloc(alloc_sz + 64);
    if (!raw) return NULL;

    /* Align start of Slab structure for cleanliness */
    uintptr_t base = (uintptr_t)raw;
    uintptr_t aligned_base = (base + 63) & ~(uintptr_t)63;
    Slab* slab = (Slab*)aligned_base;

    slab->next = NULL;
    slab->system_base = raw; /* MUST retain: system_free(raw), not system_free(slab) */
    slab->class_idx = (uint32_t)cls;
    slab->num_blocks = (uint32_t)num_blocks;
    slab->used_blocks = 0;

    /* Build intrusive free list for this slab directly into the central for this class */
    char* p = (char*)slab->data;
    FreeNode* first = NULL;
    FreeNode* prev = NULL;

    for (uint32_t i = 0; i < num_blocks; ++i) {
        AllocHeader* hdr = (AllocHeader*)p;
        hdr->size = 0;
        hdr->class_idx = (uint16_t)cls;
        hdr->magic = ALLOC_MAGIC;

        FreeNode* node = (FreeNode*)(p + CML_HEADER_SIZE);
        node->next = NULL;

        if (!first) first = node;
        if (prev) prev->next = node;
        prev = node;

        p += CML_HEADER_SIZE + bin_size;
    }

    /* Insert the whole chain into central under caller lock (or we can do it here) */
    if (first) {
        pthread_mutex_lock(&g_central[cls].lock);
        prev->next = g_central[cls].head;
        g_central[cls].head = first;
        g_central[cls].total_slabs++;
        pthread_mutex_unlock(&g_central[cls].lock);
    }

    /* Register slab so its system_base is never lost (process-lifetime today). */
    pthread_mutex_lock(&g_slab_list_lock);
    slab->next = g_slab_list;
    g_slab_list = slab;
    pthread_mutex_unlock(&g_slab_list_lock);

    return slab;
}

/* ---------------- Refill / flush ---------------- */

static void refill_from_central(int cls) {
    /* Steal a batch from central into thread local */
    const int want = 32; /* batch size */
    FreeNode* stolen = NULL;
    int got = 0;

    pthread_mutex_lock(&g_central[cls].lock);
    FreeNode* cur = g_central[cls].head;
    FreeNode* prev = NULL;
    while (cur && got < want) {
        FreeNode* next = cur->next;
        /* unlink */
        if (prev) prev->next = next;
        else g_central[cls].head = next;
        cur->next = stolen;
        stolen = cur;
        cur = next;
        ++got;
    }
    pthread_mutex_unlock(&g_central[cls].lock);

    if (got > 0) {
        tl_cache.heads[cls] = stolen;
        tl_cache.counts[cls] = (uint16_t)got;
    }
}

static void flush_local_to_central(int cls, int keep) {
    FreeNode* list = tl_cache.heads[cls];
    uint16_t cnt = tl_cache.counts[cls];
    if (!list || cnt <= (uint16_t)keep) return;

    /* Detach the excess tail */
    FreeNode* keep_head = list;
    FreeNode* tail = list;
    int keep_cnt = 0;
    while (keep_cnt < keep && tail) {
        ++keep_cnt;
        if (keep_cnt < keep) tail = tail->next;
    }
    if (!tail) {
        /* nothing to flush */
        return;
    }
    FreeNode* flush_head = tail->next;
    tail->next = NULL;
    tl_cache.heads[cls] = keep_head;
    tl_cache.counts[cls] = (uint16_t)keep_cnt;

    if (flush_head) {
        /* Append flush list to central */
        pthread_mutex_lock(&g_central[cls].lock);
        /* Find end of flush list to splice */
        FreeNode* f = flush_head;
        while (f->next) f = f->next;
        f->next = g_central[cls].head;
        g_central[cls].head = flush_head;
        pthread_mutex_unlock(&g_central[cls].lock);
    }
}

/* ---------------- Allocation paths ---------------- */

static void* alloc_from_class(int cls, size_t user_size) {
    ensure_init();

    /* Fast path: thread local */
    FreeNode* node = tl_cache.heads[cls];
    if (node) {
        tl_cache.heads[cls] = node->next;
        tl_cache.counts[cls]--;
        AllocHeader* hdr = (AllocHeader*)((char*)node - CML_HEADER_SIZE);
        hdr->size = user_size;
        hdr->class_idx = (uint16_t)cls;
        hdr->magic = ALLOC_MAGIC;

        /* update stats */
        __atomic_add_fetch(&g_total_allocated_bytes, user_size, __ATOMIC_RELAXED);
        __atomic_add_fetch(&g_alloc_count, 1, __ATOMIC_RELAXED);
        size_t cur = __atomic_load_n(&g_total_allocated_bytes, __ATOMIC_RELAXED);
        size_t pk = __atomic_load_n(&g_peak_allocated_bytes, __ATOMIC_RELAXED);
        if (cur > pk) {
            __atomic_store_n(&g_peak_allocated_bytes, cur, __ATOMIC_RELAXED);
        }
        return user_from_header(hdr);
    }

    /* Slow path: refill */
    if (tl_cache.counts[cls] == 0) {
        refill_from_central(cls);
    }

    node = tl_cache.heads[cls];
    if (!node) {
        /* Still nothing: carve a new slab (this will also populate central) */
        (void)carve_new_slab(cls);
        refill_from_central(cls);
        node = tl_cache.heads[cls];
    }

    if (!node) {
        /* OOM fallback: try libc directly for this bin size */
        size_t bin = class_to_size(cls);
        void* raw = system_malloc(CML_HEADER_SIZE + bin);
        if (!raw) return NULL;
        AllocHeader* hdr = (AllocHeader*)raw;
        hdr->size = user_size;
        hdr->class_idx = (uint16_t)cls;
        hdr->magic = ALLOC_MAGIC;
        return user_from_header(hdr);
    }

    tl_cache.heads[cls] = node->next;
    tl_cache.counts[cls]--;

    AllocHeader* hdr = (AllocHeader*)((char*)node - CML_HEADER_SIZE);
    hdr->size = user_size;
    hdr->class_idx = (uint16_t)cls;
    hdr->magic = ALLOC_MAGIC;

    __atomic_add_fetch(&g_total_allocated_bytes, user_size, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_alloc_count, 1, __ATOMIC_RELAXED);
    size_t cur = __atomic_load_n(&g_total_allocated_bytes, __ATOMIC_RELAXED);
    size_t pk = __atomic_load_n(&g_peak_allocated_bytes, __ATOMIC_RELAXED);
    if (cur > pk) __atomic_store_n(&g_peak_allocated_bytes, cur, __ATOMIC_RELAXED);

    return user_from_header(hdr);
}

static void* alloc_large(size_t size) {
    ensure_init();
    /* Large: allocate with extra header using system backing (never our cml_*) */
    size_t total = CML_HEADER_SIZE + size;
    /* Force good alignment for large data (64B) */
    void* raw = NULL;
    if (system_posix_memalign(&raw, 64, total) != 0) {
        raw = system_malloc(total);
        if (!raw) return NULL;
    }
    AllocHeader* hdr = (AllocHeader*)raw;
    hdr->size = size;
    hdr->class_idx = 0xffff;
    hdr->magic = ALLOC_MAGIC;

    __atomic_add_fetch(&g_total_allocated_bytes, size, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_alloc_count, 1, __ATOMIC_RELAXED);
    size_t cur = __atomic_load_n(&g_total_allocated_bytes, __ATOMIC_RELAXED);
    size_t pk = __atomic_load_n(&g_peak_allocated_bytes, __ATOMIC_RELAXED);
    if (cur > pk) __atomic_store_n(&g_peak_allocated_bytes, cur, __ATOMIC_RELAXED);

    return user_from_header(hdr);
}

void* cml_malloc(size_t size) {
    if (size == 0) size = 1; /* classic */

    if (size >= CML_LARGE_THRESHOLD) {
        return alloc_large(size);
    }

    int cls = size_to_class(size);
    if (cls < 0) {
        return alloc_large(size);
    }
    return alloc_from_class(cls, size);
}

void* cml_calloc(size_t nmemb, size_t size) {
    size_t bytes;
    if (__builtin_mul_overflow(nmemb, size, &bytes)) return NULL;
    void* p = cml_malloc(bytes);
    if (p) {
        memset(p, 0, bytes);
    }
    return p;
}

void* cml_realloc(void* ptr, size_t new_size) {
    if (!ptr) return cml_malloc(new_size);
    if (new_size == 0) {
        cml_free(ptr);
        return NULL;
    }

    AllocHeader* hdr = header_from_user(ptr);
    if (!hdr || hdr->magic != ALLOC_MAGIC) {
        /* Not from us or corrupted. Fall back to libc behavior? */
        /* For safety in mixed world we could abort or delegate, but assume all go through us. */
        return NULL;
    }

    size_t old_size = hdr->size;
    if (new_size <= old_size) {
        /* Shrink: keep same block, just update header */
        hdr->size = new_size;
        /* Note: we do not give memory back to freelist for shrink here (common & fast) */
        __atomic_sub_fetch(&g_total_allocated_bytes, (old_size - new_size), __ATOMIC_RELAXED);
        return ptr;
    }

    /* Grow: allocate new + copy */
    void* newp = cml_malloc(new_size);
    if (!newp) return NULL;
    memcpy(newp, ptr, old_size < new_size ? old_size : new_size);
    cml_free(ptr);
    return newp;
}

static void free_to_class(int cls, void* user_ptr, size_t user_size) {
    (void)user_size;
    ensure_init();

    AllocHeader* hdr = header_from_user(user_ptr);
    if (!hdr || hdr->magic != ALLOC_MAGIC) return;

    /* Mark as freed for double-free detection (optional) */
    hdr->magic = 0xdead;

    FreeNode* node = (FreeNode*)user_ptr;  /* user_ptr is exactly after header */
    node->next = tl_cache.heads[cls];
    tl_cache.heads[cls] = node;
    tl_cache.counts[cls]++;

    __atomic_sub_fetch(&g_total_allocated_bytes, user_size, __ATOMIC_RELAXED);

    /* If local cache is fat, flush some back to central (keeps memory bounded per thread) */
    if (tl_cache.counts[cls] > CML_MAX_LOCAL_CACHE) {
        flush_local_to_central(cls, CML_MAX_LOCAL_CACHE / 2);
    }
}

static void free_large(void* ptr) {
    AllocHeader* hdr = header_from_user(ptr);
    if (!hdr || hdr->magic != ALLOC_MAGIC) return;
    size_t sz = hdr->size;
    hdr->magic = 0xdead;

    __atomic_sub_fetch(&g_total_allocated_bytes, sz, __ATOMIC_RELAXED);

    /* Use system free on the raw header start (we used posix_memalign or system_malloc) */
    system_free(hdr);
}

void cml_free(void* ptr) {
    if (!ptr) return;

    AllocHeader* hdr = header_from_user(ptr);
    if (!hdr || hdr->magic != ALLOC_MAGIC) {
        /* Foreign pointer: ignore or optionally call system free. For purity we ignore. */
        return;
    }

    int cls = hdr->class_idx;
    size_t sz = hdr->size;

    if (cls == 0xffff || sz >= CML_LARGE_THRESHOLD) {
        free_large(ptr);
        return;
    }

    free_to_class(cls, ptr, sz);
}

char* cml_strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* p = (char*)cml_malloc(len);
    if (p) memcpy(p, s, len);
    return p;
}

void* cml_aligned_alloc(size_t size, size_t alignment) {
    if (alignment < CML_MIN_ALIGN) alignment = CML_MIN_ALIGN;
    if ((alignment & (alignment - 1)) != 0) {
        /* not power of 2: normalize */
        alignment--;
        alignment |= alignment >> 1;
        alignment |= alignment >> 2;
        alignment |= alignment >> 4;
        alignment |= alignment >> 8;
        alignment |= alignment >> 16;
        alignment++;
    }

    /* We allocate extra space so we can store header + satisfy alignment */
    size_t header_and_pad = CML_HEADER_SIZE + alignment - 1;
    void* raw = cml_malloc(size + header_and_pad);
    if (!raw) return NULL;

    /* Find aligned user position after header */
    uintptr_t addr = (uintptr_t)raw + CML_HEADER_SIZE;
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);

    /* We need to store the original base (raw) so free can find the header.
     * To keep things simple we over-allocate in the header a "delta".
     * For speed we store the delta to the real header start in the first bytes after padding.
     * Simpler approach: use a slightly larger header area for aligned.
     *
     * For this impl we record the "backing header" by writing a small prefix right before the aligned user ptr.
     * We will store at (aligned - 8) the distance back to our AllocHeader.
     */
    ptrdiff_t delta = (ptrdiff_t)(aligned - (uintptr_t)raw);
    /* Store delta just before user data. We use 8 bytes for delta (fits in the alignment padding). */
    *((ptrdiff_t*)((char*)aligned - sizeof(ptrdiff_t))) = delta;

    /* Return the aligned address. The header lives at (aligned - delta) */
    return (void*)aligned;
}

void cml_aligned_free(void* ptr) {
    if (!ptr) return;
    /* Recover the original header using stored delta */
    ptrdiff_t delta = *((ptrdiff_t*)((char*)ptr - sizeof(ptrdiff_t)));
    void* real_user = (char*)ptr - delta;
    cml_free(real_user);
}

void cml_allocator_get_stats(size_t* bytes_allocated, size_t* peak_bytes, size_t* alloc_count) {
    if (bytes_allocated) *bytes_allocated = __atomic_load_n(&g_total_allocated_bytes, __ATOMIC_RELAXED);
    if (peak_bytes)      *peak_bytes      = __atomic_load_n(&g_peak_allocated_bytes, __ATOMIC_RELAXED);
    if (alloc_count)     *alloc_count     = __atomic_load_n(&g_alloc_count, __ATOMIC_RELAXED);
}

void cml_allocator_flush_thread_cache(void) {
    if (!tl_cache.initialized) return;
    for (int c = 0; c < NUM_SIZE_CLASSES; ++c) {
        if (tl_cache.counts[c] > 0) {
            flush_local_to_central(c, 0);
        }
    }
}
