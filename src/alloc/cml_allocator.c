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
#include <stddef.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdio.h> /* only for optional stats print, remove if want zero dep */

/* System backing allocator for internal slab/large acquisition only.
 * We must NEVER call our own cml_* here for bootstrapping the allocator itself.
 * On Windows there is no posix_memalign and _aligned_malloc'd memory must be
 * freed with _aligned_free, so every system_* allocation goes through the
 * aligned pair to keep alloc/free consistent. */
#ifdef _WIN32
#include <malloc.h>
/** System backing malloc (Windows); 16-byte aligned so it pairs with system_free. */
static inline void* system_malloc(size_t sz) { return _aligned_malloc(sz ? sz : 1, 16); }
/** Release memory from any system_* allocation (Windows requires _aligned_free). */
static inline void system_free(void* p) { _aligned_free(p); }
/** Aligned system allocation (Windows); returns 0 or ENOMEM(12) like posix_memalign. */
static inline int system_posix_memalign(void** memptr, size_t alignment, size_t size) {
    void* q = _aligned_malloc(size ? size : 1, alignment);
    *memptr = q;
    return q ? 0 : 12; /* 12 == ENOMEM */
}
#else
/** System backing malloc for slab/large acquisition; never routes through cml_*. */
static inline void* system_malloc(size_t sz) { return malloc(sz); }
/** Release memory obtained from the system_* backing allocator. */
static inline void system_free(void* p) { free(p); }
/** Aligned system allocation; thin wrapper over posix_memalign. */
static inline int system_posix_memalign(void** memptr, size_t alignment, size_t size) {
    return posix_memalign(memptr, alignment, size);
}
#endif

/* Tunables for "fast as fuck" */
#define CML_SLAB_SIZE (256 * 1024) /* 256 KiB slabs - sweet spot for cache + TLB */
#define CML_MAX_LOCAL_CACHE 64 /* max free objects kept per class in TLS before flushing batch */
#define CML_LARGE_THRESHOLD (128 * 1024) /* >=128KiB: direct path */
#define CML_MIN_ALIGN 16
#define CML_HEADER_SIZE 16 /* 16B header => good default alignment for returned ptrs */

_Static_assert(CML_HEADER_SIZE >= 16 && (CML_HEADER_SIZE % 16) == 0,
               "header must preserve alignment");

/* Per-allocation header. Lives immediately before user pointer.
 * size is size_t so large tensors / buffers well beyond 4 GiB are representable. */
typedef struct {
    size_t size;        /* requested user bytes */
    uint16_t class_idx; /* which size class (0xffff for large/direct) */
    uint16_t magic;     /* 0xC4A1 for sanity */
} AllocHeader;

_Static_assert(sizeof(AllocHeader) <= CML_HEADER_SIZE, "AllocHeader must fit in CML_HEADER_SIZE");

#define ALLOC_MAGIC 0xC4A1

/** Recover the AllocHeader sitting CML_HEADER_SIZE bytes before a user pointer; NULL-safe. */
static inline AllocHeader* header_from_user(void* user) {
    if (!user)
        return NULL;
    return (AllocHeader*)((char*)user - CML_HEADER_SIZE);
}

/** User payload pointer for a header: the bytes immediately past the fixed-size header. */
static inline void* user_from_header(AllocHeader* h) { return (char*)h + CML_HEADER_SIZE; }

/* ---------------- Size classes ----------------
 * We use a compact table of increasing sizes. Good balance of internal fragmentation vs #classes.
 * Classes chosen so small objects (common for structs, nodes, small temps) have tight fits.
 */
static const size_t SIZE_CLASSES[] = {
    /* 0-15: 16B granularity for tiny */
    16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
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
    49152, 65536, 98304};
#define NUM_SIZE_CLASSES (sizeof(SIZE_CLASSES) / sizeof(SIZE_CLASSES[0]))

/** Map a byte size to its size-class index, or -1 when it exceeds the largest class
 *  (caller then takes the large/direct path). Linear scan, used only off the hot path. */
static inline int size_to_class(size_t size) {
    if (size <= SIZE_CLASSES[0])
        return 0;
    /* Linear scan is fine: NUM_SIZE_CLASSES ~ 65, called only on slow paths or first alloc per
     * size. */
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        if (size <= SIZE_CLASSES[i])
            return i;
    }
    return -1; /* large */
}

/** Byte size held by a size class, or 0 if the index is out of range. */
static inline size_t class_to_size(int cls) {
    if (cls < 0 || (size_t)cls >= NUM_SIZE_CLASSES)
        return 0;
    return SIZE_CLASSES[cls];
}

/* ---------------- Slab + block structures ---------------- */

typedef struct FreeNode {
    struct FreeNode* next;
} FreeNode;

typedef struct Slab {
    struct Slab* next;
    void* system_base; /* original pointer from system_malloc; used for system_free */
    uint32_t class_idx;
    uint32_t num_blocks;
    uint32_t used_blocks;
    /* Without this pad the fields above total 28 bytes, so data[] -- and hence
     * every AllocHeader and every user pointer carved from the slab -- landed
     * on a 4-mod-8 address. That is undefined behaviour on the header's size_t
     * and silently broke the alignment guarantee callers expect for double and
     * for SIMD loads. Every size class is a multiple of 16, so a 32-byte slab
     * header keeps every block 16-aligned. */
    uint32_t _pad;
    char data[]; /* flexible: the carved blocks start here */
} Slab;

_Static_assert(offsetof(Slab, data) % 16 == 0, "slab payload must start 16-byte aligned");

/* Track live slabs so we can eventually system_free their original base.
 * Currently slabs are process-lifetime (standard for this style of allocator),
 * but we must not lose the system_malloc pointer after alignment. */
static pthread_mutex_t g_slab_list_lock = PTHREAD_MUTEX_INITIALIZER;
static Slab* g_slab_list                = NULL;

/* One central freelist + mutex per size class */
typedef struct {
    pthread_mutex_t lock;
    FreeNode* head;
    size_t total_slabs; /* approx */
} CentralBin;

/* Thread-local cache */
typedef struct {
    FreeNode* heads[NUM_SIZE_CLASSES];
    uint16_t counts[NUM_SIZE_CLASSES];
    bool initialized;
} ThreadCache;

static CentralBin g_central[NUM_SIZE_CLASSES];
static pthread_mutex_t g_init_lock = PTHREAD_MUTEX_INITIALIZER;
static bool g_initialized          = false;

/* Thread local */
static __thread ThreadCache tl_cache = {0};

/* Stats (best effort, not perfectly accurate under races) */
static _Atomic size_t g_total_allocated_bytes = 0;
static _Atomic size_t g_peak_allocated_bytes  = 0;
static _Atomic size_t g_alloc_count           = 0;

/* Fault injection: -1 = disabled, >=0 = fail after this many more allocs */
static _Atomic long g_fault_countdown = -1;
/* Monotonic allocation index (for pinpointing the failing site) */
static _Atomic long g_alloc_index = 0;

/* Forward */
static void* alloc_from_class(int cls, size_t user_size);
static void free_to_class(int cls, void* user_ptr, size_t user_size);
static void* alloc_large(size_t size);
static void free_large(void* ptr);
static void init_once(void);
static Slab* carve_new_slab(int cls);
static void refill_from_central(int cls);
static void flush_local_to_central(int cls, int keep);

/* ---------------- Initialization ---------------- */

/** Initialize every central bin's mutex and empty its freelist; run once at startup. */
static void init_central_bins(void) {
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        pthread_mutex_init(&g_central[i].lock, NULL);
        g_central[i].head        = NULL;
        g_central[i].total_slabs = 0;
    }
}

/** Idempotent one-time init of the central bins, serialized by g_init_lock. */
static void init_once(void) {
    pthread_mutex_lock(&g_init_lock);
    if (!g_initialized) {
        init_central_bins();
        g_initialized = true;
    }
    pthread_mutex_unlock(&g_init_lock);
}

/** Ensure the global bins and this thread's TLS cache are initialized before use. */
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

/** Carve a fresh CML_SLAB_SIZE slab for class `cls` from the system allocator, thread the
 *  carved blocks onto the class's central freelist, and register the slab so its original
 *  system_base is never lost (slabs are process-lifetime today). Returns NULL on OOM. */
static Slab* carve_new_slab(int cls) {
    size_t bin_size = class_to_size(cls);
    if (bin_size == 0)
        return NULL;

    /* We allocate the slab header + data region with libc malloc.
     * This cost is amortized over hundreds of objects per slab.
     */
    size_t usable     = CML_SLAB_SIZE - sizeof(Slab);
    size_t num_blocks = usable / (CML_HEADER_SIZE + bin_size);
    if (num_blocks < 1) {
        /* Extremely large class inside "small" path - fall back */
        num_blocks = 1;
    }

    size_t alloc_sz = sizeof(Slab) + num_blocks * (CML_HEADER_SIZE + bin_size);
    /* Align the slab allocation a bit; keep the original system pointer so we can free it. */
    void* raw = system_malloc(alloc_sz + 64);
    if (!raw)
        return NULL;

    /* Align start of Slab structure for cleanliness */
    uintptr_t base         = (uintptr_t)raw;
    uintptr_t aligned_base = (base + 63) & ~(uintptr_t)63;
    Slab* slab             = (Slab*)aligned_base;

    slab->next        = NULL;
    slab->system_base = raw; /* MUST retain: system_free(raw), not system_free(slab) */
    slab->class_idx   = (uint32_t)cls;
    slab->num_blocks  = (uint32_t)num_blocks;
    slab->used_blocks = 0;

    /* Build intrusive free list for this slab directly into the central for this class */
    char* p         = (char*)slab->data;
    FreeNode* first = NULL;
    FreeNode* prev  = NULL;

    for (uint32_t i = 0; i < num_blocks; ++i) {
        AllocHeader* hdr = (AllocHeader*)p;
        hdr->size        = 0;
        hdr->class_idx   = (uint16_t)cls;
        hdr->magic       = ALLOC_MAGIC;

        FreeNode* node = (FreeNode*)(p + CML_HEADER_SIZE);
        node->next     = NULL;

        if (!first)
            first = node;
        if (prev)
            prev->next = node;
        prev = node;

        p += CML_HEADER_SIZE + bin_size;
    }

    /* Insert the whole chain into central under caller lock (or we can do it here) */
    if (first) {
        pthread_mutex_lock(&g_central[cls].lock);
        prev->next          = g_central[cls].head;
        g_central[cls].head = first;
        g_central[cls].total_slabs++;
        pthread_mutex_unlock(&g_central[cls].lock);
    }

    /* Register slab so its system_base is never lost (process-lifetime today). */
    pthread_mutex_lock(&g_slab_list_lock);
    slab->next  = g_slab_list;
    g_slab_list = slab;
    pthread_mutex_unlock(&g_slab_list_lock);

    return slab;
}

/* ---------------- Refill / flush ---------------- */

/** Move a batch of free blocks from the class's central freelist into this thread's cache,
 *  taking the central lock only for the splice. */
static void refill_from_central(int cls) {
    /* Steal a batch from central into thread local */
    const int want   = 32; /* batch size */
    FreeNode* stolen = NULL;
    int got          = 0;

    pthread_mutex_lock(&g_central[cls].lock);
    FreeNode* cur  = g_central[cls].head;
    FreeNode* prev = NULL;
    while (cur && got < want) {
        FreeNode* next = cur->next;
        /* unlink */
        if (prev)
            prev->next = next;
        else
            g_central[cls].head = next;
        cur->next = stolen;
        stolen    = cur;
        cur       = next;
        ++got;
    }
    pthread_mutex_unlock(&g_central[cls].lock);

    if (got > 0) {
        tl_cache.heads[cls]  = stolen;
        tl_cache.counts[cls] = (uint16_t)got;
    }
}

/** Return this thread's surplus blocks for `cls` to the central freelist, keeping at most
 *  `keep` cached locally so per-thread memory stays bounded. */
static void flush_local_to_central(int cls, int keep) {
    FreeNode* list = tl_cache.heads[cls];
    uint16_t cnt   = tl_cache.counts[cls];
    if (!list || cnt <= (uint16_t)keep)
        return;

    /* Detach the excess tail */
    FreeNode* keep_head = list;
    FreeNode* tail      = list;
    int keep_cnt        = 0;
    while (keep_cnt < keep && tail) {
        ++keep_cnt;
        if (keep_cnt < keep)
            tail = tail->next;
    }
    if (!tail) {
        /* nothing to flush */
        return;
    }
    FreeNode* flush_head = tail->next;
    tail->next           = NULL;
    tl_cache.heads[cls]  = keep_head;
    tl_cache.counts[cls] = (uint16_t)keep_cnt;

    if (flush_head) {
        /* Append flush list to central */
        pthread_mutex_lock(&g_central[cls].lock);
        /* Find end of flush list to splice */
        FreeNode* f = flush_head;
        while (f->next)
            f = f->next;
        f->next             = g_central[cls].head;
        g_central[cls].head = flush_head;
        pthread_mutex_unlock(&g_central[cls].lock);
    }
}

/* ---------------- Pool bypass (diagnostics) ----------------
 *
 * Build with -DCML_ALLOC_PASSTHROUGH=1 to route every allocation straight to
 * the system allocator. The pool hands out blocks from its own arenas, so a
 * heap tool sees one giant valid mapping and cannot tell a use-after-free or an
 * overflow from ordinary traffic -- corruption only surfaces later as a crash
 * inside alloc_from_class, walking a free list some earlier write clobbered.
 * With this on, AddressSanitizer sees each allocation individually and reports
 * the write that actually caused it. Diagnostics only: the pool is what makes
 * per-node allocation cheap. */
#if defined(CML_ALLOC_PASSTHROUGH) && CML_ALLOC_PASSTHROUGH
#include <stdlib.h>
#include <string.h>

/* Fault injection is mirrored here, not just in the pooled allocator. Omitting
 * it made sim_test fail to link under CML_ALLOC_PASSTHROUGH, which is exactly
 * the configuration ASAN needs -- so the one suite that exercises
 * allocation-failure paths was also the one suite ASAN never covered. Semantics
 * match the pooled path: -1 disables, >=0 fails after that many more allocs and
 * then re-disables itself. The counters themselves are declared unconditionally
 * near the top of the file, so this branch reuses them rather than shadowing. */
static int pt_fault_hit(void) {
    long cd = atomic_load_explicit(&g_fault_countdown, memory_order_relaxed);
    if (cd >= 0) {
        long prev = atomic_fetch_sub_explicit(&g_fault_countdown, 1, memory_order_relaxed);
        if (prev == 0) {
            atomic_store_explicit(&g_fault_countdown, -1, memory_order_relaxed);
            return 1;
        }
    }
    atomic_fetch_add_explicit(&g_alloc_index, 1, memory_order_relaxed);
    return 0;
}

/** Arm fault injection: the nth subsequent allocation returns NULL, then it disarms itself. */
void cml_malloc_fault_after(int n) {
    atomic_store_explicit(&g_fault_countdown, (long)n, memory_order_relaxed);
    atomic_store_explicit(&g_alloc_index, 0, memory_order_relaxed);
}
/** Disarm allocation fault injection so every allocation succeeds again. */
void cml_malloc_fault_reset(void) {
    atomic_store_explicit(&g_fault_countdown, -1L, memory_order_relaxed);
}
/** Allocations observed since the last fault_after, for pinpointing a failing call site. */
long cml_malloc_alloc_index(void) {
    return atomic_load_explicit(&g_alloc_index, memory_order_relaxed);
}

/** Passthrough cml_malloc: straight to libc malloc (0 becomes 1) so ASAN sees each block. */
void* cml_malloc(size_t size) {
    if (pt_fault_hit())
        return NULL;
    return malloc(size ? size : 1);
}
/** Passthrough cml_calloc: zeroing libc calloc, honoring the fault-injection hook. */
void* cml_calloc(size_t n, size_t sz) {
    if (pt_fault_hit())
        return NULL;
    return calloc(n ? n : 1, sz ? sz : 1);
}
/** Passthrough cml_realloc over libc realloc, honoring the fault-injection hook. */
void* cml_realloc(void* p, size_t n) {
    if (pt_fault_hit())
        return NULL;
    return realloc(p, n ? n : 1);
}
/** Passthrough cml_free: plain libc free. */
void cml_free(void* p) { free(p); }
/** Passthrough cml_strdup over libc strdup; NULL-safe and fault-injectable. */
char* cml_strdup(const char* s) {
    if (!s)
        return NULL;
    if (pt_fault_hit())
        return NULL;
    return strdup(s);
}
/** Passthrough aligned allocation via posix_memalign; release with cml_aligned_free. */
void* cml_aligned_alloc(size_t size, size_t al) {
    void* p = NULL;
    if (al < sizeof(void*))
        al = sizeof(void*);
    if (posix_memalign(&p, al, size ? size : 1) != 0)
        return NULL;
    return p;
}
/** Passthrough free for cml_aligned_alloc memory (plain libc free). */
void cml_aligned_free(void* p) { free(p); }
#else

/* ---------------- Allocation paths ---------------- */

/** Allocate one object of size class `cls` for a `user_size` request. Pops the thread-local
 *  cache first (lockless fast path), else refills from central / carves a slab, and falls
 *  back to a lone system_malloc block on OOM. Stamps the header and bumps stats. */
static void* alloc_from_class(int cls, size_t user_size) {
    ensure_init();

    /* Fast path: thread local */
    FreeNode* node = tl_cache.heads[cls];
    if (node) {
        tl_cache.heads[cls] = node->next;
        tl_cache.counts[cls]--;
        AllocHeader* hdr = (AllocHeader*)((char*)node - CML_HEADER_SIZE);
        hdr->size        = user_size;
        hdr->class_idx   = (uint16_t)cls;
        hdr->magic       = ALLOC_MAGIC;

        /* update stats */
        atomic_fetch_add_explicit(&g_total_allocated_bytes, user_size, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_alloc_count, 1, memory_order_relaxed);
        size_t cur = atomic_load_explicit(&g_total_allocated_bytes, memory_order_relaxed);
        size_t pk  = atomic_load_explicit(&g_peak_allocated_bytes, memory_order_relaxed);
        if (cur > pk) {
            atomic_store_explicit(&g_peak_allocated_bytes, cur, memory_order_relaxed);
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
        void* raw  = system_malloc(CML_HEADER_SIZE + bin);
        if (!raw)
            return NULL;
        AllocHeader* hdr = (AllocHeader*)raw;
        hdr->size        = user_size;
        hdr->class_idx   = (uint16_t)cls;
        hdr->magic       = ALLOC_MAGIC;
        return user_from_header(hdr);
    }

    tl_cache.heads[cls] = node->next;
    tl_cache.counts[cls]--;

    AllocHeader* hdr = (AllocHeader*)((char*)node - CML_HEADER_SIZE);
    hdr->size        = user_size;
    hdr->class_idx   = (uint16_t)cls;
    hdr->magic       = ALLOC_MAGIC;

    atomic_fetch_add_explicit(&g_total_allocated_bytes, user_size, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_alloc_count, 1, memory_order_relaxed);
    size_t cur = atomic_load_explicit(&g_total_allocated_bytes, memory_order_relaxed);
    size_t pk  = atomic_load_explicit(&g_peak_allocated_bytes, memory_order_relaxed);
    if (cur > pk)
        atomic_store_explicit(&g_peak_allocated_bytes, cur, memory_order_relaxed);

    return user_from_header(hdr);
}

/* Large-block cache. Large allocations otherwise hit the system allocator on
 * every alloc and free; ML workloads re-allocate the same big activation and
 * gradient buffers every step, so that round-trip is pure overhead. Keep a small
 * bounded LIFO of freed large blocks and reuse one whose backing is big enough
 * (but not wildly oversized, to limit waste). Overflow goes back to the system.
 * Entries stay reachable through this global, so they are not leaks at exit. */
#define CML_LARGE_CACHE_MAX 64
#define CML_LARGE_CACHE_BUDGET ((size_t)256 * 1024 * 1024)

typedef struct {
    void* raw;    /* header start, as returned by the system allocator */
    size_t total; /* CML_HEADER_SIZE + requested bytes when it was cached */
} LargeCacheEntry;

static LargeCacheEntry g_large_cache[CML_LARGE_CACHE_MAX];
static int g_large_cache_count            = 0;
static size_t g_large_cache_bytes         = 0;
static pthread_mutex_t g_large_cache_lock = PTHREAD_MUTEX_INITIALIZER;

/** Take a cached block with `need <= total <= 2*need` (tightest fit), or NULL.
 *  The 2x cap keeps a large block from being wasted on a much smaller request. */
static void* large_cache_take(size_t need) {
    void* found = NULL;
    pthread_mutex_lock(&g_large_cache_lock);
    int best = -1;
    for (int i = 0; i < g_large_cache_count; i++) {
        size_t t = g_large_cache[i].total;
        if (t >= need && t <= need * 2 && (best < 0 || t < g_large_cache[best].total))
            best = i;
    }
    if (best >= 0) {
        found = g_large_cache[best].raw;
        g_large_cache_bytes -= g_large_cache[best].total;
        g_large_cache[best] = g_large_cache[--g_large_cache_count]; /* swap-remove */
    }
    pthread_mutex_unlock(&g_large_cache_lock);
    return found;
}

/** Keep a freed large block for reuse; returns false (caller frees it) when the
 *  cache is at its count or byte budget. */
static bool large_cache_put(void* raw, size_t total) {
    bool kept = false;
    pthread_mutex_lock(&g_large_cache_lock);
    if (g_large_cache_count < CML_LARGE_CACHE_MAX &&
        g_large_cache_bytes + total <= CML_LARGE_CACHE_BUDGET) {
        g_large_cache[g_large_cache_count].raw   = raw;
        g_large_cache[g_large_cache_count].total = total;
        g_large_cache_count++;
        g_large_cache_bytes += total;
        kept = true;
    }
    pthread_mutex_unlock(&g_large_cache_lock);
    return kept;
}

/** Direct path for large requests: a single 64-byte-aligned system allocation carrying the
 *  header, tagged with class_idx 0xffff so cml_free routes it back to free_large. */
static void* alloc_large(size_t size) {
    ensure_init();
    /* Large: allocate with extra header using system backing (never our cml_*) */
    size_t total = CML_HEADER_SIZE + size;
    /* Reuse a cached block before touching the system allocator. A cached block
     * is at least `total` bytes and keeps its original >=16B alignment. */
    void* raw = large_cache_take(total);
    /* Force good alignment for large data (64B) */
    if (!raw && system_posix_memalign(&raw, 64, total) != 0) {
        raw = system_malloc(total);
        if (!raw)
            return NULL;
    }
    AllocHeader* hdr = (AllocHeader*)raw;
    hdr->size        = size;
    hdr->class_idx   = 0xffff;
    hdr->magic       = ALLOC_MAGIC;

    atomic_fetch_add_explicit(&g_total_allocated_bytes, size, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_alloc_count, 1, memory_order_relaxed);
    size_t cur = atomic_load_explicit(&g_total_allocated_bytes, memory_order_relaxed);
    size_t pk  = atomic_load_explicit(&g_peak_allocated_bytes, memory_order_relaxed);
    if (cur > pk)
        atomic_store_explicit(&g_peak_allocated_bytes, cur, memory_order_relaxed);

    return user_from_header(hdr);
}

/** Primary allocation entry point: 0 is treated as 1, requests >= CML_LARGE_THRESHOLD (or
 *  beyond the largest size class) take the large/direct path, the rest go to a size class.
 *  Also drives fault injection when armed. Returns NULL on failure. */
void* cml_malloc(size_t size) {
    if (size == 0)
        size = 1; /* classic */

    /* Fault injection: if a countdown is active, decrement it and fail when it hits 0. */
    long cd = atomic_load_explicit(&g_fault_countdown, memory_order_relaxed);
    if (cd >= 0) {
        long prev = atomic_fetch_sub_explicit(&g_fault_countdown, 1, memory_order_relaxed);
        if (prev == 0) {
            /* Reset to disabled so subsequent calls succeed, then return OOM. */
            atomic_store_explicit(&g_fault_countdown, -1, memory_order_relaxed);
            return NULL;
        }
    }
    atomic_fetch_add_explicit(&g_alloc_index, 1, memory_order_relaxed);

    if (size >= CML_LARGE_THRESHOLD) {
        return alloc_large(size);
    }

    int cls = size_to_class(size);
    if (cls < 0) {
        return alloc_large(size);
    }
    return alloc_from_class(cls, size);
}

/** Zero-initialized allocation; returns NULL if nmemb*size overflows size_t. */
void* cml_calloc(size_t nmemb, size_t size) {
    size_t bytes;
    if (__builtin_mul_overflow(nmemb, size, &bytes))
        return NULL;
    void* p = cml_malloc(bytes);
    if (p) {
        memset(p, 0, bytes);
    }
    return p;
}

/** Resize a block: NULL ptr acts as cml_malloc, size 0 frees and returns NULL. Shrinks in
 *  place (no return to the freelist); grows by allocate-copy-free. Returns NULL and leaves
 *  the original block intact on a foreign/corrupt header or allocation failure. */
void* cml_realloc(void* ptr, size_t new_size) {
    if (!ptr)
        return cml_malloc(new_size);
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
        atomic_fetch_sub_explicit(&g_total_allocated_bytes, (old_size - new_size),
                                  memory_order_relaxed);
        return ptr;
    }

    /* Grow: allocate new + copy */
    void* newp = cml_malloc(new_size);
    if (!newp)
        return NULL;
    memcpy(newp, ptr, old_size < new_size ? old_size : new_size);
    cml_free(ptr);
    return newp;
}

/** Return a small block to its thread-local cache, poisoning the magic to catch double-frees
 *  and flushing half the cache to central once it grows past CML_MAX_LOCAL_CACHE. */
static void free_to_class(int cls, void* user_ptr, size_t user_size) {
    (void)user_size;
    ensure_init();

    AllocHeader* hdr = header_from_user(user_ptr);
    if (!hdr || hdr->magic != ALLOC_MAGIC)
        return;

    /* Mark as freed for double-free detection (optional) */
    hdr->magic = 0xdead;

    FreeNode* node      = (FreeNode*)user_ptr; /* user_ptr is exactly after header */
    node->next          = tl_cache.heads[cls];
    tl_cache.heads[cls] = node;
    tl_cache.counts[cls]++;

    atomic_fetch_sub_explicit(&g_total_allocated_bytes, user_size, memory_order_relaxed);

    /* If local cache is fat, flush some back to central (keeps memory bounded per thread) */
    if (tl_cache.counts[cls] > CML_MAX_LOCAL_CACHE) {
        flush_local_to_central(cls, CML_MAX_LOCAL_CACHE / 2);
    }
}

/** Release a large/direct allocation back to the system allocator via its header start;
 *  no-op on a foreign or corrupt header. */
static void free_large(void* ptr) {
    AllocHeader* hdr = header_from_user(ptr);
    if (!hdr || hdr->magic != ALLOC_MAGIC)
        return;
    size_t sz  = hdr->size;
    hdr->magic = 0xdead;

    atomic_fetch_sub_explicit(&g_total_allocated_bytes, sz, memory_order_relaxed);

    /* Offer the block to the cache for reuse; only truly free it when the cache
     * is full. The raw header start is what the system allocator handed out. */
    if (!large_cache_put(hdr, CML_HEADER_SIZE + sz))
        system_free(hdr);
}

/** Free a cml_* allocation, dispatching to the large or size-class path by its header.
 *  NULL is ignored, as is any pointer whose header magic does not match (foreign pointer). */
void cml_free(void* ptr) {
    if (!ptr)
        return;

    AllocHeader* hdr = header_from_user(ptr);
    if (!hdr || hdr->magic != ALLOC_MAGIC) {
        /* Foreign pointer: ignore or optionally call system free. For purity we ignore. */
        return;
    }

    int cls   = hdr->class_idx;
    size_t sz = hdr->size;

    if (cls == 0xffff || sz >= CML_LARGE_THRESHOLD) {
        free_large(ptr);
        return;
    }

    free_to_class(cls, ptr, sz);
}

/** Duplicate a C string into a cml_malloc'd buffer the caller must cml_free; NULL in, NULL out. */
char* cml_strdup(const char* s) {
    if (!s)
        return NULL;
    size_t len = strlen(s) + 1;
    char* p    = (char*)cml_malloc(len);
    if (p)
        memcpy(p, s, len);
    return p;
}

/** Allocate `size` bytes aligned to `alignment` (rounded up to a power of two, min
 *  CML_MIN_ALIGN). Over-allocates and stores the offset back to the real block just before
 *  the returned pointer; it MUST be released with cml_aligned_free, not cml_free. */
void* cml_aligned_alloc(size_t size, size_t alignment) {
    if (alignment < CML_MIN_ALIGN)
        alignment = CML_MIN_ALIGN;
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
    void* raw             = cml_malloc(size + header_and_pad);
    if (!raw)
        return NULL;

    /* Find aligned user position after header */
    uintptr_t addr    = (uintptr_t)raw + CML_HEADER_SIZE;
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);

    /* We need to store the original base (raw) so free can find the header.
     * To keep things simple we over-allocate in the header a "delta".
     * For speed we store the delta to the real header start in the first bytes after padding.
     * Simpler approach: use a slightly larger header area for aligned.
     *
     * For this impl we record the "backing header" by writing a small prefix right before the
     * aligned user ptr. We will store at (aligned - 8) the distance back to our AllocHeader.
     */
    ptrdiff_t delta = (ptrdiff_t)(aligned - (uintptr_t)raw);
    /* Store delta just before user data. We use 8 bytes for delta (fits in the alignment padding).
     */
    *((ptrdiff_t*)((char*)aligned - sizeof(ptrdiff_t))) = delta;

    /* Return the aligned address. The header lives at (aligned - delta) */
    return (void*)aligned;
}

/** Free memory from cml_aligned_alloc by recovering the real block via the stored offset. */
void cml_aligned_free(void* ptr) {
    if (!ptr)
        return;
    /* Recover the original header using stored delta */
    ptrdiff_t delta = *((ptrdiff_t*)((char*)ptr - sizeof(ptrdiff_t)));
    void* real_user = (char*)ptr - delta;
    cml_free(real_user);
}

/** Snapshot the best-effort allocator counters into any non-NULL out params (racy, not exact). */
void cml_allocator_get_stats(size_t* bytes_allocated, size_t* peak_bytes, size_t* alloc_count) {
    if (bytes_allocated)
        *bytes_allocated = atomic_load_explicit(&g_total_allocated_bytes, memory_order_relaxed);
    if (peak_bytes)
        *peak_bytes = atomic_load_explicit(&g_peak_allocated_bytes, memory_order_relaxed);
    if (alloc_count)
        *alloc_count = atomic_load_explicit(&g_alloc_count, memory_order_relaxed);
}

/** Return every block in the calling thread's cache to the central freelists; call before a
 *  thread exits so its cached memory is reusable by others. */
void cml_allocator_flush_thread_cache(void) {
    if (!tl_cache.initialized)
        return;
    for (size_t c = 0; c < NUM_SIZE_CLASSES; ++c) {
        if (tl_cache.counts[c] > 0) {
            flush_local_to_central(c, 0);
        }
    }
}

/* --- Fault injection API --- */

/** Arm fault injection: the nth subsequent cml_malloc returns NULL, then it disarms itself. */
void cml_malloc_fault_after(int n) {
    atomic_store_explicit(&g_fault_countdown, (long)n, memory_order_relaxed);
    atomic_store_explicit(&g_alloc_index, 0, memory_order_relaxed);
}

/** Disarm allocation fault injection so every allocation succeeds again. */
void cml_malloc_fault_reset(void) {
    atomic_store_explicit(&g_fault_countdown, -1L, memory_order_relaxed);
}

/** Allocations observed since the last fault_after, for pinpointing a failing call site. */
long cml_malloc_alloc_index(void) {
    return atomic_load_explicit(&g_alloc_index, memory_order_relaxed);
}

#endif /* CML_ALLOC_PASSTHROUGH */
