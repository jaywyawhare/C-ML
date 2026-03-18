#ifndef CML_TLSF_ALLOC_H
#define CML_TLSF_ALLOC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** TLSF configuration */
#define TLSF_FL_INDEX_COUNT  32  /* First-level: log2 of size */
#define TLSF_SL_INDEX_COUNT  16  /* Second-level: subdivisions per FL */
#define TLSF_FL_INDEX_SHIFT  4   /* Minimum block size = 2^4 = 16 bytes */
#define TLSF_MIN_BLOCK_SIZE  16
#define TLSF_ALIGN           16  /* Allocation alignment */

/** Block header */
typedef struct TLSFBlock {
    struct TLSFBlock* prev_phys;  /* Previous physical block */
    size_t size;                   /* Block size (includes header) */
    struct TLSFBlock* next_free;   /* Next free block in segregated list */
    struct TLSFBlock* prev_free;   /* Previous free block */
} TLSFBlock;

/** TLSF allocator */
typedef struct CMLTLSFAllocator {
    /* First-level bitmap */
    uint32_t fl_bitmap;
    /* Second-level bitmaps (one per first-level index) */
    uint32_t sl_bitmap[TLSF_FL_INDEX_COUNT];
    /* Segregated free lists [fl][sl] */
    TLSFBlock* blocks[TLSF_FL_INDEX_COUNT][TLSF_SL_INDEX_COUNT];

    /* Pool management */
    void* pool;                /* Backing memory pool */
    size_t pool_size;          /* Total pool size */
    bool owns_pool;            /* Whether we allocated the pool */

    /* Statistics */
    size_t used_bytes;
    size_t peak_bytes;
    size_t num_allocs;
    size_t num_frees;
    size_t num_splits;
    size_t num_merges;
} CMLTLSFAllocator;

/** Timeline allocation record */
typedef struct {
    int tensor_id;
    size_t size;
    size_t offset;     /* Assigned offset in memory pool */
    int alloc_time;    /* Step when tensor is first needed */
    int free_time;     /* Step when tensor is last used */
} CMLTimelineRecord;

/** Timeline memory planner */
typedef struct CMLTimelinePlanner {
    CMLTimelineRecord* records;
    int num_records;
    int record_capacity;

    size_t total_required;     /* Minimum memory needed */
    size_t peak_usage;         /* Peak concurrent memory */
    int num_steps;             /* Total computation steps */
} CMLTimelinePlanner;

CMLTLSFAllocator* cml_tlsf_create(size_t pool_size);

CMLTLSFAllocator* cml_tlsf_create_with_pool(void* pool, size_t pool_size);

void cml_tlsf_destroy(CMLTLSFAllocator* alloc);

void* cml_tlsf_alloc(CMLTLSFAllocator* alloc, size_t size);

void* cml_tlsf_alloc_aligned(CMLTLSFAllocator* alloc, size_t size, size_t alignment);

void cml_tlsf_free(CMLTLSFAllocator* alloc, void* ptr);

void* cml_tlsf_realloc(CMLTLSFAllocator* alloc, void* ptr, size_t new_size);

size_t cml_tlsf_alloc_size(CMLTLSFAllocator* alloc, void* ptr);

void cml_tlsf_stats(const CMLTLSFAllocator* alloc, size_t* used, size_t* peak,
                     size_t* num_allocs, size_t* num_frees);

bool cml_tlsf_check(const CMLTLSFAllocator* alloc);

CMLTimelinePlanner* cml_timeline_planner_create(int initial_capacity);

void cml_timeline_planner_destroy(CMLTimelinePlanner* planner);

int cml_timeline_planner_add(CMLTimelinePlanner* planner, int tensor_id,
                              size_t size, int alloc_time, int free_time);

int cml_timeline_planner_solve(CMLTimelinePlanner* planner);

const CMLTimelineRecord* cml_timeline_planner_get(const CMLTimelinePlanner* planner, int tensor_id);

size_t cml_timeline_planner_total_memory(const CMLTimelinePlanner* planner);

size_t cml_timeline_planner_peak_usage(const CMLTimelinePlanner* planner);

void cml_timeline_planner_print(const CMLTimelinePlanner* planner);

#ifdef __cplusplus
}
#endif

#endif /* CML_TLSF_ALLOC_H */
