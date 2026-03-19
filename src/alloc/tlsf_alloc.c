/*
 * TLSF (Two-Level Segregated Fit) memory allocator with timeline planning.
 *
 * O(1) alloc/free with bounded fragmentation. The two-level segregated free
 * list maps each free block into FL = floor(log2(size)) and SL = subdivision
 * within that power-of-two range.
 *
 * Memory layout: [Block Header | User Data] ... [Sentinel Header]
 * Block headers store prev_phys pointer and size (flag bits in 2 LSBs).
 * Free blocks store next_free/prev_free pointers in the user data area.
 *
 * Timeline planner uses greedy first-fit-by-offset to pack tensor lifetimes
 * into the smallest contiguous memory region.
 */

#include "alloc/tlsf_alloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int tlsf_fls(uint32_t x)
{
    if (x == 0) return -1;
    return 31 - __builtin_clz(x);
}

static inline int tlsf_ffs(uint32_t x)
{
    if (x == 0) return -1;
    return __builtin_ctz(x);
}



#define BLOCK_HEADER_SIZE  ((size_t)offsetof(TLSFBlock, next_free))
#define MIN_USER_SIZE      (sizeof(void*) * 2)  /* Space for next_free + prev_free */
#define MIN_BLOCK_TOTAL    (BLOCK_HEADER_SIZE + MIN_USER_SIZE)

/* Flags stored in the 2 LSBs of the size field */
#define FLAG_FREE     ((size_t)1)
#define FLAG_PREVFREE ((size_t)2)
#define FLAG_MASK     (FLAG_FREE | FLAG_PREVFREE)

static inline size_t block_get_size(const TLSFBlock* b)
{
    return b->size & ~FLAG_MASK;
}

static inline void block_set_size(TLSFBlock* b, size_t sz)
{
    b->size = sz | (b->size & FLAG_MASK);
}

static inline bool block_is_free(const TLSFBlock* b)
{
    return (b->size & FLAG_FREE) != 0;
}

static inline void block_mark_free(TLSFBlock* b)
{
    b->size |= FLAG_FREE;
}

static inline void block_mark_used(TLSFBlock* b)
{
    b->size &= ~FLAG_FREE;
}

static inline bool block_prev_is_free(const TLSFBlock* b)
{
    return (b->size & FLAG_PREVFREE) != 0;
}

static inline void block_set_prev_free(TLSFBlock* b)
{
    b->size |= FLAG_PREVFREE;
}

static inline void block_set_prev_used(TLSFBlock* b)
{
    b->size &= ~FLAG_PREVFREE;
}

static inline void* block_to_user(TLSFBlock* b)
{
    return (char*)b + BLOCK_HEADER_SIZE;
}

static inline TLSFBlock* user_to_block(void* ptr)
{
    return (TLSFBlock*)((char*)ptr - BLOCK_HEADER_SIZE);
}

static inline TLSFBlock* block_next_phys(TLSFBlock* b)
{
    return (TLSFBlock*)((char*)b + BLOCK_HEADER_SIZE + block_get_size(b));
}

static inline bool block_in_pool(const CMLTLSFAllocator* a, const TLSFBlock* b)
{
    return (const char*)b >= (const char*)a->pool &&
           (const char*)b < (const char*)a->pool + a->pool_size;
}

static inline size_t align_up(size_t x, size_t a)
{
    return (x + a - 1) & ~(a - 1);
}

static void mapping_insert(size_t size, int* fl, int* sl)
{
    int f;
    if (size < (size_t)(1 << (TLSF_FL_INDEX_SHIFT + 1))) {
        f = TLSF_FL_INDEX_SHIFT;
    } else {
        f = tlsf_fls((uint32_t)size);
    }
    int s = (int)((size >> (f > 4 ? f - 4 : 0)) & (TLSF_SL_INDEX_COUNT - 1));

    if (f >= TLSF_FL_INDEX_COUNT) f = TLSF_FL_INDEX_COUNT - 1;
    if (s >= TLSF_SL_INDEX_COUNT) s = TLSF_SL_INDEX_COUNT - 1;

    *fl = f;
    *sl = s;
}

static void mapping_search(size_t size, int* fl, int* sl)
{
    if (size >= (size_t)(1 << (TLSF_FL_INDEX_SHIFT + 1))) {
        int f = tlsf_fls((uint32_t)size);
        size_t round = ((size_t)1 << (f > 4 ? f - 4 : 0)) - 1;
        size += round;
    }
    mapping_insert(size, fl, sl);
}

static void freelist_insert(CMLTLSFAllocator* a, TLSFBlock* block)
{
    int fl, sl;
    mapping_insert(block_get_size(block), &fl, &sl);

    TLSFBlock* head = a->blocks[fl][sl];
    block->next_free = head;
    block->prev_free = NULL;
    if (head) head->prev_free = block;
    a->blocks[fl][sl] = block;

    a->fl_bitmap |= (1U << fl);
    a->sl_bitmap[fl] |= (1U << sl);

    block_mark_free(block);
}

static void freelist_remove(CMLTLSFAllocator* a, TLSFBlock* block)
{
    int fl, sl;
    mapping_insert(block_get_size(block), &fl, &sl);

    if (block->prev_free) {
        block->prev_free->next_free = block->next_free;
    } else {
        a->blocks[fl][sl] = block->next_free;
    }
    if (block->next_free) {
        block->next_free->prev_free = block->prev_free;
    }

    if (a->blocks[fl][sl] == NULL) {
        a->sl_bitmap[fl] &= ~(1U << sl);
        if (a->sl_bitmap[fl] == 0) {
            a->fl_bitmap &= ~(1U << fl);
        }
    }

    block->next_free = NULL;
    block->prev_free = NULL;
}


static TLSFBlock* try_split(CMLTLSFAllocator* a, TLSFBlock* block, size_t wanted)
{
    size_t cur = block_get_size(block);
    if (cur < wanted + MIN_BLOCK_TOTAL) {
        /* Not enough room for a remainder block */
        return NULL;
    }

    size_t remain_user = cur - wanted - BLOCK_HEADER_SIZE;

    /* Shrink original */
    block_set_size(block, wanted);

    /* Create remainder block right after */
    TLSFBlock* rest = block_next_phys(block);
    rest->prev_phys = block;
    rest->size = 0;
    block_set_size(rest, remain_user);
    rest->next_free = NULL;
    rest->prev_free = NULL;

    /* Fix up the block after 'rest' so its prev_phys points to rest */
    TLSFBlock* after = block_next_phys(rest);
    if (block_in_pool(a, after)) {
        after->prev_phys = rest;
    }

    a->num_splits++;
    return rest;
}

static TLSFBlock* merge_prev(CMLTLSFAllocator* a, TLSFBlock* block)
{
    if (!block_prev_is_free(block)) return block;
    TLSFBlock* prev = block->prev_phys;
    if (!prev || !block_is_free(prev)) return block;

    freelist_remove(a, prev);

    size_t new_size = block_get_size(prev) + BLOCK_HEADER_SIZE + block_get_size(block);
    block_set_size(prev, new_size);

    TLSFBlock* after = block_next_phys(prev);
    if (block_in_pool(a, after)) {
        after->prev_phys = prev;
    }

    a->num_merges++;
    return prev;
}

static TLSFBlock* merge_next(CMLTLSFAllocator* a, TLSFBlock* block)
{
    TLSFBlock* next = block_next_phys(block);
    if (!block_in_pool(a, next)) return block;
    if (!block_is_free(next)) return block;

    freelist_remove(a, next);

    size_t new_size = block_get_size(block) + BLOCK_HEADER_SIZE + block_get_size(next);
    block_set_size(block, new_size);

    TLSFBlock* after = block_next_phys(block);
    if (block_in_pool(a, after)) {
        after->prev_phys = block;
    }

    a->num_merges++;
    return block;
}

static TLSFBlock* find_suitable(CMLTLSFAllocator* a, size_t size)
{
    int fl, sl;
    mapping_search(size, &fl, &sl);

    /* Try to find a block at [fl][sl] or a higher SL at the same FL */
    uint32_t sl_map = a->sl_bitmap[fl] & (~0U << sl);
    if (sl_map == 0) {
        /* Nothing at this FL level; try higher FL */
        uint32_t fl_map = a->fl_bitmap & (~0U << (fl + 1));
        if (fl_map == 0) return NULL;
        fl = tlsf_ffs(fl_map);
        sl_map = a->sl_bitmap[fl];
        if (sl_map == 0) return NULL;
    }
    sl = tlsf_ffs(sl_map);

    TLSFBlock* block = a->blocks[fl][sl];
    /* Walk the list to find one that actually fits */
    while (block && block_get_size(block) < size) {
        block = block->next_free;
    }
    return block;
}


static int init_pool(CMLTLSFAllocator* a)
{
    /* We need at least: one real block (header + min user) + sentinel (header only) */
    if (a->pool_size < MIN_BLOCK_TOTAL + BLOCK_HEADER_SIZE) {
        return -1;
    }

    memset(a->pool, 0, a->pool_size);

    /* Initialize bitmaps and free lists */
    a->fl_bitmap = 0;
    for (int i = 0; i < TLSF_FL_INDEX_COUNT; i++) {
        a->sl_bitmap[i] = 0;
        for (int j = 0; j < TLSF_SL_INDEX_COUNT; j++) {
            a->blocks[i][j] = NULL;
        }
    }

    /* The sentinel takes BLOCK_HEADER_SIZE bytes at the end of the pool. */
    size_t user_size = a->pool_size - BLOCK_HEADER_SIZE - BLOCK_HEADER_SIZE;

    /* First (and only) free block */
    TLSFBlock* first = (TLSFBlock*)a->pool;
    first->prev_phys = NULL;
    first->size = 0;
    block_set_size(first, user_size);
    first->next_free = NULL;
    first->prev_free = NULL;

    /* Sentinel block: zero-size, marked used, sits right after the first block */
    TLSFBlock* sentinel = block_next_phys(first);
    sentinel->prev_phys = first;
    sentinel->size = 0; /* Zero size, used (FLAG_FREE not set) */
    /* Mark sentinel as having a free previous block */
    /* (will be updated properly by freelist_insert) */

    /* Insert the first block into the free list */
    freelist_insert(a, first);

    /* Mark sentinel's prev_free flag since first block is now free */
    block_set_prev_free(sentinel);

    return 0;
}

CMLTLSFAllocator* cml_tlsf_create(size_t pool_size)
{
    if (pool_size < MIN_BLOCK_TOTAL + BLOCK_HEADER_SIZE) {
        return NULL;
    }

    /* Allocate metadata and pool separately to avoid confusion with
     * malloc's bookkeeping (the pool contains our own block headers
     * which could alias malloc metadata if placed in the same allocation). */
    CMLTLSFAllocator* a = (CMLTLSFAllocator*)calloc(1, sizeof(CMLTLSFAllocator));
    if (!a) return NULL;

    a->pool = malloc(pool_size);
    if (!a->pool) {
        free(a);
        return NULL;
    }
    a->pool_size = pool_size;
    a->owns_pool = true;

    if (init_pool(a) != 0) {
        free(a->pool);
        free(a);
        return NULL;
    }

    return a;
}

CMLTLSFAllocator* cml_tlsf_create_with_pool(void* pool, size_t pool_size)
{
    if (!pool || pool_size < MIN_BLOCK_TOTAL + BLOCK_HEADER_SIZE) {
        return NULL;
    }

    CMLTLSFAllocator* a = (CMLTLSFAllocator*)calloc(1, sizeof(CMLTLSFAllocator));
    if (!a) return NULL;

    a->pool = pool;
    a->pool_size = pool_size;
    a->owns_pool = false;

    if (init_pool(a) != 0) {
        free(a);
        return NULL;
    }

    return a;
}

void cml_tlsf_destroy(CMLTLSFAllocator* a)
{
    if (!a) return;
    if (a->owns_pool) {
        free(a->pool);
    }
    free(a);
}

void* cml_tlsf_alloc(CMLTLSFAllocator* a, size_t size)
{
    if (!a || size == 0) return NULL;

    /* Round up and enforce minimum */
    size = align_up(size, TLSF_ALIGN);
    if (size < MIN_USER_SIZE) size = MIN_USER_SIZE;

    TLSFBlock* block = find_suitable(a, size);
    if (!block) return NULL;

    freelist_remove(a, block);

    /* Split off excess */
    TLSFBlock* rest = try_split(a, block, size);
    if (rest) {
        /* rest is a new block, mark its prev (block) as used */
        block_set_prev_used(rest);
        freelist_insert(a, rest);
        /* The block after rest needs to know rest is free */
        TLSFBlock* after_rest = block_next_phys(rest);
        if (block_in_pool(a, after_rest)) {
            block_set_prev_free(after_rest);
        }
    }

    block_mark_used(block);

    /* Next physical block: mark prev as used */
    TLSFBlock* next = block_next_phys(block);
    if (block_in_pool(a, next)) {
        block_set_prev_used(next);
    }

    /* Stats */
    size_t alloc_sz = block_get_size(block);
    a->used_bytes += alloc_sz;
    if (a->used_bytes > a->peak_bytes) a->peak_bytes = a->used_bytes;
    a->num_allocs++;

    return block_to_user(block);
}

void* cml_tlsf_alloc_aligned(CMLTLSFAllocator* a, size_t size, size_t alignment)
{
    if (!a || size == 0) return NULL;

    /* Ensure alignment is at least TLSF_ALIGN and a power of two */
    if (alignment < TLSF_ALIGN) alignment = TLSF_ALIGN;
    if ((alignment & (alignment - 1)) != 0) {
        alignment--;
        alignment |= alignment >> 1;
        alignment |= alignment >> 2;
        alignment |= alignment >> 4;
        alignment |= alignment >> 8;
        alignment |= alignment >> 16;
        alignment++;
    }

    size = align_up(size, TLSF_ALIGN);
    if (size < MIN_USER_SIZE) size = MIN_USER_SIZE;

    /* Over-allocate to guarantee alignment.
     * Worst case: we need (alignment - 1) extra bytes for alignment
     * plus space for a gap block header. */
    size_t padded = size + alignment + MIN_BLOCK_TOTAL;
    padded = align_up(padded, TLSF_ALIGN);

    TLSFBlock* block = find_suitable(a, padded);
    if (!block) return NULL;

    freelist_remove(a, block);

    void* user_ptr = block_to_user(block);
    uintptr_t user_addr = (uintptr_t)user_ptr;

    if ((user_addr & (alignment - 1)) != 0) {
        /* Need to align forward.
         * Find the next aligned address that leaves enough room for a gap block. */
        uintptr_t base = (uintptr_t)block;
        uintptr_t aligned_user = align_up(user_addr, alignment);

        /* Make sure there's room for a gap block between block and aligned_user */
        while (aligned_user - base < MIN_BLOCK_TOTAL + BLOCK_HEADER_SIZE) {
            aligned_user += alignment;
        }

        /* The aligned block header sits BLOCK_HEADER_SIZE before aligned_user */
        TLSFBlock* aligned_block = (TLSFBlock*)(aligned_user - BLOCK_HEADER_SIZE);

        size_t gap_bytes = (uintptr_t)aligned_block - (uintptr_t)block;
        size_t gap_user = gap_bytes - BLOCK_HEADER_SIZE;
        size_t orig_size = block_get_size(block);
        size_t aligned_user_size = orig_size - gap_bytes;

        /* Set up the gap block (the original block shrinks) */
        block_set_size(block, gap_user);

        /* Set up the aligned block */
        aligned_block->prev_phys = block;
        aligned_block->size = 0;
        block_set_size(aligned_block, aligned_user_size);
        aligned_block->next_free = NULL;
        aligned_block->prev_free = NULL;

        /* Insert the gap block back as free */
        freelist_insert(a, block);
        block_set_prev_free(aligned_block);

        block = aligned_block;
        /* Note: don't update after-block's prev_phys yet; try_split will handle it */
    }

    /* Split off any excess at the end */
    TLSFBlock* rest = try_split(a, block, size);
    if (rest) {
        block_set_prev_used(rest);
        freelist_insert(a, rest);
        TLSFBlock* after_rest = block_next_phys(rest);
        if (block_in_pool(a, after_rest)) {
            block_set_prev_free(after_rest);
        }
    }

    block_mark_used(block);
    TLSFBlock* next = block_next_phys(block);
    if (block_in_pool(a, next)) {
        block_set_prev_used(next);
    }

    size_t alloc_sz = block_get_size(block);
    a->used_bytes += alloc_sz;
    if (a->used_bytes > a->peak_bytes) a->peak_bytes = a->used_bytes;
    a->num_allocs++;

    return block_to_user(block);
}

void cml_tlsf_free(CMLTLSFAllocator* a, void* ptr)
{
    if (!a || !ptr) return;

    TLSFBlock* block = user_to_block(ptr);
    if (block_is_free(block)) return; /* Double-free guard */

    size_t freed = block_get_size(block);

    block_mark_free(block);

    /* Tell the next physical block that we are now free */
    TLSFBlock* next = block_next_phys(block);
    if (block_in_pool(a, next)) {
        block_set_prev_free(next);
    }

    /* Merge with adjacent free blocks */
    block = merge_next(a, block);
    block = merge_prev(a, block);

    /* Insert the (possibly merged) block into the free list */
    freelist_insert(a, block);

    /* Update the block after the merged region */
    next = block_next_phys(block);
    if (block_in_pool(a, next)) {
        block_set_prev_free(next);
    }

    a->used_bytes -= freed;
    a->num_frees++;
}

void* cml_tlsf_realloc(CMLTLSFAllocator* a, void* ptr, size_t new_size)
{
    if (!a) return NULL;
    if (!ptr) return cml_tlsf_alloc(a, new_size);
    if (new_size == 0) { cml_tlsf_free(a, ptr); return NULL; }

    TLSFBlock* block = user_to_block(ptr);
    size_t old_size = block_get_size(block);

    new_size = align_up(new_size, TLSF_ALIGN);
    if (new_size < MIN_USER_SIZE) new_size = MIN_USER_SIZE;

    /* Already big enough? */
    if (old_size >= new_size) {
        TLSFBlock* rest = try_split(a, block, new_size);
        if (rest) {
            block_set_prev_used(rest);
            freelist_insert(a, rest);
            TLSFBlock* after = block_next_phys(rest);
            if (block_in_pool(a, after)) {
                block_set_prev_free(after);
            }
            a->used_bytes -= (old_size - block_get_size(block));
        }
        return ptr;
    }

    /* Try to expand into the next block if it is free */
    TLSFBlock* next = block_next_phys(block);
    if (block_in_pool(a, next) && block_is_free(next)) {
        size_t combined = old_size + BLOCK_HEADER_SIZE + block_get_size(next);
        if (combined >= new_size) {
            freelist_remove(a, next);
            block_set_size(block, combined);

            TLSFBlock* after = block_next_phys(block);
            if (block_in_pool(a, after)) {
                after->prev_phys = block;
                block_set_prev_used(after);
            }

            TLSFBlock* rest = try_split(a, block, new_size);
            if (rest) {
                block_set_prev_used(rest);
                freelist_insert(a, rest);
                TLSFBlock* after_rest = block_next_phys(rest);
                if (block_in_pool(a, after_rest)) {
                    block_set_prev_free(after_rest);
                }
            }

            a->used_bytes += (block_get_size(block) - old_size);
            if (a->used_bytes > a->peak_bytes) a->peak_bytes = a->used_bytes;
            return ptr;
        }
    }

    /* Allocate new, copy, free old */
    void* new_ptr = cml_tlsf_alloc(a, new_size);
    if (!new_ptr) return NULL;
    memcpy(new_ptr, ptr, old_size < new_size ? old_size : new_size);
    cml_tlsf_free(a, ptr);
    return new_ptr;
}

size_t cml_tlsf_alloc_size(CMLTLSFAllocator* a, void* ptr)
{
    if (!a || !ptr) return 0;
    TLSFBlock* block = user_to_block(ptr);
    return block_get_size(block);
}

void cml_tlsf_stats(const CMLTLSFAllocator* a, size_t* used, size_t* peak,
                     size_t* num_allocs, size_t* num_frees)
{
    if (!a) return;
    if (used)       *used = a->used_bytes;
    if (peak)       *peak = a->peak_bytes;
    if (num_allocs) *num_allocs = a->num_allocs;
    if (num_frees)  *num_frees = a->num_frees;
}

bool cml_tlsf_check(const CMLTLSFAllocator* a)
{
    if (!a || !a->pool) return false;

    /* Walk the physical block chain */
    TLSFBlock* block = (TLSFBlock*)a->pool;
    TLSFBlock* prev = NULL;

    while (block_in_pool(a, block)) {
        if (block->prev_phys != prev) return false;

        size_t bsz = block_get_size(block);

        /* Sentinel check: zero-size block that is not the first block */
        if (bsz == 0 && block != (TLSFBlock*)a->pool) {
            /* This is the sentinel; it should be at the expected position */
            break;
        }

        /* Consistency: if prev was free, block's FLAG_PREVFREE should be set */
        if (prev) {
            if (block_is_free(prev) != block_prev_is_free(block)) {
                return false;
            }
        }

        prev = block;
        block = block_next_phys(block);
    }

    /* Verify bitmap consistency */
    for (int fl = 0; fl < TLSF_FL_INDEX_COUNT; fl++) {
        bool fl_set = (a->fl_bitmap & (1U << fl)) != 0;

        if (!fl_set) {
            if (a->sl_bitmap[fl] != 0) return false;
            continue;
        }
        if (a->sl_bitmap[fl] == 0) return false;

        for (int sl = 0; sl < TLSF_SL_INDEX_COUNT; sl++) {
            bool sl_set = (a->sl_bitmap[fl] & (1U << sl)) != 0;
            if (sl_set && !a->blocks[fl][sl]) return false;
            if (!sl_set && a->blocks[fl][sl]) return false;
        }
    }

    return true;
}

CMLTimelinePlanner* cml_timeline_planner_create(int initial_capacity)
{
    if (initial_capacity <= 0) initial_capacity = 16;

    CMLTimelinePlanner* p = (CMLTimelinePlanner*)calloc(1, sizeof(CMLTimelinePlanner));
    if (!p) return NULL;

    p->records = (CMLTimelineRecord*)calloc((size_t)initial_capacity, sizeof(CMLTimelineRecord));
    if (!p->records) { free(p); return NULL; }

    p->record_capacity = initial_capacity;
    return p;
}

void cml_timeline_planner_destroy(CMLTimelinePlanner* p)
{
    if (!p) return;
    free(p->records);
    free(p);
}

int cml_timeline_planner_add(CMLTimelinePlanner* p, int tensor_id,
                              size_t size, int alloc_time, int free_time)
{
    if (!p) return -1;
    if (alloc_time > free_time) return -1;

    if (p->num_records >= p->record_capacity) {
        int new_cap = p->record_capacity * 2;
        CMLTimelineRecord* tmp = (CMLTimelineRecord*)realloc(
            p->records, (size_t)new_cap * sizeof(CMLTimelineRecord));
        if (!tmp) return -1;
        p->records = tmp;
        p->record_capacity = new_cap;
    }

    CMLTimelineRecord* r = &p->records[p->num_records];
    r->tensor_id = tensor_id;
    r->size = align_up(size, TLSF_ALIGN);
    r->offset = 0;
    r->alloc_time = alloc_time;
    r->free_time = free_time;
    p->num_records++;

    if (free_time + 1 > p->num_steps) p->num_steps = free_time + 1;

    return 0;
}

static int timeline_cmp(const void* a, const void* b)
{
    const CMLTimelineRecord* ra = (const CMLTimelineRecord*)a;
    const CMLTimelineRecord* rb = (const CMLTimelineRecord*)b;
    if (ra->alloc_time != rb->alloc_time) return ra->alloc_time - rb->alloc_time;
    if (rb->size > ra->size) return 1;
    if (rb->size < ra->size) return -1;
    return 0;
}

static bool time_overlaps(const CMLTimelineRecord* a, const CMLTimelineRecord* b)
{
    return a->alloc_time <= b->free_time && b->alloc_time <= a->free_time;
}

static bool space_conflicts(const CMLTimelineRecord* rec, size_t offset,
                             const CMLTimelineRecord* placed)
{
    if (!time_overlaps(rec, placed)) return false;
    return offset < placed->offset + placed->size &&
           placed->offset < offset + rec->size;
}

int cml_timeline_planner_solve(CMLTimelinePlanner* p)
{
    if (!p || p->num_records == 0) return -1;

    qsort(p->records, (size_t)p->num_records, sizeof(CMLTimelineRecord), timeline_cmp);

    p->total_required = 0;

    for (int i = 0; i < p->num_records; i++) {
        CMLTimelineRecord* rec = &p->records[i];
        size_t offset = 0;
        bool placed = false;

        while (!placed) {
            bool conflict = false;
            for (int j = 0; j < i; j++) {
                if (space_conflicts(rec, offset, &p->records[j])) {
                    size_t past = align_up(p->records[j].offset + p->records[j].size, TLSF_ALIGN);
                    if (past > offset) offset = past;
                    conflict = true;
                    break;
                }
            }
            if (!conflict) placed = true;
        }

        rec->offset = offset;
        size_t end = offset + rec->size;
        if (end > p->total_required) p->total_required = end;
    }

    /* Compute peak concurrent usage */
    p->peak_usage = 0;
    for (int t = 0; t < p->num_steps; t++) {
        size_t usage = 0;
        for (int i = 0; i < p->num_records; i++) {
            if (p->records[i].alloc_time <= t && p->records[i].free_time >= t) {
                usage += p->records[i].size;
            }
        }
        if (usage > p->peak_usage) p->peak_usage = usage;
    }

    return 0;
}

const CMLTimelineRecord* cml_timeline_planner_get(const CMLTimelinePlanner* p, int tensor_id)
{
    if (!p) return NULL;
    for (int i = 0; i < p->num_records; i++) {
        if (p->records[i].tensor_id == tensor_id) return &p->records[i];
    }
    return NULL;
}

size_t cml_timeline_planner_total_memory(const CMLTimelinePlanner* p)
{
    return p ? p->total_required : 0;
}

size_t cml_timeline_planner_peak_usage(const CMLTimelinePlanner* p)
{
    return p ? p->peak_usage : 0;
}

void cml_timeline_planner_print(const CMLTimelinePlanner* p)
{
    if (!p) { printf("Timeline planner: (null)\n"); return; }

    printf("Timeline Memory Plan\n");
    printf("Records: %d, Steps: %d\n", p->num_records, p->num_steps);
    printf("Total memory required: %zu bytes\n", p->total_required);
    printf("Peak concurrent usage: %zu bytes\n", p->peak_usage);
    printf("\n");

    printf("  %-10s %-12s %-12s %-10s %-10s\n",
           "TensorID", "Size", "Offset", "Alloc", "Free");
    printf("  %-10s %-12s %-12s %-10s %-10s\n",
           "--------", "----", "------", "-----", "----");

    for (int i = 0; i < p->num_records; i++) {
        const CMLTimelineRecord* r = &p->records[i];
        printf("  %-10d %-12zu %-12zu %-10d %-10d\n",
               r->tensor_id, r->size, r->offset, r->alloc_time, r->free_time);
    }

    if (p->num_steps > 0 && p->num_steps <= 100 && p->total_required > 0) {
        printf("\nTimeline (rows=tensors, cols=time steps):\n");
        for (int i = 0; i < p->num_records; i++) {
            const CMLTimelineRecord* r = &p->records[i];
            printf("  T%-3d [off=%4zu]: ", r->tensor_id, r->offset);
            for (int t = 0; t < p->num_steps; t++) {
                printf("%c", (t >= r->alloc_time && t <= r->free_time) ? '#' : '.');
            }
            printf("\n");
        }
    }

    printf("\n");
}
