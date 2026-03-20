/*
 * Schedule-level memory planner.
 * Performs liveness analysis on scheduled buffers and assigns them to
 * physical memory slots via greedy graph coloring, minimizing peak usage.
 */

#include "ops/ir/memory_planner.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    int index;
    size_t size;
} BufferEntry;

static int cmp_by_size_desc(const void* a, const void* b) {
    const BufferEntry* ea = (const BufferEntry*)a;
    const BufferEntry* eb = (const BufferEntry*)b;
    if (eb->size > ea->size) return 1;
    if (eb->size < ea->size) return -1;
    return ea->index - eb->index;
}

static int lifetimes_overlap(int a_first, int a_last, int b_first, int b_last) {
    return !(a_last < b_first || b_last < a_first);
}

CMLMemoryPlan* cml_memory_plan_create(int num_buffers, size_t* sizes,
                                       int* first_use, int* last_use) {
    if (num_buffers <= 0 || !sizes || !first_use || !last_use)
        return NULL;

    CMLMemoryPlan* plan = calloc(1, sizeof(CMLMemoryPlan));
    if (!plan) return NULL;

    plan->num_buffers    = num_buffers;
    plan->buffer_sizes   = malloc((size_t)num_buffers * sizeof(size_t));
    plan->buffer_offsets = calloc((size_t)num_buffers, sizeof(size_t));
    plan->buffer_reuse_map = malloc((size_t)num_buffers * sizeof(int));
    plan->buffer_first_use = malloc((size_t)num_buffers * sizeof(int));
    plan->buffer_last_use  = malloc((size_t)num_buffers * sizeof(int));

    if (!plan->buffer_sizes || !plan->buffer_offsets || !plan->buffer_reuse_map ||
        !plan->buffer_first_use || !plan->buffer_last_use) {
        cml_memory_plan_free(plan);
        return NULL;
    }

    memcpy(plan->buffer_sizes, sizes, (size_t)num_buffers * sizeof(size_t));
    memcpy(plan->buffer_first_use, first_use, (size_t)num_buffers * sizeof(int));
    memcpy(plan->buffer_last_use, last_use, (size_t)num_buffers * sizeof(int));

    for (int i = 0; i < num_buffers; i++)
        plan->buffer_reuse_map[i] = -1;

    size_t naive_total = 0;
    for (int i = 0; i < num_buffers; i++)
        naive_total += sizes[i];

    /* Sort buffers by decreasing size for greedy coloring */
    BufferEntry* sorted = malloc((size_t)num_buffers * sizeof(BufferEntry));
    if (!sorted) {
        cml_memory_plan_free(plan);
        return NULL;
    }
    for (int i = 0; i < num_buffers; i++) {
        sorted[i].index = i;
        sorted[i].size  = sizes[i];
    }
    qsort(sorted, (size_t)num_buffers, sizeof(BufferEntry), cmp_by_size_desc);

    /*
     * Slots: each slot has a size (the largest buffer assigned to it) and
     * a list of assigned buffer indices for conflict checking.
     *
     * We cap at num_buffers slots (worst case: no reuse).
     */
    int num_slots = 0;
    size_t* slot_sizes = calloc((size_t)num_buffers, sizeof(size_t));
    int* slot_owner    = malloc((size_t)num_buffers * sizeof(int));
    /* For each slot, track the merged lifetime [earliest first_use, latest last_use] */
    int* slot_first = malloc((size_t)num_buffers * sizeof(int));
    int* slot_last  = malloc((size_t)num_buffers * sizeof(int));
    /* Per-buffer slot assignment */
    int* buf_slot = malloc((size_t)num_buffers * sizeof(int));

    if (!slot_sizes || !slot_owner || !slot_first || !slot_last || !buf_slot) {
        free(sorted);
        free(slot_sizes); free(slot_owner);
        free(slot_first); free(slot_last); free(buf_slot);
        cml_memory_plan_free(plan);
        return NULL;
    }

    /*
     * Track per-slot buffer lists so we can check conflicts against all
     * buffers in a slot, not just the merged interval. Merged intervals
     * are overly conservative when a slot hosts multiple non-overlapping
     * buffers with a gap between them.
     */
    int** slot_bufs   = calloc((size_t)num_buffers, sizeof(int*));
    int*  slot_nbuf   = calloc((size_t)num_buffers, sizeof(int));
    int*  slot_bufcap = calloc((size_t)num_buffers, sizeof(int));

    if (!slot_bufs || !slot_nbuf || !slot_bufcap) {
        free(sorted); free(slot_sizes); free(slot_owner);
        free(slot_first); free(slot_last); free(buf_slot);
        free(slot_bufs); free(slot_nbuf); free(slot_bufcap);
        cml_memory_plan_free(plan);
        return NULL;
    }

    for (int si = 0; si < num_buffers; si++) {
        int idx    = sorted[si].index;
        size_t sz  = sizes[idx];
        int fu     = first_use[idx];
        int lu     = last_use[idx];

        int best_slot = -1;
        size_t best_waste = (size_t)-1;

        for (int s = 0; s < num_slots; s++) {
            if (slot_sizes[s] < sz)
                continue;

            /* Check conflict against every buffer already in this slot */
            int conflict = 0;
            for (int bi = 0; bi < slot_nbuf[s]; bi++) {
                int other = slot_bufs[s][bi];
                if (lifetimes_overlap(fu, lu, first_use[other], last_use[other])) {
                    conflict = 1;
                    break;
                }
            }
            if (conflict)
                continue;

            size_t waste = slot_sizes[s] - sz;
            if (waste < best_waste) {
                best_waste = waste;
                best_slot  = s;
            }
        }

        if (best_slot >= 0) {
            buf_slot[idx] = best_slot;
            plan->buffer_reuse_map[idx] = slot_owner[best_slot];

            /* Extend merged interval */
            if (fu < slot_first[best_slot]) slot_first[best_slot] = fu;
            if (lu > slot_last[best_slot])  slot_last[best_slot]  = lu;

            /* Append to slot's buffer list */
            if (slot_nbuf[best_slot] >= slot_bufcap[best_slot]) {
                int nc = slot_bufcap[best_slot] ? slot_bufcap[best_slot] * 2 : 4;
                int* tmp = realloc(slot_bufs[best_slot], (size_t)nc * sizeof(int));
                if (tmp) {
                    slot_bufs[best_slot]   = tmp;
                    slot_bufcap[best_slot] = nc;
                }
            }
            slot_bufs[best_slot][slot_nbuf[best_slot]++] = idx;
        } else {
            int s = num_slots++;
            slot_sizes[s] = sz;
            slot_owner[s] = idx;
            slot_first[s] = fu;
            slot_last[s]  = lu;
            buf_slot[idx] = s;

            slot_bufcap[s] = 4;
            slot_bufs[s]   = malloc(4 * sizeof(int));
            slot_nbuf[s]   = 1;
            if (slot_bufs[s])
                slot_bufs[s][0] = idx;
        }
    }

    /* Compute offsets: pack slots contiguously */
    size_t offset = 0;
    for (int s = 0; s < num_slots; s++) {
        for (int i = 0; i < num_buffers; i++) {
            if (buf_slot[i] == s)
                plan->buffer_offsets[i] = offset;
        }
        offset += slot_sizes[s];
    }

    plan->total_memory = offset;
    plan->saved_memory = (naive_total > offset) ? naive_total - offset : 0;

    /* Peak memory: max sum of alive buffer sizes at any schedule step */
    int min_step = first_use[0], max_step = last_use[0];
    for (int i = 1; i < num_buffers; i++) {
        if (first_use[i] < min_step) min_step = first_use[i];
        if (last_use[i] > max_step)  max_step = last_use[i];
    }

    size_t peak = 0;
    for (int step = min_step; step <= max_step; step++) {
        size_t alive = 0;
        for (int i = 0; i < num_buffers; i++) {
            if (first_use[i] <= step && step <= last_use[i])
                alive += sizes[i];
        }
        if (alive > peak) peak = alive;
    }
    plan->peak_memory = peak;

    for (int s = 0; s < num_slots; s++)
        free(slot_bufs[s]);
    free(slot_bufs);
    free(slot_nbuf);
    free(slot_bufcap);
    free(sorted);
    free(slot_sizes);
    free(slot_owner);
    free(slot_first);
    free(slot_last);
    free(buf_slot);

    return plan;
}

void cml_memory_plan_free(CMLMemoryPlan* plan) {
    if (!plan) return;
    free(plan->buffer_sizes);
    free(plan->buffer_offsets);
    free(plan->buffer_reuse_map);
    free(plan->buffer_first_use);
    free(plan->buffer_last_use);
    free(plan);
}

void cml_memory_plan_print(const CMLMemoryPlan* plan) {
    if (!plan) {
        printf("MemoryPlan: (null)\n");
        return;
    }

    printf("CML Memory Plan\n");
    printf("  Buffers:      %d\n", plan->num_buffers);
    printf("  Total alloc:  %zu bytes\n", plan->total_memory);
    printf("  Peak live:    %zu bytes\n", plan->peak_memory);
    printf("  Saved:        %zu bytes\n", plan->saved_memory);
    printf("\n");

    for (int i = 0; i < plan->num_buffers; i++) {
        printf("  buf[%d]: size=%-8zu  offset=%-8zu  life=[%d,%d]",
               i, plan->buffer_sizes[i], plan->buffer_offsets[i],
               plan->buffer_first_use[i], plan->buffer_last_use[i]);
        if (plan->buffer_reuse_map[i] >= 0)
            printf("  reuses=%d", plan->buffer_reuse_map[i]);
        printf("\n");
    }
    printf("\n");
}
