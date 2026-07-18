#include "ops/ir/schedule_allreduce.h"
#include "core/logging.h"
#include "backend/device.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "alloc/cml_allocator.h"

static AllReduceAlgo choose_algo(size_t bytes, int ndevices) {
    
    if (bytes >= 1024 * 1024) return AR_ALGO_RING;
    
    if (ndevices >= 4) return AR_ALGO_TREE;
    return AR_ALGO_RING;
}

static int build_ring_steps(ScheduleAllReduce* ar) {
    int n = ar->num_devices;
    size_t buf_bytes = ar->buffer_bytes;

    int chunks = n;
    size_t chunk_bytes = (buf_bytes + (size_t)(chunks - 1)) / (size_t)chunks;
    ar->chunk_count = chunks;

    
    int num_steps = 2 * (n - 1);
    ar->steps = cml_calloc((size_t)num_steps, sizeof(AllReduceStep));
    if (!ar->steps) return -1;

    int s = 0;
    for (int r = 0; r < n - 1; ++r) {
        for (int rank = 0; rank < n; ++rank) {
            int chunk_idx = ((rank - r - 1 + n) % n);
            ar->steps[s].src_rank     = rank;
            ar->steps[s].dst_rank     = (rank + 1) % n;
            ar->steps[s].chunk_offset = (size_t)chunk_idx * chunk_bytes;
            ar->steps[s].chunk_bytes  = (chunk_idx == chunks - 1)
                                        ? buf_bytes - (size_t)(chunks-1)*chunk_bytes
                                        : chunk_bytes;
            ar->steps[s].is_reduce    = true;
            ++s;
            if (s >= num_steps) break;
        }
        if (s >= num_steps) break;
    }
    for (int r = 0; r < n - 1 && s < num_steps; ++r) {
        for (int rank = 0; rank < n && s < num_steps; ++rank) {
            int chunk_idx = ((rank - r + n) % n);
            ar->steps[s].src_rank     = rank;
            ar->steps[s].dst_rank     = (rank + 1) % n;
            ar->steps[s].chunk_offset = (size_t)chunk_idx * chunk_bytes;
            ar->steps[s].chunk_bytes  = (chunk_idx == chunks - 1)
                                        ? buf_bytes - (size_t)(chunks-1)*chunk_bytes
                                        : chunk_bytes;
            ar->steps[s].is_reduce    = false;
            ++s;
        }
    }
    ar->num_steps = s;
    return 0;
}

static int build_flat_steps(ScheduleAllReduce* ar) {
    int n = ar->num_devices;
    int ns = 2 * (n - 1);
    ar->steps = cml_calloc((size_t)ns, sizeof(AllReduceStep));
    if (!ar->steps) return -1;
    int s = 0;
    for (int rank = 1; rank < n; ++rank) {
        ar->steps[s].src_rank    = rank;
        ar->steps[s].dst_rank    = 0;
        ar->steps[s].chunk_offset = 0;
        ar->steps[s].chunk_bytes  = ar->buffer_bytes;
        ar->steps[s].is_reduce    = true;
        ++s;
    }
    for (int rank = 1; rank < n; ++rank) {
        ar->steps[s].src_rank    = 0;
        ar->steps[s].dst_rank    = rank;
        ar->steps[s].chunk_offset = 0;
        ar->steps[s].chunk_bytes  = ar->buffer_bytes;
        ar->steps[s].is_reduce    = false;
        ++s;
    }
    ar->num_steps    = s;
    ar->chunk_count  = 1;
    return 0;
}

/* Binomial tree: reduce up to rank 0 (log2(n) rounds), then broadcast back
 * down. 2*(n-1) full-buffer transfers; latency-optimal for small buffers. */
static int build_tree_steps(ScheduleAllReduce* ar) {
    int n = ar->num_devices;
    int ns = 2 * (n - 1);
    ar->steps = cml_calloc((size_t)(ns > 0 ? ns : 1), sizeof(AllReduceStep));
    if (!ar->steps) return -1;

    int s = 0;
    for (int stride = 1; stride < n; stride <<= 1) {
        for (int rank = 0; rank + stride < n; rank += 2 * stride) {
            ar->steps[s].src_rank     = rank + stride;
            ar->steps[s].dst_rank     = rank;
            ar->steps[s].chunk_offset = 0;
            ar->steps[s].chunk_bytes  = ar->buffer_bytes;
            ar->steps[s].is_reduce    = true;
            ++s;
        }
    }
    int top = 1;
    while (top < n) top <<= 1;
    for (int stride = top >> 1; stride >= 1; stride >>= 1) {
        for (int rank = 0; rank + stride < n; rank += 2 * stride) {
            ar->steps[s].src_rank     = rank;
            ar->steps[s].dst_rank     = rank + stride;
            ar->steps[s].chunk_offset = 0;
            ar->steps[s].chunk_bytes  = ar->buffer_bytes;
            ar->steps[s].is_reduce    = false;
            ++s;
        }
    }
    ar->num_steps   = s;
    ar->chunk_count = 1;
    return 0;
}

static int build_recursive_halving_steps(ScheduleAllReduce* ar) {
    
    int n = ar->num_devices;
    if (n <= 0 || (n & (n - 1)) != 0)
        return build_ring_steps(ar);

    int rounds = 0;
    int tmp = n;
    while (tmp > 1) { rounds++; tmp >>= 1; }

    int ns = rounds * (n / 2) * 2;
    ar->steps = cml_calloc((size_t)(ns + 1), sizeof(AllReduceStep));
    if (!ar->steps) return -1;

    int s = 0;
    for (int r = 0; r < rounds && s < ns; ++r) {
        int stride = n >> (r + 1);
        size_t chunk = ar->buffer_bytes >> (r + 1);
        for (int rank = 0; rank < n && s < ns; rank += 2 * stride) {
            for (int i = 0; i < stride && s < ns; ++i) {
                ar->steps[s].src_rank     = rank + i + stride;
                ar->steps[s].dst_rank     = rank + i;
                ar->steps[s].chunk_offset = (size_t)(rank + i + stride) * chunk;
                ar->steps[s].chunk_bytes  = chunk;
                ar->steps[s].is_reduce    = true;
                ++s;
            }
        }
    }
    for (int r = rounds - 1; r >= 0 && s < ns; --r) {
        int stride = n >> (r + 1);
        size_t chunk = ar->buffer_bytes >> (r + 1);
        for (int rank = 0; rank < n && s < ns; rank += 2 * stride) {
            for (int i = 0; i < stride && s < ns; ++i) {
                ar->steps[s].src_rank     = rank + i;
                ar->steps[s].dst_rank     = rank + i + stride;
                ar->steps[s].chunk_offset = (size_t)(rank + i) * chunk;
                ar->steps[s].chunk_bytes  = chunk;
                ar->steps[s].is_reduce    = false;
                ++s;
            }
        }
    }
    ar->num_steps   = s;
    ar->chunk_count = n;
    return 0;
}

ScheduleAllReduce* schedule_allreduce_build(Tensor* t,
                                             AllReduceOp op,
                                             AllReduceAlgo algo,
                                             const int* device_ids,
                                             int num_devices) {
    if (!t || !device_ids || num_devices <= 0) return NULL;

    ScheduleAllReduce* ar = cml_calloc(1, sizeof(ScheduleAllReduce));
    if (!ar) return NULL;

    ar->input       = t;
    ar->output      = t;  
    ar->op          = op;
    ar->num_devices = num_devices;
    ar->buffer_bytes = t->numel * cml_dtype_size(t->dtype);

    ar->device_ids  = cml_malloc((size_t)num_devices * sizeof(int));
    if (!ar->device_ids) { schedule_allreduce_free(ar); return NULL; }
    memcpy(ar->device_ids, device_ids, (size_t)num_devices * sizeof(int));

    if (algo == AR_ALGO_AUTO)
        algo = choose_algo(ar->buffer_bytes, num_devices);
    ar->algo = algo;

    int rc = 0;
    switch (algo) {
        case AR_ALGO_RING:              rc = build_ring_steps(ar);               break;
        case AR_ALGO_FLAT:              rc = build_flat_steps(ar);               break;
        case AR_ALGO_RECURSIVE_HALVING: rc = build_recursive_halving_steps(ar);  break;
        case AR_ALGO_TREE:              rc = build_tree_steps(ar);               break;
        default:                         rc = build_ring_steps(ar); break;
    }
    if (rc != 0) { schedule_allreduce_free(ar); return NULL; }
    return ar;
}

void schedule_allreduce_free(ScheduleAllReduce* ar) {
    if (!ar) return;
    cml_free(ar->device_ids);
    cml_free(ar->steps);
    cml_free(ar->overlap_kernels);
    cml_free(ar);
}

int schedule_allreduce_run(ScheduleAllReduce* ar) {
    if (!ar || !ar->input || !ar->input->data) return -1;
    if (ar->num_devices <= 1) return 0;   /* nothing to reduce */

    /* Single-process simulation. This module has ONE physical buffer, so the
     * `num_devices` logical replicas all hold the same `input`; an all-reduce
     * over identical replicas has a closed form applied in place:
     *   SUM  -> x * num_devices
     *   PROD -> x ^ num_devices
     *   MAX/MIN -> unchanged (reduce of equal values)
     *
     * The previous implementation set dst_ptr = src_ptr and did `x += x`,
     * doubling the buffer regardless of device count (data corruption). A true
     * cross-device reduce needs one buffer per device; that is out of scope for
     * this single-process planner, whose real job is the cost model below. */
    if (ar->input->dtype != DTYPE_FLOAT32) return 0;   /* only f32 modeled */

    float* buf = (float*)ar->input->data;
    size_t n = ar->input->numel;
    float nd = (float)ar->num_devices;
    switch (ar->op) {
    case AR_OP_SUM:
        for (size_t i = 0; i < n; ++i) buf[i] *= nd;
        break;
    case AR_OP_PROD:
        for (size_t i = 0; i < n; ++i) buf[i] = powf(buf[i], nd);
        break;
    case AR_OP_MAX:
    case AR_OP_MIN:
        break;
    }
    return 0;
}

/* Insert the all-reduce's communication into a schedule as SCHED_COPY items so
 * the schedule's kernel/byte accounting reflects the injected collective. Grows
 * items + dependencies + dep_counts together to keep cml_schedule_free (which
 * iterates dependencies[0..num_items)) consistent. */
int schedule_allreduce_inject(CMLSchedule* sched, ScheduleAllReduce* ar) {
    if (!sched || !ar || ar->num_steps <= 0) return -1;

    int add   = ar->num_steps;
    int new_n = sched->num_items + add;

    CMLScheduleItem** it = (CMLScheduleItem**)cml_realloc(
        sched->items, (size_t)new_n * sizeof(*it));
    if (!it) return -1;
    sched->items = it;

    if (sched->dependencies) {
        int** dep = (int**)cml_realloc(sched->dependencies, (size_t)new_n * sizeof(*dep));
        if (!dep) return -1;
        sched->dependencies = dep;
    }
    if (sched->dep_counts) {
        int* dc = (int*)cml_realloc(sched->dep_counts, (size_t)new_n * sizeof(*dc));
        if (!dc) return -1;
        sched->dep_counts = dc;
    }

    for (int s = 0; s < add; ++s) {
        CMLScheduleItem* item = (CMLScheduleItem*)cml_calloc(1, sizeof(CMLScheduleItem));
        if (!item) return -1;   /* prior appends stay consistent; caller frees sched */
        item->type         = SCHED_COPY;
        item->memory_bytes = ar->steps[s].chunk_bytes;
        item->device_id    = (ar->steps[s].dst_rank >= 0 &&
                              ar->steps[s].dst_rank < ar->num_devices)
                             ? ar->device_ids[ar->steps[s].dst_rank] : 0;
        /* ops / inputs / outputs stay NULL — sched_item_free frees NULLs safely. */

        int idx = sched->num_items;
        sched->items[idx] = item;
        if (sched->dependencies) sched->dependencies[idx] = NULL;
        if (sched->dep_counts)   sched->dep_counts[idx]   = 0;
        sched->num_items++;
    }
    if (sched->item_capacity < new_n) sched->item_capacity = new_n;
    sched->total_kernels += add;
    return 0;
}

size_t schedule_allreduce_comm_bytes(const ScheduleAllReduce* ar) {
    if (!ar) return 0;
    size_t total = 0;
    for (int s = 0; s < ar->num_steps; ++s)
        total += ar->steps[s].chunk_bytes;
    return total;
}

double schedule_allreduce_latency_us(const ScheduleAllReduce* ar,
                                      double bandwidth_gbps,
                                      double latency_us) {
    if (!ar || bandwidth_gbps <= 0) return 0.0;
    double comm = (double)schedule_allreduce_comm_bytes(ar);
    double bw_bytes_us = bandwidth_gbps * 1e9 / 1e6;  
    return latency_us * ar->num_steps + comm / bw_bytes_us;
}

void schedule_allreduce_print(const ScheduleAllReduce* ar) {
    if (!ar) { fprintf(stderr, "ScheduleAllReduce(NULL)\n"); return; }
    static const char* algo_names[] = {"RING","TREE","FLAT","RECURSIVE_HALVING","AUTO"};
    static const char* op_names[]   = {"SUM","MAX","MIN","PROD"};
    fprintf(stderr, "ScheduleAllReduce: op=%s algo=%s devices=%d steps=%d "
                    "chunks=%d buf=%zu bytes\n",
            op_names[ar->op], algo_names[ar->algo],
            ar->num_devices, ar->num_steps,
            ar->chunk_count, ar->buffer_bytes);
}
