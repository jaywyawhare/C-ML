#include "ops/ir/schedule_allreduce.h"
#include "core/logging.h"
#include "backend/device.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

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
    ar->steps = calloc((size_t)num_steps, sizeof(AllReduceStep));
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
    ar->steps = calloc((size_t)ns, sizeof(AllReduceStep));
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

static int build_recursive_halving_steps(ScheduleAllReduce* ar) {
    
    int n = ar->num_devices;
    if (n <= 0 || (n & (n - 1)) != 0)
        return build_ring_steps(ar);

    int rounds = 0;
    int tmp = n;
    while (tmp > 1) { rounds++; tmp >>= 1; }

    int ns = rounds * (n / 2) * 2;
    ar->steps = calloc((size_t)(ns + 1), sizeof(AllReduceStep));
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

    ScheduleAllReduce* ar = calloc(1, sizeof(ScheduleAllReduce));
    if (!ar) return NULL;

    ar->input       = t;
    ar->output      = t;  
    ar->op          = op;
    ar->num_devices = num_devices;
    ar->buffer_bytes = t->numel * cml_dtype_size(t->dtype);

    ar->device_ids  = malloc((size_t)num_devices * sizeof(int));
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
        case AR_ALGO_TREE:              rc = build_ring_steps(ar); break; 
        default:                         rc = build_ring_steps(ar); break;
    }
    if (rc != 0) { schedule_allreduce_free(ar); return NULL; }
    return ar;
}

void schedule_allreduce_free(ScheduleAllReduce* ar) {
    if (!ar) return;
    free(ar->device_ids);
    free(ar->steps);
    free(ar->overlap_kernels);
    free(ar);
}

int schedule_allreduce_run(ScheduleAllReduce* ar) {
    if (!ar || !ar->input || !ar->input->data) return -1;
    size_t elem_size = cml_dtype_size(ar->input->dtype);

    for (int s = 0; s < ar->num_steps; ++s) {
        AllReduceStep* step = &ar->steps[s];
        if (step->src_rank < 0 || step->src_rank >= ar->num_devices) continue;
        if (step->dst_rank < 0 || step->dst_rank >= ar->num_devices) continue;

        int src_dev = ar->device_ids[step->src_rank];
        int dst_dev = ar->device_ids[step->dst_rank];

        char* src_ptr = (char*)ar->input->data + step->chunk_offset;
        char* dst_ptr = src_ptr;  

        if (src_dev != dst_dev) {
            int rc = device_copy(dst_ptr, src_ptr, step->chunk_bytes,
                                 (DeviceType)dst_dev, (DeviceType)src_dev);
            if (rc != 0) return rc;
        }

        if (step->is_reduce && ar->op == AR_OP_SUM) {
            float* fa = (float*)dst_ptr;
            float* fb = (float*)src_ptr;
            size_t n = step->chunk_bytes / elem_size;
            for (size_t i = 0; i < n; ++i) fa[i] += fb[i];
        }
    }
    return 0;
}

int schedule_allreduce_inject(CMLSchedule* sched, ScheduleAllReduce* ar) {
    if (!sched || !ar) return -1;
    
    (void)sched; (void)ar;
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
