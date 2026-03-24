/*
 * schedule_allreduce — AllReduce operations integrated into the IR schedule.
 *
 * Mirrors TinyGrad's tinygrad/schedule/allreduce.py.
 *
 * Unlike the standalone ring-allreduce in distributed/ring_allreduce.h,
 * this module generates AllReduce as first-class IR schedule items so the
 * compiler can fuse them with surrounding elementwise kernels and overlap
 * communication with computation.
 *
 * Supported algorithms:
 *   RING    — bandwidth-optimal ring allreduce (default)
 *   TREE    — latency-optimal double-binary-tree reduce-scatter + allgather
 *   FLAT    — reduce to rank 0, broadcast back (simple, high latency)
 *   RECURSIVE_HALVING — bandwidth-optimal for powers of 2
 */

#ifndef CML_OPS_IR_SCHEDULE_ALLREDUCE_H
#define CML_OPS_IR_SCHEDULE_ALLREDUCE_H

#include "ops/ir/schedule.h"
#include "ops/uops.h"    /* for UOpType (SUM/MAX/MIN) */
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Reduction operation ---- */
typedef enum {
    AR_OP_SUM = 0,
    AR_OP_MAX,
    AR_OP_MIN,
    AR_OP_PROD,
} AllReduceOp;

/* ---- Algorithm selection ---- */
typedef enum {
    AR_ALGO_RING              = 0,  /* default */
    AR_ALGO_TREE,
    AR_ALGO_FLAT,
    AR_ALGO_RECURSIVE_HALVING,
    AR_ALGO_AUTO,               /* chosen by heuristic based on message size */
} AllReduceAlgo;

/* ---- One communication step in the ring / tree decomposition ---- */
typedef struct AllReduceStep {
    int     src_rank;
    int     dst_rank;
    size_t  chunk_offset;   /* byte offset within the buffer */
    size_t  chunk_bytes;
    bool    is_reduce;      /* true = reduce-scatter chunk, false = allgather */
} AllReduceStep;

/* ---- Complete AllReduce schedule item ---- */
typedef struct ScheduleAllReduce {
    Tensor*           input;
    Tensor*           output;         /* may alias input for in-place */
    AllReduceOp       op;
    AllReduceAlgo     algo;

    int*              device_ids;     /* participating devices [num_devices] */
    int               num_devices;

    AllReduceStep*    steps;          /* communication micro-steps */
    int               num_steps;

    /* Fusion: kernels that can run concurrently with communication. */
    CMLScheduleItem** overlap_kernels;
    int               num_overlap_kernels;

    size_t            buffer_bytes;   /* total message size */
    int               chunk_count;   /* number of chunks (ring segments) */
} ScheduleAllReduce;

/* ---- Construction ---- */

/*
 * Build an AllReduce schedule for tensor t across the given devices.
 * algo == AR_ALGO_AUTO selects ring for large messages, tree for small.
 *
 * Returns NULL on error.
 */
ScheduleAllReduce* schedule_allreduce_build(Tensor* t,
                                            AllReduceOp op,
                                            AllReduceAlgo algo,
                                            const int* device_ids,
                                            int num_devices);

void schedule_allreduce_free(ScheduleAllReduce* ar);

/* ---- Execution ---- */

/* Execute all steps of the allreduce.  Returns 0 on success. */
int schedule_allreduce_run(ScheduleAllReduce* ar);

/* Inject this allreduce into an existing CMLSchedule at the right position
 * (after producers, before consumers of t).  Returns 0 on success. */
int schedule_allreduce_inject(CMLSchedule* sched, ScheduleAllReduce* ar);

/* ---- Inspection ---- */

/* Estimated total communication volume in bytes. */
size_t schedule_allreduce_comm_bytes(const ScheduleAllReduce* ar);

/* Estimated latency in microseconds (rough model). */
double schedule_allreduce_latency_us(const ScheduleAllReduce* ar,
                                     double bandwidth_gbps,
                                     double latency_us);

void schedule_allreduce_print(const ScheduleAllReduce* ar);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_SCHEDULE_ALLREDUCE_H */
