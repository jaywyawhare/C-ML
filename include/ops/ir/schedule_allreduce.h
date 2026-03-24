

#ifndef CML_OPS_IR_SCHEDULE_ALLREDUCE_H
#define CML_OPS_IR_SCHEDULE_ALLREDUCE_H

#include "ops/ir/schedule.h"
#include "ops/uops.h"    
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AR_OP_SUM = 0,
    AR_OP_MAX,
    AR_OP_MIN,
    AR_OP_PROD,
} AllReduceOp;

typedef enum {
    AR_ALGO_RING              = 0,  
    AR_ALGO_TREE,
    AR_ALGO_FLAT,
    AR_ALGO_RECURSIVE_HALVING,
    AR_ALGO_AUTO,               
} AllReduceAlgo;

typedef struct AllReduceStep {
    int     src_rank;
    int     dst_rank;
    size_t  chunk_offset;   
    size_t  chunk_bytes;
    bool    is_reduce;      
} AllReduceStep;

typedef struct ScheduleAllReduce {
    Tensor*           input;
    Tensor*           output;         
    AllReduceOp       op;
    AllReduceAlgo     algo;

    int*              device_ids;     
    int               num_devices;

    AllReduceStep*    steps;         
    int               num_steps;

    CMLScheduleItem** overlap_kernels;
    int               num_overlap_kernels;

    size_t            buffer_bytes;   
    int               chunk_count;   
} ScheduleAllReduce;

ScheduleAllReduce* schedule_allreduce_build(Tensor* t,
                                            AllReduceOp op,
                                            AllReduceAlgo algo,
                                            const int* device_ids,
                                            int num_devices);

void schedule_allreduce_free(ScheduleAllReduce* ar);

int schedule_allreduce_run(ScheduleAllReduce* ar);

int schedule_allreduce_inject(CMLSchedule* sched, ScheduleAllReduce* ar);

size_t schedule_allreduce_comm_bytes(const ScheduleAllReduce* ar);

double schedule_allreduce_latency_us(const ScheduleAllReduce* ar,
                                     double bandwidth_gbps,
                                     double latency_us);

void schedule_allreduce_print(const ScheduleAllReduce* ar);

#ifdef __cplusplus
}
#endif

#endif 
