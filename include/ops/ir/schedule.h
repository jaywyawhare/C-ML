/**
 * @file schedule.h
 * @brief Automatic kernel scheduling and fusion
 *
 * Analyzes UOp graphs to determine optimal kernel boundaries, automatically
 * fusing compatible operations into minimal GPU kernels. Inspired by tinygrad's
 * schedule-based approach.
 */

#ifndef CML_SCHEDULE_H
#define CML_SCHEDULE_H

#include "ops/ir/ir.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum ops per fused kernel */
#define CML_SCHEDULE_MAX_FUSED_OPS 64

/** Schedule item type */
typedef enum {
    SCHED_ELEMENTWISE = 0,  /* Fused chain of elementwise ops */
    SCHED_REDUCE,           /* Reduction op (breaks fusion boundary) */
    SCHED_MATMUL,           /* Matrix multiply kernel */
    SCHED_CONV,             /* Convolution kernel */
    SCHED_MOVEMENT,         /* View/reshape (zero-cost, no kernel) */
    SCHED_COPY,             /* Memory copy between devices */
    SCHED_CUSTOM,           /* Custom/unfuseable op */
} CMLScheduleItemType;

/** A single scheduled kernel (group of fused ops) */
typedef struct CMLScheduleItem {
    CMLScheduleItemType type;
    struct IRNode** ops;        /* Array of IR nodes in this kernel */
    int num_ops;                /* Number of ops fused into this kernel */
    int op_capacity;            /* Allocated capacity */

    Tensor** inputs;            /* External inputs to this kernel */
    int num_inputs;
    Tensor** outputs;           /* Outputs produced by this kernel */
    int num_outputs;

    size_t flops;               /* Estimated FLOPs for this kernel */
    size_t memory_bytes;        /* Estimated memory traffic */
    float arithmetic_intensity; /* flops / memory_bytes */

    bool is_fuseable;           /* Can be further fused with neighbors */
    int device_id;              /* Target device */
} CMLScheduleItem;

/** Complete execution schedule */
typedef struct CMLSchedule {
    CMLScheduleItem** items;    /* Array of schedule items (topological order) */
    int num_items;
    int item_capacity;

    /* Statistics */
    int total_ops;              /* Total ops before fusion */
    int total_kernels;          /* Kernels after fusion */
    float fusion_ratio;         /* total_ops / total_kernels */
    size_t total_flops;
    size_t peak_memory;

    /* Dependency tracking */
    int** dependencies;         /* dependencies[i] = list of item indices that item i depends on */
    int* dep_counts;            /* Number of dependencies per item */
} CMLSchedule;

/** Scheduling options */
typedef struct {
    bool enable_fusion;          /* Enable op fusion (default: true) */
    bool enable_movement_fold;   /* Fold movements into loads/stores (default: true) */
    int max_fused_ops;           /* Max ops per kernel (default: 64) */
    bool estimate_costs;         /* Compute FLOP/memory estimates (default: true) */
    bool topological_sort;       /* Sort items in dependency order (default: true) */
} CMLScheduleOptions;

/** Create default schedule options */
CMLScheduleOptions cml_schedule_default_options(void);

/** Create execution schedule from IR graph */
CMLSchedule* cml_schedule_create(CMLGraph_t graph, const CMLScheduleOptions* opts);

/** Free schedule */
void cml_schedule_free(CMLSchedule* sched);

/** Get number of kernels in schedule */
int cml_schedule_num_kernels(const CMLSchedule* sched);

/** Get a specific schedule item */
const CMLScheduleItem* cml_schedule_get_item(const CMLSchedule* sched, int index);

/** Check if two ops can be fused together */
bool cml_schedule_can_fuse(UOpType a, UOpType b);

/** Check if a UOp is elementwise (can be fused freely) */
bool cml_schedule_is_elementwise(UOpType type);

/** Check if a UOp is a reduction (breaks fusion boundary) */
bool cml_schedule_is_reduction(UOpType type);

/** Check if a UOp is a movement/view (zero-cost, no kernel needed) */
bool cml_schedule_is_movement(UOpType type);

/** Print schedule summary to stdout */
void cml_schedule_print(const CMLSchedule* sched);

/** Convert schedule to string */
char* cml_schedule_to_string(const CMLSchedule* sched);

#ifdef __cplusplus
}
#endif

#endif /* CML_SCHEDULE_H */
