/*
 * Automatic kernel scheduling and fusion.
 * Analyzes UOp graphs to determine optimal kernel boundaries, automatically
 * fusing compatible operations into minimal GPU kernels.
 */

#ifndef CML_SCHEDULE_H
#define CML_SCHEDULE_H

#include "ops/ir/ir.h"
#include "ops/ir/memory_planner.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_SCHEDULE_MAX_FUSED_OPS 64

typedef enum {
    SCHED_ELEMENTWISE = 0,  /* Fused chain of elementwise ops */
    SCHED_REDUCE,           /* Reduction op (breaks fusion boundary) */
    SCHED_MATMUL,           /* Matrix multiply kernel */
    SCHED_CONV,             /* Convolution kernel */
    SCHED_MOVEMENT,         /* View/reshape (zero-cost, no kernel) */
    SCHED_COPY,             /* Memory copy between devices */
    SCHED_CUSTOM,           /* Custom/unfuseable op */
} CMLScheduleItemType;

typedef struct CMLScheduleItem {
    CMLScheduleItemType type;
    struct IRNode** ops;
    int num_ops;
    int op_capacity;

    Tensor** inputs;
    int num_inputs;
    Tensor** outputs;
    int num_outputs;

    size_t flops;               /* Estimated FLOPs for this kernel */
    size_t memory_bytes;        /* Estimated memory traffic */
    float arithmetic_intensity; /* flops / memory_bytes */

    bool is_fuseable;
    int device_id;
} CMLScheduleItem;

typedef struct CMLSchedule {
    CMLScheduleItem** items;    /* Topological order */
    int num_items;
    int item_capacity;

    int total_ops;              /* Total ops before fusion */
    int total_kernels;          /* Kernels after fusion */
    float fusion_ratio;         /* total_ops / total_kernels */
    size_t total_flops;
    size_t peak_memory;

    int** dependencies;         /* dependencies[i] = list of item indices that item i depends on */
    int* dep_counts;
} CMLSchedule;

typedef enum {
    CML_SCHEDULE_ORDER_TOPO = 0,
    CML_SCHEDULE_ORDER_BFS,
} CMLScheduleOrder;

typedef struct {
    bool enable_fusion;          /* default: true */
    bool enable_movement_fold;   /* Fold movements into loads/stores (default: true) */
    int max_fused_ops;           /* default: 64 */
    bool estimate_costs;         /* Compute FLOP/memory estimates (default: true) */
    bool topological_sort;       /* Sort items in dependency order (default: true) */
    bool allow_reduce_elem_fusion; /* Allow reduce->elem fusion when safe (default: true) */
    CMLScheduleOrder schedule_order; /* Kernel execution ordering (default: TOPO) */
} CMLScheduleOptions;

CMLScheduleOptions cml_schedule_default_options(void);
CMLSchedule* cml_schedule_create(CMLGraph_t graph, const CMLScheduleOptions* opts);
void cml_schedule_free(CMLSchedule* sched);
int cml_schedule_num_kernels(const CMLSchedule* sched);
const CMLScheduleItem* cml_schedule_get_item(const CMLSchedule* sched, int index);
bool cml_schedule_can_fuse(UOpType a, UOpType b);
bool cml_schedule_is_elementwise(UOpType type);
bool cml_schedule_is_reduction(UOpType type);
bool cml_schedule_is_movement(UOpType type);
void cml_schedule_print(const CMLSchedule* sched);
char* cml_schedule_to_string(const CMLSchedule* sched);

/* ── V2 Fusion Scheduler ── */

typedef struct CMLFusionAnalysis {
    bool can_fuse;
    float benefit;           /* Estimated speedup from fusion */
    size_t memory_saved;     /* Bytes of intermediate storage eliminated */
    bool eliminates_buffer;  /* True if fusion removes an intermediate buffer */
} CMLFusionAnalysis;

typedef struct CMLFusionGroup {
    struct IRNode** nodes;     /* Ops in this group (topological order) */
    int num_nodes;
    int node_capacity;

    int* eliminated_buffers;   /* Indices of intermediate buffers kept in registers */
    int num_eliminated;
    int elim_capacity;

    CMLScheduleItemType type;
    size_t total_flops;
    size_t total_memory;
    int color;                 /* Graph coloring ID */
} CMLFusionGroup;

typedef struct CMLScheduleV2 {
    CMLFusionGroup** groups;
    int num_groups;
    int group_capacity;

    int* execution_order;      /* Indices into groups[] in execution order */
    int num_ordered;

    int total_ops_before;
    int total_groups_after;
    float fusion_ratio;
    size_t memory_saved;

    CMLMemoryPlan* memory_plan;
} CMLScheduleV2;

CMLScheduleV2* cml_schedule_v2_create(CMLGraph_t graph, const CMLScheduleOptions* opts);
void cml_schedule_v2_free(CMLScheduleV2* sched);
CMLFusionAnalysis cml_schedule_analyze_fusion(struct IRNode* a, struct IRNode* b);
void cml_schedule_v2_print(const CMLScheduleV2* sched);

#ifdef __cplusplus
}
#endif

#endif /* CML_SCHEDULE_H */
