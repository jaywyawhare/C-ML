/*
 * Cross-boundary fusion between forward and backward passes.
 * Identifies patterns where forward computation intermediates can be
 * kept in registers for the backward pass, avoiding materialization.
 */

#ifndef CML_CROSS_BOUNDARY_FUSION_H
#define CML_CROSS_BOUNDARY_FUSION_H

#include "ops/ir/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CBF_SOFTMAX_CE = 0,     /* softmax fwd + cross-entropy bwd -> softmax(x) - one_hot(y) */
    CBF_LAYERNORM_BWD,      /* layernorm fwd + bwd -> keep mean/var in registers */
    CBF_GELU_BWD,           /* gelu fwd + bwd -> save intermediate sigmoid */
    CBF_PATTERN_COUNT
} CMLCrossBoundaryPatternType;

typedef struct CMLCrossBoundaryFusion {
    int forward_node_idx;
    int backward_node_idx;
    int pattern_type;
} CMLCrossBoundaryFusion;

typedef struct CMLCrossBoundaryStats {
    int patterns_found;
    int patterns_applied;
    size_t memory_saved;
    size_t flops_saved;
} CMLCrossBoundaryStats;

int cml_cross_boundary_analyze(CMLScheduleV2* sched,
                               CMLCrossBoundaryFusion** out, int* count);

int cml_cross_boundary_fuse(CMLScheduleV2* sched,
                            CMLCrossBoundaryFusion* fusions, int count);

void cml_cross_boundary_fusions_free(CMLCrossBoundaryFusion* fusions);

CMLCrossBoundaryStats cml_cross_boundary_stats(const CMLCrossBoundaryFusion* fusions,
                                               int count);

#ifdef __cplusplus
}
#endif

#endif /* CML_CROSS_BOUNDARY_FUSION_H */
