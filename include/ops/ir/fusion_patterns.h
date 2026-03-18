/*
 * Pluggable fusion pattern registry for backend-specific optimizations.
 * Each pattern has match() and emit() functions, registered per backend target.
 * Built-in patterns: MatMul+Bias+ReLU, Conv+BN+ReLU, Softmax+Mask, LayerNorm+Residual.
 */

#ifndef CML_FUSION_PATTERNS_H
#define CML_FUSION_PATTERNS_H

#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct IRNode;
struct CMLGraph;

typedef enum {
    FUSION_PATTERN_MATMUL_BIAS_RELU = 0,
    FUSION_PATTERN_CONV_BN_RELU,
    FUSION_PATTERN_SOFTMAX_MASK,
    FUSION_PATTERN_LAYERNORM_RESIDUAL,
    FUSION_PATTERN_ELEMENTWISE_CHAIN,
    FUSION_PATTERN_REDUCE_ELEMENTWISE,
    FUSION_PATTERN_USER_DEFINED,
    FUSION_PATTERN_COUNT
} FusionPatternKind;

typedef enum {
    FUSION_TARGET_CPU = 0,
    FUSION_TARGET_CUDA,
    FUSION_TARGET_ROCM,
    FUSION_TARGET_COUNT
} FusionTarget;

typedef struct {
    struct IRNode** matched_nodes; /* Array of matched nodes (owned) */
    int num_matched;
    void* match_data;              /* Pattern-specific data for emit */
} FusionMatch;

typedef FusionMatch* (*FusionMatchFn)(struct IRNode* start, struct CMLGraph* ir);
typedef int (*FusionEmitFn)(FusionMatch* match, struct CMLGraph* ir);

typedef struct FusionPattern {
    const char* name;
    FusionPatternKind kind;
    FusionTarget target;
    int priority;
    FusionMatchFn match;
    FusionEmitFn emit;
    struct FusionPattern* next;
} FusionPattern;

typedef struct FusionPatternRegistry {
    FusionPattern* patterns[FUSION_TARGET_COUNT];
    int total_patterns;
} FusionPatternRegistry;

FusionPatternRegistry* cml_fusion_registry_create(void);
void cml_fusion_registry_free(FusionPatternRegistry* registry);
FusionPatternRegistry* cml_fusion_registry_get_default(void);

int cml_fusion_register_pattern(FusionPatternRegistry* registry,
                                const char* name,
                                FusionPatternKind kind,
                                FusionTarget target,
                                int priority,
                                FusionMatchFn match,
                                FusionEmitFn emit);

int cml_fusion_apply_patterns(FusionPatternRegistry* registry,
                              struct CMLGraph* ir,
                              FusionTarget target);

void cml_fusion_match_free(FusionMatch* match);

#ifdef __cplusplus
}
#endif

#endif /* CML_FUSION_PATTERNS_H */
