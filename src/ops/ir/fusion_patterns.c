#include "ops/ir/fusion_patterns.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

static FusionMatch* match_matmul_bias_relu(struct IRNode* start, struct CMLGraph* ir) {
    (void)ir;
    if (!start || start->type != UOP_MATMUL)
        return NULL;

    struct IRNode* add_node = NULL;
    struct IRNode* relu_node = NULL;

    if (start->users && start->use_count > 0) {
        add_node = start->users[0];
        if (!add_node || add_node->type != UOP_ADD)
            return NULL;
    } else {
        return NULL;
    }

    if (add_node->users && add_node->use_count > 0) {
        relu_node = add_node->users[0];
        if (!relu_node)
            return NULL;
        if (relu_node->type != UOP_MAX && relu_node->type != UOP_SIGMOID)
            return NULL;
    } else {
        return NULL;
    }

    FusionMatch* match = calloc(1, sizeof(FusionMatch));
    if (!match)
        return NULL;

    match->num_matched = 3;
    match->matched_nodes = malloc(3 * sizeof(struct IRNode*));
    if (!match->matched_nodes) {
        free(match);
        return NULL;
    }
    match->matched_nodes[0] = start;
    match->matched_nodes[1] = add_node;
    match->matched_nodes[2] = relu_node;

    LOG_DEBUG("Fusion pattern matched: MatMul + Bias + Activation");
    return match;
}

static int emit_matmul_bias_relu(FusionMatch* match, struct CMLGraph* ir) {
    if (!match || !ir || match->num_matched < 3)
        return -1;

    for (int i = 0; i < match->num_matched; i++) {
        match->matched_nodes[i]->is_fused = true;
        match->matched_nodes[i]->fusion_type = FUSION_FMA;
    }

    LOG_DEBUG("Emitted fused MatMul+Bias+Activation kernel");
    return 0;
}

static FusionMatch* match_elementwise_chain(struct IRNode* start, struct CMLGraph* ir) {
    (void)ir;
    if (!start)
        return NULL;

    bool is_elementwise = false;
    switch (start->type) {
    case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
    case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
    case UOP_SIGMOID: case UOP_TANH: case UOP_ABS:
    case UOP_SIN: case UOP_COS: case UOP_TAN:
        is_elementwise = true;
        break;
    default:
        break;
    }

    if (!is_elementwise)
        return NULL;

    struct IRNode* chain[32];
    int chain_len = 0;
    struct IRNode* current = start;

    while (current && chain_len < 32) {
        bool current_is_ew = false;
        switch (current->type) {
        case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
        case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
        case UOP_SIGMOID: case UOP_TANH: case UOP_ABS:
        case UOP_SIN: case UOP_COS: case UOP_TAN:
            current_is_ew = true;
            break;
        default:
            break;
        }
        if (!current_is_ew)
            break;

        chain[chain_len++] = current;

        if (current->use_count == 1 && current->users)
            current = current->users[0];
        else
            break;
    }

    if (chain_len < 2)
        return NULL;

    FusionMatch* match = calloc(1, sizeof(FusionMatch));
    if (!match)
        return NULL;

    match->num_matched = chain_len;
    match->matched_nodes = malloc(chain_len * sizeof(struct IRNode*));
    if (!match->matched_nodes) {
        free(match);
        return NULL;
    }
    memcpy(match->matched_nodes, chain, chain_len * sizeof(struct IRNode*));

    LOG_DEBUG("Fusion pattern matched: Elementwise chain (%d ops)", chain_len);
    return match;
}

static int emit_elementwise_chain(FusionMatch* match, struct CMLGraph* ir) {
    if (!match || !ir)
        return -1;

    for (int i = 0; i < match->num_matched; i++) {
        match->matched_nodes[i]->is_fused = true;
        match->matched_nodes[i]->fusion_type = FUSION_CHAIN_ELEMENTWISE;
        match->matched_nodes[i]->chain_id = match->matched_nodes[0]->chain_id;
    }

    LOG_DEBUG("Emitted fused elementwise chain (%d ops)", match->num_matched);
    return 0;
}

static FusionMatch* match_softmax_ce_bwd(struct IRNode* start, struct CMLGraph* ir) {
    (void)ir;
    if (!start || start->type != UOP_EXP) return NULL;

    struct IRNode* sum_node = NULL;
    if (start->use_count >= 1 && start->users)
        sum_node = start->users[0];
    if (!sum_node || sum_node->type != UOP_SUM) return NULL;

    struct IRNode* recip_node = NULL;
    if (sum_node->use_count >= 1 && sum_node->users)
        recip_node = sum_node->users[0];
    if (!recip_node || recip_node->type != UOP_RECIP) return NULL;

    struct IRNode* mul_node = NULL;
    if (recip_node->use_count >= 1 && recip_node->users)
        mul_node = recip_node->users[0];
    if (!mul_node || mul_node->type != UOP_MUL) return NULL;

    /* Check for SUB consumer (backward: softmax - one_hot) */
    struct IRNode* sub_node = NULL;
    if (mul_node->use_count >= 1 && mul_node->users)
        sub_node = mul_node->users[0];
    if (!sub_node || sub_node->type != UOP_SUB) return NULL;

    FusionMatch* match = calloc(1, sizeof(FusionMatch));
    if (!match) return NULL;

    match->num_matched = 5;
    match->matched_nodes = malloc(5 * sizeof(struct IRNode*));
    if (!match->matched_nodes) { free(match); return NULL; }
    match->matched_nodes[0] = start;
    match->matched_nodes[1] = sum_node;
    match->matched_nodes[2] = recip_node;
    match->matched_nodes[3] = mul_node;
    match->matched_nodes[4] = sub_node;

    LOG_DEBUG("Fusion pattern matched: Softmax + CrossEntropy backward");
    return match;
}

static int emit_softmax_ce_bwd(FusionMatch* match, struct CMLGraph* ir) {
    if (!match || !ir || match->num_matched < 5) return -1;

    for (int i = 0; i < match->num_matched; i++) {
        match->matched_nodes[i]->is_fused = true;
        match->matched_nodes[i]->fusion_type = FUSION_REDUCE_ELEMENTWISE;
    }

    struct IRNode* fwd = match->matched_nodes[0];
    struct IRNode* bwd = match->matched_nodes[4];
    fwd->backward_node = bwd;
    bwd->forward_node = fwd;

    LOG_DEBUG("Emitted fused Softmax+CE backward kernel");
    return 0;
}

static FusionMatch* match_layernorm_bwd(struct IRNode* start, struct CMLGraph* ir) {
    (void)ir;
    if (!start || start->type != UOP_MEAN) return NULL;

    struct IRNode* sub = NULL;
    if (start->use_count >= 1 && start->users)
        sub = start->users[0];
    if (!sub || sub->type != UOP_SUB) return NULL;

    struct IRNode* sq = NULL;
    if (sub->use_count >= 1 && sub->users) {
        for (int i = 0; i < sub->use_count; i++) {
            if (sub->users[i] && sub->users[i]->type == UOP_SQUARE) {
                sq = sub->users[i];
                break;
            }
        }
    }
    if (!sq) return NULL;

    struct IRNode* var_mean = NULL;
    if (sq->use_count >= 1 && sq->users)
        var_mean = sq->users[0];
    if (!var_mean || var_mean->type != UOP_MEAN) return NULL;

    struct IRNode* rsqrt = NULL;
    if (var_mean->use_count >= 1 && var_mean->users) {
        for (int i = 0; i < var_mean->use_count; i++) {
            struct IRNode* u = var_mean->users[i];
            if (u && (u->type == UOP_RSQRT || u->type == UOP_SQRT)) {
                rsqrt = u;
                break;
            }
        }
    }
    if (!rsqrt) return NULL;

    FusionMatch* match = calloc(1, sizeof(FusionMatch));
    if (!match) return NULL;

    match->num_matched = 5;
    match->matched_nodes = malloc(5 * sizeof(struct IRNode*));
    if (!match->matched_nodes) { free(match); return NULL; }
    match->matched_nodes[0] = start;
    match->matched_nodes[1] = sub;
    match->matched_nodes[2] = sq;
    match->matched_nodes[3] = var_mean;
    match->matched_nodes[4] = rsqrt;

    LOG_DEBUG("Fusion pattern matched: LayerNorm + backward");
    return match;
}

static int emit_layernorm_bwd(FusionMatch* match, struct CMLGraph* ir) {
    if (!match || !ir || match->num_matched < 5) return -1;

    for (int i = 0; i < match->num_matched; i++) {
        match->matched_nodes[i]->is_fused = true;
        match->matched_nodes[i]->fusion_type = FUSION_CHAIN_ELEMENTWISE;
    }

    LOG_DEBUG("Emitted fused LayerNorm+backward kernel (mean/var in registers)");
    return 0;
}

static FusionMatch* match_gelu_bwd(struct IRNode* start, struct CMLGraph* ir) {
    (void)ir;
    if (!start) return NULL;

    if (start->type == UOP_QUICK_GELU || start->type == UOP_SILU) {
        if (start->backward_node || (start->use_count > 0 && start->users)) {
            FusionMatch* match = calloc(1, sizeof(FusionMatch));
            if (!match) return NULL;

            match->num_matched = 1;
            match->matched_nodes = malloc(sizeof(struct IRNode*));
            if (!match->matched_nodes) { free(match); return NULL; }
            match->matched_nodes[0] = start;

            LOG_DEBUG("Fusion pattern matched: GELU/SiLU + backward");
            return match;
        }
    }

    return NULL;
}

static int emit_gelu_bwd(FusionMatch* match, struct CMLGraph* ir) {
    if (!match || !ir || match->num_matched < 1) return -1;

    match->matched_nodes[0]->is_fused = true;
    match->matched_nodes[0]->fusion_type = FUSION_CHAIN_ELEMENTWISE;

    LOG_DEBUG("Emitted fused GELU+backward kernel (sigmoid reuse)");
    return 0;
}

static FusionPatternRegistry* g_default_registry = NULL;

FusionPatternRegistry* cml_fusion_registry_create(void) {
    FusionPatternRegistry* reg = calloc(1, sizeof(FusionPatternRegistry));
    if (!reg)
        return NULL;

    for (int t = 0; t < FUSION_TARGET_COUNT; t++) {
        cml_fusion_register_pattern(reg, "matmul_bias_relu",
                                    FUSION_PATTERN_MATMUL_BIAS_RELU,
                                    (FusionTarget)t, 100,
                                    match_matmul_bias_relu,
                                    emit_matmul_bias_relu);

        cml_fusion_register_pattern(reg, "elementwise_chain",
                                    FUSION_PATTERN_ELEMENTWISE_CHAIN,
                                    (FusionTarget)t, 50,
                                    match_elementwise_chain,
                                    emit_elementwise_chain);

        cml_fusion_register_pattern(reg, "softmax_ce_bwd",
                                    FUSION_PATTERN_SOFTMAX_CE_BWD,
                                    (FusionTarget)t, 90,
                                    match_softmax_ce_bwd,
                                    emit_softmax_ce_bwd);

        cml_fusion_register_pattern(reg, "layernorm_bwd",
                                    FUSION_PATTERN_LAYERNORM_BWD,
                                    (FusionTarget)t, 85,
                                    match_layernorm_bwd,
                                    emit_layernorm_bwd);

        cml_fusion_register_pattern(reg, "gelu_bwd",
                                    FUSION_PATTERN_GELU_BWD,
                                    (FusionTarget)t, 80,
                                    match_gelu_bwd,
                                    emit_gelu_bwd);
    }

    return reg;
}

void cml_fusion_registry_free(FusionPatternRegistry* registry) {
    if (!registry)
        return;

    for (int t = 0; t < FUSION_TARGET_COUNT; t++) {
        FusionPattern* p = registry->patterns[t];
        while (p) {
            FusionPattern* next = p->next;
            free(p);
            p = next;
        }
    }
    free(registry);
}

FusionPatternRegistry* cml_fusion_registry_get_default(void) {
    if (!g_default_registry)
        g_default_registry = cml_fusion_registry_create();
    return g_default_registry;
}

int cml_fusion_register_pattern(FusionPatternRegistry* registry,
                                const char* name,
                                FusionPatternKind kind,
                                FusionTarget target,
                                int priority,
                                FusionMatchFn match,
                                FusionEmitFn emit) {
    if (!registry || !match || !emit || target >= FUSION_TARGET_COUNT)
        return -1;

    FusionPattern* pattern = calloc(1, sizeof(FusionPattern));
    if (!pattern)
        return -1;

    pattern->name = name;
    pattern->kind = kind;
    pattern->target = target;
    pattern->priority = priority;
    pattern->match = match;
    pattern->emit = emit;

    FusionPattern** pp = &registry->patterns[target];
    while (*pp && (*pp)->priority >= priority)
        pp = &(*pp)->next;
    pattern->next = *pp;
    *pp = pattern;

    registry->total_patterns++;
    return 0;
}

int cml_fusion_apply_patterns(FusionPatternRegistry* registry,
                              struct CMLGraph* ir,
                              FusionTarget target) {
    if (!registry || !ir || target >= FUSION_TARGET_COUNT)
        return 0;

    int applied = 0;
    FusionPattern* pattern = registry->patterns[target];

    while (pattern) {
        struct IRNode* node = ir->head;
        while (node) {
            if (node->is_fused) {
                node = node->next;
                continue;
            }

            FusionMatch* match = pattern->match(node, ir);
            if (match) {
                int result = pattern->emit(match, ir);
                if (result == 0)
                    applied++;
                cml_fusion_match_free(match);
            }
            node = node->next;
        }
        pattern = pattern->next;
    }

    LOG_INFO("Applied %d fusion patterns for target %d", applied, target);
    return applied;
}

void cml_fusion_match_free(FusionMatch* match) {
    if (!match)
        return;
    free(match->matched_nodes);
    free(match->match_data);
    free(match);
}
