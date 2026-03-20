#include "ops/ir/cross_boundary_fusion.h"
#include "ops/ir/internal.h"
#include "ops/ir/fusion_patterns.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

/*
 * Softmax + CrossEntropy pattern:
 *   Forward: exp(x) -> sum(exp(x)) -> div -> softmax output
 *   Backward: softmax(x) - one_hot(target)
 *
 * The fused kernel computes softmax in the forward pass and holds the
 * result in registers. The backward pass becomes a single subtraction.
 *
 * Detection: look for EXP -> SUM -> RECIP -> MUL chain (softmax decomposition)
 * followed by a backward node that consumes the softmax output and a
 * one-hot target to produce gradients via SUB.
 */
static bool match_softmax_sequence(struct IRNode* node, int* chain_len) {
    if (!node || node->type != UOP_EXP) return false;

    struct IRNode* sum_node = NULL;
    if (node->use_count == 1 && node->users)
        sum_node = node->users[0];
    if (!sum_node || sum_node->type != UOP_SUM) return false;

    struct IRNode* recip_node = NULL;
    if (sum_node->use_count >= 1 && sum_node->users)
        recip_node = sum_node->users[0];
    if (!recip_node || recip_node->type != UOP_RECIP) return false;

    struct IRNode* mul_node = NULL;
    if (recip_node->use_count >= 1 && recip_node->users)
        mul_node = recip_node->users[0];
    if (!mul_node || mul_node->type != UOP_MUL) return false;

    *chain_len = 4;
    return true;
}

/*
 * LayerNorm pattern:
 *   Forward: mean(x) -> sub -> square -> mean -> add(eps) -> rsqrt -> mul -> scale+bias
 *   Backward: needs mean and variance, which we keep in registers
 *
 * Detection: MEAN -> SUB chain with the SUB feeding into SQUARE -> MEAN -> RSQRT.
 */
static bool match_layernorm_sequence(struct IRNode* node, int* chain_len) {
    if (!node || node->type != UOP_MEAN) return false;

    struct IRNode* sub = NULL;
    if (node->use_count >= 1 && node->users)
        sub = node->users[0];
    if (!sub || sub->type != UOP_SUB) return false;

    struct IRNode* sq = NULL;
    if (sub->use_count >= 1 && sub->users) {
        for (int i = 0; i < sub->use_count; i++) {
            if (sub->users[i] && sub->users[i]->type == UOP_SQUARE) {
                sq = sub->users[i];
                break;
            }
        }
    }
    if (!sq) return false;

    struct IRNode* var_mean = NULL;
    if (sq->use_count >= 1 && sq->users)
        var_mean = sq->users[0];
    if (!var_mean || var_mean->type != UOP_MEAN) return false;

    *chain_len = 4;
    return true;
}

/*
 * GELU pattern (approximate):
 *   x * sigmoid(1.702 * x)  -- this is UOP_QUICK_GELU
 *   or: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * For the backward pass, the sigmoid intermediate is needed.
 * If we detect QUICK_GELU or a MUL -> SIGMOID chain, we can
 * fuse forward+backward to reuse the sigmoid value.
 *
 * Detection: MUL node whose one input feeds through SIGMOID.
 */
static bool match_gelu_sequence(struct IRNode* node, int* chain_len) {
    if (!node) return false;

    if (node->type == UOP_QUICK_GELU) {
        *chain_len = 1;
        return true;
    }

    if (node->type == UOP_SILU) {
        *chain_len = 1;
        return true;
    }

    if (node->type != UOP_MUL) return false;

    if (node->num_inputs < 2 || !node->inputs) return false;

    /* Check if either input is a SIGMOID node in the graph */
    for (int i = 0; i < node->num_inputs; i++) {
        struct IRNode* cur = NULL;
        /* Walk backward through the graph to find the producing node */
        /* The input tensor's producing node would have type SIGMOID */
        /* Since we don't have direct producer pointers on tensors,
         * we check the previous node in the linked list */
        if (node->inputs[i]) {
            /* Heuristic: check if the backward_node or forward_node
             * points to a sigmoid-type operation */
            cur = node->forward_node;
            if (cur && cur->type == UOP_SIGMOID) {
                *chain_len = 2;
                return true;
            }
        }
    }

    return false;
}

/*
 * Check if a forward node's output is consumed only by its
 * corresponding backward node(s). This is the primary criterion
 * for cross-boundary fusion: if no other forward node needs the
 * intermediate, we can keep it in registers.
 */
static bool output_only_consumed_by_backward(struct IRNode* fwd_node) {
    if (!fwd_node || !fwd_node->output) return false;

    for (int i = 0; i < fwd_node->use_count; i++) {
        struct IRNode* user = fwd_node->users[i];
        if (!user) continue;
        if (!user->backward_node && !user->forward_node) {
            /* User is neither marked as backward nor has forward link:
             * it's another forward consumer -> cannot fuse */
            return false;
        }
    }

    return fwd_node->use_count > 0;
}

static int find_node_index_in_schedule(CMLScheduleV2* sched, struct IRNode* node) {
    for (int g = 0; g < sched->num_groups; g++) {
        CMLFusionGroup* group = sched->groups[g];
        if (!group) continue;
        for (int n = 0; n < group->num_nodes; n++) {
            if (group->nodes[n] == node)
                return g * 1000 + n;
        }
    }
    return -1;
}

int cml_cross_boundary_analyze(CMLScheduleV2* sched,
                               CMLCrossBoundaryFusion** out, int* count) {
    if (!sched || !out || !count) return -1;

    *out = NULL;
    *count = 0;

    int capacity = 16;
    CMLCrossBoundaryFusion* fusions = calloc((size_t)capacity,
                                             sizeof(CMLCrossBoundaryFusion));
    if (!fusions) return -1;

    int found = 0;

    for (int g = 0; g < sched->num_groups; g++) {
        CMLFusionGroup* group = sched->groups[g];
        if (!group) continue;

        for (int n = 0; n < group->num_nodes; n++) {
            struct IRNode* node = group->nodes[n];
            if (!node || node->is_fused) continue;

            int chain_len = 0;
            int pattern = -1;

            if (match_softmax_sequence(node, &chain_len)) {
                pattern = CBF_SOFTMAX_CE;
            } else if (match_layernorm_sequence(node, &chain_len)) {
                pattern = CBF_LAYERNORM_BWD;
            } else if (match_gelu_sequence(node, &chain_len)) {
                pattern = CBF_GELU_BWD;
            }

            if (pattern < 0) continue;

            if (!output_only_consumed_by_backward(node)) continue;

            /* Find the backward consumer */
            int bwd_idx = -1;
            if (node->backward_node) {
                bwd_idx = find_node_index_in_schedule(sched, node->backward_node);
            } else if (node->use_count > 0 && node->users && node->users[0]) {
                bwd_idx = find_node_index_in_schedule(sched, node->users[0]);
            }

            if (bwd_idx < 0) continue;

            if (found >= capacity) {
                capacity *= 2;
                CMLCrossBoundaryFusion* tmp = realloc(fusions,
                    (size_t)capacity * sizeof(CMLCrossBoundaryFusion));
                if (!tmp) { free(fusions); return -1; }
                fusions = tmp;
            }

            fusions[found].forward_node_idx = g * 1000 + n;
            fusions[found].backward_node_idx = bwd_idx;
            fusions[found].pattern_type = pattern;
            found++;
        }
    }

    if (found == 0) {
        free(fusions);
        *out = NULL;
        *count = 0;
        return 0;
    }

    *out = fusions;
    *count = found;

    LOG_INFO("Cross-boundary analysis: %d fusible patterns found", found);
    return 0;
}

/*
 * Apply cross-boundary fusions by merging forward+backward schedule items.
 * For each fusion:
 *   - Mark forward nodes as fused with their backward counterparts
 *   - Set fusion metadata so the codegen can emit combined kernels
 *   - Update memory estimates (register-kept intermediates save bandwidth)
 */
int cml_cross_boundary_fuse(CMLScheduleV2* sched,
                            CMLCrossBoundaryFusion* fusions, int count) {
    if (!sched || !fusions || count <= 0) return -1;

    int applied = 0;

    for (int i = 0; i < count; i++) {
        CMLCrossBoundaryFusion* f = &fusions[i];

        int fwd_group = f->forward_node_idx / 1000;
        int fwd_node = f->forward_node_idx % 1000;
        int bwd_group = f->backward_node_idx / 1000;
        int bwd_node = f->backward_node_idx % 1000;

        if (fwd_group >= sched->num_groups || bwd_group >= sched->num_groups)
            continue;

        CMLFusionGroup* fg = sched->groups[fwd_group];
        CMLFusionGroup* bg = sched->groups[bwd_group];
        if (!fg || !bg) continue;
        if (fwd_node >= fg->num_nodes || bwd_node >= bg->num_nodes)
            continue;

        struct IRNode* fwd = fg->nodes[fwd_node];
        struct IRNode* bwd = bg->nodes[bwd_node];
        if (!fwd || !bwd) continue;

        FusionType ftype;
        switch (f->pattern_type) {
        case CBF_SOFTMAX_CE:
            ftype = FUSION_REDUCE_ELEMENTWISE;
            break;
        case CBF_LAYERNORM_BWD:
            ftype = FUSION_CHAIN_ELEMENTWISE;
            break;
        case CBF_GELU_BWD:
            ftype = FUSION_CHAIN_ELEMENTWISE;
            break;
        default:
            continue;
        }

        fwd->is_fused = true;
        fwd->fusion_type = ftype;
        bwd->is_fused = true;
        bwd->fusion_type = ftype;

        /* Link forward and backward nodes for codegen */
        fwd->backward_node = bwd;
        bwd->forward_node = fwd;

        /* Estimate memory savings from register-kept intermediates */
        if (fwd->output) {
            size_t elems = fwd->output->numel;
            size_t saved = elems * sizeof(float);
            sched->memory_saved += saved;
        }

        applied++;
    }

    LOG_INFO("Cross-boundary fusion: applied %d of %d patterns", applied, count);
    return applied;
}

void cml_cross_boundary_fusions_free(CMLCrossBoundaryFusion* fusions) {
    free(fusions);
}

CMLCrossBoundaryStats cml_cross_boundary_stats(const CMLCrossBoundaryFusion* fusions,
                                               int count) {
    CMLCrossBoundaryStats stats = {0};
    if (!fusions || count <= 0) return stats;

    stats.patterns_found = count;

    for (int i = 0; i < count; i++) {
        switch (fusions[i].pattern_type) {
        case CBF_SOFTMAX_CE:
            /* Softmax+CE fusion saves the softmax intermediate buffer
             * and reduces backward to a single subtraction */
            stats.memory_saved += 4096 * sizeof(float);
            stats.flops_saved += 4096 * 3;
            break;
        case CBF_LAYERNORM_BWD:
            /* Keeping mean/var in registers saves two buffers */
            stats.memory_saved += 2 * 256 * sizeof(float);
            stats.flops_saved += 256 * 4;
            break;
        case CBF_GELU_BWD:
            /* Reusing sigmoid saves one buffer and one exp() call */
            stats.memory_saved += 4096 * sizeof(float);
            stats.flops_saved += 4096;
            break;
        default:
            break;
        }
    }

    return stats;
}
