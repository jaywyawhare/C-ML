#include "ops/ir/multi_output.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

#define MAX_INPUTS_TRACKED 32

typedef struct {
    Tensor* inputs[MAX_INPUTS_TRACKED];
    int num_inputs;
    CMLScheduleItemType type;
} GroupSignature;

static void collect_group_inputs(const CMLFusionGroup* g, GroupSignature* sig) {
    sig->num_inputs = 0;
    sig->type = g->type;

    for (int i = 0; i < g->num_nodes; i++) {
        struct IRNode* node = g->nodes[i];
        if (!node) continue;

        for (int j = 0; j < node->num_inputs; j++) {
            if (!node->inputs || !node->inputs[j]) continue;
            Tensor* inp = node->inputs[j];

            int is_internal = 0;
            for (int k = 0; k < g->num_nodes; k++) {
                if (g->nodes[k] && g->nodes[k]->output == inp) {
                    is_internal = 1;
                    break;
                }
            }
            if (is_internal) continue;

            int already_tracked = 0;
            for (int k = 0; k < sig->num_inputs; k++) {
                if (sig->inputs[k] == inp) {
                    already_tracked = 1;
                    break;
                }
            }
            if (!already_tracked && sig->num_inputs < MAX_INPUTS_TRACKED) {
                sig->inputs[sig->num_inputs++] = inp;
            }
        }
    }
}

static int signatures_match(const GroupSignature* a, const GroupSignature* b) {
    if (a->type != b->type) return 0;
    if (a->num_inputs != b->num_inputs) return 0;

    for (int i = 0; i < a->num_inputs; i++) {
        int found = 0;
        for (int j = 0; j < b->num_inputs; j++) {
            if (a->inputs[i] == b->inputs[j]) {
                found = 1;
                break;
            }
        }
        if (!found) return 0;
    }
    return 1;
}

static int groups_have_distinct_outputs(const CMLFusionGroup* a, const CMLFusionGroup* b) {
    for (int i = 0; i < a->num_nodes; i++) {
        if (!a->nodes[i] || !a->nodes[i]->output) continue;
        for (int j = 0; j < b->num_nodes; j++) {
            if (!b->nodes[j] || !b->nodes[j]->output) continue;
            if (a->nodes[i]->output == b->nodes[j]->output) return 0;
        }
    }
    return 1;
}

int cml_multi_output_analyze(CMLScheduleV2* sched, int** merge_groups, int* num_merges) {
    if (!sched || !merge_groups || !num_merges) return -1;
    *merge_groups = NULL;
    *num_merges = 0;

    int ng = sched->num_groups;
    if (ng < 2) return 0;

    GroupSignature* sigs = calloc((size_t)ng, sizeof(GroupSignature));
    if (!sigs) return -1;

    for (int i = 0; i < ng; i++) {
        if (sched->groups[i])
            collect_group_inputs(sched->groups[i], &sigs[i]);
    }

    int* pairs = calloc((size_t)(ng * 2), sizeof(int));
    if (!pairs) { free(sigs); return -1; }
    int count = 0;

    int* merged = calloc((size_t)ng, sizeof(int));
    if (!merged) { free(sigs); free(pairs); return -1; }

    for (int i = 0; i < ng; i++) {
        if (merged[i]) continue;
        if (!sched->groups[i]) continue;
        if (sigs[i].type != SCHED_ELEMENTWISE) continue;

        for (int j = i + 1; j < ng; j++) {
            if (merged[j]) continue;
            if (!sched->groups[j]) continue;

            if (signatures_match(&sigs[i], &sigs[j]) &&
                groups_have_distinct_outputs(sched->groups[i], sched->groups[j])) {
                pairs[count * 2] = i;
                pairs[count * 2 + 1] = j;
                count++;
                merged[j] = 1;
                break;
            }
        }
    }

    free(merged);
    free(sigs);

    if (count == 0) {
        free(pairs);
        return 0;
    }

    *merge_groups = realloc(pairs, (size_t)(count * 2) * sizeof(int));
    if (!*merge_groups) *merge_groups = pairs;
    *num_merges = count;
    return count;
}

int cml_multi_output_fuse(CMLScheduleV2* sched, int* merge_groups, int num_merges) {
    if (!sched || !merge_groups || num_merges <= 0) return -1;

    for (int m = 0; m < num_merges; m++) {
        int gi = merge_groups[m * 2];
        int gj = merge_groups[m * 2 + 1];

        CMLFusionGroup* primary = sched->groups[gi];
        CMLFusionGroup* secondary = sched->groups[gj];
        if (!primary || !secondary) continue;

        for (int i = 0; i < secondary->num_nodes; i++) {
            struct IRNode* node = secondary->nodes[i];
            if (!node) continue;

            int duplicate = 0;
            for (int k = 0; k < primary->num_nodes; k++) {
                if (primary->nodes[k] == node) {
                    duplicate = 1;
                    break;
                }
            }
            if (duplicate) continue;

            if (primary->num_nodes >= primary->node_capacity) {
                int nc = primary->node_capacity * 2;
                struct IRNode** tmp = realloc(primary->nodes,
                                              (size_t)nc * sizeof(struct IRNode*));
                if (!tmp) return -1;
                primary->nodes = tmp;
                primary->node_capacity = nc;
            }
            primary->nodes[primary->num_nodes++] = node;
            primary->total_flops += secondary->total_flops / secondary->num_nodes;
            primary->total_memory += secondary->total_memory / secondary->num_nodes;
        }

        free(secondary->nodes);
        free(secondary->eliminated_buffers);
        free(secondary);
        sched->groups[gj] = NULL;
    }

    int write = 0;
    for (int i = 0; i < sched->num_groups; i++) {
        if (sched->groups[i]) {
            sched->groups[write++] = sched->groups[i];
        }
    }
    sched->num_groups = write;
    sched->total_groups_after = write;

    if (sched->execution_order) {
        free(sched->execution_order);
        sched->execution_order = calloc((size_t)write, sizeof(int));
        if (sched->execution_order) {
            for (int i = 0; i < write; i++)
                sched->execution_order[i] = i;
            sched->num_ordered = write;
        }
    }

    if (sched->total_ops_before > 0 && write > 0)
        sched->fusion_ratio = (float)sched->total_ops_before / (float)write;

    return 0;
}
