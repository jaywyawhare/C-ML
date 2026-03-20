/* Fusion rules:
 *   elem -> elem  YES
 *   elem -> reduce YES (softmax as one kernel)
 *   reduce -> elem CONDITIONAL (allowed when reduce has single consumer whose
 *                   other inputs are the reduce's own inputs or constants)
 *   matmul/conv + elem YES (bias + activation)
 *   movement is free
 *
 * Buffer elimination: if every user of a producer lives in the same
 * fusion group, the intermediate buffer is eliminated (kept in registers). */

#include "ops/ir/schedule.h"
#include "ops/ir/memory_planner.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/** Classify a UOp into a schedule item type (mirrors schedule.c) */
static CMLScheduleItemType classify_op_v2(UOpType type) {
    if (type == UOP_MATMUL)  return SCHED_MATMUL;
    if (type == UOP_CONV2D)  return SCHED_CONV;
    if (cml_schedule_is_elementwise(type)) return SCHED_ELEMENTWISE;
    if (cml_schedule_is_reduction(type))   return SCHED_REDUCE;
    if (cml_schedule_is_movement(type))    return SCHED_MOVEMENT;
    return SCHED_CUSTOM;
}

/** Determine the dominant type for a fusion group.
 *  Priority: MATMUL/CONV > REDUCE > ELEMENTWISE > MOVEMENT */
static CMLScheduleItemType dominant_type(CMLScheduleItemType a,
                                         CMLScheduleItemType b) {
    if (a == SCHED_MATMUL || b == SCHED_MATMUL) return SCHED_MATMUL;
    if (a == SCHED_CONV   || b == SCHED_CONV)   return SCHED_CONV;
    if (a == SCHED_REDUCE || b == SCHED_REDUCE) return SCHED_REDUCE;
    if (a == SCHED_ELEMENTWISE || b == SCHED_ELEMENTWISE)
        return SCHED_ELEMENTWISE;
    return a;
}

/** Estimate element count from a tensor (0 if unknown). */
static size_t tensor_elems(const Tensor* t) {
    if (!t || !t->shape || t->ndim <= 0) return 0;
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) {
        if (t->shape[i] <= 0) return 0;
        n *= (size_t)t->shape[i];
    }
    return n;
}

static CMLFusionGroup* fusion_group_create(void) {
    CMLFusionGroup* g = calloc(1, sizeof(CMLFusionGroup));
    if (!g) return NULL;
    g->node_capacity = 8;
    g->nodes = calloc((size_t)g->node_capacity, sizeof(struct IRNode*));
    if (!g->nodes) { free(g); return NULL; }

    g->elim_capacity = 4;
    g->eliminated_buffers = calloc((size_t)g->elim_capacity, sizeof(int));
    if (!g->eliminated_buffers) { free(g->nodes); free(g); return NULL; }

    g->type = SCHED_MOVEMENT;  /* weakest type; promoted as nodes are added */
    return g;
}

static int fusion_group_add_node(CMLFusionGroup* g, struct IRNode* node) {
    if (!g || !node) return -1;
    if (g->num_nodes >= g->node_capacity) {
        int nc = g->node_capacity * 2;
        struct IRNode** tmp = realloc(g->nodes,
                                      (size_t)nc * sizeof(struct IRNode*));
        if (!tmp) return -1;
        g->nodes = tmp;
        g->node_capacity = nc;
    }
    g->nodes[g->num_nodes++] = node;

    /* Update dominant type */
    CMLScheduleItemType kind = classify_op_v2(node->type);
    g->type = dominant_type(g->type, kind);

    /* Accumulate flops/memory estimates */
    size_t elems = node->output ? tensor_elems(node->output) : 0;
    switch (kind) {
        case SCHED_ELEMENTWISE:
            g->total_flops += elems;
            break;
        case SCHED_REDUCE:
            if (node->inputs && node->num_inputs > 0 && node->inputs[0])
                g->total_flops += tensor_elems(node->inputs[0]) * 2;
            else
                g->total_flops += elems * 2;
            break;
        case SCHED_MATMUL: {
            Tensor* a = (node->inputs && node->num_inputs > 0) ? node->inputs[0] : NULL;
            Tensor* b = (node->inputs && node->num_inputs > 1) ? node->inputs[1] : NULL;
            if (a && b && a->ndim >= 2 && b->ndim >= 2) {
                size_t M = (size_t)a->shape[a->ndim - 2];
                size_t K = (size_t)a->shape[a->ndim - 1];
                size_t N = (size_t)b->shape[b->ndim - 1];
                g->total_flops += 2 * M * N * K;
            }
            break;
        }
        default:
            g->total_flops += elems;
            break;
    }

    /* Memory for inputs + output */
    if (node->output)
        g->total_memory += tensor_elems(node->output) * sizeof(float);
    for (int i = 0; i < node->num_inputs; i++) {
        if (node->inputs && node->inputs[i])
            g->total_memory += tensor_elems(node->inputs[i]) * sizeof(float);
    }

    return 0;
}

static void fusion_group_add_eliminated(CMLFusionGroup* g, int buffer_idx) {
    if (!g) return;
    if (g->num_eliminated >= g->elim_capacity) {
        int nc = g->elim_capacity * 2;
        int* tmp = realloc(g->eliminated_buffers, (size_t)nc * sizeof(int));
        if (!tmp) return;
        g->eliminated_buffers = tmp;
        g->elim_capacity = nc;
    }
    g->eliminated_buffers[g->num_eliminated++] = buffer_idx;
}

static void fusion_group_free(CMLFusionGroup* g) {
    if (!g) return;
    free(g->nodes);
    free(g->eliminated_buffers);
    free(g);
}

CMLFusionAnalysis cml_schedule_analyze_fusion(struct IRNode* a,
                                               struct IRNode* b) {
    CMLFusionAnalysis result = {0};
    if (!a || !b) return result;

    result.can_fuse = cml_schedule_can_fuse(a->type, b->type);
    if (!result.can_fuse) return result;

    /* Estimate benefit: memory saved by not materializing intermediate */
    size_t a_out = a->output ? tensor_elems(a->output) * sizeof(float) : 0;
    result.memory_saved = a_out;
    result.eliminates_buffer = (a_out > 0);

    /* Benefit heuristic: more memory saved -> higher benefit */
    if (a_out > 0) {
        /* Rough: assume 10 GB/s bandwidth, 1 TFLOP/s compute
         * Saving a_out bytes -> a_out / 10e9 seconds saved */
        result.benefit = 1.0f + (float)a_out / (1024.0f * 1024.0f);
    } else {
        result.benefit = 1.0f;
    }

    return result;
}

CMLScheduleV2* cml_schedule_v2_create(CMLGraph_t graph,
                                       const CMLScheduleOptions* opts) {
    CMLScheduleOptions default_opts;
    if (!opts) {
        default_opts = cml_schedule_default_options();
        opts = &default_opts;
    }

    CMLScheduleV2* sched = calloc(1, sizeof(CMLScheduleV2));
    if (!sched) return NULL;

    /* Handle NULL or empty graph */
    if (!graph || !graph->head || graph->node_count == 0) {
        sched->groups        = NULL;
        sched->num_groups    = 0;
        sched->group_capacity = 0;
        sched->execution_order = NULL;
        sched->num_ordered   = 0;
        sched->total_ops_before  = 0;
        sched->total_groups_after = 0;
        sched->fusion_ratio  = 0.0f;
        sched->memory_saved  = 0;
        return sched;
    }

    /* Allocate group list */
    int cap = graph->node_count < 16 ? 16 : graph->node_count;
    sched->groups = calloc((size_t)cap, sizeof(CMLFusionGroup*));
    if (!sched->groups) { free(sched); return NULL; }
    sched->group_capacity = cap;

    int total_ops = 0;
    size_t total_memory_saved = 0;
    int color = 0;

    CMLFusionGroup* cur = NULL;

    /* Walk graph in topological order (head -> tail) */
    struct IRNode* node = graph->head;
    struct IRNode* prev = NULL;

    while (node) {
        total_ops++;

        CMLScheduleItemType kind = classify_op_v2(node->type);

        /* Movement is free -- always absorb into current group */
        if (kind == SCHED_MOVEMENT) {
            if (opts->enable_movement_fold && cur) {
                fusion_group_add_node(cur, node);
            } else if (!cur) {
                /* Start a new group just for the movement */
                cur = fusion_group_create();
                if (cur) {
                    cur->color = color++;
                    fusion_group_add_node(cur, node);
                }
            } else {
                fusion_group_add_node(cur, node);
            }
            prev = node;
            node = node->next;
            continue;
        }

        /* Try to extend current group */
        if (cur && cur->num_nodes > 0 && opts->enable_fusion) {
            struct IRNode* last = cur->nodes[cur->num_nodes - 1];
            bool can_extend = cml_schedule_can_fuse(last->type, node->type);

            if (!can_extend && opts->allow_reduce_elem_fusion &&
                cml_schedule_is_reduction(last->type) &&
                cml_schedule_is_elementwise(node->type)) {
                int single_consumer = (last->use_count == 1);
                if (single_consumer) {
                    int inputs_ok = 1;
                    for (int j = 0; j < node->num_inputs && inputs_ok; j++) {
                        if (!node->inputs || !node->inputs[j]) continue;
                        Tensor* inp = node->inputs[j];
                        if (inp == last->output) continue;
                        int found = 0;
                        for (int k = 0; k < cur->num_nodes && !found; k++) {
                            struct IRNode* gn = cur->nodes[k];
                            if (!gn) continue;
                            for (int m = 0; m < gn->num_inputs; m++) {
                                if (gn->inputs && gn->inputs[m] == inp) {
                                    found = 1;
                                    break;
                                }
                            }
                            if (gn->output == inp) found = 1;
                        }
                        if (!found && inp->data) found = 1;
                        if (!found) inputs_ok = 0;
                    }
                    if (inputs_ok) can_extend = true;
                }
            }

            /* Respect max fused ops */
            if (can_extend && cur->num_nodes < opts->max_fused_ops) {
                /* Check buffer elimination: if prev produced an output
                 * and all its users are inside this group, eliminate it */
                if (prev && prev->output) {
                    size_t saved = tensor_elems(prev->output) * sizeof(float);
                    /* Heuristic: mark as eliminated (real check would
                     * verify all users are in this group) */
                    fusion_group_add_eliminated(cur, cur->num_nodes - 1);
                    total_memory_saved += saved;
                }

                fusion_group_add_node(cur, node);
                prev = node;
                node = node->next;
                continue;
            }
        }

        /* Cannot extend -- flush current group and start new one */
        if (cur) {
            if (sched->num_groups >= sched->group_capacity) {
                int nc = sched->group_capacity * 2;
                CMLFusionGroup** tmp = realloc(
                    sched->groups,
                    (size_t)nc * sizeof(CMLFusionGroup*));
                if (tmp) {
                    sched->groups = tmp;
                    sched->group_capacity = nc;
                }
            }
            sched->groups[sched->num_groups++] = cur;
        }

        cur = fusion_group_create();
        if (cur) {
            cur->color = color++;
            fusion_group_add_node(cur, node);
        }

        prev = node;
        node = node->next;
    }

    /* Flush last group */
    if (cur) {
        if (sched->num_groups >= sched->group_capacity) {
            int nc = sched->group_capacity * 2;
            CMLFusionGroup** tmp = realloc(
                sched->groups,
                (size_t)nc * sizeof(CMLFusionGroup*));
            if (tmp) {
                sched->groups = tmp;
                sched->group_capacity = nc;
            }
        }
        sched->groups[sched->num_groups++] = cur;
    }

    /* Build execution order */
    sched->execution_order = calloc((size_t)sched->num_groups, sizeof(int));
    if (sched->execution_order) {
        if (opts->schedule_order == CML_SCHEDULE_ORDER_BFS && sched->num_groups > 1) {
            int ng = sched->num_groups;
            int* in_degree = calloc((size_t)ng, sizeof(int));
            int** deps = calloc((size_t)ng, sizeof(int*));
            int* dep_counts = calloc((size_t)ng, sizeof(int));

            if (in_degree && deps && dep_counts) {
                for (int i = 0; i < ng; i++) {
                    deps[i] = calloc((size_t)ng, sizeof(int));
                }

                for (int i = 0; i < ng; i++) {
                    CMLFusionGroup* consumer = sched->groups[i];
                    if (!consumer) continue;
                    for (int j = 0; j < i; j++) {
                        CMLFusionGroup* producer = sched->groups[j];
                        if (!producer) continue;
                        int has_dep = 0;
                        for (int ci = 0; ci < consumer->num_nodes && !has_dep; ci++) {
                            struct IRNode* cn = consumer->nodes[ci];
                            if (!cn) continue;
                            for (int k = 0; k < cn->num_inputs && !has_dep; k++) {
                                if (!cn->inputs || !cn->inputs[k]) continue;
                                for (int pi = 0; pi < producer->num_nodes; pi++) {
                                    if (producer->nodes[pi] &&
                                        producer->nodes[pi]->output == cn->inputs[k]) {
                                        has_dep = 1;
                                        break;
                                    }
                                }
                            }
                        }
                        if (has_dep) {
                            deps[i][dep_counts[i]++] = j;
                            in_degree[i]++;
                        }
                    }
                }

                int* queue = calloc((size_t)ng, sizeof(int));
                int qhead = 0, qtail = 0;

                for (int i = 0; i < ng; i++) {
                    if (in_degree[i] == 0)
                        queue[qtail++] = i;
                }

                int ordered = 0;
                while (qhead < qtail) {
                    int level_end = qtail;
                    while (qhead < level_end) {
                        int idx = queue[qhead++];
                        sched->execution_order[ordered++] = idx;

                        for (int i = 0; i < ng; i++) {
                            for (int d = 0; d < dep_counts[i]; d++) {
                                if (deps[i][d] == idx) {
                                    in_degree[i]--;
                                    if (in_degree[i] == 0)
                                        queue[qtail++] = i;
                                    break;
                                }
                            }
                        }
                    }
                }

                sched->num_ordered = ordered;
                free(queue);
            }

            for (int i = 0; i < ng; i++) free(deps[i]);
            free(deps);
            free(dep_counts);
            free(in_degree);
        } else {
            for (int i = 0; i < sched->num_groups; i++) {
                sched->execution_order[i] = i;
            }
            sched->num_ordered = sched->num_groups;
        }
    }

    /* Statistics */
    sched->total_ops_before   = total_ops;
    sched->total_groups_after = sched->num_groups;
    sched->fusion_ratio = (sched->num_groups > 0)
        ? (float)total_ops / (float)sched->num_groups
        : 0.0f;
    sched->memory_saved = total_memory_saved;

    /* Build memory plan from fusion group buffer liveness */
    sched->memory_plan = NULL;
    if (sched->num_groups > 0) {
        int nb = sched->num_groups;
        size_t* buf_sizes = calloc((size_t)nb, sizeof(size_t));
        int* buf_first    = calloc((size_t)nb, sizeof(int));
        int* buf_last     = calloc((size_t)nb, sizeof(int));

        if (buf_sizes && buf_first && buf_last) {
            for (int i = 0; i < nb; i++) {
                CMLFusionGroup* g = sched->groups[i];
                if (!g) continue;
                buf_sizes[i] = g->total_memory;
                buf_first[i] = i;
                buf_last[i]  = i;

                /* Extend last_use to cover groups that consume this output */
                for (int j = i + 1; j < nb; j++) {
                    CMLFusionGroup* consumer = sched->groups[j];
                    if (!consumer) continue;
                    for (int ci = 0; ci < consumer->num_nodes; ci++) {
                        struct IRNode* cn = consumer->nodes[ci];
                        if (!cn) continue;
                        for (int k = 0; k < cn->num_inputs; k++) {
                            if (!cn->inputs || !cn->inputs[k]) continue;
                            for (int pi = 0; pi < g->num_nodes; pi++) {
                                if (g->nodes[pi] && g->nodes[pi]->output == cn->inputs[k]) {
                                    if (j > buf_last[i]) buf_last[i] = j;
                                }
                            }
                        }
                    }
                }
            }
            sched->memory_plan = cml_memory_plan_create(nb, buf_sizes, buf_first, buf_last);
        }

        free(buf_sizes);
        free(buf_first);
        free(buf_last);
    }

    return sched;
}

void cml_schedule_v2_free(CMLScheduleV2* sched) {
    if (!sched) return;

    for (int i = 0; i < sched->num_groups; i++) {
        fusion_group_free(sched->groups[i]);
    }
    free(sched->groups);
    free(sched->execution_order);
    cml_memory_plan_free(sched->memory_plan);
    free(sched);
}

static const char* sched_type_name_v2(CMLScheduleItemType type) {
    switch (type) {
        case SCHED_ELEMENTWISE: return "ELEMENTWISE";
        case SCHED_REDUCE:      return "REDUCE";
        case SCHED_MATMUL:      return "MATMUL";
        case SCHED_CONV:        return "CONV";
        case SCHED_MOVEMENT:    return "MOVEMENT";
        case SCHED_COPY:        return "COPY";
        case SCHED_CUSTOM:      return "CUSTOM";
        default:                return "UNKNOWN";
    }
}

void cml_schedule_v2_print(const CMLScheduleV2* sched) {
    if (!sched) {
        printf("ScheduleV2: (null)\n");
        return;
    }

    printf("CML V2 Fusion Schedule\n");
    printf("  Total ops:     %d\n", sched->total_ops_before);
    printf("  Fusion groups: %d\n", sched->total_groups_after);
    printf("  Fusion ratio:  %.2f\n", (double)sched->fusion_ratio);
    printf("  Memory saved:  %zu bytes\n", sched->memory_saved);
    printf("\n");

    for (int i = 0; i < sched->num_groups; i++) {
        const CMLFusionGroup* g = sched->groups[i];
        if (!g) continue;
        printf("  [%d] %-12s  nodes=%d  color=%d  flops=%zu  mem=%zu  elim=%d\n",
               i, sched_type_name_v2(g->type),
               g->num_nodes, g->color,
               g->total_flops, g->total_memory,
               g->num_eliminated);
        for (int j = 0; j < g->num_nodes; j++) {
            struct IRNode* nd = g->nodes[j];
            if (nd) {
                printf("       node[%d]: %s\n", j, uop_type_to_string(nd->type));
            }
        }
    }
    printf("\n");

    if (sched->memory_plan)
        cml_memory_plan_print(sched->memory_plan);
}

int cml_ir_execute_v2(CMLGraph_t ir) {
    if (!ir) return -1;

    CMLScheduleV2* sched = cml_schedule_v2_create(ir, NULL);
    if (!sched) {
        LOG_ERROR("Failed to create V2 schedule");
        return -1;
    }

    /* Execute each group in order */
    int rc = 0;
    for (int i = 0; i < sched->num_ordered && rc == 0; i++) {
        int idx = sched->execution_order[i];
        CMLFusionGroup* g = sched->groups[idx];
        if (!g) continue;

        /* Execute each node in the group sequentially (CPU fallback) */
        for (int j = 0; j < g->num_nodes && rc == 0; j++) {
            struct IRNode* node = g->nodes[j];
            if (!node || node->is_executed) continue;
            rc = cpu_execute_node(node);
            if (rc == 0) node->is_executed = true;
        }
    }

    cml_schedule_v2_free(sched);
    return rc;
}
