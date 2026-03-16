/**
 * @file optimization.c
 * @brief IR optimization passes (fusion, DCE, reordering)
 */

#include "ops/ir/ir.h"
#include "ops/ir/optimization.h"
#include "ops/ir/pattern_matcher.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static struct IRNode* find_node_by_output(CMLGraph_t ir, const char* output_name) {
    if (!ir || !output_name)
        return NULL;

    struct IRNode* node = ir->head;
    while (node) {
        if (node->output_name && strcmp(node->output_name, output_name) == 0) {
            return node;
        }
        node = node->next;
    }
    return NULL;
}

static int build_dependency_graph(CMLGraph_t ir) {
    if (!ir)
        return -1;

    struct IRNode* node = ir->head;
    while (node) {
        node->use_count = 0;
        if (node->users) {
            free(node->users);
            node->users          = NULL;
            node->users_capacity = 0;
        }
        node = node->next;
    }

    node = ir->head;
    while (node) {
        for (int i = 0; i < node->num_inputs; i++) {
            struct IRNode* producer = find_node_by_output(ir, node->input_names[i]);
            if (producer) {
                if (producer->use_count >= producer->users_capacity) {
                    int new_capacity =
                        producer->users_capacity == 0 ? 4 : producer->users_capacity * 2;
                    struct IRNode** new_users =
                        realloc(producer->users, (size_t)new_capacity * sizeof(struct IRNode*));
                    if (!new_users)
                        return -1;
                    producer->users          = new_users;
                    producer->users_capacity = new_capacity;
                }
                producer->users[producer->use_count++] = node;
            }
        }
        node = node->next;
    }

    return 0;
}

static void mark_reachable_nodes(CMLGraph_t ir) {
    if (!ir)
        return;

    struct IRNode* node = ir->head;
    while (node) {
        node->is_used = false;
        node          = node->next;
    }

    struct IRNode* stack[256]; // Stack for DFS
    int stack_top = 0;

    if (ir->tail) {
        stack[stack_top++] = ir->tail;
        ir->tail->is_used  = true;
    }

    while (stack_top > 0) {
        struct IRNode* current = stack[--stack_top];

        for (int i = 0; i < current->num_inputs; i++) {
            struct IRNode* producer = find_node_by_output(ir, current->input_names[i]);
            if (producer && !producer->is_used) {
                producer->is_used = true;
                if (stack_top < 256) {
                    stack[stack_top++] = producer;
                }
            }
        }
    }
}

static int remove_dead_nodes(CMLGraph_t ir) {
    if (!ir)
        return -1;

    struct IRNode* prev = NULL;
    struct IRNode* node = ir->head;
    int removed         = 0;

    while (node) {
        struct IRNode* next = node->next;

        if (!node->is_used && node->use_count == 0) {
            if (prev) {
                prev->next = next;
            } else {
                ir->head = next;
            }

            if (node == ir->tail) {
                ir->tail = prev;
            }

            // CRITICAL: Clear tensor's ir_node pointer before freeing the node
            // This prevents dangling pointers when tensors are later freed
            if (node->output) {
                node->output->ir_node    = NULL;
                node->output->ir_context = NULL;
            }

            if (node->input_names) {
                for (int i = 0; i < node->num_inputs; i++) {
                    if (node->input_names[i]) {
                        free(node->input_names[i]);
                    }
                }
                free(node->input_names);
            }
            if (node->output_name) {
                free(node->output_name);
            }
            if (node->users) {
                free(node->users);
            }
            if (node->inputs) {
                free(node->inputs);
            }
            if (node->input_shapes) {
                free(node->input_shapes);
            }
            if (node->input_ndims) {
                free(node->input_ndims);
            }
            if (node->output_shape) {
                free(node->output_shape);
            }
            if (node->broadcast) {
                if (node->broadcast->broadcast_dims) {
                    free(node->broadcast->broadcast_dims);
                }
                if (node->broadcast->broadcast_strides) {
                    free(node->broadcast->broadcast_strides);
                }
                free(node->broadcast);
            }
            if (node->saved_for_backward) {
                free(node->saved_for_backward);
            }
            free(node);

            ir->node_count--;
            removed++;
        } else {
            prev = node;
        }

        node = next;
    }

    if (removed > 0) {
        LOG_DEBUG("Removed %d dead nodes", removed);
    }

    return 0;
}

static bool node_uses_output(struct IRNode* node1, struct IRNode* node2) {
    if (!node1 || !node2 || !node1->output_name)
        return false;

    for (int i = 0; i < node2->num_inputs; i++) {
        if (node2->input_names[i] && strcmp(node2->input_names[i], node1->output_name) == 0) {
            return true;
        }
    }
    return false;
}

static bool can_fuse_operations(struct IRNode* node1, struct IRNode* node2,
                                FusionType* fusion_type) {
    if (!node1 || !node2 || !fusion_type)
        return false;

    if (!node_uses_output(node1, node2))
        return false;

    // Pattern 1: MUL + ADD -> FMA (Fused Multiply-Add)
    if (node1->type == UOP_MUL && node2->type == UOP_ADD) {
        *fusion_type = FUSION_FMA;
        return true;
    }

    // Pattern 2: NEG + ADD -> SUB
    if (node1->type == UOP_NEG && node2->type == UOP_ADD) {
        *fusion_type = FUSION_NEG_ADD;
        return true;
    }

    // Pattern 3: EXP + LOG -> identity (if same input)
    if (node1->type == UOP_EXP && node2->type == UOP_LOG) {
        *fusion_type = FUSION_EXP_LOG;
        return true;
    }

    // Pattern 4: MUL + DIV -> identity (if same operand)
    if (node1->type == UOP_MUL && node2->type == UOP_DIV) {
        // Check if dividing by same operand
        if (node1->num_inputs >= 1 && node2->num_inputs >= 2) {
            // If MUL(a, b) and DIV(mul_result, a) or DIV(mul_result, b)
            if ((node2->input_names[1] && node1->input_names[0] &&
                 strcmp(node2->input_names[1], node1->input_names[0]) == 0) ||
                (node2->input_names[1] && node1->input_names[1] &&
                 strcmp(node2->input_names[1], node1->input_names[1]) == 0)) {
                *fusion_type = FUSION_MUL_DIV;
                return true;
            }
        }
    }

    // Pattern 5: SQRT + MUL -> sqrt_mul
    if (node1->type == UOP_SQRT && node2->type == UOP_MUL) {
        *fusion_type = FUSION_SQRT_MUL;
        return true;
    }

    // Pattern 6: EXP + RECIP -> exp_recip
    if (node1->type == UOP_EXP && node2->type == UOP_RECIP) {
        *fusion_type = FUSION_EXP_RECIP;
        return true;
    }

    // Pattern 7: Elementwise chain (multiple elementwise ops in sequence)
    // Check if both are elementwise and can be chained
    bool node1_is_elementwise =
        (node1->type == UOP_ADD || node1->type == UOP_SUB || node1->type == UOP_MUL ||
         node1->type == UOP_DIV || node1->type == UOP_EXP || node1->type == UOP_LOG ||
         node1->type == UOP_SQRT || node1->type == UOP_NEG || node1->type == UOP_RECIP ||
         node1->type == UOP_ABS);
    bool node2_is_elementwise =
        (node2->type == UOP_ADD || node2->type == UOP_SUB || node2->type == UOP_MUL ||
         node2->type == UOP_DIV || node2->type == UOP_EXP || node2->type == UOP_LOG ||
         node2->type == UOP_SQRT || node2->type == UOP_NEG || node2->type == UOP_RECIP ||
         node2->type == UOP_ABS);

    if (node1_is_elementwise && node2_is_elementwise) {
        // Check if node2 only uses node1's output (single consumer)
        if (node1->use_count == 1) {
            *fusion_type = FUSION_CHAIN_ELEMENTWISE;
            return true;
        }
    }

    // Pattern 8: ADD + MUL -> can be fused for better cache locality
    if (node1->type == UOP_ADD && node2->type == UOP_MUL) {
        // If ADD result is only used by MUL, can fuse
        if (node1->use_count == 1) {
            *fusion_type = FUSION_CHAIN_ELEMENTWISE;
            return true;
        }
    }

    // Pattern 9: NEG + EXP -> exp(-x) (common in sigmoid/softmax)
    if (node1->type == UOP_NEG && node2->type == UOP_EXP) {
        *fusion_type = FUSION_CHAIN_ELEMENTWISE;
        return true;
    }

    // Pattern 10: LOG + MUL -> log(x) * y (common in loss functions)
    if (node1->type == UOP_LOG && node2->type == UOP_MUL) {
        if (node1->use_count == 1) {
            *fusion_type = FUSION_CHAIN_ELEMENTWISE;
            return true;
        }
    }

    // Pattern 11: Reduction + elementwise (e.g., SUM + DIV for mean)
    bool node1_is_reduction =
        (node1->type == UOP_SUM || node1->type == UOP_MEAN || node1->type == UOP_MAX_REDUCE);
    if (node1_is_reduction && node2_is_elementwise) {
        if (node1->use_count == 1) {
            *fusion_type = FUSION_REDUCE_ELEMENTWISE;
            return true;
        }
    }

    return false;
}

static FusedKernel* create_fused_kernel(struct IRNode** ops, int num_ops, FusionType fusion_type) {
    if (!ops || num_ops < 2)
        return NULL;

    FusedKernel* kernel = malloc(sizeof(FusedKernel));
    if (!kernel)
        return NULL;

    kernel->ops = malloc((size_t)num_ops * sizeof(struct IRNode*));
    if (!kernel->ops) {
        free(kernel);
        return NULL;
    }

    memcpy(kernel->ops, ops, (size_t)num_ops * sizeof(struct IRNode*));
    kernel->num_ops     = num_ops;
    kernel->capacity    = num_ops;
    kernel->fusion_type = fusion_type;
    kernel->is_chained  = (fusion_type == FUSION_CHAIN_ELEMENTWISE);

    return kernel;
}

void free_fused_kernel(FusedKernel* kernel) {
    if (!kernel)
        return;
    // CRITICAL: Clear fused_kernel pointers on all nodes in this kernel
    // to prevent dangling pointers after this kernel is freed
    if (kernel->ops) {
        for (int i = 0; i < kernel->num_ops; i++) {
            if (kernel->ops[i]) {
                kernel->ops[i]->fused_kernel = NULL;
            }
        }
        free(kernel->ops);
    }
    free(kernel);
}

static char* find_other_input(struct IRNode* producer, struct IRNode* consumer) {
    if (!producer || !consumer || !producer->output_name)
        return NULL;

    for (int i = 0; i < consumer->num_inputs; i++) {
        if (consumer->input_names[i] &&
            strcmp(consumer->input_names[i], producer->output_name) != 0) {
            return consumer->input_names[i];
        }
    }
    return NULL;
}

static int apply_fusion(struct IRNode* node1, struct IRNode* node2, FusionType fusion_type) {
    if (!node1 || !node2)
        return -1;

    switch (fusion_type) {
    case FUSION_NONE:
        break;
    case FUSION_FMA: {
        // MUL + ADD -> FMA: a * b + c
        char* other_input = find_other_input(node1, node2);
        if (other_input && node1->num_inputs >= 2) {
            LOG_DEBUG("Fused MUL+ADD -> FMA: %s * %s + %s", node1->input_names[0],
                      node1->input_names[1], other_input);
            // Create fused kernel
            struct IRNode* ops[] = {node1, node2};
            FusedKernel* kernel  = create_fused_kernel(ops, 2, FUSION_FMA);
            if (kernel) {
                node1->fused_kernel = kernel;
                node2->fused_kernel = kernel;
                node1->is_fused     = true;
                node2->is_fused     = true;
                node1->fusion_type  = fusion_type;
                node2->fusion_type  = fusion_type;
            }
        }
        break;
    }

    case FUSION_NEG_ADD: {
        // NEG + ADD -> SUB: -a + b -> b - a
        char* other_input = find_other_input(node1, node2);
        if (other_input && node1->num_inputs >= 1) {
            // Transform ADD to SUB
            node2->type = UOP_SUB;
            // Swap inputs
            if (node2->input_names[0] && node1->output_name &&
                strcmp(node2->input_names[0], node1->output_name) == 0) {
                free(node2->input_names[0]);
                node2->input_names[0] = malloc(32);
                if (node2->input_names[0]) {
                    strncpy(node2->input_names[0], other_input, 31);
                    node2->input_names[0][31] = '\0';
                }

                if (node2->num_inputs >= 2) {
                    free(node2->input_names[1]);
                    node2->input_names[1] = malloc(32);
                    if (node2->input_names[1] && node1->input_names[0]) {
                        strncpy(node2->input_names[1], node1->input_names[0], 31);
                        node2->input_names[1][31] = '\0';
                    }
                }
            }
            LOG_DEBUG("Fused NEG+ADD -> SUB: %s - %s", other_input, node1->input_names[0]);
            struct IRNode* ops[] = {node1, node2};
            FusedKernel* kernel  = create_fused_kernel(ops, 2, FUSION_NEG_ADD);
            if (kernel) {
                node1->fused_kernel = kernel;
                node2->fused_kernel = kernel;
                node1->is_fused     = true;
                node2->is_fused     = true;
                node1->fusion_type  = fusion_type;
                node2->fusion_type  = fusion_type;
            }
        }
        break;
    }

    case FUSION_EXP_LOG: {
        // EXP + LOG -> identity: log(exp(a)) -> a
        LOG_DEBUG("Fused EXP+LOG -> identity");
        // Mark for removal (identity operation)
        struct IRNode* ops[] = {node1, node2};
        FusedKernel* kernel  = create_fused_kernel(ops, 2, FUSION_EXP_LOG);
        if (kernel) {
            node1->fused_kernel = kernel;
            node2->fused_kernel = kernel;
            node1->is_fused     = true;
            node2->is_fused     = true;
            node1->fusion_type  = fusion_type;
            node2->fusion_type  = fusion_type;
        }
        break;
    }

    case FUSION_MUL_DIV: {
        // MUL + DIV -> identity: (a * b) / a -> b or (a * b) / b -> a
        LOG_DEBUG("Fused MUL+DIV -> identity");
        struct IRNode* ops[] = {node1, node2};
        FusedKernel* kernel  = create_fused_kernel(ops, 2, FUSION_MUL_DIV);
        if (kernel) {
            node1->fused_kernel = kernel;
            node2->fused_kernel = kernel;
            node1->is_fused     = true;
            node2->is_fused     = true;
            node1->fusion_type  = fusion_type;
            node2->fusion_type  = fusion_type;
        }
        break;
    }

    case FUSION_SQRT_MUL: {
        // SQRT + MUL -> sqrt_mul: sqrt(a) * b
        LOG_DEBUG("Fused SQRT+MUL -> sqrt_mul");
        struct IRNode* ops[] = {node1, node2};
        FusedKernel* kernel  = create_fused_kernel(ops, 2, FUSION_SQRT_MUL);
        if (kernel) {
            node1->fused_kernel = kernel;
            node2->fused_kernel = kernel;
            node1->is_fused     = true;
            node2->is_fused     = true;
            node1->fusion_type  = fusion_type;
            node2->fusion_type  = fusion_type;
        }
        break;
    }

    case FUSION_EXP_RECIP: {
        // EXP + RECIP -> exp_recip: 1 / exp(a)
        LOG_DEBUG("Fused EXP+RECIP -> exp_recip");
        struct IRNode* ops[] = {node1, node2};
        FusedKernel* kernel  = create_fused_kernel(ops, 2, FUSION_EXP_RECIP);
        if (kernel) {
            node1->fused_kernel = kernel;
            node2->fused_kernel = kernel;
            node1->is_fused     = true;
            node2->is_fused     = true;
            node1->fusion_type  = fusion_type;
            node2->fusion_type  = fusion_type;
        }
        break;
    }

    case FUSION_CHAIN_ELEMENTWISE:
    case FUSION_REDUCE_ELEMENTWISE: {
        // Chain elementwise operations or reduction + elementwise
        LOG_DEBUG("Fused chain: %d operations", 2);
        struct IRNode* ops[] = {node1, node2};
        FusedKernel* kernel  = create_fused_kernel(ops, 2, fusion_type);
        if (kernel) {
            node1->fused_kernel = kernel;
            node2->fused_kernel = kernel;
            node1->is_fused     = true;
            node2->is_fused     = true;
            node1->fusion_type  = fusion_type;
            node2->fusion_type  = fusion_type;
        }
        break;
    }

    default:
        break;
    }

    return 0;
}

static int find_fusable_chain(struct IRNode* start, struct IRNode** chain, int max_chain) {
    if (!start || !chain || max_chain < 1)
        return 0;

    chain[0]               = start;
    int chain_len          = 1;
    struct IRNode* current = start;

    while (chain_len < max_chain && current->use_count == 1) {
        struct IRNode* next = current->users[0];
        if (!next || next->is_fused)
            break;

        if (!node_uses_output(current, next)) {
            break;
        }

        FusionType fusion_type;
        if (can_fuse_operations(current, next, &fusion_type)) {
            if (fusion_type == FUSION_CHAIN_ELEMENTWISE || chain_len == 1) {
                if (current->type == next->type && fusion_type == FUSION_CHAIN_ELEMENTWISE) {
                    break;
                }
                chain[chain_len++] = next;
                current            = next;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    return chain_len;
}

static int fuse_operations(CMLGraph_t ir) {
    if (!ir)
        return -1;

    int fused           = 0;
    struct IRNode* node = ir->head;

    while (node) {
        if (node->is_fused) {
            node = node->next;
            continue;
        }

#define MAX_FUSION_CHAIN 1024
        struct IRNode* chain[MAX_FUSION_CHAIN];
        int chain_len = find_fusable_chain(node, chain, MAX_FUSION_CHAIN);

        bool all_same = true;
        if (chain_len >= 2) {
            for (int i = 1; i < chain_len; i++) {
                if (chain[i]->type != chain[0]->type) {
                    all_same = false;
                    break;
                }
            }
        }

        if (chain_len >= 2 && !all_same) {
            FusedKernel* kernel = create_fused_kernel(chain, chain_len, FUSION_CHAIN_ELEMENTWISE);
            if (kernel) {
                for (int i = 0; i < chain_len; i++) {
                    chain[i]->is_fused     = true;
                    chain[i]->fusion_type  = FUSION_CHAIN_ELEMENTWISE;
                    chain[i]->fused_kernel = kernel;
                    chain[i]->chain_id     = fused; // Same chain ID for all
                }
                fused++;
                LOG_DEBUG("Created fused kernel with %d chained operations", chain_len);
            }
            node = chain[chain_len - 1]->next;
            continue;
        } else if (all_same && chain_len >= 2) {
            LOG_DEBUG("Skipping fusion for chain of %d identical %s operations", chain_len,
                      uop_type_to_string(chain[0]->type));
        }

        for (int i = 0; i < node->use_count; i++) {
            struct IRNode* user = node->users[i];
            if (!user || user->is_fused)
                continue;

            if (node->type == user->type) {
                continue;
            }

            FusionType fusion_type;
            if (can_fuse_operations(node, user, &fusion_type)) {
                if (apply_fusion(node, user, fusion_type) == 0) {
                    fused++;
                }
                break; // Only fuse with first matching user
            }
        }

        node = node->next;
    }

    if (fused > 0) {
        LOG_DEBUG("Fused %d operation groups", fused);
    }

    return 0;
}

static int reorder_for_cache_locality(CMLGraph_t ir) {
    if (!ir || ir->node_count <= 0)
        return -1;

    int* in_degree = calloc((size_t)ir->node_count, sizeof(int));
    if (!in_degree)
        return -1;

    struct IRNode** all_nodes = malloc((size_t)ir->node_count * sizeof(struct IRNode*));
    if (!all_nodes) {
        free(in_degree);
        return -1;
    }

    struct IRNode* node = ir->head;
    int idx             = 0;
    while (node) {
        all_nodes[idx++] = node;
        node             = node->next;
    }

    for (int i = 0; i < ir->node_count; i++) {
        in_degree[i] = 0;
        for (int j = 0; j < all_nodes[i]->num_inputs; j++) {
            struct IRNode* producer = find_node_by_output(ir, all_nodes[i]->input_names[j]);
            if (producer) {
                in_degree[i]++;
            }
        }
    }

    struct IRNode** queue = malloc((size_t)ir->node_count * sizeof(struct IRNode*));
    if (!queue) {
        free(in_degree);
        free(all_nodes);
        return -1;
    }

    int queue_front = 0;
    int queue_back  = 0;

    for (int i = 0; i < ir->node_count; i++) {
        if (in_degree[i] == 0) {
            queue[queue_back++] = all_nodes[i];
        }
    }

    struct IRNode** sorted = malloc((size_t)ir->node_count * sizeof(struct IRNode*));
    if (!sorted) {
        free(in_degree);
        free(all_nodes);
        free(queue);
        return -1;
    }

    int sorted_count = 0;

    while (queue_front < queue_back) {
        struct IRNode* current = queue[queue_front++];
        sorted[sorted_count++] = current;

        for (int i = 0; i < ir->node_count; i++) {
            if (in_degree[i] > 0) {
                for (int j = 0; j < all_nodes[i]->num_inputs; j++) {
                    if (all_nodes[i]->input_names[j] && current->output_name &&
                        strcmp(all_nodes[i]->input_names[j], current->output_name) == 0) {
                        in_degree[i]--;
                        if (in_degree[i] == 0) {
                            queue[queue_back++] = all_nodes[i];
                        }
                    }
                }
            }
        }
    }

    if (sorted_count == ir->node_count) {
        ir->head = sorted[0];
        ir->tail = sorted[sorted_count - 1];

        for (int i = 0; i < sorted_count - 1; i++) {
            sorted[i]->next = sorted[i + 1];
        }
        sorted[sorted_count - 1]->next = NULL;
    } else {
        LOG_WARNING("Topological sort incomplete: %d/%d nodes sorted (possible cycle?)",
                    sorted_count, ir->node_count);
        // Don't reorder if sort failed
        free(in_degree);
        free(all_nodes);
        free(queue);
        free(sorted);
        return -1;
    }

    free(in_degree);
    free(all_nodes);
    free(queue);
    free(sorted);

    return 0;
}

int cml_ir_optimize(CMLGraph_t ir) {
    if (!ir)
        return -1;
    if (build_dependency_graph(ir) != 0) {
        LOG_ERROR("Failed to build dependency graph");
        return -1;
    }

    mark_reachable_nodes(ir);

    if (remove_dead_nodes(ir) != 0) {
        LOG_ERROR("Failed to remove dead nodes");
        return -1;
    }

    if (build_dependency_graph(ir) != 0) {
        LOG_ERROR("Failed to rebuild dependency graph");
        return -1;
    }

    {
        CMLRewriteRegistry* builtin = cml_rewrite_builtin_rules();
        if (builtin) {
            int rewrites = cml_rewrite_apply(builtin, ir, 0);
            if (rewrites > 0) {
                build_dependency_graph(ir);
                mark_reachable_nodes(ir);
                remove_dead_nodes(ir);
                build_dependency_graph(ir);
            }
            cml_rewrite_registry_free(builtin);
        }
    }

    if (fuse_operations(ir) != 0) {
        LOG_WARNING("Operation fusion encountered issues, continuing...");
    }

    if (reorder_for_cache_locality(ir) != 0) {
        LOG_WARNING("Cache locality optimization encountered issues, continuing...");
    }
    return 0;
}
