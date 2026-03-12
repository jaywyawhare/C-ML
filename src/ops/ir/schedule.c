/**
 * @file schedule.c
 * @brief Automatic kernel scheduling and fusion implementation
 *
 * Walks the IR graph and groups ops into fused kernels based on fusibility
 * rules. Elementwise chains are fused greedily; reductions, matmuls, and
 * convolutions break the fusion boundary and get their own schedule items.
 */

#include "ops/ir/schedule.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -----------------------------------------------------------------------
 * Classification helpers
 * ----------------------------------------------------------------------- */

bool cml_schedule_is_elementwise(UOpType type) {
    switch (type) {
        /* Core binary */
        case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
        case UOP_MAX: case UOP_CMPLT: case UOP_POW:
        /* Core unary */
        case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
        case UOP_RECIP: case UOP_ABS: case UOP_SIN: case UOP_COS:
        case UOP_TAN: case UOP_TANH: case UOP_SIGMOID:
        /* Additional unary */
        case UOP_SIGN: case UOP_FLOOR: case UOP_CEIL: case UOP_ROUND:
        case UOP_LOG2: case UOP_EXP2: case UOP_ASIN: case UOP_ACOS:
        case UOP_ATAN: case UOP_SQUARE: case UOP_RSQRT: case UOP_ERF:
        case UOP_CLAMP:
        /* Tinygrad-parity unary */
        case UOP_LOG10: case UOP_SINH: case UOP_COSH:
        case UOP_ASINH: case UOP_ACOSH: case UOP_ATANH:
        case UOP_TRUNC: case UOP_ISINF: case UOP_ISNAN: case UOP_ISFINITE:
        case UOP_LOGICAL_NOT: case UOP_ERFC:
        /* Additional binary */
        case UOP_IDIV: case UOP_MOD: case UOP_MINIMUM:
        case UOP_COPYSIGN: case UOP_LOGADDEXP:
        case UOP_LERP:
        /* Bitwise */
        case UOP_BITWISE_AND: case UOP_BITWISE_OR:
        case UOP_BITWISE_XOR: case UOP_BITWISE_NOT:
        case UOP_LSHIFT: case UOP_RSHIFT:
        /* Logical */
        case UOP_LOGICAL_AND: case UOP_LOGICAL_OR:
        /* Comparison */
        case UOP_CMPEQ: case UOP_CMPNE: case UOP_CMPLE:
        case UOP_CMPGT: case UOP_CMPGE:
        /* Masking / selection */
        case UOP_WHERE: case UOP_FILL: case UOP_MASKED_FILL:
        /* Activation functions */
        case UOP_RELU6: case UOP_HARD_SIGMOID: case UOP_HARD_TANH:
        case UOP_CELU: case UOP_QUICK_GELU: case UOP_SOFTPLUS:
        case UOP_SOFTSIGN: case UOP_LOGSIGMOID:
        case UOP_ELU: case UOP_SELU: case UOP_MISH:
        case UOP_SILU: case UOP_HARDSWISH:
            return true;
        default:
            return false;
    }
}

bool cml_schedule_is_reduction(UOpType type) {
    switch (type) {
        case UOP_SUM: case UOP_MAX_REDUCE: case UOP_MEAN:
        case UOP_PROD: case UOP_ARGMAX: case UOP_ARGMIN:
        case UOP_MIN_REDUCE: case UOP_VAR: case UOP_STD:
        case UOP_ANY: case UOP_ALL: case UOP_LOGSUMEXP:
            return true;
        default:
            return false;
    }
}

bool cml_schedule_is_movement(UOpType type) {
    switch (type) {
        case UOP_RESHAPE: case UOP_PERMUTE: case UOP_EXPAND:
        case UOP_STRIDE: case UOP_SLICE: case UOP_FLATTEN:
        case UOP_UNFLATTEN: case UOP_SHRINK:
        case UOP_CAT: case UOP_STACK:
            return true;
        default:
            return false;
    }
}

/** Classify a UOp into a schedule item type. */
static CMLScheduleItemType classify_op(UOpType type) {
    if (type == UOP_MATMUL)  return SCHED_MATMUL;
    if (type == UOP_CONV2D)  return SCHED_CONV;
    if (cml_schedule_is_elementwise(type)) return SCHED_ELEMENTWISE;
    if (cml_schedule_is_reduction(type))   return SCHED_REDUCE;
    if (cml_schedule_is_movement(type))    return SCHED_MOVEMENT;
    return SCHED_CUSTOM;
}

bool cml_schedule_can_fuse(UOpType a, UOpType b) {
    CMLScheduleItemType ta = classify_op(a);
    CMLScheduleItemType tb = classify_op(b);

    /* movement is free -- can always be absorbed */
    if (ta == SCHED_MOVEMENT || tb == SCHED_MOVEMENT) return true;

    /* elementwise + elementwise */
    if (ta == SCHED_ELEMENTWISE && tb == SCHED_ELEMENTWISE) return true;

    /* elementwise feeding into a reduction */
    if (ta == SCHED_ELEMENTWISE && tb == SCHED_REDUCE) return true;

    /* matmul/conv followed by elementwise (bias add, activation) */
    if ((ta == SCHED_MATMUL || ta == SCHED_CONV) && tb == SCHED_ELEMENTWISE)
        return true;

    /* reduction breaks the chain -- cannot fuse reduction -> anything */
    if (ta == SCHED_REDUCE) return false;

    return false;
}

/* -----------------------------------------------------------------------
 * Default options
 * ----------------------------------------------------------------------- */

CMLScheduleOptions cml_schedule_default_options(void) {
    CMLScheduleOptions opts;
    opts.enable_fusion       = true;
    opts.enable_movement_fold = true;
    opts.max_fused_ops       = CML_SCHEDULE_MAX_FUSED_OPS;
    opts.estimate_costs      = true;
    opts.topological_sort    = true;
    return opts;
}

/* -----------------------------------------------------------------------
 * Schedule item helpers
 * ----------------------------------------------------------------------- */

static CMLScheduleItem* sched_item_create(CMLScheduleItemType type) {
    CMLScheduleItem* item = calloc(1, sizeof(CMLScheduleItem));
    if (!item) return NULL;
    item->type        = type;
    item->op_capacity = 8;
    item->ops         = calloc((size_t)item->op_capacity, sizeof(struct IRNode*));
    if (!item->ops) { free(item); return NULL; }
    item->num_ops     = 0;
    item->inputs      = NULL;
    item->num_inputs  = 0;
    item->outputs     = NULL;
    item->num_outputs = 0;
    item->is_fuseable = (type == SCHED_ELEMENTWISE);
    item->device_id   = 0;
    return item;
}

static int sched_item_add_op(CMLScheduleItem* item, struct IRNode* node) {
    if (!item || !node) return -1;
    if (item->num_ops >= item->op_capacity) {
        int new_cap = item->op_capacity * 2;
        struct IRNode** tmp = realloc(item->ops,
                                      (size_t)new_cap * sizeof(struct IRNode*));
        if (!tmp) return -1;
        item->ops = tmp;
        item->op_capacity = new_cap;
    }
    item->ops[item->num_ops++] = node;
    return 0;
}

static void sched_item_free(CMLScheduleItem* item) {
    if (!item) return;
    free(item->ops);
    free(item->inputs);
    free(item->outputs);
    free(item);
}

/* -----------------------------------------------------------------------
 * Input / output tracking for a schedule item
 * ----------------------------------------------------------------------- */

/**
 * For a given schedule item, figure out which tensors are *external* inputs
 * (produced outside this item or are graph-level inputs) and which are
 * outputs (produced by ops in this item).
 */
static void compute_item_io(CMLScheduleItem* item) {
    if (!item || item->num_ops == 0) return;

    /* Collect all tensors produced inside the item */
    int out_cap = item->num_ops;
    Tensor** produced = calloc((size_t)out_cap, sizeof(Tensor*));
    int num_produced = 0;
    if (!produced) return;

    for (int i = 0; i < item->num_ops; i++) {
        struct IRNode* nd = item->ops[i];
        if (nd && nd->output) {
            produced[num_produced++] = nd->output;
        }
    }

    /* Inputs: tensors consumed but not produced inside the item */
    int in_cap = 8;
    Tensor** ext_in = calloc((size_t)in_cap, sizeof(Tensor*));
    int num_ext = 0;
    if (!ext_in) { free(produced); return; }

    for (int i = 0; i < item->num_ops; i++) {
        struct IRNode* nd = item->ops[i];
        if (!nd) continue;
        for (int j = 0; j < nd->num_inputs; j++) {
            Tensor* t = nd->inputs ? nd->inputs[j] : NULL;
            if (!t) continue;
            /* Check if produced locally */
            bool local = false;
            for (int k = 0; k < num_produced; k++) {
                if (produced[k] == t) { local = true; break; }
            }
            if (local) continue;
            /* Check for duplicates */
            bool dup = false;
            for (int k = 0; k < num_ext; k++) {
                if (ext_in[k] == t) { dup = true; break; }
            }
            if (dup) continue;
            if (num_ext >= in_cap) {
                in_cap *= 2;
                Tensor** tmp = realloc(ext_in, (size_t)in_cap * sizeof(Tensor*));
                if (!tmp) { free(produced); free(ext_in); return; }
                ext_in = tmp;
            }
            ext_in[num_ext++] = t;
        }
    }

    item->inputs     = ext_in;
    item->num_inputs = num_ext;

    /* Outputs: we store all tensors produced by the item */
    item->outputs     = produced;
    item->num_outputs = num_produced;
}

/* -----------------------------------------------------------------------
 * Cost estimation
 * ----------------------------------------------------------------------- */

/** Estimate total elements for a tensor (returns 0 if unknown). */
static size_t tensor_total_elements(const Tensor* t) {
    if (!t || !t->shape || t->ndim <= 0) return 0;
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) {
        if (t->shape[i] <= 0) return 0;
        n *= (size_t)t->shape[i];
    }
    return n;
}

static void estimate_item_cost(CMLScheduleItem* item) {
    if (!item) return;

    size_t total_flops = 0;
    size_t total_mem   = 0;

    for (int i = 0; i < item->num_ops; i++) {
        struct IRNode* nd = item->ops[i];
        if (!nd) continue;

        size_t out_elems = nd->output ? tensor_total_elements(nd->output) : 0;
        CMLScheduleItemType kind = classify_op(nd->type);

        switch (kind) {
            case SCHED_ELEMENTWISE:
                total_flops += out_elems;  /* 1 FLOP per element per op */
                break;

            case SCHED_REDUCE:
                /* reduction scans all input elements with accumulate */
                if (nd->inputs && nd->num_inputs > 0 && nd->inputs[0]) {
                    total_flops += tensor_total_elements(nd->inputs[0]) * 2;
                } else {
                    total_flops += out_elems * 2;
                }
                break;

            case SCHED_MATMUL: {
                /* [M,K] x [K,N] -> 2*M*N*K */
                Tensor* a = (nd->inputs && nd->num_inputs > 0) ? nd->inputs[0] : NULL;
                Tensor* b = (nd->inputs && nd->num_inputs > 1) ? nd->inputs[1] : NULL;
                if (a && b && a->ndim >= 2 && b->ndim >= 2) {
                    size_t M = (size_t)a->shape[a->ndim - 2];
                    size_t K = (size_t)a->shape[a->ndim - 1];
                    size_t N = (size_t)b->shape[b->ndim - 1];
                    total_flops += 2 * M * N * K;
                }
                break;
            }

            case SCHED_CONV: {
                /* batch * out_c * out_h * out_w * kern_h * kern_w * in_c * 2 */
                if (nd->output && nd->output->ndim >= 4 &&
                    nd->inputs && nd->num_inputs > 1 && nd->inputs[1] &&
                    nd->inputs[1]->ndim >= 4) {
                    Tensor* w = nd->inputs[1];
                    size_t batch   = (size_t)nd->output->shape[0];
                    size_t out_c   = (size_t)nd->output->shape[1];
                    size_t out_h   = (size_t)nd->output->shape[2];
                    size_t out_w   = (size_t)nd->output->shape[3];
                    size_t kern_h  = (size_t)w->shape[2];
                    size_t kern_w  = (size_t)w->shape[3];
                    size_t in_c    = (size_t)w->shape[1];
                    total_flops += 2 * batch * out_c * out_h * out_w *
                                   kern_h * kern_w * in_c;
                }
                break;
            }

            case SCHED_MOVEMENT:
                /* zero-cost */
                break;

            default:
                total_flops += out_elems;
                break;
        }

        /* Memory traffic: sum of input + output tensor sizes */
        if (nd->output) {
            total_mem += tensor_total_elements(nd->output) * sizeof(float);
        }
        for (int j = 0; j < nd->num_inputs; j++) {
            if (nd->inputs && nd->inputs[j]) {
                total_mem += tensor_total_elements(nd->inputs[j]) * sizeof(float);
            }
        }
    }

    item->flops        = total_flops;
    item->memory_bytes = total_mem;
    item->arithmetic_intensity = (total_mem > 0)
        ? (float)total_flops / (float)total_mem
        : 0.0f;
}

/* -----------------------------------------------------------------------
 * Dependency tracking
 * ----------------------------------------------------------------------- */

/**
 * Build dependency lists: item i depends on item j if any input tensor of
 * item i is an output tensor of item j.
 */
static void build_dependencies(CMLSchedule* sched) {
    if (!sched || sched->num_items == 0) return;

    sched->dependencies = calloc((size_t)sched->num_items, sizeof(int*));
    sched->dep_counts   = calloc((size_t)sched->num_items, sizeof(int));
    if (!sched->dependencies || !sched->dep_counts) return;

    for (int i = 0; i < sched->num_items; i++) {
        CMLScheduleItem* consumer = sched->items[i];
        if (!consumer) continue;

        int dep_cap  = 4;
        int* deps    = calloc((size_t)dep_cap, sizeof(int));
        int dep_cnt  = 0;
        if (!deps) continue;

        for (int j = 0; j < i; j++) {
            CMLScheduleItem* producer = sched->items[j];
            if (!producer) continue;

            bool found = false;
            for (int ci = 0; ci < consumer->num_inputs && !found; ci++) {
                for (int po = 0; po < producer->num_outputs && !found; po++) {
                    if (consumer->inputs[ci] == producer->outputs[po]) {
                        found = true;
                    }
                }
            }
            if (found) {
                if (dep_cnt >= dep_cap) {
                    dep_cap *= 2;
                    int* tmp = realloc(deps, (size_t)dep_cap * sizeof(int));
                    if (!tmp) break;
                    deps = tmp;
                }
                deps[dep_cnt++] = j;
            }
        }

        sched->dependencies[i] = deps;
        sched->dep_counts[i]   = dep_cnt;
    }
}

/* -----------------------------------------------------------------------
 * Schedule creation
 * ----------------------------------------------------------------------- */

CMLSchedule* cml_schedule_create(CMLGraph_t graph, const CMLScheduleOptions* opts) {
    CMLScheduleOptions default_opts;
    if (!opts) {
        default_opts = cml_schedule_default_options();
        opts = &default_opts;
    }

    CMLSchedule* sched = calloc(1, sizeof(CMLSchedule));
    if (!sched) return NULL;

    /* Handle NULL or empty graph */
    if (!graph || !graph->head || graph->node_count == 0) {
        sched->items        = NULL;
        sched->num_items    = 0;
        sched->item_capacity = 0;
        sched->total_ops    = 0;
        sched->total_kernels = 0;
        sched->fusion_ratio = 0.0f;
        sched->total_flops  = 0;
        sched->peak_memory  = 0;
        sched->dependencies = NULL;
        sched->dep_counts   = NULL;
        return sched;
    }

    /* Allocate item list */
    int cap = graph->node_count < 16 ? 16 : graph->node_count;
    sched->items = calloc((size_t)cap, sizeof(CMLScheduleItem*));
    if (!sched->items) { free(sched); return NULL; }
    sched->item_capacity = cap;
    sched->num_items     = 0;

    int total_ops = 0;

    /* Current item being assembled (may be NULL when starting fresh) */
    CMLScheduleItem* cur = NULL;

    /* Walk graph from head to tail */
    struct IRNode* node = graph->head;
    while (node) {
        total_ops++;

        CMLScheduleItemType kind = classify_op(node->type);

        /* ----- movement: fold into current or skip ----- */
        if (kind == SCHED_MOVEMENT) {
            if (opts->enable_movement_fold && cur) {
                /* Absorb movement into current item */
                sched_item_add_op(cur, node);
            } else {
                /* Create dedicated movement item */
                CMLScheduleItem* mv = sched_item_create(SCHED_MOVEMENT);
                if (mv) {
                    sched_item_add_op(mv, node);
                    /* Grow items array if needed */
                    if (sched->num_items >= sched->item_capacity) {
                        int nc = sched->item_capacity * 2;
                        CMLScheduleItem** tmp = realloc(
                            sched->items, (size_t)nc * sizeof(CMLScheduleItem*));
                        if (tmp) { sched->items = tmp; sched->item_capacity = nc; }
                    }
                    sched->items[sched->num_items++] = mv;
                }
            }
            node = node->next;
            continue;
        }

        /* ----- elementwise: extend current or start new ----- */
        if (kind == SCHED_ELEMENTWISE && opts->enable_fusion) {
            if (cur &&
                (cur->type == SCHED_ELEMENTWISE ||
                 cur->type == SCHED_MATMUL ||
                 cur->type == SCHED_CONV) &&
                cur->num_ops < opts->max_fused_ops) {
                sched_item_add_op(cur, node);
                node = node->next;
                continue;
            }
            /* Start a new elementwise item */
            if (cur) {
                /* Finish old item */
                if (sched->num_items >= sched->item_capacity) {
                    int nc = sched->item_capacity * 2;
                    CMLScheduleItem** tmp = realloc(
                        sched->items, (size_t)nc * sizeof(CMLScheduleItem*));
                    if (tmp) { sched->items = tmp; sched->item_capacity = nc; }
                }
                sched->items[sched->num_items++] = cur;
            }
            cur = sched_item_create(SCHED_ELEMENTWISE);
            if (cur) sched_item_add_op(cur, node);
            node = node->next;
            continue;
        }

        /* ----- reduction: finish current, make dedicated item ----- */
        if (kind == SCHED_REDUCE) {
            /* Flush current */
            if (cur) {
                if (opts->enable_fusion &&
                    cur->type == SCHED_ELEMENTWISE &&
                    cur->num_ops < opts->max_fused_ops) {
                    /* Fuse elementwise chain into the reduce item */
                    CMLScheduleItem* red = sched_item_create(SCHED_REDUCE);
                    if (red) {
                        /* Copy existing ops into reduce item */
                        for (int i = 0; i < cur->num_ops; i++) {
                            sched_item_add_op(red, cur->ops[i]);
                        }
                        sched_item_add_op(red, node);
                        sched_item_free(cur);
                        cur = NULL;
                        if (sched->num_items >= sched->item_capacity) {
                            int nc = sched->item_capacity * 2;
                            CMLScheduleItem** tmp = realloc(
                                sched->items,
                                (size_t)nc * sizeof(CMLScheduleItem*));
                            if (tmp) {
                                sched->items = tmp;
                                sched->item_capacity = nc;
                            }
                        }
                        sched->items[sched->num_items++] = red;
                    }
                } else {
                    /* Flush current as-is */
                    if (sched->num_items >= sched->item_capacity) {
                        int nc = sched->item_capacity * 2;
                        CMLScheduleItem** tmp = realloc(
                            sched->items,
                            (size_t)nc * sizeof(CMLScheduleItem*));
                        if (tmp) {
                            sched->items = tmp;
                            sched->item_capacity = nc;
                        }
                    }
                    sched->items[sched->num_items++] = cur;
                    cur = NULL;

                    /* Create standalone reduce item */
                    CMLScheduleItem* red = sched_item_create(SCHED_REDUCE);
                    if (red) {
                        sched_item_add_op(red, node);
                        if (sched->num_items >= sched->item_capacity) {
                            int nc = sched->item_capacity * 2;
                            CMLScheduleItem** tmp = realloc(
                                sched->items,
                                (size_t)nc * sizeof(CMLScheduleItem*));
                            if (tmp) {
                                sched->items = tmp;
                                sched->item_capacity = nc;
                            }
                        }
                        sched->items[sched->num_items++] = red;
                    }
                }
            } else {
                /* No current item -- standalone reduce */
                CMLScheduleItem* red = sched_item_create(SCHED_REDUCE);
                if (red) {
                    sched_item_add_op(red, node);
                    if (sched->num_items >= sched->item_capacity) {
                        int nc = sched->item_capacity * 2;
                        CMLScheduleItem** tmp = realloc(
                            sched->items,
                            (size_t)nc * sizeof(CMLScheduleItem*));
                        if (tmp) {
                            sched->items = tmp;
                            sched->item_capacity = nc;
                        }
                    }
                    sched->items[sched->num_items++] = red;
                }
            }
            node = node->next;
            continue;
        }

        /* ----- matmul / conv: flush current, make dedicated item ----- */
        if (kind == SCHED_MATMUL || kind == SCHED_CONV) {
            if (cur) {
                if (sched->num_items >= sched->item_capacity) {
                    int nc = sched->item_capacity * 2;
                    CMLScheduleItem** tmp = realloc(
                        sched->items,
                        (size_t)nc * sizeof(CMLScheduleItem*));
                    if (tmp) { sched->items = tmp; sched->item_capacity = nc; }
                }
                sched->items[sched->num_items++] = cur;
            }
            cur = sched_item_create(kind);
            if (cur) sched_item_add_op(cur, node);
            node = node->next;
            continue;
        }

        /* ----- custom / unknown: flush and create standalone ----- */
        if (cur) {
            if (sched->num_items >= sched->item_capacity) {
                int nc = sched->item_capacity * 2;
                CMLScheduleItem** tmp = realloc(
                    sched->items,
                    (size_t)nc * sizeof(CMLScheduleItem*));
                if (tmp) { sched->items = tmp; sched->item_capacity = nc; }
            }
            sched->items[sched->num_items++] = cur;
            cur = NULL;
        }
        {
            CMLScheduleItem* cust = sched_item_create(SCHED_CUSTOM);
            if (cust) {
                sched_item_add_op(cust, node);
                if (sched->num_items >= sched->item_capacity) {
                    int nc = sched->item_capacity * 2;
                    CMLScheduleItem** tmp = realloc(
                        sched->items,
                        (size_t)nc * sizeof(CMLScheduleItem*));
                    if (tmp) { sched->items = tmp; sched->item_capacity = nc; }
                }
                sched->items[sched->num_items++] = cust;
            }
        }
        node = node->next;
    }

    /* Flush any remaining current item */
    if (cur) {
        if (sched->num_items >= sched->item_capacity) {
            int nc = sched->item_capacity * 2;
            CMLScheduleItem** tmp = realloc(
                sched->items, (size_t)nc * sizeof(CMLScheduleItem*));
            if (tmp) { sched->items = tmp; sched->item_capacity = nc; }
        }
        sched->items[sched->num_items++] = cur;
        cur = NULL;
    }

    /* Compute IO and cost for each item */
    size_t grand_flops  = 0;
    size_t peak_mem     = 0;
    for (int i = 0; i < sched->num_items; i++) {
        compute_item_io(sched->items[i]);
        if (opts->estimate_costs) {
            estimate_item_cost(sched->items[i]);
            grand_flops += sched->items[i]->flops;
            size_t item_mem = sched->items[i]->memory_bytes;
            if (item_mem > peak_mem) peak_mem = item_mem;
        }
    }

    /* Build dependency graph */
    build_dependencies(sched);

    /* Statistics */
    sched->total_ops     = total_ops;
    sched->total_kernels = sched->num_items;
    sched->fusion_ratio  = (sched->num_items > 0)
                            ? (float)total_ops / (float)sched->num_items
                            : 0.0f;
    sched->total_flops   = grand_flops;
    sched->peak_memory   = peak_mem;

    return sched;
}

/* -----------------------------------------------------------------------
 * Accessors
 * ----------------------------------------------------------------------- */

int cml_schedule_num_kernels(const CMLSchedule* sched) {
    return sched ? sched->num_items : 0;
}

const CMLScheduleItem* cml_schedule_get_item(const CMLSchedule* sched, int index) {
    if (!sched || index < 0 || index >= sched->num_items) return NULL;
    return sched->items[index];
}

/* -----------------------------------------------------------------------
 * Printing / string conversion
 * ----------------------------------------------------------------------- */

static const char* sched_type_name(CMLScheduleItemType type) {
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

void cml_schedule_print(const CMLSchedule* sched) {
    if (!sched) {
        printf("Schedule: (null)\n");
        return;
    }
    printf("=== CML Execution Schedule ===\n");
    printf("  Total ops:    %d\n", sched->total_ops);
    printf("  Kernels:      %d\n", sched->total_kernels);
    printf("  Fusion ratio: %.2f\n", (double)sched->fusion_ratio);
    printf("  Total FLOPs:  %zu\n", sched->total_flops);
    printf("  Peak memory:  %zu bytes\n", sched->peak_memory);
    printf("  ---\n");

    for (int i = 0; i < sched->num_items; i++) {
        const CMLScheduleItem* it = sched->items[i];
        if (!it) continue;
        printf("  [%d] %-12s  ops=%d  in=%d  out=%d  flops=%zu  mem=%zu",
               i, sched_type_name(it->type),
               it->num_ops, it->num_inputs, it->num_outputs,
               it->flops, it->memory_bytes);
        if (it->arithmetic_intensity > 0.0f) {
            printf("  AI=%.2f", (double)it->arithmetic_intensity);
        }
        printf("\n");
        /* Print individual ops */
        for (int j = 0; j < it->num_ops; j++) {
            struct IRNode* nd = it->ops[j];
            if (nd) {
                printf("       op[%d]: %s\n", j, uop_type_to_string(nd->type));
            }
        }
        /* Print dependencies */
        if (sched->dep_counts && sched->dep_counts[i] > 0) {
            printf("       deps: [");
            for (int d = 0; d < sched->dep_counts[i]; d++) {
                if (d > 0) printf(", ");
                printf("%d", sched->dependencies[i][d]);
            }
            printf("]\n");
        }
    }
    printf("==============================\n");
}

char* cml_schedule_to_string(const CMLSchedule* sched) {
    if (!sched) return NULL;

    /* Rough upper bound for buffer size */
    size_t buf_size = 512 + (size_t)sched->num_items * 256;
    for (int i = 0; i < sched->num_items; i++) {
        if (sched->items[i]) {
            buf_size += (size_t)sched->items[i]->num_ops * 64;
        }
    }

    char* buf = malloc(buf_size);
    if (!buf) return NULL;

    int off = 0;
    off += snprintf(buf + off, buf_size - (size_t)off,
                    "=== CML Execution Schedule ===\n"
                    "  Total ops:    %d\n"
                    "  Kernels:      %d\n"
                    "  Fusion ratio: %.2f\n"
                    "  Total FLOPs:  %zu\n"
                    "  Peak memory:  %zu bytes\n"
                    "  ---\n",
                    sched->total_ops, sched->total_kernels,
                    (double)sched->fusion_ratio,
                    sched->total_flops, sched->peak_memory);

    for (int i = 0; i < sched->num_items && (size_t)off < buf_size - 128; i++) {
        const CMLScheduleItem* it = sched->items[i];
        if (!it) continue;
        off += snprintf(buf + off, buf_size - (size_t)off,
                        "  [%d] %-12s  ops=%d  in=%d  out=%d  flops=%zu  mem=%zu\n",
                        i, sched_type_name(it->type),
                        it->num_ops, it->num_inputs, it->num_outputs,
                        it->flops, it->memory_bytes);
        for (int j = 0; j < it->num_ops && (size_t)off < buf_size - 128; j++) {
            struct IRNode* nd = it->ops[j];
            if (nd) {
                off += snprintf(buf + off, buf_size - (size_t)off,
                                "       op[%d]: %s\n", j,
                                uop_type_to_string(nd->type));
            }
        }
    }
    off += snprintf(buf + off, buf_size - (size_t)off,
                    "==============================\n");
    return buf;
}

/* -----------------------------------------------------------------------
 * Free
 * ----------------------------------------------------------------------- */

void cml_schedule_free(CMLSchedule* sched) {
    if (!sched) return;

    for (int i = 0; i < sched->num_items; i++) {
        sched_item_free(sched->items[i]);
    }
    free(sched->items);

    if (sched->dependencies) {
        for (int i = 0; i < sched->num_items; i++) {
            free(sched->dependencies[i]);
        }
        free(sched->dependencies);
    }
    free(sched->dep_counts);
    free(sched);
}
