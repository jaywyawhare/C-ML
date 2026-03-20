#include "ops/ir/spec.h"
#include "ops/ir/internal.h"
#include "ops/ir/schedule.h"
#include "ops/ir/linearize.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static CMLSpecResult* spec_result_create(void) {
    CMLSpecResult* r = calloc(1, sizeof(CMLSpecResult));
    if (!r) return NULL;
    r->valid = true;
    r->capacity = 8;
    r->errors = calloc((size_t)r->capacity, sizeof(CMLSpecError));
    if (!r->errors) { free(r); return NULL; }
    return r;
}

static void spec_add_error(CMLSpecResult* r, int node_id,
                           const char* msg, CMLSpecLevel level) {
    if (!r) return;
    r->valid = false;
    if (r->num_errors >= r->capacity) {
        int nc = r->capacity * 2;
        CMLSpecError* tmp = realloc(r->errors, (size_t)nc * sizeof(CMLSpecError));
        if (!tmp) return;
        r->errors = tmp;
        r->capacity = nc;
    }
    r->errors[r->num_errors].node_id = node_id;
    r->errors[r->num_errors].message = msg;
    r->errors[r->num_errors].level = level;
    r->num_errors++;
}

static bool is_binary_op(UOpType t) {
    switch (t) {
        case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
        case UOP_MAX: case UOP_CMPLT: case UOP_POW:
        case UOP_IDIV: case UOP_MOD: case UOP_MINIMUM:
        case UOP_COPYSIGN: case UOP_LOGADDEXP:
        case UOP_LSHIFT: case UOP_RSHIFT:
        case UOP_LOGICAL_AND: case UOP_LOGICAL_OR:
        case UOP_CMPEQ: case UOP_CMPNE: case UOP_CMPLE:
        case UOP_CMPGT: case UOP_CMPGE:
        case UOP_BITWISE_AND: case UOP_BITWISE_OR: case UOP_BITWISE_XOR:
            return true;
        default:
            return false;
    }
}

static bool is_unary_op(UOpType t) {
    switch (t) {
        case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
        case UOP_RECIP: case UOP_ABS: case UOP_SIN: case UOP_COS:
        case UOP_TAN: case UOP_TANH: case UOP_SIGMOID:
        case UOP_SIGN: case UOP_FLOOR: case UOP_CEIL: case UOP_ROUND:
        case UOP_LOG2: case UOP_EXP2: case UOP_ASIN: case UOP_ACOS:
        case UOP_ATAN: case UOP_SQUARE: case UOP_RSQRT: case UOP_ERF:
        case UOP_BITWISE_NOT: case UOP_LOGICAL_NOT:
        case UOP_LOG10: case UOP_SINH: case UOP_COSH:
        case UOP_ASINH: case UOP_ACOSH: case UOP_ATANH:
        case UOP_TRUNC: case UOP_ISINF: case UOP_ISNAN: case UOP_ISFINITE:
        case UOP_ERFC: case UOP_RELU6: case UOP_HARD_SIGMOID:
        case UOP_HARD_TANH: case UOP_QUICK_GELU: case UOP_SOFTPLUS:
        case UOP_SOFTSIGN: case UOP_LOGSIGMOID:
        case UOP_SELU: case UOP_MISH: case UOP_SILU: case UOP_HARDSWISH:
            return true;
        default:
            return false;
    }
}

static bool is_reduce_op(UOpType t) {
    switch (t) {
        case UOP_SUM: case UOP_MAX_REDUCE: case UOP_MEAN:
        case UOP_PROD: case UOP_ARGMAX: case UOP_ARGMIN:
        case UOP_MIN_REDUCE: case UOP_VAR: case UOP_STD:
        case UOP_ANY: case UOP_ALL: case UOP_LOGSUMEXP:
            return true;
        default:
            return false;
    }
}

static bool is_float_dtype(DType d) {
    return d == DTYPE_FLOAT32 || d == DTYPE_FLOAT64 ||
           d == DTYPE_FLOAT16 || d == DTYPE_BFLOAT16 ||
           d == DTYPE_FLOAT8_E4M3 || d == DTYPE_FLOAT8_E5M2 ||
           d == DTYPE_FLOAT8_E4M3_FNUZ || d == DTYPE_FLOAT8_E5M2_FNUZ;
}

static bool is_int_dtype(DType d) {
    return d == DTYPE_INT32 || d == DTYPE_INT64 ||
           d == DTYPE_INT8 || d == DTYPE_INT16 ||
           d == DTYPE_UINT8 || d == DTYPE_UINT16 ||
           d == DTYPE_UINT32 || d == DTYPE_UINT64;
}

static void validate_tensor_level(CMLGraph_t graph, CMLSpecResult* result) {
    if (!graph || !graph->head) return;

    struct IRNode* node = graph->head;
    int idx = 0;
    int visited_cap = graph->node_count > 0 ? graph->node_count : 16;
    struct IRNode** visited = calloc((size_t)visited_cap, sizeof(struct IRNode*));
    int num_visited = 0;

    while (node) {
        if ((int)node->type < 0 || node->type >= UOP_COUNT) {
            spec_add_error(result, idx, "invalid UOpType", CML_SPEC_TENSOR);
            node = node->next;
            idx++;
            continue;
        }

        if (is_binary_op(node->type) && node->num_inputs != 2) {
            spec_add_error(result, idx,
                "binary op requires exactly 2 inputs", CML_SPEC_TENSOR);
        }
        if (is_unary_op(node->type) && node->num_inputs != 1) {
            spec_add_error(result, idx,
                "unary op requires exactly 1 input", CML_SPEC_TENSOR);
        }

        if (node->num_inputs >= 2 && node->inputs &&
            node->inputs[0] && node->inputs[1]) {
            DType d0 = node->inputs[0]->dtype;
            DType d1 = node->inputs[1]->dtype;
            if ((is_float_dtype(d0) && is_int_dtype(d1)) ||
                (is_int_dtype(d0) && is_float_dtype(d1))) {
                if (is_binary_op(node->type)) {
                    spec_add_error(result, idx,
                        "mixed int/float without cast", CML_SPEC_TENSOR);
                }
            }
        }

        if (is_binary_op(node->type) && node->num_inputs == 2 &&
            node->inputs && node->inputs[0] && node->inputs[1]) {
            Tensor* a = node->inputs[0];
            Tensor* b = node->inputs[1];
            if (a->ndim > 0 && b->ndim > 0 && a->shape && b->shape) {
                int max_nd = a->ndim > b->ndim ? a->ndim : b->ndim;
                for (int d = 0; d < max_nd; d++) {
                    int ai = a->ndim - 1 - d;
                    int bi = b->ndim - 1 - d;
                    int da = (ai >= 0) ? a->shape[ai] : 1;
                    int db = (bi >= 0) ? b->shape[bi] : 1;
                    if (da != db && da != 1 && db != 1) {
                        spec_add_error(result, idx,
                            "shapes not broadcast-compatible", CML_SPEC_TENSOR);
                        break;
                    }
                }
            }
        }

        if (is_reduce_op(node->type) && node->params) {
            ReduceParams* rp = (ReduceParams*)node->params;
            if (rp->dims && node->inputs && node->inputs[0]) {
                int ndim = node->inputs[0]->ndim;
                for (int d = 0; d < rp->num_dims; d++) {
                    if (rp->dims[d] < 0 || rp->dims[d] >= ndim) {
                        spec_add_error(result, idx,
                            "reduce axis out of bounds", CML_SPEC_TENSOR);
                        break;
                    }
                }
            }
        }

        /* Cycle detection: check if this node appears in its own ancestry */
        for (int v = 0; v < num_visited; v++) {
            if (visited[v] == node) {
                spec_add_error(result, idx,
                    "cycle detected in graph", CML_SPEC_TENSOR);
                goto done_cycle;
            }
        }
        if (num_visited < visited_cap) {
            visited[num_visited++] = node;
        }
done_cycle:

        node = node->next;
        idx++;
    }

    free(visited);
}

static void validate_kernel_level(CMLGraph_t graph, CMLSpecResult* result) {
    if (!graph || !graph->head) return;

    CMLScheduleV2* sched = cml_schedule_v2_create(graph, NULL);
    if (!sched) {
        spec_add_error(result, -1, "failed to create schedule", CML_SPEC_KERNEL);
        return;
    }

    struct IRNode* node = graph->head;
    int idx = 0;
    while (node) {
        bool found = false;
        for (int g = 0; g < sched->num_groups && !found; g++) {
            CMLFusionGroup* grp = sched->groups[g];
            if (!grp) continue;
            for (int n = 0; n < grp->num_nodes; n++) {
                if (grp->nodes[n] == node) { found = true; break; }
            }
        }
        if (!found) {
            spec_add_error(result, idx,
                "node not assigned to any fusion group", CML_SPEC_KERNEL);
        }
        node = node->next;
        idx++;
    }

    for (int g = 0; g < sched->num_groups; g++) {
        CMLFusionGroup* grp = sched->groups[g];
        if (!grp) continue;
        for (int n = 0; n < grp->num_nodes; n++) {
            if (!grp->nodes[n]) {
                spec_add_error(result, g,
                    "group references freed node", CML_SPEC_KERNEL);
            }
        }
    }

    for (int g = 0; g < sched->num_groups; g++) {
        CMLFusionGroup* grp = sched->groups[g];
        if (!grp) continue;
        for (int n = 0; n < grp->num_nodes; n++) {
            struct IRNode* nd = grp->nodes[n];
            if (!nd || !nd->output) continue;
            Tensor* out = nd->output;
            if (out->ndim > 0 && out->shape) {
                size_t sz = 1;
                for (int d = 0; d < out->ndim; d++) {
                    if (out->shape[d] < 0) {
                        spec_add_error(result, g,
                            "negative buffer dimension", CML_SPEC_KERNEL);
                        break;
                    }
                    sz *= (size_t)out->shape[d];
                }
                (void)sz;
            }
        }
    }

    cml_schedule_v2_free(sched);
}

static void validate_linear_level(CMLGraph_t graph, CMLSpecResult* result) {
    if (!graph || !graph->head) return;

    CMLScheduleV2* sched = cml_schedule_v2_create(graph, NULL);
    if (!sched) return;

    for (int g = 0; g < sched->num_groups; g++) {
        CMLFusionGroup* grp = sched->groups[g];
        if (!grp) continue;

        LinearProgram* prog = linearize_group(grp);
        if (!prog) continue;

        bool* has_store = calloc((size_t)(prog->next_vreg + 1), sizeof(bool));
        bool* has_load_reg = calloc((size_t)(prog->next_vreg + 1), sizeof(bool));
        if (!has_store || !has_load_reg) {
            free(has_store);
            free(has_load_reg);
            linear_program_free(prog);
            continue;
        }

        for (int i = 0; i < prog->num_ops; i++) {
            LinearOp* op = &prog->ops[i];
            if (op->kind == LINOP_STORE)
                has_store[op->dest_reg] = true;
            if (op->kind == LINOP_LOAD)
                has_load_reg[op->dest_reg] = true;
        }

        for (int i = 0; i < prog->num_ops; i++) {
            LinearOp* op = &prog->ops[i];
            if (op->kind == LINOP_LOAD && !op->is_eliminated) {
                bool matched = false;
                for (int j = 0; j < prog->num_ops; j++) {
                    if (prog->ops[j].kind == LINOP_STORE) {
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    spec_add_error(result, g,
                        "LOAD without any matching STORE", CML_SPEC_LINEAR);
                }
                break;
            }
        }

        int loop_depth = 0;
        for (int i = 0; i < prog->num_ops; i++) {
            LinearOp* op = &prog->ops[i];
            if (op->kind == LINOP_LOOP) {
                loop_depth++;
            } else if (op->kind == LINOP_ENDLOOP) {
                loop_depth--;
                if (loop_depth < 0) {
                    spec_add_error(result, g,
                        "ENDLOOP without matching LOOP", CML_SPEC_LINEAR);
                    break;
                }
            }
        }
        if (loop_depth != 0) {
            spec_add_error(result, g,
                "mismatched LOOP/ENDLOOP nesting", CML_SPEC_LINEAR);
        }

        free(has_store);
        free(has_load_reg);
        linear_program_free(prog);
    }

    cml_schedule_v2_free(sched);
}

static void validate_program_level(CMLGraph_t graph, CMLSpecResult* result) {
    if (!graph || !graph->head) return;

    CMLScheduleV2* sched = cml_schedule_v2_create(graph, NULL);
    if (!sched) return;

    for (int g = 0; g < sched->num_groups; g++) {
        CMLFusionGroup* grp = sched->groups[g];
        if (!grp) continue;

        for (int n = 0; n < grp->num_nodes; n++) {
            struct IRNode* nd = grp->nodes[n];
            if (!nd) continue;
            if (nd->output && nd->output->ndim > 0 && nd->output->shape) {
                size_t sz = 1;
                bool valid = true;
                for (int d = 0; d < nd->output->ndim; d++) {
                    if (nd->output->shape[d] <= 0) { valid = false; break; }
                    sz *= (size_t)nd->output->shape[d];
                }
                if (!valid || sz == 0) {
                    spec_add_error(result, g,
                        "buffer has zero or negative size", CML_SPEC_PROGRAM);
                }
            }
        }

        LinearProgram* prog = linearize_group(grp);
        if (!prog) continue;

        for (int d = 0; d < 3; d++) {
            if (prog->group_dims[d] <= 0) {
                spec_add_error(result, g,
                    "kernel launch dimension is non-positive", CML_SPEC_PROGRAM);
                break;
            }
        }

        if (prog->next_vreg > MAX_VIRTUAL_REGS) {
            spec_add_error(result, g,
                "register count exceeds hardware limit", CML_SPEC_PROGRAM);
        }

        linear_program_free(prog);
    }

    cml_schedule_v2_free(sched);
}

CMLSpecResult* cml_spec_validate(CMLGraph_t graph, CMLSpecLevel level) {
    CMLSpecResult* result = spec_result_create();
    if (!result) return NULL;

    switch (level) {
        case CML_SPEC_PROGRAM:
            validate_program_level(graph, result);
            /* fallthrough */
        case CML_SPEC_LINEAR:
            validate_linear_level(graph, result);
            /* fallthrough */
        case CML_SPEC_KERNEL:
            validate_kernel_level(graph, result);
            /* fallthrough */
        case CML_SPEC_TENSOR:
            validate_tensor_level(graph, result);
            break;
    }

    return result;
}

void cml_spec_result_free(CMLSpecResult* result) {
    if (!result) return;
    free(result->errors);
    free(result);
}

void cml_spec_result_print(const CMLSpecResult* result) {
    if (!result) {
        printf("SpecResult: (null)\n");
        return;
    }

    static const char* level_names[] = {
        "TENSOR", "KERNEL", "LINEAR", "PROGRAM"
    };

    printf("Spec Validation: %s (%d error%s)\n",
           result->valid ? "PASS" : "FAIL",
           result->num_errors,
           result->num_errors == 1 ? "" : "s");

    for (int i = 0; i < result->num_errors; i++) {
        const CMLSpecError* e = &result->errors[i];
        printf("  [%s] node %d: %s\n",
               level_names[e->level], e->node_id, e->message);
    }
}
