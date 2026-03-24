#include "ops/ir/validate.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

CMLValidateOpts cml_validate_default_opts(void) {
    CMLValidateOpts o;
    o.check_shapes = true;
    o.check_dtypes = true;
    o.check_cycles = true;
    o.check_dead   = false;
    o.max_diags    = 0;
    return o;
}

static void push_diag(CMLValidateDiag* diags, int max_diags, int* count,
                      CMLValidateCode code, int node_idx, const char* fmt, ...) {
    (*count)++;
    if (!diags) return;
    if (max_diags > 0 && (*count) > max_diags) return;
    CMLValidateDiag* d = &diags[(*count) - 1];
    d->code       = code;
    d->node_index = node_idx;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(d->message, CML_VALIDATE_MSG_LEN, fmt, ap);
    va_end(ap);
}

static struct IRNode** collect_nodes(CMLGraph_t ir, int* out_count) {
    *out_count = 0;
    if (!ir || ir->node_count == 0) return NULL;
    struct IRNode** arr = malloc((size_t)ir->node_count * sizeof(struct IRNode*));
    if (!arr) return NULL;
    int n = 0;
    struct IRNode* cur = ir->head;
    while (cur && n < ir->node_count) {
        arr[n++] = cur;
        cur = cur->next;
    }
    *out_count = n;
    return arr;
}

static int expected_inputs(UOpType t) {
    switch (t) {
        case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
        case UOP_MAX: case UOP_CMPLT: case UOP_POW: case UOP_MATMUL:
        case UOP_CONV2D: case UOP_WHERE:
            return 2;
        case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
        case UOP_RECIP: case UOP_ABS: case UOP_SIN: case UOP_COS:
        case UOP_TAN: case UOP_TANH: case UOP_SIGMOID:
        case UOP_SIGN: case UOP_FLOOR: case UOP_CEIL: case UOP_ROUND:
        case UOP_LOG2: case UOP_EXP2: case UOP_ASIN: case UOP_ACOS:
        case UOP_ATAN: case UOP_SQUARE: case UOP_RSQRT: case UOP_ERF:
            return 1;
        case UOP_SUM: case UOP_MEAN: case UOP_MAX_REDUCE:
        case UOP_PROD: case UOP_ARGMAX: case UOP_ARGMIN: case UOP_CUMSUM:
            return 1;
        case UOP_RESHAPE: case UOP_PERMUTE: case UOP_EXPAND:
        case UOP_STRIDE: case UOP_SLICE:
            return 1;
        case UOP_FILL:  return 0;
        case UOP_CLAMP: return 1;
        case UOP_GATHER: return 2;
        default: return -1;
    }
}

static bool is_binary(UOpType t) {
    return expected_inputs(t) == 2;
}

static bool dfs_has_cycle(struct IRNode** arr, int n, uint8_t* color, int idx,
                          CMLValidateDiag* diags, int max_diags, int* ndiags) {
    if (color[idx] == 2) return false;
    if (color[idx] == 1) {
        push_diag(diags, max_diags, ndiags, CML_VALID_CYCLE, idx,
                  "Cycle detected at node %d (%s)", idx,
                  uop_type_to_string(arr[idx]->type));
        return true;
    }
    color[idx] = 1;
    struct IRNode* node = arr[idx];
    for (int s = 0; s < node->num_inputs; ++s) {
        Tensor* inp = node->inputs ? node->inputs[s] : NULL;
        if (!inp || !inp->ir_node) continue;
        for (int j = 0; j < n; ++j) {
            if (arr[j] == (struct IRNode*)inp->ir_node) {
                if (dfs_has_cycle(arr, n, color, j, diags, max_diags, ndiags))
                    return true;
                break;
            }
        }
    }
    color[idx] = 2;
    return false;
}

CMLValidateCode cml_validate_graph(CMLGraph_t ir,
                                   const CMLValidateOpts* opts,
                                   CMLValidateDiag* diags,
                                   int max_diag_count,
                                   int* num_diags_out) {
    int ndiags = 0;
    CMLValidateCode first = CML_VALID_OK;

#define RECORD(code, idx, ...) do { \
    push_diag(diags, max_diag_count, &ndiags, (code), (idx), __VA_ARGS__); \
    if (first == CML_VALID_OK) first = (code); \
} while(0)

    if (!ir) {
        RECORD(CML_VALID_NULL_GRAPH, -1, "IR graph pointer is NULL");
        goto done;
    }

    CMLValidateOpts def = cml_validate_default_opts();
    if (!opts) opts = &def;

    int num_nodes = 0;
    struct IRNode** nodes = collect_nodes(ir, &num_nodes);

    if (num_nodes == 0) {
        RECORD(CML_VALID_EMPTY_GRAPH, -1, "IR graph is empty");
        free(nodes);
        goto done;
    }

    for (int i = 0; i < num_nodes; ++i) {
        if (!nodes[i]) {
            RECORD(CML_VALID_NULL_NODE, i, "Node slot %d is NULL", i);
        }
    }
    if (first != CML_VALID_OK) { free(nodes); goto done; }

    for (int i = 0; i < num_nodes; ++i) {
        struct IRNode* n = nodes[i];
        int exp = expected_inputs(n->type);
        if (exp >= 0 && n->num_inputs < exp) {
            RECORD(CML_VALID_MISSING_INPUT, i,
                   "Node %d (%s): has %d inputs, expected %d",
                   i, uop_type_to_string(n->type), n->num_inputs, exp);
        }
        if (n->inputs) {
            for (int s = 0; s < n->num_inputs; ++s) {
                if (!n->inputs[s]) {
                    RECORD(CML_VALID_MISSING_INPUT, i,
                           "Node %d (%s): input slot %d is NULL",
                           i, uop_type_to_string(n->type), s);
                }
            }
        }
    }

    if (opts->check_cycles) {
        uint8_t* color = calloc((size_t)num_nodes, 1);
        if (color) {
            for (int i = 0; i < num_nodes; ++i)
                if (color[i] == 0)
                    dfs_has_cycle(nodes, num_nodes, color, i,
                                  diags, max_diag_count, &ndiags);
            free(color);
        }
    }

    if (opts->check_dtypes) {
        for (int i = 0; i < num_nodes; ++i) {
            struct IRNode* n = nodes[i];
            if (!is_binary(n->type)) continue;
            if (n->num_inputs < 2 || !n->inputs) continue;
            Tensor* a = n->inputs[0];
            Tensor* b = n->inputs[1];
            if (a && b && a->dtype != b->dtype) {
                RECORD(CML_VALID_DTYPE_MISMATCH, i,
                       "Node %d (%s): input dtypes mismatch (%d vs %d)",
                       i, uop_type_to_string(n->type), a->dtype, b->dtype);
            }
        }
    }

    if (opts->check_shapes) {
        for (int i = 0; i < num_nodes; ++i) {
            struct IRNode* n = nodes[i];
            if (n->is_executed && !n->output) {
                RECORD(CML_VALID_SHAPE_MISMATCH, i,
                       "Node %d (%s): executed but output tensor is NULL",
                       i, uop_type_to_string(n->type));
            }
        }
    }

    free(nodes);

done:
    if (num_diags_out) *num_diags_out = ndiags;
    return first;
#undef RECORD
}

void cml_validate_graph_or_die(CMLGraph_t ir) {
    CMLValidateDiag diags[32];
    int ndiags = 0;
    CMLValidateCode rc = cml_validate_graph(ir, NULL, diags, 32, &ndiags);
    if (rc != CML_VALID_OK) {
        cml_validate_print_diags(diags, ndiags);
        LOG_ERROR("IR graph validation failed: %s", cml_validate_code_str(rc));
        abort();
    }
}

void cml_validate_print_diags(const CMLValidateDiag* diags, int num_diags) {
    for (int i = 0; i < num_diags; ++i) {
        const CMLValidateDiag* d = &diags[i];
        fprintf(stderr, "[validate] node=%d %s: %s\n",
                d->node_index, cml_validate_code_str(d->code), d->message);
    }
}

const char* cml_validate_code_str(CMLValidateCode code) {
    switch (code) {
        case CML_VALID_OK:              return "OK";
        case CML_VALID_NULL_GRAPH:      return "NULL_GRAPH";
        case CML_VALID_NULL_NODE:       return "NULL_NODE";
        case CML_VALID_CYCLE:           return "CYCLE";
        case CML_VALID_MISSING_INPUT:   return "MISSING_INPUT";
        case CML_VALID_TOO_MANY_INPUTS: return "TOO_MANY_INPUTS";
        case CML_VALID_SHAPE_MISMATCH:  return "SHAPE_MISMATCH";
        case CML_VALID_DTYPE_MISMATCH:  return "DTYPE_MISMATCH";
        case CML_VALID_INVALID_AXIS:    return "INVALID_AXIS";
        case CML_VALID_UNKNOWN_OP:      return "UNKNOWN_OP";
        case CML_VALID_EMPTY_GRAPH:     return "EMPTY_GRAPH";
        case CML_VALID_DEAD_OUTPUT:     return "DEAD_OUTPUT";
        case CML_VALID_INTERNAL:        return "INTERNAL";
        default:                         return "UNKNOWN";
    }
}
