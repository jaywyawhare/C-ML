/**
 * @file ir.c
 * @brief Intermediate Representation (IR) implementation
 */

#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "autograd/autograd.h"
#include "core/logging.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <math.h>

const char* uop_type_to_string(UOpType type) {
    switch (type) {
    case UOP_ADD:
        return "ADD";
    case UOP_SUB:
        return "SUB";
    case UOP_MUL:
        return "MUL";
    case UOP_DIV:
        return "DIV";
    case UOP_MAX:
        return "MAX";
    case UOP_NEG:
        return "NEG";
    case UOP_EXP:
        return "EXP";
    case UOP_LOG:
        return "LOG";
    case UOP_SQRT:
        return "SQRT";
    case UOP_RECIP:
        return "RECIP";
    case UOP_ABS:
        return "ABS";
    case UOP_SIN:
        return "SIN";
    case UOP_COS:
        return "COS";
    case UOP_TAN:
        return "TAN";
    case UOP_POW:
        return "POW";
    case UOP_SUM:
        return "SUM";
    case UOP_MAX_REDUCE:
        return "MAX_REDUCE";
    case UOP_MEAN:
        return "MEAN";
    case UOP_MATMUL:
        return "MATMUL";
    case UOP_CONV2D:
        return "CONV2D";
    case UOP_WHERE:
        return "WHERE";
    case UOP_CMPLT:
        return "CMPLT";
    case UOP_FILL:
        return "FILL";
    case UOP_GATHER:
        return "GATHER";
    case UOP_RESHAPE:
        return "RESHAPE";
    case UOP_PERMUTE:
        return "PERMUTE";
    case UOP_EXPAND:
        return "EXPAND";
    case UOP_STRIDE:
        return "STRIDE";
    case UOP_SLICE:
        return "SLICE";
    case UOP_SIGN:
        return "SIGN";
    case UOP_FLOOR:
        return "FLOOR";
    case UOP_CEIL:
        return "CEIL";
    case UOP_ROUND:
        return "ROUND";
    case UOP_LOG2:
        return "LOG2";
    case UOP_EXP2:
        return "EXP2";
    case UOP_ASIN:
        return "ASIN";
    case UOP_ACOS:
        return "ACOS";
    case UOP_ATAN:
        return "ATAN";
    case UOP_SQUARE:
        return "SQUARE";
    case UOP_RSQRT:
        return "RSQRT";
    case UOP_ERF:
        return "ERF";
    case UOP_CLAMP:
        return "CLAMP";
    case UOP_PROD:
        return "PROD";
    case UOP_ARGMAX:
        return "ARGMAX";
    case UOP_ARGMIN:
        return "ARGMIN";
    case UOP_CUMSUM:
        return "CUMSUM";
    case UOP_TRIU:
        return "TRIU";
    case UOP_TRIL:
        return "TRIL";
    case UOP_PAD:
        return "PAD";
    case UOP_SORT:
        return "SORT";
    case UOP_ARGSORT:
        return "ARGSORT";
    case UOP_TOPK:
        return "TOPK";
    case UOP_CUMPROD:
        return "CUMPROD";
    case UOP_BITWISE_AND:
        return "BITWISE_AND";
    case UOP_BITWISE_OR:
        return "BITWISE_OR";
    case UOP_BITWISE_XOR:
        return "BITWISE_XOR";
    case UOP_BITWISE_NOT:
        return "BITWISE_NOT";
    case UOP_NONZERO:
        return "NONZERO";
    case UOP_MASKED_FILL:
        return "MASKED_FILL";
    case UOP_LOG10:
        return "LOG10";
    case UOP_SINH:
        return "SINH";
    case UOP_COSH:
        return "COSH";
    case UOP_ASINH:
        return "ASINH";
    case UOP_ACOSH:
        return "ACOSH";
    case UOP_ATANH:
        return "ATANH";
    case UOP_TRUNC:
        return "TRUNC";
    case UOP_ISINF:
        return "ISINF";
    case UOP_ISNAN:
        return "ISNAN";
    case UOP_ISFINITE:
        return "ISFINITE";
    case UOP_LOGICAL_NOT:
        return "LOGICAL_NOT";
    case UOP_IDIV:
        return "IDIV";
    case UOP_MOD:
        return "MOD";
    case UOP_MINIMUM:
        return "MINIMUM";
    case UOP_COPYSIGN:
        return "COPYSIGN";
    case UOP_LOGADDEXP:
        return "LOGADDEXP";
    case UOP_LSHIFT:
        return "LSHIFT";
    case UOP_RSHIFT:
        return "RSHIFT";
    case UOP_LOGICAL_AND:
        return "LOGICAL_AND";
    case UOP_LOGICAL_OR:
        return "LOGICAL_OR";
    case UOP_CMPEQ:
        return "CMPEQ";
    case UOP_CMPNE:
        return "CMPNE";
    case UOP_CMPLE:
        return "CMPLE";
    case UOP_CMPGT:
        return "CMPGT";
    case UOP_CMPGE:
        return "CMPGE";
    case UOP_MIN_REDUCE:
        return "MIN_REDUCE";
    case UOP_VAR:
        return "VAR";
    case UOP_STD:
        return "STD";
    case UOP_ANY:
        return "ANY";
    case UOP_ALL:
        return "ALL";
    case UOP_LOGSUMEXP:
        return "LOGSUMEXP";
    case UOP_CUMMAX:
        return "CUMMAX";
    case UOP_CUMMIN:
        return "CUMMIN";
    case UOP_CAT:
        return "CAT";
    case UOP_STACK:
        return "STACK";
    case UOP_SCATTER:
        return "SCATTER";
    case UOP_ROLL:
        return "ROLL";
    case UOP_FLATTEN:
        return "FLATTEN";
    case UOP_UNFLATTEN:
        return "UNFLATTEN";
    case UOP_DIAG:
        return "DIAG";
    case UOP_ONE_HOT:
        return "ONE_HOT";
    case UOP_ERFC:
        return "ERFC";
    case UOP_LOGCUMSUMEXP:
        return "LOGCUMSUMEXP";
    case UOP_LERP:
        return "LERP";
    case UOP_TILE:
        return "TILE";
    case UOP_REPEAT_INTERLEAVE:
        return "REPEAT_INTERLEAVE";
    case UOP_TRACE:
        return "TRACE";
    case UOP_SHRINK:
        return "SHRINK";
    case UOP_RELU6:
        return "RELU6";
    case UOP_HARD_SIGMOID:
        return "HARD_SIGMOID";
    case UOP_HARD_TANH:
        return "HARD_TANH";
    case UOP_CELU:
        return "CELU";
    case UOP_QUICK_GELU:
        return "QUICK_GELU";
    case UOP_SOFTPLUS:
        return "SOFTPLUS";
    case UOP_SOFTSIGN:
        return "SOFTSIGN";
    case UOP_LOGSIGMOID:
        return "LOGSIGMOID";
    case UOP_UNFOLD:
        return "UNFOLD";
    case UOP_COUNT:
        return "COUNT";
    default:
        return "UNKNOWN";
    }
}

CMLGraph_t cml_ir_new(IRTarget target) {
    CMLGraph_t ir = malloc(sizeof(struct CMLGraph));
    if (!ir)
        return NULL;

    ir->target        = target;
    ir->head          = NULL;
    ir->tail          = NULL;
    ir->backward_head = NULL;
    ir->node_count    = 0;

    // Execution state
    ir->is_executed                = false;
    ir->is_optimized               = false;
    ir->is_decomposed              = false;
    ir->execution_results          = NULL;
    ir->execution_results_count    = 0;
    ir->execution_results_capacity = 0;

    // Tensor tracking
    ir->tensor_names         = NULL;
    ir->tensor_count         = 0;
    ir->tensor_capacity      = 0;
    ir->tensor_refs          = NULL;
    ir->tensor_refs_count    = 0;
    ir->tensor_refs_capacity = 0;

    return ir;
}

static void free_node_params(struct IRNode* node) {
    if (!node || !node->params)
        return;

    switch (node->type) {
    case UOP_ADD:
    case UOP_SUB:
    case UOP_MUL:
    case UOP_DIV:
    case UOP_MAX:
    case UOP_NEG:
    case UOP_EXP:
    case UOP_LOG:
    case UOP_SQRT:
    case UOP_RECIP:
    case UOP_ABS:
    case UOP_SIN:
    case UOP_COS:
    case UOP_TAN:
    case UOP_POW:
    case UOP_SUM:
    case UOP_MAX_REDUCE:
    case UOP_MATMUL:
    case UOP_WHERE:
    case UOP_CMPLT:
    case UOP_SIGN:
    case UOP_FLOOR:
    case UOP_CEIL:
    case UOP_ROUND:
    case UOP_LOG2:
    case UOP_EXP2:
    case UOP_ASIN:
    case UOP_ACOS:
    case UOP_ATAN:
    case UOP_SQUARE:
    case UOP_RSQRT:
    case UOP_ERF:
    case UOP_COUNT:
        // No params to free for these operations
        break;
    case UOP_CLAMP: {
        ClampParams* p = (ClampParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_PROD:
    case UOP_ARGMAX:
    case UOP_ARGMIN: {
        ReduceParams* p = (ReduceParams*)node->params;
        if (p) {
            if (p->dims) free(p->dims);
            free(p);
        }
        break;
    }
    case UOP_CUMSUM: {
        CumsumParams* p = (CumsumParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_TRIU:
    case UOP_TRIL: {
        TriParams* p = (TriParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_PAD: {
        PadParams* p = (PadParams*)node->params;
        if (p) {
            if (p->pad_widths) free(p->pad_widths);
            free(p);
        }
        break;
    }
    case UOP_FILL: {
        FillParams* p = (FillParams*)node->params;
        if (p) {
            if (p->shape)
                free(p->shape);
            free(p);
        }
        break;
    }
    case UOP_GATHER: {
        GatherParams* p = (GatherParams*)node->params;
        if (p) {
            free(p);
        }
        break;
    }
    case UOP_RESHAPE: {
        ReshapeParams* p = (ReshapeParams*)node->params;
        if (p) {
            if (p->new_shape)
                free(p->new_shape);
            free(p);
        }
        break;
    }
    case UOP_PERMUTE: {
        PermuteParams* p = (PermuteParams*)node->params;
        if (p) {
            if (p->perm)
                free(p->perm);
            free(p);
        }
        break;
    }
    case UOP_EXPAND: {
        ExpandParams* p = (ExpandParams*)node->params;
        if (p) {
            if (p->new_shape)
                free(p->new_shape);
            free(p);
        }
        break;
    }
    case UOP_STRIDE: {
        StrideParams* p = (StrideParams*)node->params;
        if (p) {
            if (p->new_strides)
                free(p->new_strides);
            free(p);
        }
        break;
    }
    case UOP_SLICE: {
        SliceParams* p = (SliceParams*)node->params;
        if (p) {
            if (p->start)
                free(p->start);
            if (p->end)
                free(p->end);
            if (p->step)
                free(p->step);
            free(p);
        }
        break;
    }
    case UOP_MEAN: {
        ReduceParams* p = (ReduceParams*)node->params;
        if (p) {
            if (p->dims)
                free(p->dims);
            free(p);
        }
        break;
    }
    case UOP_CONV2D: {
        Conv2DParams* p = (Conv2DParams*)node->params;
        if (p) {
            if (p->kernel_size)
                free(p->kernel_size);
            if (p->stride)
                free(p->stride);
            if (p->padding)
                free(p->padding);
            if (p->dilation)
                free(p->dilation);
            free(p);
        }
        break;
    }
    case UOP_SORT:
    case UOP_ARGSORT: {
        SortParams* p = (SortParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_TOPK: {
        TopkParams* p = (TopkParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_CUMPROD:
    case UOP_CUMMAX:
    case UOP_CUMMIN: {
        CumsumParams* p = (CumsumParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_MASKED_FILL: {
        MaskedFillParams* p = (MaskedFillParams*)node->params;
        if (p) free(p);
        break;
    }
    // New reduction ops using ReduceParams
    case UOP_MIN_REDUCE:
    case UOP_VAR:
    case UOP_STD:
    case UOP_ANY:
    case UOP_ALL:
    case UOP_LOGSUMEXP: {
        ReduceParams* p = (ReduceParams*)node->params;
        if (p) {
            if (p->dims) free(p->dims);
            free(p);
        }
        break;
    }
    // New movement ops
    case UOP_CAT: {
        CatParams* p = (CatParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_STACK: {
        StackParams* p = (StackParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_SCATTER: {
        ScatterParams* p = (ScatterParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_ROLL: {
        RollParams* p = (RollParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_FLATTEN: {
        FlattenParams* p = (FlattenParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_UNFLATTEN: {
        UnflattenParams* p = (UnflattenParams*)node->params;
        if (p) {
            if (p->sizes) free(p->sizes);
            free(p);
        }
        break;
    }
    case UOP_DIAG: {
        DiagParams* p = (DiagParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_ONE_HOT: {
        OneHotParams* p = (OneHotParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_TILE: {
        TileParams* p = (TileParams*)node->params;
        if (p) {
            if (p->repeats) free(p->repeats);
            free(p);
        }
        break;
    }
    case UOP_REPEAT_INTERLEAVE: {
        RepeatInterleaveParams* p = (RepeatInterleaveParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_SHRINK: {
        ShrinkParams* p = (ShrinkParams*)node->params;
        if (p) {
            if (p->starts) free(p->starts);
            if (p->ends) free(p->ends);
            free(p);
        }
        break;
    }
    case UOP_UNFOLD: {
        UnfoldParams* p = (UnfoldParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_LOGCUMSUMEXP: {
        CumsumParams* p = (CumsumParams*)node->params;
        if (p) free(p);
        break;
    }
    case UOP_CELU: {
        ClampParams* p = (ClampParams*)node->params;
        if (p) free(p);
        break;
    }
    // Ops with no params to free
    case UOP_ERFC:
    case UOP_LERP:
    case UOP_TRACE:
    case UOP_RELU6:
    case UOP_HARD_SIGMOID:
    case UOP_HARD_TANH:
    case UOP_QUICK_GELU:
    case UOP_SOFTPLUS:
    case UOP_SOFTSIGN:
    case UOP_LOGSIGMOID:
    case UOP_LOG10:
    case UOP_SINH:
    case UOP_COSH:
    case UOP_ASINH:
    case UOP_ACOSH:
    case UOP_ATANH:
    case UOP_TRUNC:
    case UOP_ISINF:
    case UOP_ISNAN:
    case UOP_ISFINITE:
    case UOP_LOGICAL_NOT:
    case UOP_IDIV:
    case UOP_MOD:
    case UOP_MINIMUM:
    case UOP_COPYSIGN:
    case UOP_LOGADDEXP:
    case UOP_LSHIFT:
    case UOP_RSHIFT:
    case UOP_LOGICAL_AND:
    case UOP_LOGICAL_OR:
    case UOP_CMPEQ:
    case UOP_CMPNE:
    case UOP_CMPLE:
    case UOP_CMPGT:
    case UOP_CMPGE:
    case UOP_BITWISE_AND:
    case UOP_BITWISE_OR:
    case UOP_BITWISE_XOR:
    case UOP_BITWISE_NOT:
    case UOP_NONZERO:
    case UOP_TANH:
    case UOP_SIGMOID:
        break;
    default:
        break;
    }
    node->params = NULL;
}

static void free_ir_node(struct IRNode* node) {
    if (!node)
        return;

    // Validate num_inputs is reasonable to prevent wild loop
    if (node->num_inputs < 0 || node->num_inputs > 1000) {
        fprintf(stderr,
                "WARNING: free_ir_node: invalid num_inputs=%d, skipping input_names cleanup\n",
                node->num_inputs);
        node->input_names = NULL; // Prevent invalid access
    }

    // Free input names
    if (node->input_names && node->num_inputs >= 0 && node->num_inputs <= 1000) {
        for (int i = 0; i < node->num_inputs; i++) {
            if (node->input_names[i]) {
                free(node->input_names[i]);
                node->input_names[i] = NULL;
            }
        }
        free(node->input_names);
        node->input_names = NULL;
    } else if (node->input_names) {
        // Just free the array, skip individual names
        free(node->input_names);
        node->input_names = NULL;
    }

    // Free output name
    if (node->output_name) {
        free(node->output_name);
        node->output_name = NULL;
    }

    // Free users array
    if (node->users) {
        free(node->users);
        node->users = NULL;
    }

    // Free inputs array (array of pointers, not the tensors themselves)
    if (node->inputs) {
        free(node->inputs);
        node->inputs = NULL;
    }

    // Free shape tracking arrays
    if (node->input_shapes) {
        free(node->input_shapes);
        node->input_shapes = NULL;
    }
    if (node->input_ndims) {
        free(node->input_ndims);
        node->input_ndims = NULL;
    }
    if (node->output_shape) {
        free(node->output_shape);
        node->output_shape = NULL;
    }

    // Free broadcast info
    if (node->broadcast) {
        if (node->broadcast->broadcast_dims) {
            free(node->broadcast->broadcast_dims);
        }
        if (node->broadcast->broadcast_strides) {
            free(node->broadcast->broadcast_strides);
        }
        free(node->broadcast);
        node->broadcast = NULL;
    }

    // Free saved_for_backward
    if (node->saved_for_backward) {
        free(node->saved_for_backward);
        node->saved_for_backward = NULL;
    }

    // Free fused kernel if this is the first node in the kernel (owns it)
    // Note: free_fused_kernel clears fused_kernel pointers on all nodes in the kernel
    if (node->fused_kernel && node->fused_kernel->ops && node->fused_kernel->ops[0] == node) {
        free_fused_kernel(node->fused_kernel);
        // fused_kernel is now NULL (set by free_fused_kernel)
    }

    // Free params
    free_node_params(node);

    // Note: execution_result points to tensor data, which is owned by the tensor
    // Do NOT free it here - the tensor will free it when tensor_free is called

    free(node);
}

void cml_ir_free(CMLGraph_t ir) {
    if (!ir)
        return;

    // PHASE 1: Detach output tensors from IR (but DON'T free them!)
    // Output tensors are returned to user code and managed by the user.
    // We just clear the IR references so tensors know they're detached.
    // Freeing them here would cause double-free when user code also frees them.

    // Detach output tensors from forward graph
    struct IRNode* node = ir->head;
    int node_idx        = 0;
    while (node) {
        // Safety check: validate node looks reasonable
        if (node->num_inputs < 0 || node->num_inputs > 1000) {
            fprintf(stderr, "WARNING: cml_ir_free phase 1: corrupt node %d, num_inputs=%d\n",
                    node_idx, node->num_inputs);
            break; // Stop traversing corrupt list
        }
        if (node->output) {
            // Clear IR references so tensor knows it's detached
            // But DON'T free - user code owns these tensors
            node->output->ir_node    = NULL;
            node->output->ir_context = NULL;
            node->output             = NULL;
        }
        node = node->next;
        node_idx++;
    }

    // Detach output tensors from backward graph
    node     = ir->backward_head;
    node_idx = 0;
    while (node) {
        if (node->num_inputs < 0 || node->num_inputs > 1000) {
            fprintf(stderr, "WARNING: cml_ir_free phase 1b: corrupt backward node %d\n", node_idx);
            break;
        }
        if (node->output) {
            // Clear IR references - gradients are stored in tensor->grad
            // and managed by tensor lifecycle
            node->output->ir_node    = NULL;
            node->output->ir_context = NULL;
            node->output             = NULL;
        }
        node = node->next;
        node_idx++;
    }

    // Release tensor refs (external inputs) - decrement refcount
    // These tensors were created by user code, IR just holds a reference
    if (ir->tensor_refs) {
        for (int i = 0; i < ir->tensor_refs_count; i++) {
            if (ir->tensor_refs[i]) {
                ir->tensor_refs[i]->ir_node    = NULL;
                ir->tensor_refs[i]->ir_context = NULL;
                // Decrement refcount - IR incremented it when tensor was added
                // This releases IR's reference, doesn't free if user still has one
                tensor_free(ir->tensor_refs[i]);
                ir->tensor_refs[i] = NULL;
            }
        }
        free(ir->tensor_refs);
        ir->tensor_refs       = NULL;
        ir->tensor_refs_count = 0;
    }

    // PHASE 2: Free all nodes (now safe since tensor pointers are cleared)

    // Free forward graph
    node     = ir->head;
    node_idx = 0;
    while (node) {
        struct IRNode* next = node->next;
        // Safety check
        if (node->num_inputs < 0 || node->num_inputs > 1000) {
            fprintf(stderr, "WARNING: cml_ir_free phase 2: corrupt node %d, stopping\n", node_idx);
            break;
        }
        free_ir_node(node);
        node = next;
        node_idx++;
    }
    ir->head       = NULL;
    ir->tail       = NULL;
    ir->node_count = 0;

    // Free backward graph
    node     = ir->backward_head;
    node_idx = 0;
    while (node) {
        struct IRNode* next = node->next;
        if (node->num_inputs < 0 || node->num_inputs > 1000) {
            fprintf(stderr, "WARNING: cml_ir_free phase 2b: corrupt backward node %d, stopping\n",
                    node_idx);
            break;
        }
        free_ir_node(node);
        node = next;
        node_idx++;
    }
    ir->backward_head = NULL;

    // PHASE 3: Free tensor names
    if (ir->tensor_names) {
        for (int i = 0; i < ir->tensor_count; i++) {
            if (ir->tensor_names[i]) {
                free(ir->tensor_names[i]);
                ir->tensor_names[i] = NULL;
            }
        }
        free(ir->tensor_names);
        ir->tensor_names = NULL;
        ir->tensor_count = 0;
    }

    // Free execution results array (array of pointers, not the data itself)
    if (ir->execution_results) {
        free(ir->execution_results);
        ir->execution_results       = NULL;
        ir->execution_results_count = 0;
    }

    free(ir);
}

int cml_ir_add_uop(CMLGraph_t ir, UOpType type, Tensor** inputs, int num_inputs, void* params) {
    // Allow source nodes (like UOP_FILL) with 0 inputs
    if (!ir || (num_inputs > 0 && !inputs) || num_inputs < 0) {
        LOG_ERROR("Invalid parameters for cml_ir_add_uop");
        return -1;
    }

    struct IRNode* node = malloc(sizeof(struct IRNode));
    if (!node)
        return -1;

    node->type       = type;
    node->num_inputs = num_inputs;

    // Handle source nodes with no inputs
    if (num_inputs == 0) {
        node->input_names = NULL;
    } else {
        node->input_names = malloc((size_t)num_inputs * sizeof(char*));
        if (!node->input_names) {
            free(node);
            return -1;
        }
    }

    // Ensure capacity for tensor refs and names (for external inputs)
    // We might add up to num_inputs new external tensors
    if (ir->tensor_count + num_inputs > ir->tensor_capacity) {
        int new_capacity = ir->tensor_capacity == 0 ? 16 : ir->tensor_capacity * 2;
        while (new_capacity < ir->tensor_count + num_inputs) {
            new_capacity *= 2;
        }

        Tensor** new_refs = realloc(ir->tensor_refs, (size_t)new_capacity * sizeof(Tensor*));
        char** new_names  = realloc(ir->tensor_names, (size_t)new_capacity * sizeof(char*));

        if (!new_refs || !new_names) {
            if (new_refs)
                free(new_refs);
            if (new_names)
                free(new_names);
            free(node->input_names);
            free(node);
            return -1;
        }

        ir->tensor_refs          = new_refs;
        ir->tensor_names         = new_names;
        ir->tensor_capacity      = new_capacity;
        ir->tensor_refs_capacity = new_capacity; // Keep sync
    }

    // Process inputs
    for (int i = 0; i < num_inputs; i++) {
        Tensor* t  = inputs[i];
        char* name = NULL;

        if (t) {
            // Case 1: Internal node output (already has a name in this IR)
            if (t->ir_node && t->ir_context == ir && t->ir_node->output_name) {
                name = strdup(t->ir_node->output_name);
            }
            // Case 2: External tensor (check if already registered)
            else {
                for (int j = 0; j < ir->tensor_count; j++) {
                    if (ir->tensor_refs[j] == t) {
                        name = strdup(ir->tensor_names[j]);
                        break;
                    }
                }

                // Case 3: New external tensor
                if (!name) {
                    // Register new external tensor
                    ir->tensor_refs[ir->tensor_count] = t;
                    t->ref_count++; // Increment ref count as we hold a reference in IR

                    // Generate unique name
                    // Unique ID = tensor_count (external) + node_count (internal)
                    char* new_name = malloc(32);
                    if (new_name) {
                        snprintf(new_name, 32, "t%d", ir->tensor_count + ir->node_count);
                        ir->tensor_names[ir->tensor_count] = new_name;
                        name                               = strdup(new_name);

                        ir->tensor_count++;
                        ir->tensor_refs_count = ir->tensor_count; // Keep sync
                    }
                }
            }
        } else {
            name = strdup("null");
        }

        if (!name) {
            for (int j = 0; j < i; j++)
                free(node->input_names[j]);
            free(node->input_names);
            free(node);
            return -1;
        }
        node->input_names[i] = name;
    }

    // Generate output name
    char* output_name = malloc(32);
    if (!output_name) {
        for (int i = 0; i < num_inputs; i++)
            free(node->input_names[i]);
        free(node->input_names);
        free(node);
        return -1;
    }
    // Unique ID = tensor_count (external) + node_count (internal)
    snprintf(output_name, 32, "t%d", ir->tensor_count + ir->node_count);
    node->output_name = output_name;

    node->params = params;
    node->next   = NULL;

    // Initialize tensor references
    if (num_inputs > 0) {
        node->inputs = malloc((size_t)num_inputs * sizeof(Tensor*));
        if (!node->inputs) {
            for (int i = 0; i < num_inputs; i++) {
                free(node->input_names[i]);
            }
            free(node->input_names);
            free(node);
            return -1;
        }
        memcpy(node->inputs, inputs, (size_t)num_inputs * sizeof(Tensor*));
    } else {
        node->inputs = NULL; // Source node with no inputs
    }
    node->output = NULL; // Will be set when tensor is created

    // Initialize broadcasting fields
    node->input_shapes = NULL;
    node->input_ndims  = NULL;
    node->output_shape = NULL;
    node->output_ndim  = 0;
    node->broadcast    = NULL;

    // Initialize autograd fields
    node->requires_grad = false;
    memset(node->needs_input_grad, 0, sizeof(node->needs_input_grad));

    // Check if any input requires gradient
    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i] && inputs[i]->requires_grad) {
            node->requires_grad       = true;
            node->needs_input_grad[i] = true;
        }
    }

    node->backward_node      = NULL;
    node->forward_node       = NULL;
    node->saved_for_backward = NULL;

    // Initialize execution state
    node->is_executed      = false;
    node->execution_result = NULL;

    // Initialize optimization fields
    node->is_used        = false;
    node->is_fused       = false;
    node->fusion_type    = FUSION_NONE;
    node->fused_kernel   = NULL;
    node->use_count      = 0;
    node->users          = NULL;
    node->users_capacity = 0;
    node->chain_id       = -1;

    if (!ir->head) {
        ir->head = node;
        ir->tail = node;
    } else {
        ir->tail->next = node;
        ir->tail       = node;
    }

    ir->node_count++;

    return 0;
}

struct IRNode* cml_ir_get_tail(CMLGraph_t ir) {
    if (!ir)
        return NULL;
    return ir->tail;
}

char* cml_ir_compile(CMLGraph_t ir, const char* output_file) {
    if (!ir)
        return NULL;

    LOG_ERROR("Legacy codegen has been removed. Please use the LLVM backend.");
    (void)output_file;
    return NULL;
}

char* cml_ir_to_string(CMLGraph_t ir) {
    if (!ir)
        return NULL;

    size_t buffer_size = 2048;
    char* str          = malloc(buffer_size);
    if (!str)
        return NULL;

    int offset = 0;
    offset += snprintf(str + offset, (size_t)buffer_size - (size_t)offset,
                       "IR (target: %d, nodes: %d):\n", ir->target, ir->node_count);

    struct IRNode* node = ir->head;
    size_t idx          = 0;
    while (node && offset < (int)(buffer_size - 100)) {
        const char* op_name = uop_type_to_string(node->type);
        offset += snprintf(str + offset, (size_t)buffer_size - (size_t)offset, "  %zu: %s(", idx++,
                           op_name);

        for (int i = 0; i < node->num_inputs; i++) {
            offset += snprintf(str + offset, (size_t)buffer_size - (size_t)offset, "%s%s",
                               i > 0 ? ", " : "", node->input_names[i]);
        }

        offset += snprintf(str + offset, (size_t)buffer_size - (size_t)offset, ") -> %s\n",
                           node->output_name);
        node = node->next;
    }

    return str;
}

int cml_ir_compute_broadcast_shape(struct IRNode* node) {
    if (!node || node->num_inputs < 2)
        return -1;

    // Get input shapes from input tensors
    int** input_shapes = malloc((size_t)node->num_inputs * sizeof(int*));
    if (!input_shapes)
        return -1;

    int* input_ndims = malloc((size_t)node->num_inputs * sizeof(int));
    if (!input_ndims) {
        free(input_shapes);
        return -1;
    }

    for (int i = 0; i < node->num_inputs; i++) {
        if (!node->inputs[i]) {
            free(input_shapes);
            free(input_ndims);
            return -1;
        }
        input_shapes[i] = node->inputs[i]->shape;
        input_ndims[i]  = node->inputs[i]->ndim;
    }

    // Compute broadcasted output shape (NumPy-style)
    int max_ndim = 0;
    for (int i = 0; i < node->num_inputs; i++) {
        if (input_ndims[i] > max_ndim)
            max_ndim = input_ndims[i];
    }

    int* output_shape = malloc((size_t)max_ndim * sizeof(int));
    if (!output_shape) {
        free(input_shapes);
        free(input_ndims);
        return -1;
    }

    for (int d = 0; d < max_ndim; d++) {
        int max_dim = 1;
        for (int i = 0; i < node->num_inputs; i++) {
            int dim_idx = input_ndims[i] - max_ndim + d;
            if (dim_idx >= 0) {
                int dim = input_shapes[i][dim_idx];
                if (dim != 1 && max_dim != 1 && dim != max_dim) {
                    // Incompatible shapes
                    free(output_shape);
                    free(input_shapes);
                    free(input_ndims);
                    return -1;
                }
                if (dim > max_dim)
                    max_dim = dim;
            }
        }
        output_shape[d] = max_dim;
    }

    // Store in IR node
    node->output_shape = output_shape;
    node->output_ndim  = max_ndim;
    node->input_shapes = input_shapes;
    node->input_ndims  = input_ndims;

    return 0;
}
