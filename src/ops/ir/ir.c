
#ifndef _WIN32
#include <execinfo.h>
#endif
#include "ops/ir/flamegraph.h"

/* Defined below, next to the interning table it uses. */
static char* cml_capture_build_stack(void);
static int cml_stack_is_internal(const char* folded);
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "core/cml_flags.h"
#include "ops/ir/context.h"
#include "ops/uops.h"
#include "autograd/autograd.h"
#include "core/logging.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <math.h>
#include "alloc/cml_allocator.h"

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
    case UOP_FOLD:
        return "FOLD";
    case UOP_IM2COL:
        return "IM2COL";
    case UOP_COL2IM:
        return "COL2IM";
    case UOP_SCATTER_ADD:
        return "SCATTER_ADD";
    case UOP_SPMM:
        return "SPMM";
    case UOP_FUSED_ELEMENTWISE:
        return "FUSED_ELEMENTWISE";
    case UOP_CONST:
        return "CONST";
    case UOP_RAND_UNIFORM:
        return "RAND_UNIFORM";
    case UOP_RAND_NORMAL:
        return "RAND_NORMAL";
    case UOP_ARANGE_OP:
        return "ARANGE_OP";
    case UOP_EYE_OP:
        return "EYE_OP";
    case UOP_RAND_INT:
        return "RAND_INT";
    case UOP_COUNT:
        return "COUNT";
    case UOP_RELU:
        return "RELU";
    case UOP_LEAKY_RELU:
        return "LEAKY_RELU";
    case UOP_GELU:
        return "GELU";
    case UOP_SILU:
        return "SILU";
    case UOP_ELU:
        return "ELU";
    case UOP_SELU:
        return "SELU";
    case UOP_MISH:
        return "MISH";
    case UOP_HARDSWISH:
        return "HARDSWISH";
    case UOP_LINEAR:
        return "LINEAR";
    case UOP_ALLOC:
        return "ALLOC";
    case UOP_CHUNK:
        return "CHUNK";
    case UOP_SPLIT:
        return "SPLIT";
    case UOP_DIAGONAL:
        return "DIAGONAL";
    case UOP_MASKED_SELECT:
        return "MASKED_SELECT";
    case UOP_MESHGRID:
        return "MESHGRID";
    /* These were missing, so every one of them printed as "UNKNOWN" wherever a
     * uop is named -- the profile, the kernel studio, the graph export and the
     * logs. A run using sigmoid or an optimizer step had unlabelled nodes. */
    case UOP_TANH:
        return "TANH";
    case UOP_SIGMOID:
        return "SIGMOID";
    case UOP_MAXPOOL2D:
        return "MAXPOOL2D";
    case UOP_AVGPOOL2D:
        return "AVGPOOL2D";
    case UOP_CONV3D:
        return "CONV3D";
    case UOP_CONV_TRANSPOSE2D:
        return "CONV_TRANSPOSE2D";
    case UOP_CONV_TRANSPOSE3D:
        return "CONV_TRANSPOSE3D";
    case UOP_SGD_STEP:
        return "SGD_STEP";
    case UOP_ADAM_STEP:
        return "ADAM_STEP";
    default:
        return "UNKNOWN";
    }
}

CMLGraph_t cml_ir_new(IRTarget target) {
    CMLGraph_t ir = cml_malloc(sizeof(struct CMLGraph));
    if (!ir)
        return NULL;

    ir->target        = target;
    ir->head          = NULL;
    ir->tail          = NULL;
    ir->last_result   = NULL;
    ir->backward_head = NULL;
    ir->node_count    = 0;

    ir->is_executed                = false;
    ir->is_optimized               = false;
    ir->is_decomposed              = false;
    /* Allocator blocks are recycled without zeroing (see cml_malloc), so
     * every field must be set explicitly -- a stale has_backward_nodes from
     * a freed context made the next context skip its forward decompose. */
    ir->has_backward_nodes         = false;
    ir->decomposed_frontier        = NULL;
    ir->grad_publish_log           = NULL;
    ir->grad_publish_count         = 0;
    ir->grad_publish_cap           = 0;
    ir->execution_results          = NULL;
    ir->execution_results_count    = 0;
    ir->execution_results_capacity = 0;

    ir->tensor_names         = NULL;
    ir->tensor_count         = 0;
    ir->tensor_capacity      = 0;
    ir->tensor_refs          = NULL;
    ir->tensor_refs_count    = 0;
    ir->tensor_refs_capacity = 0;

    ir->intern_table = cml_intern_table_create();

    return ir;
}

void cml_ir_free_node_params(struct IRNode* node) {
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
        break;
    case UOP_ELU:            /* ELU/CLAMP both store ClampParams */
    case UOP_CLAMP: {
        ClampParams* p = (ClampParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_PROD:
    case UOP_ARGMAX:
    case UOP_ARGMIN:
    case UOP_SUM:
    case UOP_MAX_REDUCE:
    case UOP_MEAN: {
        ReduceParams* p = (ReduceParams*)node->params;
        if (p) {
            if (p->dims) cml_free(p->dims);
            cml_free(p);
        }
        break;
    }
    case UOP_CUMSUM: {
        CumsumParams* p = (CumsumParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_TRIU:
    case UOP_TRIL: {
        TriParams* p = (TriParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_PAD: {
        PadParams* p = (PadParams*)node->params;
        if (p) {
            if (p->pad_widths) cml_free(p->pad_widths);
            cml_free(p);
        }
        break;
    }
    case UOP_FILL: {
        FillParams* p = (FillParams*)node->params;
        if (p) {
            if (p->shape)
                cml_free(p->shape);
            cml_free(p);
        }
        break;
    }
    case UOP_GATHER: {
        GatherParams* p = (GatherParams*)node->params;
        if (p) {
            cml_free(p);
        }
        break;
    }
    case UOP_RESHAPE: {
        ReshapeParams* p = (ReshapeParams*)node->params;
        if (p) {
            if (p->new_shape)
                cml_free(p->new_shape);
            cml_free(p);
        }
        break;
    }
    case UOP_PERMUTE: {
        PermuteParams* p = (PermuteParams*)node->params;
        if (p) {
            if (p->perm)
                cml_free(p->perm);
            cml_free(p);
        }
        break;
    }
    case UOP_EXPAND: {
        ExpandParams* p = (ExpandParams*)node->params;
        if (p) {
            if (p->new_shape)
                cml_free(p->new_shape);
            cml_free(p);
        }
        break;
    }
    case UOP_STRIDE: {
        StrideParams* p = (StrideParams*)node->params;
        if (p) {
            if (p->new_strides)
                cml_free(p->new_strides);
            cml_free(p);
        }
        break;
    }
    case UOP_SLICE: {
        SliceParams* p = (SliceParams*)node->params;
        if (p) {
            if (p->start)
                cml_free(p->start);
            if (p->end)
                cml_free(p->end);
            if (p->step)
                cml_free(p->step);
            cml_free(p);
        }
        break;
    }
    case UOP_CONV2D: {
        Conv2DParams* p = (Conv2DParams*)node->params;
        if (p) {
            if (p->kernel_size)
                cml_free(p->kernel_size);
            if (p->stride)
                cml_free(p->stride);
            if (p->padding)
                cml_free(p->padding);
            if (p->dilation)
                cml_free(p->dilation);
            cml_free(p);
        }
        break;
    }
    case UOP_SORT:
    case UOP_ARGSORT: {
        SortParams* p = (SortParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_TOPK: {
        TopkParams* p = (TopkParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_CUMPROD:
    case UOP_CUMMAX:
    case UOP_CUMMIN: {
        CumsumParams* p = (CumsumParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_MASKED_FILL: {
        MaskedFillParams* p = (MaskedFillParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_MIN_REDUCE:
    case UOP_VAR:
    case UOP_STD:
    case UOP_ANY:
    case UOP_ALL:
    case UOP_LOGSUMEXP: {
        ReduceParams* p = (ReduceParams*)node->params;
        if (p) {
            if (p->dims) cml_free(p->dims);
            cml_free(p);
        }
        break;
    }
    case UOP_CAT: {
        CatParams* p = (CatParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_STACK: {
        StackParams* p = (StackParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_SCATTER: {
        ScatterParams* p = (ScatterParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_ROLL: {
        RollParams* p = (RollParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_FLATTEN: {
        FlattenParams* p = (FlattenParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_UNFLATTEN: {
        UnflattenParams* p = (UnflattenParams*)node->params;
        if (p) {
            if (p->sizes) cml_free(p->sizes);
            cml_free(p);
        }
        break;
    }
    case UOP_DIAGONAL:       /* DIAG/DIAGONAL both store DiagParams */
    case UOP_DIAG: {
        DiagParams* p = (DiagParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_ALLOC: {
        AllocParams* p = (AllocParams*)node->params;
        if (p) {
            if (p->shape) cml_free(p->shape);
            cml_free(p);
        }
        break;
    }
    case UOP_ONE_HOT: {
        OneHotParams* p = (OneHotParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_TILE: {
        TileParams* p = (TileParams*)node->params;
        if (p) {
            if (p->repeats) cml_free(p->repeats);
            cml_free(p);
        }
        break;
    }
    case UOP_REPEAT_INTERLEAVE: {
        RepeatInterleaveParams* p = (RepeatInterleaveParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_SHRINK: {
        ShrinkParams* p = (ShrinkParams*)node->params;
        if (p) {
            if (p->starts) cml_free(p->starts);
            if (p->ends) cml_free(p->ends);
            cml_free(p);
        }
        break;
    }
    case UOP_UNFOLD: {
        UnfoldParams* p = (UnfoldParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_FOLD: {
        FoldParams* p = (FoldParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_IM2COL: {
        Im2colParams* p = (Im2colParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_COL2IM: {
        Col2imParams* p = (Col2imParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_SCATTER_ADD: {
        ScatterAddParams* p = (ScatterAddParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_SPMM: {
        SpMMParams* p = (SpMMParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_FUSED_ELEMENTWISE: {
        FusedElementwiseParams* p = (FusedElementwiseParams*)node->params;
        if (p) {
            cml_free(p->op); cml_free(p->a); cml_free(p->b);
            cml_free(p->c); cml_free(p->konst); cml_free(p);
        }
        break;
    }
    case UOP_MATMUL: {
        /* Plain matmul has no params; a fused epilogue (cml_ir_fuse_matmul_epilogue)
         * attaches a FusedElementwiseParams to be freed here. */
        FusedElementwiseParams* p = (FusedElementwiseParams*)node->params;
        if (p) {
            cml_free(p->op); cml_free(p->a); cml_free(p->b);
            cml_free(p->c); cml_free(p->konst); cml_free(p);
        }
        break;
    }
    case UOP_MAXPOOL2D:
    case UOP_AVGPOOL2D: {
        Pool2DParams* p = (Pool2DParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_CONV3D: {
        Conv3DParams* p = (Conv3DParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_CONV_TRANSPOSE2D: {
        ConvTranspose2DParams* p = (ConvTranspose2DParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_CONV_TRANSPOSE3D: {
        ConvTranspose3DParams* p = (ConvTranspose3DParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_LOGCUMSUMEXP: {
        CumsumParams* p = (CumsumParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_CELU: {
        ClampParams* p = (ClampParams*)node->params;
        if (p) cml_free(p);
        break;
    }
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
    case UOP_CONST: {
        ConstParams* p = (ConstParams*)node->params;
        if (p) {
            if (p->data)  cml_free(p->data);
            if (p->shape) cml_free(p->shape);
            cml_free(p);
        }
        break;
    }
    case UOP_RAND_UNIFORM:
    case UOP_RAND_NORMAL: {
        RandParams* p = (RandParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_ARANGE_OP: {
        ArangeParams* p = (ArangeParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_EYE_OP: {
        EyeParams* p = (EyeParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_RAND_INT: {
        RandIntParams* p = (RandIntParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_SGD_STEP: {
        SgdStepParams* p = (SgdStepParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    case UOP_ADAM_STEP: {
        AdamStepParams* p = (AdamStepParams*)node->params;
        if (p) cml_free(p);
        break;
    }
    default:
        break;
    }
    node->params = NULL;
}

static void free_ir_node(struct IRNode* node);
static void free_ir_node(struct IRNode* node) {
    if (!node)
        return;

    if (--node->ref_count > 0)
        return;

    cml_ir_release_node_storage(node);

    /* execution_result is owned by the tensor -- do not free here */
    cml_free(node);
}

/* Canonical teardown of a node's owned storage, WITHOUT freeing the node
 * itself and without ref_count accounting. This is the single source of
 * truth for which fields a node owns (params, scope, build_stack, per-input
 * shape arrays, ...); custom free paths must call this instead of open-coding
 * the field list — the DCE removal path used to hand-roll it and leaked
 * params/scope/build_stack/input_shapes on every removed node. */
void cml_ir_release_node_storage(struct IRNode* node) {
    if (!node)
        return;

    if (node->num_inputs < 0 || node->num_inputs > 1000) {
        fprintf(stderr,
                "WARNING: free_ir_node: invalid num_inputs=%d, skipping input_names cleanup\n",
                node->num_inputs);
        node->input_names = NULL; // Prevent invalid access
    }

    if (node->input_names && node->num_inputs >= 0 && node->num_inputs <= 1000) {
        for (int i = 0; i < node->num_inputs; i++) {
            if (node->input_names[i]) {
                cml_free(node->input_names[i]);
                node->input_names[i] = NULL;
            }
        }
        cml_free(node->input_names);
        node->input_names = NULL;
    } else if (node->input_names) {
        cml_free(node->input_names);
        node->input_names = NULL;
    }

    if (node->output_name) {
        cml_free(node->output_name);
        node->output_name = NULL;
    }
    if (node->scope) {
        cml_free(node->scope);
        node->scope = NULL;
    }
    if (node->build_stack) {
        cml_free(node->build_stack);
        node->build_stack = NULL;
    }

    if (node->users) {
        cml_free(node->users);
        node->users = NULL;
    }

    if (node->inputs) {
        cml_free(node->inputs);
        node->inputs = NULL;
    }

    if (node->input_shapes) {
        for (int i = 0; i < node->num_inputs; i++)
            cml_free(node->input_shapes[i]);
        cml_free(node->input_shapes);
        node->input_shapes = NULL;
    }
    if (node->input_ndims) {
        cml_free(node->input_ndims);
        node->input_ndims = NULL;
    }
    if (node->output_shape) {
        cml_free(node->output_shape);
        node->output_shape = NULL;
    }

    if (node->broadcast) {
        if (node->broadcast->broadcast_dims) {
            cml_free(node->broadcast->broadcast_dims);
        }
        if (node->broadcast->broadcast_strides) {
            cml_free(node->broadcast->broadcast_strides);
        }
        cml_free(node->broadcast);
        node->broadcast = NULL;
    }

    if (node->saved_for_backward) {
        cml_free(node->saved_for_backward);
        node->saved_for_backward = NULL;
    }

    if (node->fused_kernel && node->fused_kernel->ops && node->fused_kernel->ops[0] == node) {
        free_fused_kernel(node->fused_kernel);
    }

    cml_ir_free_node_params(node);
}

/* Pointers destroyed by the output phases of cml_ir_free.
 *
 * A tensor can be reachable both as node->output and as an ir->tensor_refs
 * entry. The output phases force ref_count to 1 and free unconditionally, so
 * the tensor_refs pass would then read (tr->ir_context) and free an object that
 * is already gone. A freed pointer cannot be tested, so the identities have to
 * be recorded while they are still valid. */
typedef struct {
    Tensor** p;
    int      n, cap;
} FreedSet;

static void freed_set_add(FreedSet* fs, Tensor* t) {
    if (!t || !fs->p) return;
    if (fs->n == fs->cap) {
        int cap    = fs->cap ? fs->cap * 2 : 64;
        Tensor** np = cml_realloc(fs->p, (size_t)cap * sizeof(Tensor*));
        if (!np) { cml_free(fs->p); fs->p = NULL; fs->n = 0; fs->cap = 0; return; }
        fs->p   = np;
        fs->cap = cap;
    }
    fs->p[fs->n++] = t;
}

static int freed_set_cmp(const void* a, const void* b) {
    Tensor* x = *(Tensor* const*)a;
    Tensor* y = *(Tensor* const*)b;
    return (x > y) - (x < y);
}

static bool freed_set_contains(const FreedSet* fs, Tensor* t) {
    if (!fs->p || fs->n == 0) return false;
    int lo = 0, hi = fs->n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (fs->p[mid] == t) return true;
        if (fs->p[mid] < t) lo = mid + 1;
        else                hi = mid - 1;
    }
    return false;
}

void cml_ir_free(CMLGraph_t ir) {
    if (!ir)
        return;

    cml_ir_clear_global_if_current(ir);

    FreedSet freed = {0};
    freed.cap = 64;
    freed.p   = cml_malloc((size_t)freed.cap * sizeof(Tensor*));

    /* Phase 1: Free forward output tensors.
     * Do not auto-execute pending nodes during teardown: lazy tensors can be
     * freed safely without materializing their outputs. */
    struct IRNode* node = ir->head;
    int node_idx        = 0;
    while (node) {
        if (node->num_inputs < 0 || node->num_inputs > 1000) {
            fprintf(stderr, "WARNING: cml_ir_free phase 1: corrupt node %d, num_inputs=%d\n",
                    node_idx, node->num_inputs);
            break;
        }
        if (node->output) {
            Tensor* out = node->output;
            freed_set_add(&freed, out);
            if (out->external_refs > 0) {
                /* An external owner (language binding) still holds this
                 * output: detach it (copying borrowed plan data) instead of
                 * destroying it under them. */
                tensor_detach_keep(out);
            } else {
                out->ref_count = 1;
                tensor_free(out);
            }
            node->output = NULL;
        }
        node = node->next;
        node_idx++;
    }

    /* Phase 1b: Free backward output tensors */
    node     = ir->backward_head;
    node_idx = 0;
    while (node) {
        if (node->num_inputs < 0 || node->num_inputs > 1000) {
            fprintf(stderr, "WARNING: cml_ir_free phase 1b: corrupt backward node %d\n", node_idx);
            break;
        }
        if (node->output) {
            Tensor* out = node->output;
            freed_set_add(&freed, out);
            if (out->external_refs > 0) {
                tensor_detach_keep(out);
            } else {
                out->ref_count = 1;
                tensor_free(out);
            }
            node->output = NULL;
        }
        node = node->next;
        node_idx++;
    }

    if (ir->tensor_refs) {
        if (freed.p && freed.n > 1)
            qsort(freed.p, (size_t)freed.n, sizeof(Tensor*), freed_set_cmp);
        for (int i = 0; i < ir->tensor_refs_count; i++) {
            if (ir->tensor_refs[i]) {
                Tensor* tr = ir->tensor_refs[i];
                /* Already destroyed as a node output above: the pointer is
                 * dangling, so it must not be dereferenced or freed again. */
                if (freed_set_contains(&freed, tr)) {
                    ir->tensor_refs[i] = NULL;
                    continue;
                }
                /* Only detach from THIS graph's context. Do NOT clear ir_node
                 * before tensor_free — tensor_free needs ir_node to clear the
                 * original node's output pointer when ref_count reaches 0. */
                if (tr->ir_context == ir)
                    tr->ir_context = NULL;
                tensor_free(tr);
                ir->tensor_refs[i] = NULL;
            }
        }
        cml_free(ir->tensor_refs);
        ir->tensor_refs       = NULL;
        ir->tensor_refs_count = 0;
    }

    cml_free(freed.p);

    cml_intern_table_free(ir->intern_table);
    ir->intern_table = NULL;

    /* Phase 2: free all nodes (safe now that tensor pointers are cleared) */
    node     = ir->head;
    node_idx = 0;
    while (node) {
        struct IRNode* next = node->next;
        if (node->num_inputs < 0 || node->num_inputs > 1000) {
            fprintf(stderr, "WARNING: cml_ir_free phase 2: corrupt node %d, stopping\n", node_idx);
            break;
        }
        node->ref_count = 1;
        free_ir_node(node);
        node = next;
        node_idx++;
    }
    ir->head       = NULL;
    ir->tail       = NULL;
    ir->node_count = 0;

    node     = ir->backward_head;
    node_idx = 0;
    while (node) {
        struct IRNode* next = node->next;
        if (node->num_inputs < 0 || node->num_inputs > 1000) {
            fprintf(stderr, "WARNING: cml_ir_free phase 2b: corrupt backward node %d, stopping\n",
                    node_idx);
            break;
        }
        node->ref_count = 1;
        free_ir_node(node);
        node = next;
        node_idx++;
    }
    ir->backward_head = NULL;

    if (ir->tensor_names) {
        for (int i = 0; i < ir->tensor_count; i++) {
            if (ir->tensor_names[i]) {
                cml_free(ir->tensor_names[i]);
                ir->tensor_names[i] = NULL;
            }
        }
        cml_free(ir->tensor_names);
        ir->tensor_names = NULL;
        ir->tensor_count = 0;
    }

    /* Log of values that received lazy grads (see autodiff.c publish). The
     * entries are plain Tensor* borrows owned elsewhere — only the array. */
    if (ir->grad_publish_log) {
        cml_free(ir->grad_publish_log);
        ir->grad_publish_log   = NULL;
        ir->grad_publish_count = 0;
        ir->grad_publish_cap   = 0;
    }

    if (ir->execution_results) {
        cml_free(ir->execution_results);
        ir->execution_results       = NULL;
        ir->execution_results_count = 0;
    }

    cml_free(ir);
}

/** Undo refcount bumps on inputs wired as lazy tensors in this IR graph. */
static void cml_ir_release_same_graph_input_refs(CMLGraph_t ir, Tensor** inputs, int n) {
    for (int j = 0; j < n; j++) {
        Tensor* u = inputs[j];
        if (u && u->ir_node && u->ir_context == ir && u->ir_node->output_name)
            u->ref_count--;
    }
}

int cml_ir_add_uop(CMLGraph_t ir, UOpType type, Tensor** inputs, int num_inputs, void* params) {
    if (!ir || (num_inputs > 0 && !inputs) || num_inputs < 0) {
        LOG_ERROR("Invalid parameters for cml_ir_add_uop");
        return -1;
    }

    if (ir->intern_table && num_inputs >= 0) {
        /* Skip interning for parameterized ops: the intern hash does not
           include param contents (arg_len is always 0), so ops with different
           params but the same type/inputs would collide.  Also skip interning
           for zero-input ops (e.g. FILL) to avoid conflating different
           constants.  Finally, never reuse nodes that have already been
           executed -- the underlying input data may have changed or tensor
           pointers may have been recycled by malloc. */
        bool can_intern = (params == NULL) && (num_inputs > 0);
        if (can_intern) {
            struct IRNode* input_nodes[8];
            Tensor* raw_inputs_arr[8];
            int lookup_count = num_inputs < 8 ? num_inputs : 8;
            for (int i = 0; i < lookup_count; i++) {
                input_nodes[i] = (inputs[i] && inputs[i]->ir_node) ? inputs[i]->ir_node : NULL;
                raw_inputs_arr[i] = inputs[i];
            }

            int dtype = (num_inputs > 0 && inputs[0]) ? (int)inputs[0]->dtype : 0;
            uint64_t hash = cml_intern_hash_node_ex((int)type, dtype, input_nodes,
                                                    raw_inputs_arr, lookup_count, params, 0);

            struct IRNode* existing = cml_intern_lookup_ex(ir->intern_table, hash, (int)type,
                                                           dtype, input_nodes, raw_inputs_arr,
                                                           lookup_count, params, 0);
            if (existing && existing->output && !existing->is_executed) {
                existing->ref_count++;
                ir->last_result = existing;
                return 0;
            }
        }
    }

    struct IRNode* node = cml_malloc(sizeof(struct IRNode));
    if (!node)
        return -1;

    node->type       = type;
    node->num_inputs = num_inputs;

    if (num_inputs == 0) {
        node->input_names = NULL;
    } else {
        node->input_names = cml_malloc((size_t)num_inputs * sizeof(char*));
        if (!node->input_names) {
            cml_free(node);
            return -1;
        }
    }

    if (ir->tensor_count + num_inputs > ir->tensor_capacity) {
        int new_capacity = ir->tensor_capacity == 0 ? 16 : ir->tensor_capacity * 2;
        while (new_capacity < ir->tensor_count + num_inputs) {
            new_capacity *= 2;
        }

        Tensor** new_refs = cml_realloc(ir->tensor_refs, (size_t)new_capacity * sizeof(Tensor*));
        char** new_names  = cml_realloc(ir->tensor_names, (size_t)new_capacity * sizeof(char*));

        if (!new_refs || !new_names) {
            if (new_refs)
                cml_free(new_refs);
            if (new_names)
                cml_free(new_names);
            cml_free(node->input_names);
            cml_free(node);
            return -1;
        }

        ir->tensor_refs          = new_refs;
        ir->tensor_names         = new_names;
        ir->tensor_capacity      = new_capacity;
        ir->tensor_refs_capacity = new_capacity;
    }

    for (int i = 0; i < num_inputs; i++) {
        Tensor* t  = inputs[i];
        char* name = NULL;

        if (t) {
            if (t->ir_node && t->ir_context == ir && t->ir_node->output_name) {
                name = cml_strdup(t->ir_node->output_name);
                if (!name) {
                    cml_ir_release_same_graph_input_refs(ir, inputs, i);
                    for (int j = 0; j < i; j++)
                        cml_free(node->input_names[j]);
                    cml_free(node->input_names);
                    cml_free(node);
                    return -1;
                }
                t->ref_count++;
            } else {
                for (int j = 0; j < ir->tensor_count; j++) {
                    if (ir->tensor_refs[j] == t) {
                        name = cml_strdup(ir->tensor_names[j]);
                        break;
                    }
                }

                if (!name) {
                    char* new_name = cml_malloc(32);
                    if (!new_name) {
                        cml_ir_release_same_graph_input_refs(ir, inputs, i);
                        for (int j = 0; j < i; j++)
                            cml_free(node->input_names[j]);
                        cml_free(node->input_names);
                        cml_free(node);
                        return -1;
                    }
                    snprintf(new_name, 32, "t%d", ir->tensor_count + ir->node_count);
                    ir->tensor_refs[ir->tensor_count] = t;
                    t->ref_count++;
                    ir->tensor_names[ir->tensor_count] = new_name;
                    name                               = cml_strdup(new_name);
                    if (!name) {
                        cml_free(new_name);
                        ir->tensor_refs[ir->tensor_count] = NULL;
                        t->ref_count--;
                        cml_ir_release_same_graph_input_refs(ir, inputs, i);
                        for (int j = 0; j < i; j++)
                            cml_free(node->input_names[j]);
                        cml_free(node->input_names);
                        cml_free(node);
                        return -1;
                    }
                    ir->tensor_count++;
                    ir->tensor_refs_count = ir->tensor_count;
                }
            }
        } else {
            name = cml_strdup("null");
        }

        if (!name) {
            cml_ir_release_same_graph_input_refs(ir, inputs, i);
            for (int j = 0; j < i; j++)
                cml_free(node->input_names[j]);
            cml_free(node->input_names);
            cml_free(node);
            return -1;
        }
        node->input_names[i] = name;
    }

    char* output_name = cml_malloc(32);
    if (!output_name) {
        cml_ir_release_same_graph_input_refs(ir, inputs, num_inputs);
        for (int i = 0; i < num_inputs; i++)
            cml_free(node->input_names[i]);
        cml_free(node->input_names);
        cml_free(node);
        return -1;
    }
    snprintf(output_name, 32, "t%d", ir->tensor_count + ir->node_count);
    node->output_name = output_name;

    node->params = params;
    node->next   = NULL;

    if (num_inputs > 0) {
        node->inputs = cml_malloc((size_t)num_inputs * sizeof(Tensor*));
        if (!node->inputs) {
            cml_ir_release_same_graph_input_refs(ir, inputs, num_inputs);
            for (int i = 0; i < num_inputs; i++) {
                cml_free(node->input_names[i]);
            }
            cml_free(node->input_names);
            cml_free(node);
            return -1;
        }
        memcpy(node->inputs, inputs, (size_t)num_inputs * sizeof(Tensor*));
    } else {
        node->inputs = NULL;
    }
    node->output = NULL;

    node->input_shapes = NULL;
    node->input_ndims  = NULL;
    node->output_shape = NULL;
    node->output_ndim  = 0;
    node->broadcast    = NULL;

    node->requires_grad = false;
    memset(node->needs_input_grad, 0, sizeof(node->needs_input_grad));

    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i] && inputs[i]->requires_grad) {
            node->requires_grad       = true;
            node->needs_input_grad[i] = true;
        }
    }

    node->backward_node      = NULL;
    node->forward_node       = NULL;
    node->saved_for_backward = NULL;

    node->is_executed      = false;
    node->execution_result = NULL;

    node->is_used        = false;
    node->is_fused       = false;
    node->fusion_type    = FUSION_NONE;
    node->fused_kernel   = NULL;
    node->use_count      = 0;
    node->users          = NULL;
    node->users_capacity = 0;
    node->chain_id       = -1;

    node->ref_count = 1;
    node->scope = NULL;
    {
        const char* sc = cml_ir_scope_current();
        if (sc)
            node->scope = cml_strdup(sc);
    }
    if (cml_flame_enabled()) {
        node->build_stack = cml_capture_build_stack();
        /* Nodes manufactured by a rewrite pass (decompose lowering a composite,
         * the fuser collapsing a chain) are created from inside the executor, so
         * their own stack says "cml_ir_execute > cml_ir_decompose" -- true, and
         * useless: it attributes the work to the compiler rather than to the
         * layer whose op is being rewritten. Inherit from the first input, which
         * is the node being rewritten and already carries the model's stack.
         * Doing it here covers every pass, including ones not written yet. */
        if (cml_stack_is_internal(node->build_stack)) {
            const char* inherited = NULL;
            for (int i = 0; i < num_inputs && !inherited; i++) {
                struct IRNode* in = (inputs[i] && inputs[i]->ir_node)
                                        ? (struct IRNode*)inputs[i]->ir_node : NULL;
                if (in && in->build_stack) inherited = in->build_stack;
            }
            if (inherited) {
                cml_free(node->build_stack);
                node->build_stack = cml_strdup(inherited);
            }
        }
    } else {
        node->build_stack = NULL;
    }
    {
        struct IRNode* input_nodes[8];
        int hash_count = num_inputs < 8 ? num_inputs : 8;
        for (int i = 0; i < hash_count; i++)
            input_nodes[i] = (inputs[i] && inputs[i]->ir_node) ? inputs[i]->ir_node : NULL;

        int dtype = (num_inputs > 0 && inputs[0]) ? (int)inputs[0]->dtype : 0;
        node->hash = cml_intern_hash_node_ex((int)type, dtype, input_nodes,
                                             inputs, hash_count, params, 0);
    }

    if (ir->intern_table)
        cml_intern_insert(ir->intern_table, node);

    if (!ir->head) {
        ir->head = node;
        ir->tail = node;
    } else {
        ir->tail->next = node;
        ir->tail       = node;
    }

    ir->node_count++;
    ir->last_result = node;

    ir->is_executed = false;

    return 0;
}

struct IRNode* cml_ir_get_tail(CMLGraph_t ir) {
    if (!ir)
        return NULL;
    return ir->last_result ? ir->last_result : ir->tail;
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
    char* str          = cml_malloc(buffer_size);
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

#define IR_TENSOR_MAX_NDIM 16

static bool ir_tensor_resolve_shape(const Tensor* t, const int** out_shape, int* out_ndim) {
    if (!t || !out_shape || !out_ndim)
        return false;

    if (t->ir_node && t->ir_node->output_shape && t->ir_node->output_ndim > 0 &&
        t->ir_node->output_ndim <= IR_TENSOR_MAX_NDIM) {
        *out_shape = t->ir_node->output_shape;
        *out_ndim  = t->ir_node->output_ndim;
        return true;
    }

    if (t->shape && t->ndim > 0 && t->ndim <= IR_TENSOR_MAX_NDIM) {
        *out_shape = t->shape;
        *out_ndim  = t->ndim;
        return true;
    }

    return false;
}

int cml_ir_compute_broadcast_shape(struct IRNode* node) {
    if (!node || node->num_inputs < 2)
        return -1;

    if (node->input_shapes) {
        for (int i = 0; i < node->num_inputs; i++)
            cml_free(node->input_shapes[i]);
        cml_free(node->input_shapes);
        node->input_shapes = NULL;
    }
    cml_free(node->input_ndims);
    node->input_ndims = NULL;
    if (node->output_shape) {
        cml_free(node->output_shape);
        node->output_shape = NULL;
        node->output_ndim  = 0;
    }

    int** input_shapes = cml_malloc((size_t)node->num_inputs * sizeof(int*));
    if (!input_shapes)
        return -1;

    int* input_ndims = cml_malloc((size_t)node->num_inputs * sizeof(int));
    if (!input_ndims) {
        cml_free(input_shapes);
        return -1;
    }

    for (int i = 0; i < node->num_inputs; i++) {
        if (!node->inputs[i]) {
            for (int j = 0; j < i; j++)
                cml_free(input_shapes[j]);
            cml_free(input_shapes);
            cml_free(input_ndims);
            return -1;
        }
        const int* src = NULL;
        if (!ir_tensor_resolve_shape(node->inputs[i], &src, &input_ndims[i])) {
            for (int j = 0; j < i; j++)
                cml_free(input_shapes[j]);
            cml_free(input_shapes);
            cml_free(input_ndims);
            return -1;
        }
        input_shapes[i] = tensor_shape_copy((int*)src, input_ndims[i]);
        if (!input_shapes[i]) {
            for (int j = 0; j < i; j++)
                cml_free(input_shapes[j]);
            cml_free(input_shapes);
            cml_free(input_ndims);
            return -1;
        }
    }

    int max_ndim = 0;
    for (int i = 0; i < node->num_inputs; i++) {
        if (input_ndims[i] > max_ndim)
            max_ndim = input_ndims[i];
    }

    int* output_shape = cml_malloc((size_t)max_ndim * sizeof(int));
    if (!output_shape) {
        for (int j = 0; j < node->num_inputs; j++)
            cml_free(input_shapes[j]);
        cml_free(input_shapes);
        cml_free(input_ndims);
        return -1;
    }

    for (int d = 0; d < max_ndim; d++) {
        int max_dim = 1;
        for (int i = 0; i < node->num_inputs; i++) {
            int dim_idx = input_ndims[i] - max_ndim + d;
            if (dim_idx >= 0) {
                int dim = input_shapes[i][dim_idx];
                if (dim != 1 && max_dim != 1 && dim != max_dim) {
                    cml_free(output_shape);
                    for (int j = 0; j < node->num_inputs; j++)
                        cml_free(input_shapes[j]);
                    cml_free(input_shapes);
                    cml_free(input_ndims);
                    return -1;
                }
                if (dim > max_dim)
                    max_dim = dim;
            }
        }
        output_shape[d] = max_dim;
    }

    node->output_shape = output_shape;
    node->output_ndim  = max_ndim;
    node->input_shapes = input_shapes;
    node->input_ndims  = input_ndims;

    return 0;
}

struct IRNode* cml_ir_find_by_output(CMLGraph_t ir, const char* output_name) {
    if (!ir || !output_name)
        return NULL;
    for (struct IRNode* node = ir->head; node; node = node->next)
        if (node->output_name && strcmp(node->output_name, output_name) == 0)
            return node;
    return NULL;
}

void cml_ir_unlink_node(CMLGraph_t ir, struct IRNode* node) {
    if (!ir || !node)
        return;

    if (ir->head == node) {
        ir->head = node->next;
        if (ir->tail == node)
            ir->tail = NULL;
        ir->node_count--;
        return;
    }

    struct IRNode* prev = ir->head;
    while (prev && prev->next != node)
        prev = prev->next;
    if (prev) {
        prev->next = node->next;
        if (ir->tail == node)
            ir->tail = prev;
        ir->node_count--;
    }
}

void cml_ir_replace_refs(CMLGraph_t ir, const char* old_name, const char* new_name) {
    if (!ir || !old_name || !new_name)
        return;
    for (struct IRNode* n = ir->head; n; n = n->next) {
        for (int i = 0; i < n->num_inputs; i++) {
            if (n->input_names[i] && strcmp(n->input_names[i], old_name) == 0) {
                cml_free(n->input_names[i]);
                n->input_names[i] = cml_strdup(new_name);
            }
        }
    }
}

void cml_ir_insert_before(CMLGraph_t ir, struct IRNode* new_node, struct IRNode* before) {
    if (!ir || !new_node)
        return;
    new_node->next = NULL;

    if (!before || !ir->head) {
        if (ir->tail)
            ir->tail->next = new_node;
        else
            ir->head = new_node;
        ir->tail = new_node;
        ir->node_count++;
        return;
    }

    if (ir->head == before) {
        new_node->next = before;
        ir->head = new_node;
        ir->node_count++;
        return;
    }

    struct IRNode* prev = ir->head;
    while (prev && prev->next != before)
        prev = prev->next;

    if (prev) {
        new_node->next = before;
        prev->next = new_node;
    } else {
        ir->tail->next = new_node;
        ir->tail = new_node;
    }
    ir->node_count++;
}

/* ── Module scope stack (graph view) ─────────────────────────────────────── */

#define IR_SCOPE_MAX_DEPTH 32
#define IR_SCOPE_MAX_LEN   256

static _Thread_local char g_scope_path[IR_SCOPE_MAX_LEN];
static _Thread_local int  g_scope_ends[IR_SCOPE_MAX_DEPTH]; /* path length after each push */
static _Thread_local int  g_scope_depth = 0;
static int g_scope_enabled = -1; /* -1 = not yet probed */

bool cml_ir_scope_enabled(void) {
    if (cml_flag_enabled(CML_FLAG_NO_EXPORT))
        return false;
    if (g_scope_enabled < 0) {
        const char* viz = getenv("VIZ");
        g_scope_enabled = (viz && viz[0] != '0') ? 1 : 0;
    }
    return g_scope_enabled == 1;
}

/* Folded call stack of whoever is building this node.
 *
 * The flame graph's depth has to come from somewhere, and a lazy graph offers
 * nothing at execution time: every kernel is dispatched from the same executor
 * loop, so sampling there yields one stack for the whole program. The stack that
 * carries meaning is the one that BUILT the node -- module_forward, the layer's
 * forward, the uop helper -- so it is captured here, at construction.
 *
 * Addresses are resolved to names immediately because the strings are shared:
 * identical stacks are interned, so the resolve cost is paid once per distinct
 * call path (tens of them), not once per node.
 */
#define CML_BT_MAX 40      /* deepest stack kept */
#define CML_BT_SKIP 2      /* this function + cml_ir_add_uop itself */

/* Interned stacks: a small table keyed on the raw address vector, so the common
 * case (thousands of nodes from a handful of call paths) costs one memcmp. */
typedef struct {
    void*  addrs[CML_BT_MAX];
    int    n;
    char*  folded;
} BtEntry;
static BtEntry g_bt[64];
static int g_bt_n = 0;

/* "/path/bin(function+0x2e) [0x...]" -> "function", or empty when the frame has
 * no symbol. Static functions are absent from the dynamic table even with
 * -rdynamic, and for those backtrace_symbols reports only the object path --
 * emitting that would put the binary's own name in the stack once per
 * unresolved frame, which reads like a real function and is not one. An empty
 * name drops the frame instead. */
static void bt_symbol(const char* raw, char* out, size_t cap) {
    out[0] = '\0';
    const char* open = strchr(raw, '(');
    const char* plus = open ? strchr(open, '+') : NULL;
    const char* end  = plus ? plus : (open ? strchr(open, ')') : NULL);
    if (!open || !end || end <= open + 1)
        return;                     /* "/path/bin() [0x..]" -- no symbol */
    size_t n = (size_t)(end - open - 1);
    if (n >= cap) n = cap - 1;
    memcpy(out, open + 1, n);
    out[n] = '\0';
}

/* Frames that are pure plumbing: they appear in every stack and say nothing
 * about which part of the model is being built. */
static int bt_is_noise(const char* fn) {
    static const char* skip[] = {
        "cml_ir_add_uop", "cml_ir_create_node", "ir_node_create",
        "uop_binary_ex", "uop_unary_noparam", "uop_reduce_ex",
        "finish_source_node", "attach_movement_op",
        /* Process entry: true of every stack, so it distinguishes nothing. */
        "_start", "__libc_start_main", "__libc_start_call_main", NULL,
    };
    for (int i = 0; skip[i]; i++)
        if (strcmp(fn, skip[i]) == 0) return 1;
    return 0;
}

/* True when a stack is rooted in the compiler rather than in model code. */
static int cml_stack_is_internal(const char* folded) {
    if (!folded || !*folded) return 1;
    static const char* passes[] = {
        "cml_ir_decompose", "cml_ir_fuse_elementwise", "cml_ir_fuse_matmul_epilogue",
        "cml_ir_optimize", "cml_ir_execute", "cpu_execute_ir", "cml_ir_execute_fusion",
        "cml_ir_reexecute", "cml_ir_execute_up_to", NULL,
    };
    for (int i = 0; passes[i]; i++)
        if (strstr(folded, passes[i])) return 1;
    return 0;
}

static char* cml_capture_build_stack(void) {
#ifdef _WIN32
    return NULL;   /* no execinfo/backtrace on Windows; flame graph omits stacks */
#else
    void* addrs[CML_BT_MAX];
    int n = backtrace(addrs, CML_BT_MAX);
    if (n <= CML_BT_SKIP) return NULL;

    for (int i = 0; i < g_bt_n; i++) {
        if (g_bt[i].n == n && memcmp(g_bt[i].addrs, addrs, (size_t)n * sizeof(void*)) == 0)
            return g_bt[i].folded ? cml_strdup(g_bt[i].folded) : NULL;
    }

    char** syms = backtrace_symbols(addrs, n);
    if (!syms) return NULL;

    /* backtrace() is innermost-first; a flame graph reads root-first. */
    char buf[2048];
    size_t len = 0;
    for (int i = n - 1; i >= CML_BT_SKIP; i--) {
        char fn[192];
        bt_symbol(syms[i], fn, sizeof(fn));
        if (!fn[0] || bt_is_noise(fn)) continue;
        size_t need = strlen(fn) + (len ? 1 : 0);
        if (len + need >= sizeof(buf)) break;
        if (len) buf[len++] = ';';
        memcpy(buf + len, fn, strlen(fn));
        len += strlen(fn);
    }
    buf[len] = '\0';
    free(syms);   /* backtrace_symbols uses malloc, not the pool allocator */
    if (!len) return NULL;

    if (g_bt_n < (int)(sizeof(g_bt) / sizeof(g_bt[0]))) {
        BtEntry* e = &g_bt[g_bt_n++];
        memcpy(e->addrs, addrs, (size_t)n * sizeof(void*));
        e->n = n;
        e->folded = cml_strdup(buf);
    }
    return cml_strdup(buf);
#endif /* _WIN32 */
}

void cml_ir_scope_push(const char* name) {
    if (!cml_ir_scope_enabled() || !name || g_scope_depth >= IR_SCOPE_MAX_DEPTH)
        return;

    int len = (int)strlen(g_scope_path);
    int add = (int)strlen(name) + (len > 0 ? 1 : 0);
    if (len + add >= IR_SCOPE_MAX_LEN) {
        /* Too deep to name: still push so the pop stays balanced. */
        g_scope_ends[g_scope_depth++] = len;
        return;
    }
    if (len > 0)
        g_scope_path[len++] = '/';
    strcpy(g_scope_path + len, name);
    g_scope_ends[g_scope_depth++] = (int)strlen(g_scope_path);
}

void cml_ir_scope_pop(void) {
    if (!cml_ir_scope_enabled() || g_scope_depth <= 0)
        return;
    g_scope_depth--;
    int keep = g_scope_depth > 0 ? g_scope_ends[g_scope_depth - 1] : 0;
    g_scope_path[keep] = '\0';
}

const char* cml_ir_scope_current(void) {
    return (cml_ir_scope_enabled() && g_scope_path[0]) ? g_scope_path : NULL;
}
