#include "ops/ir/ir.h"
#include "ops/ir/context.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "core/logging.h"
#include "autograd/autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>

static _Atomic(CMLGraph_t) g_auto_capture_ir = NULL;

static CMLGraph_t g_global_ir_context = NULL;

int cml_ir_enable_auto_capture(CMLGraph_t ir) {
    if (!ir) {
        LOG_ERROR("Cannot enable auto-capture with NULL IR");
        return -1;
    }

    atomic_store(&g_auto_capture_ir, ir);
    return 0;
}

void cml_ir_disable_auto_capture(void) {
    atomic_store(&g_auto_capture_ir, NULL);
}

CMLGraph_t cml_ir_get_auto_capture_context(void) { return atomic_load(&g_auto_capture_ir); }

UOpType cml_ir_optype_to_uoptype(OpType op_type, int num_inputs) {
    switch (op_type) {
    case OP_NONE:
    case OP_RELU:
    case OP_SOFTMAX:
    case OP_LOG_SOFTMAX:
    case OP_LEAKY_RELU:
    case OP_SWISH:
    case OP_GELU:
    case OP_MIN:
    case OP_TRANSPOSE:
    case OP_VIEW:
    case OP_SQUEEZE:
    case OP_UNSQUEEZE:
    case OP_MSE_LOSS:
    case OP_MAE_LOSS:
    case OP_BCE_LOSS:
    case OP_CROSS_ENTROPY_LOSS:
    case OP_HUBER_LOSS:
    case OP_KL_DIV_LOSS:
    case OP_CLONE:
    case OP_DETACH:
    case OP_CUSTOM:
        return UOP_COUNT;
    case OP_TANH:
        return UOP_TANH;
    case OP_SIGMOID:
        return UOP_SIGMOID;
    case OP_ELU:
        return UOP_ELU;
    case OP_SELU:
        return UOP_SELU;
    case OP_MISH:
        return UOP_MISH;
    case OP_HARD_SWISH:
        return UOP_HARDSWISH;
    case OP_POW:
        return UOP_POW;
    case OP_SIN:
        return UOP_SIN;
    case OP_COS:
        return UOP_COS;
    case OP_TAN:
        return UOP_TAN;
    case OP_ADD:
        return UOP_ADD;
    case OP_SUB:
        return UOP_SUB;
    case OP_MUL:
        return UOP_MUL;
    case OP_DIV:
        return UOP_DIV;
    case OP_MATMUL:
        return UOP_MATMUL;

    case OP_NEG:
        return UOP_NEG;
    case OP_EXP:
        return UOP_EXP;
    case OP_LOG:
        return UOP_LOG;
    case OP_SQRT:
        return UOP_SQRT;

    case OP_SUM:
        return UOP_SUM;
    case OP_MEAN:
        return UOP_MEAN;
    case OP_MAX:
        return (num_inputs >= 2) ? UOP_MAX : UOP_MAX_REDUCE;

    case OP_RESHAPE:
        return UOP_RESHAPE;
    case OP_PERMUTE:
        return UOP_PERMUTE;

    default:
        return UOP_COUNT;
    }
}

int cml_ir_auto_capture_tensor_op(OpType op_type, Tensor** inputs, int num_inputs, void* params) {
    CMLGraph_t ir = cml_ir_get_auto_capture_context();
    if (!ir)
        return 0;

    UOpType uop_type = cml_ir_optype_to_uoptype(op_type, num_inputs);
    if (uop_type == UOP_COUNT)
        return 0;

    int result = cml_ir_add_uop(ir, uop_type, inputs, num_inputs, params);
    if (result != 0) {
        LOG_WARNING("Failed to auto-capture operation %d", op_type);
    }

    return result;
}

CMLGraph_t cml_ir_get_or_create_context(void) {
    CMLGraph_t auto_capture_ir = cml_ir_get_auto_capture_context();
    if (auto_capture_ir)
        return auto_capture_ir;

    if (!g_global_ir_context) {
        g_global_ir_context = cml_ir_new(IR_TARGET_C);
        if (!g_global_ir_context) {
            LOG_ERROR("Failed to create global IR context");
            return NULL;
        }
    }
    return g_global_ir_context;
}

void cml_ir_set_global_context(CMLGraph_t ir) { g_global_ir_context = ir; }

void cml_ir_clear_global_if_current(CMLGraph_t ir) {
    if (!ir || g_global_ir_context != ir)
        return;
    if (atomic_load(&g_auto_capture_ir) == ir)
        cml_ir_disable_auto_capture();
    g_global_ir_context = NULL;
}

void cml_ir_reset_global_context(void) {
    if (g_global_ir_context) {
        if (atomic_load(&g_auto_capture_ir) == g_global_ir_context) {
            cml_ir_disable_auto_capture();
        }
        cml_ir_free(g_global_ir_context);
        g_global_ir_context = NULL;
    }
    cml_cleanup_buffer_cache();
}

void cml_ir_ensure_gradients_executed(CMLGraph_t ir) {
    if (!ir || !ir->tensor_refs)
        return;

    for (int i = 0; i < ir->tensor_refs_count; i++) {
        Tensor* t = ir->tensor_refs[i];
        if (t && t->grad) {
            tensor_ensure_executed(t->grad);
        }
    }
}
