/**
 * @file context.c
 * @brief IR auto-capture and global context management
 */

#include "ops/ir/ir.h"
#include "ops/ir/context.h"
#include "ops/ir/internal.h"
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
    LOG_DEBUG("Auto-capture enabled for IR target: %d", ir->target);
    return 0;
}

void cml_ir_disable_auto_capture(void) {
    atomic_store(&g_auto_capture_ir, NULL);
    LOG_DEBUG("Auto-capture disabled");
}

CMLGraph_t cml_ir_get_auto_capture_context(void) { return atomic_load(&g_auto_capture_ir); }

// Convert OpType (from autograd) to UOpType (for IR)
UOpType cml_ir_optype_to_uoptype(OpType op_type, int num_inputs) {
    switch (op_type) {
    case OP_NONE:
    case OP_TANH:
    case OP_RELU:
    case OP_SIGMOID:
    case OP_SOFTMAX:
    case OP_LOG_SOFTMAX:
    case OP_LEAKY_RELU:
    case OP_ELU:
    case OP_SELU:
    case OP_SWISH:
    case OP_MISH:
    case OP_HARD_SWISH:
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
    case OP_POW:
        return UOP_POW;
    case OP_SIN:
        return UOP_SIN;
    case OP_COS:
        return UOP_COS;
    case OP_TAN:
        return UOP_TAN;
    // Binary operations
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

    // Unary operations
    case OP_NEG:
        return UOP_NEG;
    case OP_EXP:
        return UOP_EXP;
    case OP_LOG:
        return UOP_LOG;
    case OP_SQRT:
        return UOP_SQRT;

    // Reductions
    case OP_SUM:
        return UOP_SUM;
    case OP_MEAN:
        return UOP_MEAN;
    case OP_MAX:
        // OP_MAX can be elementwise (2 inputs) or reduction (1 input)
        return (num_inputs >= 2) ? UOP_MAX : UOP_MAX_REDUCE;

    // Shape operations
    case OP_RESHAPE:
        return UOP_RESHAPE;
    case OP_PERMUTE:
        return UOP_PERMUTE;

    // No direct mapping for other operations
    default:
        return UOP_COUNT; // Invalid/unknown
    }
}

// Helper function to automatically capture tensor operations to IR
// This is called from tensor_* functions when auto-capture is enabled
int cml_ir_auto_capture_tensor_op(OpType op_type, Tensor** inputs, int num_inputs, void* params) {
    CMLGraph_t ir = cml_ir_get_auto_capture_context();
    if (!ir) {
        // Auto-capture not enabled, silently skip
        return 0;
    }

    // Convert OpType to UOpType
    UOpType uop_type = cml_ir_optype_to_uoptype(op_type, num_inputs);
    if (uop_type == UOP_COUNT) {
        // No mapping available, skip this operation
        LOG_DEBUG("No UOp mapping for OpType %d, skipping auto-capture", op_type);
        return 0;
    }

    // Add to IR
    int result = cml_ir_add_uop(ir, uop_type, inputs, num_inputs, params);
    if (result == 0) {
        LOG_DEBUG("Auto-captured operation %d as UOp %d", op_type, uop_type);
    } else {
        LOG_WARNING("Failed to auto-capture operation %d", op_type);
    }

    return result;
}

CMLGraph_t cml_ir_get_or_create_context(void) {
    // First check if auto-capture is enabled
    CMLGraph_t auto_capture_ir = cml_ir_get_auto_capture_context();
    if (auto_capture_ir) {
        return auto_capture_ir;
    }

    // Otherwise use/create global context
    if (!g_global_ir_context) {
        // Default to CPU target, can be changed
        g_global_ir_context = cml_ir_new(IR_TARGET_C);
        if (!g_global_ir_context) {
            LOG_ERROR("Failed to create global IR context");
            return NULL;
        }
    }
    return g_global_ir_context;
}

void cml_ir_set_global_context(CMLGraph_t ir) { g_global_ir_context = ir; }

void cml_ir_reset_global_context(void) {
    if (g_global_ir_context) {
        cml_ir_free(g_global_ir_context);
        g_global_ir_context = NULL;
    }
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
