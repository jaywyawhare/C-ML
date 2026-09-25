#include "backend/thunder_executor.h"
#include "ops/uops.h"
#include "ops/ir/dispatch.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

typedef struct {
    const char* thunder_name;
    UOpType uop;
} ThunderOpMapping;

static const ThunderOpMapping op_table[] = {
    {"torch.add", UOP_ADD},
    {"torch.sub", UOP_SUB},
    {"torch.mul", UOP_MUL},
    {"torch.div", UOP_DIV},
    {"torch.matmul", UOP_MATMUL},
    {"torch.neg", UOP_NEG},
    {"torch.exp", UOP_EXP},
    {"torch.log", UOP_LOG},
    {"torch.sqrt", UOP_SQRT},
    {"torch.abs", UOP_ABS},
    {"torch.sin", UOP_SIN},
    {"torch.cos", UOP_COS},
    {"torch.tanh", UOP_TANH},
    {"torch.sigmoid", UOP_SIGMOID},
    {"torch.relu", UOP_RELU},
    {"torch.sum", UOP_SUM},
    {"torch.max", UOP_MAX_REDUCE},
    {"torch.mean", UOP_MEAN},
    {"torch.where", UOP_WHERE},
    {"torch.pow", UOP_POW},
    {"torch.sign", UOP_SIGN},
    {"torch.floor", UOP_FLOOR},
    {"torch.ceil", UOP_CEIL},
    {"torch.round", UOP_ROUND},
    {"torch.erf", UOP_ERF},
    {"torch.rsqrt", UOP_RSQRT},
    {"torch.reciprocal", UOP_RECIP},
    {"torch.mm", UOP_MATMUL},
    {"torch.bmm", UOP_MATMUL},
    {"torch.relu6", UOP_RELU6},
    /* torch.gelu and torch.leaky_relu decompose into other uops and have no
     * dedicated UOpType, so they are intentionally absent from this table
     * (thunder_lookup_op returns "unsupported" for unknown names).
     *
     * torch.reshape, torch.permute, torch.conv2d and torch.gather are absent for
     * a different reason: their uops need attributes (target shape, permutation,
     * stride/padding, gather dim) that a CMLThunderOp carries no field for, and
     * they are not derivable from the input tensors. They used to sit in this
     * table, which made thunder_lookup_op resolve them and then fail in the
     * dispatch switch with a confusing "not implemented" -- a table entry is a
     * claim the op is dispatchable. Re-add them together with an attribute field
     * on CMLThunderOp, not before. */
    {"torch.silu", UOP_SILU},
    {"torch.mish", UOP_MISH},
    {"torch.hardswish", UOP_HARDSWISH},
    {"torch.selu", UOP_SELU},
    {"torch.elu", UOP_ELU},
    {"torch.log2", UOP_LOG2},
    {"torch.exp2", UOP_EXP2},
    {"torch.maximum", UOP_MAX},
    {NULL, 0}};

static UOpType thunder_lookup_op(const char* name) {
    for (int i = 0; op_table[i].thunder_name; i++) {
        if (strcmp(op_table[i].thunder_name, name) == 0)
            return op_table[i].uop;
    }
    return (UOpType)-1;
}

static CMLBackendType parse_backend(const char* name) {
    if (!name)
        return CML_BACKEND_CPU_FALLBACK;
    if (strcmp(name, "cml_cuda") == 0)
        return CML_BACKEND_CUDA;
    if (strcmp(name, "cml_metal") == 0)
        return CML_BACKEND_METAL;
    if (strcmp(name, "cml_rocm") == 0)
        return CML_BACKEND_ROCM;
    if (strcmp(name, "cml_cpu") == 0)
        return CML_BACKEND_CPU_FALLBACK;
    return CML_BACKEND_CPU_FALLBACK;
}

CMLThunderExecutor* cml_thunder_create(const char* backend) {
    CMLThunderExecutor* exec = cml_calloc(1, sizeof(CMLThunderExecutor));
    if (!exec)
        return NULL;

    if (backend)
        strncpy(exec->backend_name, backend, sizeof(exec->backend_name) - 1);
    else
        strncpy(exec->backend_name, "cml_cpu", sizeof(exec->backend_name) - 1);

    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) {
        cml_free(exec);
        return NULL;
    }

    CMLBackendType bt = parse_backend(exec->backend_name);
    cml_dispatch_init(ctx);
    cml_dispatch_set_preferred(ctx, bt);

    exec->dispatch_ctx = ctx;
    exec->initialized  = true;

    LOG_INFO("[thunder] Executor created: backend=%s", exec->backend_name);
    return exec;
}

void cml_thunder_free(CMLThunderExecutor* exec) {
    if (!exec)
        return;
    if (exec->dispatch_ctx)
        cml_dispatch_free((CMLDispatchContext*)exec->dispatch_ctx);
    cml_free(exec);
}

int cml_thunder_execute(CMLThunderExecutor* exec, CMLThunderOp* ops, int num_ops) {
    if (!exec || !exec->initialized || !ops)
        return -1;

    CMLDispatchContext* ctx = (CMLDispatchContext*)exec->dispatch_ctx;

    for (int i = 0; i < num_ops; i++) {
        CMLThunderOp* op = &ops[i];
        UOpType uop      = thunder_lookup_op(op->op_name);
        if ((int)uop == -1) {
            LOG_ERROR("[thunder] Unsupported op: %s", op->op_name);
            return -1;
        }

        Tensor** inputs  = (Tensor**)op->inputs;
        Tensor** outputs = (Tensor**)op->outputs;

        Tensor* result = NULL;

        switch (uop) {
        case UOP_ADD:
            if (op->num_inputs >= 2)
                result = uop_add(inputs[0], inputs[1]);
            break;
        case UOP_SUB:
            if (op->num_inputs >= 2)
                result = uop_sub(inputs[0], inputs[1]);
            break;
        case UOP_MUL:
            if (op->num_inputs >= 2)
                result = uop_mul(inputs[0], inputs[1]);
            break;
        case UOP_DIV:
            if (op->num_inputs >= 2)
                result = uop_div(inputs[0], inputs[1]);
            break;
        case UOP_MATMUL:
            if (op->num_inputs >= 2)
                result = uop_matmul(inputs[0], inputs[1]);
            break;
        case UOP_NEG:
            if (op->num_inputs >= 1)
                result = uop_neg(inputs[0]);
            break;
        case UOP_EXP:
            if (op->num_inputs >= 1)
                result = uop_exp(inputs[0]);
            break;
        case UOP_LOG:
            if (op->num_inputs >= 1)
                result = uop_log(inputs[0]);
            break;
        case UOP_SQRT:
            if (op->num_inputs >= 1)
                result = uop_sqrt(inputs[0]);
            break;
        case UOP_ABS:
            if (op->num_inputs >= 1)
                result = uop_abs(inputs[0]);
            break;
        case UOP_SIN:
            if (op->num_inputs >= 1)
                result = uop_sin(inputs[0]);
            break;
        case UOP_COS:
            if (op->num_inputs >= 1)
                result = uop_cos(inputs[0]);
            break;
        case UOP_TANH:
            if (op->num_inputs >= 1)
                result = uop_tanh(inputs[0]);
            break;
        case UOP_SIGMOID:
            if (op->num_inputs >= 1)
                result = uop_sigmoid(inputs[0]);
            break;
        case UOP_SUM:
            if (op->num_inputs >= 1)
                result = uop_sum(inputs[0], 0);
            break;
        case UOP_MEAN:
            if (op->num_inputs >= 1)
                result = uop_mean(inputs[0], 0);
            break;
        case UOP_RECIP:
            if (op->num_inputs >= 1)
                result = uop_recip(inputs[0]);
            break;
        case UOP_RSQRT:
            if (op->num_inputs >= 1)
                result = uop_rsqrt(inputs[0]);
            break;
        case UOP_RELU:
            if (op->num_inputs >= 1)
                result = uop_relu(inputs[0]);
            break;
        case UOP_RELU6:
            if (op->num_inputs >= 1)
                result = uop_relu6(inputs[0]);
            break;
        case UOP_SILU:
            if (op->num_inputs >= 1)
                result = uop_silu(inputs[0]);
            break;
        case UOP_MISH:
            if (op->num_inputs >= 1)
                result = uop_mish(inputs[0]);
            break;
        case UOP_HARDSWISH:
            if (op->num_inputs >= 1)
                result = uop_hardswish(inputs[0]);
            break;
        case UOP_SELU:
            if (op->num_inputs >= 1)
                result = uop_selu(inputs[0]);
            break;
        case UOP_ELU:
            if (op->num_inputs >= 1)
                result = uop_elu(inputs[0], 1.0f);
            break;
        case UOP_MAX:
            if (op->num_inputs >= 2)
                result = uop_max(inputs[0], inputs[1]);
            break;
        case UOP_LOG2:
            if (op->num_inputs >= 1)
                result = uop_log2(inputs[0]);
            break;
        case UOP_EXP2:
            if (op->num_inputs >= 1)
                result = uop_exp2(inputs[0]);
            break;
        case UOP_SIGN:
            if (op->num_inputs >= 1)
                result = uop_sign(inputs[0]);
            break;
        case UOP_FLOOR:
            if (op->num_inputs >= 1)
                result = uop_floor(inputs[0]);
            break;
        case UOP_CEIL:
            if (op->num_inputs >= 1)
                result = uop_ceil(inputs[0]);
            break;
        case UOP_ROUND:
            if (op->num_inputs >= 1)
                result = uop_round(inputs[0]);
            break;
        case UOP_ERF:
            if (op->num_inputs >= 1)
                result = uop_erf(inputs[0]);
            break;
        case UOP_POW:
            if (op->num_inputs >= 2)
                result = uop_pow(inputs[0], inputs[1]);
            break;
        case UOP_MAX_REDUCE:
            /* NULL ReduceParams reduces every axis, matching how SUM and MEAN
             * are dispatched above. */
            if (op->num_inputs >= 1)
                result = uop_max_reduce(inputs[0], NULL);
            break;
        case UOP_WHERE: {
            /* uop_where takes its three operands in a params struct rather than
             * as positional arguments. */
            if (op->num_inputs >= 3) {
                WhereParams wp = {.cond = inputs[0], .a = inputs[1], .b = inputs[2]};
                result         = uop_where(&wp);
            }
            break;
        }
        default:
            LOG_ERROR("[thunder] Op dispatch not implemented: %s", op->op_name);
            return -1;
        }

        if (!result) {
            LOG_ERROR("[thunder] Op execution failed: %s", op->op_name);
            return -1;
        }

        if (op->num_outputs > 0 && outputs[0]) {
            Tensor* dst = outputs[0];
            if (dst->data && result->data && dst->numel == result->numel)
                memcpy(dst->data, result->data, dst->numel * sizeof(float));
            tensor_free(result);
        } else if (op->num_outputs > 0) {
            outputs[0] = result;
        } else {
            tensor_free(result);
        }
    }

    ctx->executions_total += num_ops;
    return 0;
}

int cml_thunder_register(void) {
    LOG_INFO("[thunder] C-ML registered as Thunder executor");
    return 0;
}
