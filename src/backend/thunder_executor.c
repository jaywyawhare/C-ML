#include "backend/thunder_executor.h"
#include "ops/uops.h"
#include "ops/ir/dispatch.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* thunder_name;
    UOpType uop;
} ThunderOpMapping;

static const ThunderOpMapping op_table[] = {
    {"torch.add",       UOP_ADD},
    {"torch.sub",       UOP_SUB},
    {"torch.mul",       UOP_MUL},
    {"torch.div",       UOP_DIV},
    {"torch.matmul",    UOP_MATMUL},
    {"torch.neg",       UOP_NEG},
    {"torch.exp",       UOP_EXP},
    {"torch.log",       UOP_LOG},
    {"torch.sqrt",      UOP_SQRT},
    {"torch.abs",       UOP_ABS},
    {"torch.sin",       UOP_SIN},
    {"torch.cos",       UOP_COS},
    {"torch.tanh",      UOP_TANH},
    {"torch.sigmoid",   UOP_SIGMOID},
    {"torch.relu",      UOP_MAX},
    {"torch.sum",       UOP_SUM},
    {"torch.max",       UOP_MAX_REDUCE},
    {"torch.mean",      UOP_MEAN},
    {"torch.reshape",   UOP_RESHAPE},
    {"torch.permute",   UOP_PERMUTE},
    {"torch.where",     UOP_WHERE},
    {"torch.pow",       UOP_POW},
    {"torch.conv2d",    UOP_CONV2D},
    {"torch.gather",    UOP_GATHER},
    {"torch.sign",      UOP_SIGN},
    {"torch.floor",     UOP_FLOOR},
    {"torch.ceil",      UOP_CEIL},
    {"torch.round",     UOP_ROUND},
    {"torch.erf",       UOP_ERF},
    {"torch.rsqrt",     UOP_RSQRT},
    {"torch.reciprocal", UOP_RECIP},
    {NULL,              0}
};

static UOpType thunder_lookup_op(const char* name) {
    for (int i = 0; op_table[i].thunder_name; i++) {
        if (strcmp(op_table[i].thunder_name, name) == 0)
            return op_table[i].uop;
    }
    return (UOpType)-1;
}

static CMLBackendType parse_backend(const char* name) {
    if (!name) return CML_BACKEND_CPU_FALLBACK;
    if (strcmp(name, "cml_cuda") == 0)  return CML_BACKEND_CUDA;
    if (strcmp(name, "cml_metal") == 0) return CML_BACKEND_METAL;
    if (strcmp(name, "cml_rocm") == 0)  return CML_BACKEND_ROCM;
    if (strcmp(name, "cml_cpu") == 0)   return CML_BACKEND_CPU_FALLBACK;
    return CML_BACKEND_CPU_FALLBACK;
}

CMLThunderExecutor* cml_thunder_create(const char* backend) {
    CMLThunderExecutor* exec = calloc(1, sizeof(CMLThunderExecutor));
    if (!exec) return NULL;

    if (backend)
        strncpy(exec->backend_name, backend, sizeof(exec->backend_name) - 1);
    else
        strncpy(exec->backend_name, "cml_cpu", sizeof(exec->backend_name) - 1);

    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) {
        free(exec);
        return NULL;
    }

    CMLBackendType bt = parse_backend(exec->backend_name);
    cml_dispatch_init(ctx);
    cml_dispatch_set_preferred(ctx, bt);

    exec->dispatch_ctx = ctx;
    exec->initialized = true;

    LOG_INFO("[thunder] Executor created: backend=%s", exec->backend_name);
    return exec;
}

void cml_thunder_free(CMLThunderExecutor* exec) {
    if (!exec) return;
    if (exec->dispatch_ctx)
        cml_dispatch_free((CMLDispatchContext*)exec->dispatch_ctx);
    free(exec);
}

int cml_thunder_execute(CMLThunderExecutor* exec, CMLThunderOp* ops, int num_ops) {
    if (!exec || !exec->initialized || !ops) return -1;

    CMLDispatchContext* ctx = (CMLDispatchContext*)exec->dispatch_ctx;

    for (int i = 0; i < num_ops; i++) {
        CMLThunderOp* op = &ops[i];
        UOpType uop = thunder_lookup_op(op->op_name);
        if ((int)uop == -1) {
            LOG_ERROR("[thunder] Unsupported op: %s", op->op_name);
            return -1;
        }

        Tensor** inputs = (Tensor**)op->inputs;
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
