#include "autograd/checkpointing.h"
#include "autograd/autograd.h"
#include "nn.h"
#include "nn/layers/sequential.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

static bool checkpointing_enabled = false;

typedef struct CheckpointedTensor {
    Tensor* tensor;
    struct IRNode* saved_ir_node; // Save IR node to recompute
    CMLGraph_t saved_ir_context;  // Save IR context
    Tensor** saved_inputs;        // Save inputs for recomputation
    int num_inputs;
} CheckpointedTensor;

static CheckpointedTensor** checkpointed_tensors = NULL;
static int num_checkpointed                      = 0;
static int checkpointed_capacity                 = 0;

void autograd_set_checkpointing(bool enabled) {
    checkpointing_enabled = enabled;
    LOG_DEBUG("Gradient checkpointing %s", enabled ? "enabled" : "disabled");
}

bool autograd_is_checkpointing_enabled(void) { return checkpointing_enabled; }

int autograd_checkpoint(Tensor* tensor) {
    if (!tensor || !checkpointing_enabled)
        return -1;

    if (num_checkpointed >= checkpointed_capacity) {
        int new_capacity = checkpointed_capacity == 0 ? 16 : checkpointed_capacity * 2;
        CheckpointedTensor** new_array =
            realloc(checkpointed_tensors, (size_t)new_capacity * sizeof(CheckpointedTensor*));
        if (!new_array)
            return -1;

        checkpointed_tensors  = new_array;
        checkpointed_capacity = new_capacity;
    }

    CheckpointedTensor* checkpoint = malloc(sizeof(CheckpointedTensor));
    if (!checkpoint)
        return -1;

    checkpoint->tensor           = tensor;
    checkpoint->saved_ir_node    = tensor->ir_node;
    checkpoint->saved_ir_context = tensor->ir_context;

    if (tensor->ir_node && tensor->ir_node->inputs) {
        checkpoint->num_inputs   = tensor->ir_node->num_inputs;
        checkpoint->saved_inputs = malloc((size_t)checkpoint->num_inputs * sizeof(Tensor*));
        if (!checkpoint->saved_inputs) {
            free(checkpoint);
            return -1;
        }
        for (int i = 0; i < checkpoint->num_inputs; i++) {
            checkpoint->saved_inputs[i] = tensor->ir_node->inputs[i];
        }
    } else {
        checkpoint->num_inputs   = 0;
        checkpoint->saved_inputs = NULL;
    }

    tensor->ir_node    = NULL;
    tensor->ir_context = NULL;

    checkpointed_tensors[num_checkpointed] = checkpoint;
    num_checkpointed++;

    return 0;
}

Tensor* autograd_recompute(Tensor* tensor) {
    if (!tensor || !checkpointing_enabled)
        return NULL;

    CheckpointedTensor* checkpoint = NULL;
    for (int i = 0; i < num_checkpointed; i++) {
        if (checkpointed_tensors[i] && checkpointed_tensors[i]->tensor == tensor) {
            checkpoint = checkpointed_tensors[i];
            break;
        }
    }

    if (!checkpoint || !checkpoint->saved_ir_node)
        return tensor;

    if (checkpoint->saved_inputs && checkpoint->num_inputs > 0 && checkpoint->saved_ir_node) {
        struct IRNode* node = checkpoint->saved_ir_node;
        UOpType uop_type    = node->type;

        Tensor** inputs = checkpoint->saved_inputs;
        for (int i = 0; i < checkpoint->num_inputs; i++) {
            if (inputs[i] && !inputs[i]->ir_node) {
                Tensor* recomputed_input = autograd_recompute(inputs[i]);
                if (recomputed_input && recomputed_input != inputs[i]) {
                    inputs[i] = recomputed_input;
                }
            }
        }

        Tensor* recomputed = NULL;

        switch (uop_type) {
        case UOP_ADD:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_add(inputs[0], inputs[1]);
            }
            break;
        case UOP_SUB:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_sub(inputs[0], inputs[1]);
            }
            break;
        case UOP_MUL:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_mul(inputs[0], inputs[1]);
            }
            break;
        case UOP_DIV:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_div(inputs[0], inputs[1]);
            }
            break;
        case UOP_NEG:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_neg(inputs[0]);
            }
            break;
        case UOP_EXP:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_exp(inputs[0]);
            }
            break;
        case UOP_LOG:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_log(inputs[0]);
            }
            break;
        case UOP_SQRT:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_sqrt(inputs[0]);
            }
            break;
        case UOP_MATMUL:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_matmul(inputs[0], inputs[1]);
            }
            break;
        case UOP_SUM:
            if (checkpoint->num_inputs == 1) {
                ReduceParams params = {0};
                params.dims         = NULL;
                params.num_dims     = 0;
                params.keepdim      = false;
                recomputed          = uop_sum(inputs[0], &params);
            }
            break;
        case UOP_MEAN:
            if (checkpoint->num_inputs == 1) {
                ReduceParams params = {0};
                params.dims         = NULL;
                params.num_dims     = 0;
                params.keepdim      = false;
                recomputed          = uop_mean(inputs[0], &params);
            }
            break;
        case UOP_MAX:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_max(inputs[0], inputs[1]);
            }
            break;
        case UOP_CMPLT:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_cmplt(inputs[0], inputs[1]);
            }
            break;
        case UOP_RECIP:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_recip(inputs[0]);
            }
            break;
        case UOP_ABS:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_abs(inputs[0]);
            }
            break;
        case UOP_SIN:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_sin(inputs[0]);
            }
            break;
        case UOP_COS:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_cos(inputs[0]);
            }
            break;
        case UOP_TAN:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_tan(inputs[0]);
            }
            break;
        case UOP_POW:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_pow(inputs[0], inputs[1]);
            }
            break;
        case UOP_MAX_REDUCE:
            if (checkpoint->num_inputs == 1) {
                ReduceParams max_reduce_params = {0};
                max_reduce_params.dims         = NULL;
                max_reduce_params.num_dims     = 0;
                max_reduce_params.keepdim      = false;
                recomputed = uop_max_reduce(inputs[0], &max_reduce_params);
            }
            break;
        case UOP_RESHAPE:
            if (checkpoint->num_inputs == 1 && node->output_shape) {
                ReshapeParams reshape_params;
                reshape_params.new_shape = node->output_shape;
                reshape_params.new_ndim  = node->output_ndim;
                recomputed = uop_reshape(inputs[0], &reshape_params);
            }
            break;
        case UOP_PERMUTE:
            if (checkpoint->num_inputs == 1 && node->params) {
                recomputed = uop_permute(inputs[0], (PermuteParams*)node->params);
            }
            break;
        case UOP_EXPAND:
            if (checkpoint->num_inputs == 1 && node->params) {
                recomputed = uop_expand(inputs[0], (ExpandParams*)node->params);
            }
            break;
        case UOP_STRIDE:
            if (checkpoint->num_inputs == 1 && node->params) {
                recomputed = uop_stride(inputs[0], (StrideParams*)node->params);
            }
            break;
        case UOP_SLICE:
            if (checkpoint->num_inputs == 1 && node->params) {
                recomputed = uop_slice(inputs[0], (SliceParams*)node->params);
            }
            break;
        case UOP_CONV2D:
            if (checkpoint->num_inputs >= 2 && node->params) {
                Tensor* bias = checkpoint->num_inputs >= 3 ? inputs[2] : NULL;
                recomputed = uop_conv2d(inputs[0], inputs[1], bias, (Conv2DParams*)node->params);
            }
            break;
        case UOP_WHERE:
            if (checkpoint->num_inputs == 3) {
                WhereParams where_params;
                where_params.cond = inputs[0];
                where_params.a    = inputs[1];
                where_params.b    = inputs[2];
                recomputed = uop_where(&where_params);
            }
            break;
        case UOP_COUNT:
            break;
        default:
            // For unsupported operations, restore IR node only
            LOG_DEBUG("UOpType %d not supported for recomputation, restoring IR node only",
                      uop_type);
            break;
        }

        if (recomputed) {
            if (tensor->numel == recomputed->numel && tensor->dtype == recomputed->dtype) {
                void* tensor_data     = tensor_data_ptr(tensor);
                void* recomputed_data = tensor_data_ptr(recomputed);
                if (tensor_data && recomputed_data) {
                    size_t data_size = tensor->numel * cml_dtype_size(tensor->dtype);
                    memcpy(tensor_data, recomputed_data, data_size);
                }
            }

            tensor->ir_node    = checkpoint->saved_ir_node;
            tensor->ir_context = checkpoint->saved_ir_context;
            tensor_free(recomputed);
        } else {
            tensor->ir_node    = checkpoint->saved_ir_node;
            tensor->ir_context = checkpoint->saved_ir_context;
        }
    } else {
        if (checkpoint->saved_ir_node) {
            tensor->ir_node    = checkpoint->saved_ir_node;
            tensor->ir_context = checkpoint->saved_ir_context;
        }
    }

    return tensor;
}

Tensor* checkpoint_forward(Module* module, Tensor* input) {
    if (!module || !input) return NULL;

    Tensor* output = module_forward(module, input);
    if (!output) return NULL;

    if (checkpointing_enabled) {
        autograd_checkpoint(output);
    }

    return output;
}

void sequential_apply_checkpointing(Sequential* seq, int every_n) {
    if (!seq || every_n < 0) return;

    if (every_n == 0) {
        autograd_set_checkpointing(false);
        LOG_DEBUG("Disabled checkpointing for Sequential model");
        return;
    }

    autograd_set_checkpointing(true);
    LOG_DEBUG("Applied checkpointing to Sequential model: every %d layers", every_n);
}

void autograd_checkpointing_cleanup(void) {
    if (checkpointed_tensors) {
        for (int i = 0; i < num_checkpointed; i++) {
            if (checkpointed_tensors[i]) {
                if (checkpointed_tensors[i]->saved_inputs) {
                    free(checkpointed_tensors[i]->saved_inputs);
                }
                free(checkpointed_tensors[i]);
            }
        }
        free(checkpointed_tensors);
        checkpointed_tensors  = NULL;
        num_checkpointed      = 0;
        checkpointed_capacity = 0;
    }
}
