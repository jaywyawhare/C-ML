#include "optim.h"
#include "tensor/realize.h"
#include "ops/uops.h"
#include "core/cml_flags.h"
#include "core/logging.h"
#include "core/training_metrics.h"
#include "core/error_stack.h"
#include "tensor/tensor.h"
#include "autograd/autograd.h"
#include "backend/blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alloc/cml_allocator.h"

int optimizer_init(Optimizer* optimizer, const char* name, StepFn step, ZeroGradFn zero_grad) {
    if (!optimizer || !name || !step || !zero_grad)
        return -1;

    optimizer->name                   = name;
    optimizer->step                   = step;
    optimizer->zero_grad              = zero_grad;
    optimizer->param_groups           = NULL;
    optimizer->num_param_groups       = 0;
    optimizer->param_groups_capacity  = 0;
    optimizer->use_amp                = false;
    optimizer->grad_clip_norm         = 0.0f;
    optimizer->amsgrad                = false;
    optimizer->lr_scheduler_factor    = 1.0f;
    optimizer->lr_scheduler_step_size = 0;
    optimizer->lr_scheduler_gamma     = 1.0f;
    optimizer->training_metrics       = NULL;
    optimizer->version                = "1.0.0";
    optimizer->description            = "Optimizer";

    return 0;
}

Optimizer* optimizer_create(const char* name, StepFn step, ZeroGradFn zero_grad) {
    Optimizer* optimizer = cml_malloc(sizeof(Optimizer));
    if (!optimizer) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for Optimizer",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (optimizer_init(optimizer, name, step, zero_grad) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize Optimizer", __FILE__, __LINE__,
                         __func__);
        cml_free(optimizer);
        return NULL;
    }

    return optimizer;
}

typedef struct SGDMomentumState {
    Tensor* momentum_buffer;
} SGDMomentumState;

typedef struct AdamState {
    Tensor* exp_avg;
    Tensor* exp_avg_sq;
    Tensor* max_exp_avg_sq;
} AdamState;

typedef struct RMSpropState {
    Tensor* square_avg;
} RMSpropState;

typedef struct AdagradState {
    Tensor* sum_sq_grad;
} AdagradState;

typedef struct AdaDeltaState {
    Tensor* acc_grad;   // Accumulated squared gradients
    Tensor* acc_update; // Accumulated squared updates
} AdaDeltaState;

typedef struct LAMBState {
    Tensor* exp_avg;
    Tensor* exp_avg_sq;
} LAMBState;

typedef struct LARSState {
    Tensor* momentum_buffer;
} LARSState;

typedef struct MuonState {
    Tensor* momentum_buffer;
} MuonState;

typedef void (*StateInitFn)(void* state, Tensor* tensor, TensorConfig* config);

static void* optimizer_alloc_state(ParameterGroup* group, size_t state_size, StateInitFn init_fn) {
    void** states = cml_malloc((size_t)group->num_parameters * sizeof(void*));
    if (!states)
        return NULL;
    memset(states, 0, (size_t)group->num_parameters * sizeof(void*));
    for (int i = 0; i < group->num_parameters; i++) {
        Parameter* param = group->parameters[i];
        if (!param || !param->tensor)
            continue;
        void* state = cml_calloc(1, state_size);
        if (!state)
            continue;
        TensorConfig config = {.dtype      = param->tensor->dtype,
                               .device     = param->tensor->device,
                               .has_dtype  = true,
                               .has_device = true};
        init_fn(state, param->tensor, &config);
        states[i] = state;
    }
    return states;
}

/* Helper: create a realized zero tensor for optimizer state.
 * Optimizer state tensors are long-lived (must survive IR graph resets).
 * tensor_realize() both executes the lazy fill AND detaches from the IR graph. */
static Tensor* optim_zeros(int* shape, int ndim, TensorConfig* config) {
    Tensor* t = tensor_zeros(shape, ndim, config);
    tensor_realize(t);
    return t;
}

static void sgd_state_init(void* state, Tensor* tensor, TensorConfig* config) {
    SGDMomentumState* s = (SGDMomentumState*)state;
    s->momentum_buffer  = optim_zeros(tensor->shape, tensor->ndim, config);
}

static void adam_state_init(void* state, Tensor* tensor, TensorConfig* config) {
    AdamState* s      = (AdamState*)state;
    s->exp_avg        = optim_zeros(tensor->shape, tensor->ndim, config);
    s->exp_avg_sq     = optim_zeros(tensor->shape, tensor->ndim, config);
    s->max_exp_avg_sq = NULL;
}

static void rmsprop_state_init(void* state, Tensor* tensor, TensorConfig* config) {
    RMSpropState* s = (RMSpropState*)state;
    s->square_avg   = optim_zeros(tensor->shape, tensor->ndim, config);
}

static void adagrad_state_init(void* state, Tensor* tensor, TensorConfig* config) {
    AdagradState* s = (AdagradState*)state;
    s->sum_sq_grad  = optim_zeros(tensor->shape, tensor->ndim, config);
}

static void adadelta_state_init(void* state, Tensor* tensor, TensorConfig* config) {
    AdaDeltaState* s = (AdaDeltaState*)state;
    s->acc_grad      = optim_zeros(tensor->shape, tensor->ndim, config);
    s->acc_update    = optim_zeros(tensor->shape, tensor->ndim, config);
}

static void lamb_state_init(void* state, Tensor* tensor, TensorConfig* config) {
    LAMBState* s  = (LAMBState*)state;
    s->exp_avg    = optim_zeros(tensor->shape, tensor->ndim, config);
    s->exp_avg_sq = optim_zeros(tensor->shape, tensor->ndim, config);
}

static void lars_state_init(void* state, Tensor* tensor, TensorConfig* config) {
    LARSState* s       = (LARSState*)state;
    s->momentum_buffer = optim_zeros(tensor->shape, tensor->ndim, config);
}

static void muon_state_init(void* state, Tensor* tensor, TensorConfig* config) {
    MuonState* s       = (MuonState*)state;
    s->momentum_buffer = optim_zeros(tensor->shape, tensor->ndim, config);
}

/* Widest optimizer state block: Adam's exp_avg / exp_avg_sq / max_exp_avg_sq. */
#define OPTIM_STATE_MAX_TENSORS 3

/* Fill `ptrs` with the addresses of the Tensor* slots inside one parameter's
 * `state` block for optimizer `name` and return how many were written, or -1
 * when the name has no known state layout. */
static int optim_state_tensor_ptrs(const char* name, void* state, Tensor*** ptrs);

void optimizer_free(Optimizer* optimizer) {
    if (!optimizer)
        return;

    // Remove from tracking list to prevent double-free during cleanup
    extern void cml_untrack_optimizer(Optimizer*);
    cml_untrack_optimizer(optimizer);

    if (optimizer->param_groups) {
        for (int i = 0; i < optimizer->num_param_groups; i++) {
            ParameterGroup* group = &optimizer->param_groups[i];
            if (group->parameters) {
                cml_free(group->parameters);
            }
            if (group->state) {
                /* optim_state_tensor_ptrs knows every optimizer's state layout;
                 * an unrecognised name (-1) leaves the per-parameter blocks
                 * alone and only releases the array, as before. */
                void** states = (void**)group->state;
                for (int j = 0; j < group->num_parameters; j++) {
                    if (!states[j])
                        continue;
                    Tensor** owned[OPTIM_STATE_MAX_TENSORS];
                    int n = optim_state_tensor_ptrs(optimizer->name, states[j], owned);
                    for (int k = 0; k < n; k++)
                        if (*owned[k])
                            tensor_free(*owned[k]);
                    if (n >= 0)
                        cml_free(states[j]);
                }
                cml_free(states);
            }
        }
        cml_free(optimizer->param_groups);
    }

    cml_free(optimizer);
}

int optimizer_add_param_group(Optimizer* optimizer, Parameter** parameters, int num_parameters,
                              float lr, float weight_decay) {
    if (!optimizer || !parameters || num_parameters <= 0) {
        LOG_ERROR("Invalid arguments to optimizer_add_param_group");
        return -1;
    }

    if (optimizer->num_param_groups >= optimizer->param_groups_capacity) {
        int new_capacity =
            optimizer->param_groups_capacity == 0 ? 4 : optimizer->param_groups_capacity * 2;
        ParameterGroup* new_groups =
            cml_realloc(optimizer->param_groups, (size_t)new_capacity * sizeof(ParameterGroup));
        if (!new_groups) {
            LOG_ERROR("Failed to allocate memory for parameter groups");
            return -1;
        }
        optimizer->param_groups          = new_groups;
        optimizer->param_groups_capacity = new_capacity;
    }

    ParameterGroup* group = &optimizer->param_groups[optimizer->num_param_groups];

    group->parameters = cml_malloc((size_t)num_parameters * sizeof(Parameter*));
    if (!group->parameters) {
        LOG_ERROR("Failed to allocate memory for parameter group parameters");
        return -1;
    }

    for (int i = 0; i < num_parameters; i++) {
        group->parameters[i] = parameters[i];
    }
    group->num_parameters = num_parameters;

    group->lr           = lr;
    group->weight_decay = weight_decay;
    group->momentum     = 0.0f;   // Default, can be set later
    group->beta1        = 0.9f;   // Adam default
    group->beta2        = 0.999f; // Adam default
    group->epsilon      = 1e-8f;  // Adam default

    group->state      = NULL;
    group->step_count = 0;

    optimizer->num_param_groups++;

    LOG_DEBUG("Added parameter group %d with %d parameters (lr=%.6f, wd=%.6f)",
              optimizer->num_param_groups - 1, num_parameters, (double)lr, (double)weight_decay);

    return 0;
}

int optimizer_get_param_groups(Optimizer* optimizer, ParameterGroup** groups, int* num_groups) {
    if (!optimizer)
        return -1;

    if (num_groups) {
        *num_groups = optimizer->num_param_groups;
    }

    if (groups && optimizer->num_param_groups > 0) {
        *groups = optimizer->param_groups;
    }

    return 0;
}

ParameterGroup* optimizer_get_param_group(Optimizer* optimizer, int index) {
    if (!optimizer || index < 0 || index >= optimizer->num_param_groups) {
        return NULL;
    }

    return &optimizer->param_groups[index];
}

/* --- Optimizer state (moments) serialization for checkpoints -------------
 * The per-param state structs are private to this file, so the checkpoint
 * code in model_io.c delegates here. */

/* Addresses of the tensor slots inside one per-param state struct. */
static int optim_state_tensor_ptrs(const char* name, void* state, Tensor*** ptrs) {
    if (!state)
        return -1;
    if (strcmp(name, "SGD") == 0) {
        ptrs[0] = &((SGDMomentumState*)state)->momentum_buffer;
        return 1;
    }
    if (strcmp(name, "Adam") == 0 || strcmp(name, "AdamW") == 0 || strcmp(name, "Nadam") == 0 ||
        strcmp(name, "AdaMax") == 0) {
        AdamState* s = (AdamState*)state;
        ptrs[0]      = &s->exp_avg;
        ptrs[1]      = &s->exp_avg_sq;
        ptrs[2]      = &s->max_exp_avg_sq;
        return 3;
    }
    if (strcmp(name, "RMSprop") == 0) {
        ptrs[0] = &((RMSpropState*)state)->square_avg;
        return 1;
    }
    if (strcmp(name, "Adagrad") == 0) {
        ptrs[0] = &((AdagradState*)state)->sum_sq_grad;
        return 1;
    }
    if (strcmp(name, "AdaDelta") == 0) {
        AdaDeltaState* s = (AdaDeltaState*)state;
        ptrs[0]          = &s->acc_grad;
        ptrs[1]          = &s->acc_update;
        return 2;
    }
    if (strcmp(name, "LAMB") == 0) {
        LAMBState* s = (LAMBState*)state;
        ptrs[0]      = &s->exp_avg;
        ptrs[1]      = &s->exp_avg_sq;
        return 2;
    }
    if (strcmp(name, "LARS") == 0) {
        ptrs[0] = &((LARSState*)state)->momentum_buffer;
        return 1;
    }
    if (strcmp(name, "Muon") == 0) {
        ptrs[0] = &((MuonState*)state)->momentum_buffer;
        return 1;
    }
    return -1;
}

static size_t optim_state_struct_size(const char* name) {
    if (strcmp(name, "SGD") == 0)
        return sizeof(SGDMomentumState);
    if (strcmp(name, "Adam") == 0 || strcmp(name, "AdamW") == 0 || strcmp(name, "Nadam") == 0 ||
        strcmp(name, "AdaMax") == 0)
        return sizeof(AdamState);
    if (strcmp(name, "RMSprop") == 0)
        return sizeof(RMSpropState);
    if (strcmp(name, "Adagrad") == 0)
        return sizeof(AdagradState);
    if (strcmp(name, "AdaDelta") == 0)
        return sizeof(AdaDeltaState);
    if (strcmp(name, "LAMB") == 0)
        return sizeof(LAMBState);
    if (strcmp(name, "LARS") == 0)
        return sizeof(LARSState);
    if (strcmp(name, "Muon") == 0)
        return sizeof(MuonState);
    return 0;
}

int optimizer_state_save(Optimizer* optimizer, FILE* f) {
    if (!optimizer || !f)
        return -1;

    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];
        int32_t num_params    = group->num_parameters;
        fwrite(&num_params, sizeof(int32_t), 1, f);

        /* Slot count per param: 0 when no state exists yet (nothing stepped)
         * or the optimizer keeps no serializable moments. */
        Tensor** probe[OPTIM_STATE_MAX_TENSORS];
        void** states = (void**)group->state;
        int nslots    = 0;
        if (states) {
            for (int p = 0; p < num_params && nslots <= 0; p++)
                if (states[p])
                    nslots = optim_state_tensor_ptrs(optimizer->name, states[p], probe);
            if (nslots < 0)
                nslots = 0;
        }
        int32_t nslots32 = nslots;
        fwrite(&nslots32, sizeof(int32_t), 1, f);
        if (nslots == 0)
            continue;

        for (int p = 0; p < num_params; p++) {
            Tensor** ptrs[OPTIM_STATE_MAX_TENSORS] = {0};
            int n = states[p] ? optim_state_tensor_ptrs(optimizer->name, states[p], ptrs) : 0;
            for (int s = 0; s < nslots; s++) {
                Tensor* t      = (s < n && ptrs[s]) ? *ptrs[s] : NULL;
                uint64_t numel = 0;
                if (t) {
                    tensor_ensure_executed(t);
                    if (tensor_data_ptr(t))
                        numel = (uint64_t)t->numel;
                }
                fwrite(&numel, sizeof(uint64_t), 1, f);
                if (numel > 0)
                    fwrite(tensor_data_ptr(t), sizeof(float), (size_t)numel, f);
            }
        }
    }
    return 0;
}

int optimizer_state_load(Optimizer* optimizer, FILE* f) {
    if (!optimizer || !f)
        return -1;

    size_t ssize = optim_state_struct_size(optimizer->name);

    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];
        int32_t num_params, nslots;
        if (fread(&num_params, sizeof(int32_t), 1, f) != 1)
            return 0; /* old checkpoint: no state block */
        if (fread(&nslots, sizeof(int32_t), 1, f) != 1)
            return -1;
        if (num_params != group->num_parameters) {
            LOG_WARNING("optimizer_state_load: group %d param count mismatch (%d vs %d)", g,
                        num_params, group->num_parameters);
            return -1;
        }
        if (nslots == 0)
            continue;

        if (!group->state) {
            if (ssize == 0)
                return -1;
            void** states = cml_calloc((size_t)num_params, sizeof(void*));
            if (!states)
                return -1;
            group->state = states;
        }
        void** states = (void**)group->state;

        for (int p = 0; p < num_params; p++) {
            if (!states[p]) {
                states[p] = cml_calloc(1, ssize);
                if (!states[p])
                    return -1;
            }
            Tensor** ptrs[OPTIM_STATE_MAX_TENSORS] = {0};
            int n = optim_state_tensor_ptrs(optimizer->name, states[p], ptrs);
            for (int s = 0; s < nslots; s++) {
                uint64_t numel;
                if (fread(&numel, sizeof(uint64_t), 1, f) != 1)
                    return -1;
                if (numel == 0)
                    continue;
                Tensor* t = (s < n && ptrs[s]) ? *ptrs[s] : NULL;
                if (!t && s < n && ptrs[s]) {
                    Parameter* param = group->parameters[p];
                    TensorConfig cfg = {.dtype      = DTYPE_FLOAT32,
                                        .device     = DEVICE_CPU,
                                        .has_dtype  = true,
                                        .has_device = true};
                    if (param && param->tensor && param->tensor->numel == (size_t)numel) {
                        t = optim_zeros(param->tensor->shape, param->tensor->ndim, &cfg);
                    } else {
                        int shape1[1] = {(int)numel};
                        t             = optim_zeros(shape1, 1, &cfg);
                    }
                    if (!t)
                        return -1;
                    *ptrs[s] = t;
                }
                if (t && t->numel == (size_t)numel && tensor_data_ptr(t)) {
                    if (fread(tensor_data_ptr(t), sizeof(float), (size_t)numel, f) != (size_t)numel)
                        return -1;
                } else {
                    if (fseek(f, (long)(numel * sizeof(float)), SEEK_CUR) != 0)
                        return -1;
                }
            }
        }
    }
    return 0;
}

void optimizer_step(Optimizer* optimizer) {
    if (!optimizer || !optimizer->step)
        return;

    optimizer->step(optimizer);
    training_metrics_auto_capture_optimizer(optimizer);
}

void optimizer_set_metrics(Optimizer* optimizer, void* metrics) {
    if (!optimizer) {
        LOG_ERROR("Invalid optimizer");
        return;
    }
    optimizer->training_metrics = metrics;
}

void optimizer_zero_grad(Optimizer* optimizer) {
    if (!optimizer || !optimizer->zero_grad)
        return;

    optimizer->zero_grad(optimizer);

    training_metrics_mark_zero_grad();
}

int optimizer_get_step_count(Optimizer* optimizer) {
    if (!optimizer || optimizer->num_param_groups == 0)
        return 0;

    return optimizer->param_groups[0].step_count;
}

void optimizer_set_lr(Optimizer* optimizer, float lr) {
    if (!optimizer)
        return;

    for (int i = 0; i < optimizer->num_param_groups; i++) {
        optimizer->param_groups[i].lr = lr;
    }

    LOG_DEBUG("Set learning rate to %.6f for all %d parameter groups", (double)lr,
              optimizer->num_param_groups);
}

void optimizer_set_group_lr(Optimizer* optimizer, int group_index, float lr) {
    if (!optimizer || group_index < 0 || group_index >= optimizer->num_param_groups) {
        LOG_WARNING("Invalid group index %d for optimizer with %d groups", group_index,
                    optimizer ? optimizer->num_param_groups : 0);
        return;
    }

    optimizer->param_groups[group_index].lr = lr;

    LOG_DEBUG("Set learning rate to %.6f for parameter group %d", (double)lr, group_index);
}

float optimizer_get_group_lr(Optimizer* optimizer, int group_index) {
    if (!optimizer || group_index < 0 || group_index >= optimizer->num_param_groups) {
        LOG_WARNING("Invalid group index %d for optimizer with %d groups", group_index,
                    optimizer ? optimizer->num_param_groups : 0);
        return 0.0f;
    }

    return optimizer->param_groups[group_index].lr;
}

void optimizer_set_lr_scheduler(Optimizer* optimizer, int step_size, float gamma) {
    if (!optimizer)
        return;

    optimizer->lr_scheduler_step_size = step_size;
    optimizer->lr_scheduler_gamma     = gamma;
}

void optimizer_set_amp(Optimizer* optimizer, bool use_amp) {
    if (optimizer) {
        optimizer->use_amp = use_amp;
    }
}

void optimizer_set_grad_clip_norm(Optimizer* optimizer, float norm) {
    if (optimizer) {
        optimizer->grad_clip_norm = norm;
    }
}

void optimizer_set_amsgrad(Optimizer* optimizer, bool amsgrad) {
    if (optimizer) {
        optimizer->amsgrad = amsgrad;
    }
}

const char* optimizer_get_name(Optimizer* optimizer) { return optimizer ? optimizer->name : NULL; }

int optimizer_get_total_parameters(Optimizer* optimizer) {
    if (!optimizer)
        return 0;

    int total = 0;
    for (int i = 0; i < optimizer->num_param_groups; i++) {
        total += optimizer->param_groups[i].num_parameters;
    }

    return total;
}

void optimizer_print_summary(Optimizer* optimizer, int indent) {
    if (!optimizer)
        return;

    for (int i = 0; i < indent; i++)
        printf("  ");
    printf("Optimizer: %s (Parameter Groups: %d, Total Parameters: %d)\n", optimizer->name,
           optimizer->num_param_groups, optimizer_get_total_parameters(optimizer));

    for (int i = 0; i < optimizer->num_param_groups; i++) {
        ParameterGroup* group = &optimizer->param_groups[i];
        for (int j = 0; j < indent + 1; j++)
            printf("  ");
        printf("Group %d: %d parameters, lr=%.6f, wd=%.6f\n", i, group->num_parameters,
               (double)group->lr, (double)group->weight_decay);
    }
}

bool optimizer_supports_lr_scheduling(Optimizer* optimizer) {
    if (!optimizer)
        return false;
    return optimizer->lr_scheduler_step_size > 0;
}

bool optimizer_supports_grad_clipping(Optimizer* optimizer) {
    if (!optimizer)
        return false;
    return optimizer->grad_clip_norm > 0.0f;
}

/* Move `updated`'s buffer into the live parameter tensor and hand the old one
 * back, so freeing `updated` releases it. The parameter keeps its identity --
 * modules and graph nodes still point at it. */
static void adopt_param_data(Tensor* param, Tensor* updated) {
    tensor_ensure_executed(updated);

    void* old_data             = param->data;
    bool old_owns              = param->owns_data;
    bool old_cache             = param->from_buffer_cache;
    param->data                = updated->data;
    param->owns_data           = updated->owns_data;
    param->from_buffer_cache   = updated->from_buffer_cache;
    updated->data              = old_data;
    updated->owns_data         = old_owns;
    updated->from_buffer_cache = old_cache;
    tensor_free(updated);
}

/* Momentum buffers for `group`, allocated on first use; NULL when momentum is
 * off or the allocation failed (the caller then runs plain SGD). */
static SGDMomentumState** sgd_momentum_states(ParameterGroup* group, float momentum) {
    if (momentum <= 0.0f)
        return NULL;
    if (!group->state) {
        group->state = optimizer_alloc_state(group, sizeof(SGDMomentumState), sgd_state_init);
        if (!group->state)
            LOG_ERROR("Failed to allocate SGD momentum state");
    }
    return (SGDMomentumState**)group->state;
}

/* Per-parameter Adam moments for `group`, allocated on first use along with the
 * amsgrad maxima. `name` only labels the failure log. */
static AdamState** adam_states(Optimizer* optimizer, ParameterGroup* group, const char* name) {
    if (group->state)
        return (AdamState**)group->state;

    group->state = optimizer_alloc_state(group, sizeof(AdamState), adam_state_init);
    if (!group->state) {
        LOG_ERROR("Failed to allocate %s state", name);
        return NULL;
    }

    AdamState** states = (AdamState**)group->state;
    if (optimizer->amsgrad) {
        for (int i = 0; i < group->num_parameters; i++) {
            if (!states[i] || !group->parameters[i] || !group->parameters[i]->tensor)
                continue;
            Tensor* t        = group->parameters[i]->tensor;
            TensorConfig cfg = {
                .dtype = t->dtype, .device = t->device, .has_dtype = true, .has_device = true};
            states[i]->max_exp_avg_sq = optim_zeros(t->shape, t->ndim, &cfg);
        }
    }
    return states;
}

/* Resolve the float buffers a hand-written optimizer step needs for parameter
 * `i` of `group`, or NULL when the parameter has no gradient and should be
 * skipped. `grad_data` and `numel` are only written on success. */
static float* optim_param_buffers(ParameterGroup* group, int i, float** grad_data, size_t* numel) {
    Parameter* param = group->parameters[i];
    if (!param || !param->tensor || !param->requires_grad)
        return NULL;

    Tensor* grad = tensor_get_grad(param->tensor);
    if (!grad)
        return NULL;

    float* param_data = (float*)tensor_data_ptr(param->tensor);
    *grad_data        = (float*)tensor_data_ptr(grad);
    if (!param_data || !*grad_data)
        return NULL;

    *numel = param->tensor->numel;
    return param_data;
}

static void sgd_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    /* FUSE_OPTIM defers each update's realize and runs them in one pass. The
     * executor is is_executed-guarded, so the deferred adopt below never
     * re-executes a node -- momentum stays single-applied. */
    bool fuse = cml_flag_enabled(CML_FLAG_FUSE_OPTIM);

    int total = 0;
    for (int g = 0; g < optimizer->num_param_groups; g++)
        total += optimizer->param_groups[g].num_parameters;

    Tensor** pending_dst = NULL;
    Tensor** pending_upd = NULL;
    int pending_n        = 0;
    if (fuse && total > 0) {
        pending_dst = (Tensor**)cml_malloc(sizeof(Tensor*) * (size_t)total);
        pending_upd = (Tensor**)cml_malloc(sizeof(Tensor*) * (size_t)total);
        if (!pending_dst || !pending_upd) {
            cml_free(pending_dst);
            cml_free(pending_upd);
            pending_dst = pending_upd = NULL;
            fuse                      = false;
        }
    }

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float momentum        = group->momentum;

        SGDMomentumState** states = sgd_momentum_states(group, momentum);

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor || !param->requires_grad)
                continue;

            Tensor* tensor = param->tensor;
            Tensor* grad   = tensor_get_grad(tensor);
            if (!grad)
                continue;

            Tensor* mom_buf =
                (momentum > 0.0f && states && states[i]) ? states[i]->momentum_buffer : NULL;

            /* Record SGD update as an IR node so it can be scheduled/fused
             * with the backward pass in the same execution graph. */
            SgdStepParams sp = {
                .lr           = lr,
                .momentum     = momentum,
                .weight_decay = weight_decay,
                .dampening    = 0.0f,
                .nesterov     = false,
            };
            Tensor* new_p = uop_sgd_step(tensor, grad, mom_buf, &sp);
            if (!new_p)
                continue;

            if (fuse) {
                pending_dst[pending_n] = tensor;
                pending_upd[pending_n] = new_p;
                pending_n++;
            } else {
                adopt_param_data(tensor, new_p);
            }
        }

        group->step_count++;
    }

    if (fuse) {
        /* Realizing the last-emitted update walks the graph head-to-tail, so
         * this single pass computes every earlier update too. */
        if (pending_n > 0)
            tensor_ensure_executed(pending_upd[pending_n - 1]);
        for (int k = 0; k < pending_n; k++)
            adopt_param_data(pending_dst[k], pending_upd[k]);
        cml_free(pending_dst);
        cml_free(pending_upd);
    }
}

/* In-place SGD update (lr / momentum / weight_decay) applied directly to the
 * parameter buffers from param->grad->data — allocates NO IR nodes. This is what
 * lets a static-graph training step stay zero-rebuild (the normal sgd_step emits
 * a uop_sgd_step per parameter each iteration). Matches sgd_step's math
 * (dampening=0, nesterov=false). Intended for SGD optimizers; other optimizers
 * fall back to their node-emitting step (so static mode isn't fully rebuild-free
 * for them yet). Returns 1 if it handled the step in-place, 0 otherwise. */
int optimizer_step_inplace(Optimizer* optimizer) {
    if (!optimizer)
        return 0;
    if (!optimizer->name || strcmp(optimizer->name, "SGD") != 0) {
        optimizer_step(optimizer); /* non-SGD: correct, but allocates nodes */
        return 0;
    }
    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr = group->lr, weight_decay = group->weight_decay, momentum = group->momentum;

        SGDMomentumState** states = sgd_momentum_states(group, momentum);

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor || !param->requires_grad)
                continue;
            Tensor* t    = param->tensor;
            Tensor* grad = tensor_get_grad(t);
            if (!grad || !grad->data || !t->data || t->dtype != DTYPE_FLOAT32 ||
                grad->dtype != DTYPE_FLOAT32)
                continue;
            Tensor* mom =
                (momentum > 0.0f && states && states[i]) ? states[i]->momentum_buffer : NULL;
            float* p        = (float*)t->data;
            const float* gd = (const float*)grad->data;
            float* buf      = (mom && mom->data) ? (float*)mom->data : NULL;
            size_t n        = t->numel;
            for (size_t k = 0; k < n; k++) {
                float d = gd[k] + weight_decay * p[k];
                if (buf) {
                    buf[k] = momentum * buf[k] + d;
                    d      = buf[k];
                }
                p[k] -= lr * d;
            }
        }
        group->step_count++;
    }
    return 1;
}

static void generic_zero_grad(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor)
                continue;

            tensor_zero_grad(param->tensor);
        }
    }
}

static void adam_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    /* Gradient clipping: compute global L2 norm, then scale all grads down */
    if (optimizer->grad_clip_norm > 0.0f) {
        float global_norm = 0.0f;
        for (int g = 0; g < optimizer->num_param_groups; g++) {
            ParameterGroup* grp = &optimizer->param_groups[g];
            for (int i = 0; i < grp->num_parameters; i++) {
                Parameter* p = grp->parameters[i];
                if (!p || !p->tensor || !p->requires_grad)
                    continue;
                Tensor* grad = tensor_get_grad(p->tensor);
                if (!grad)
                    continue;
                float* gd = (float*)tensor_data_ptr(grad);
                if (!gd)
                    continue;
                for (size_t j = 0; j < p->tensor->numel; j++)
                    global_norm += gd[j] * gd[j];
            }
        }
        global_norm = sqrtf(global_norm);
        if (global_norm > optimizer->grad_clip_norm) {
            float scale = optimizer->grad_clip_norm / (global_norm + 1e-6f);
            for (int g = 0; g < optimizer->num_param_groups; g++) {
                ParameterGroup* grp = &optimizer->param_groups[g];
                for (int i = 0; i < grp->num_parameters; i++) {
                    Parameter* p = grp->parameters[i];
                    if (!p || !p->tensor || !p->requires_grad)
                        continue;
                    Tensor* grad = tensor_get_grad(p->tensor);
                    if (!grad)
                        continue;
                    float* gd = (float*)tensor_data_ptr(grad);
                    if (!gd)
                        continue;
                    for (size_t j = 0; j < p->tensor->numel; j++)
                        gd[j] *= scale;
                }
            }
        }
    }

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float beta1           = group->beta1;
        float beta2           = group->beta2;
        float epsilon         = group->epsilon;

        AdamState** states = adam_states(optimizer, group, "Adam");
        if (!states)
            continue;
        int step = group->step_count + 1;

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor || !param->requires_grad)
                continue;

            Tensor* tensor = param->tensor;
            Tensor* grad   = tensor_get_grad(tensor);
            if (!grad || !states[i])
                continue;

            Tensor* max_sq = optimizer->amsgrad ? states[i]->max_exp_avg_sq : NULL;

            /* Record Adam update as an IR node so it is scheduled alongside
             * the backward pass in the same execution graph. */
            AdamStepParams ap = {
                .lr           = lr,
                .beta1        = beta1,
                .beta2        = beta2,
                .eps          = epsilon,
                .weight_decay = weight_decay,
                .step         = step,
                .amsgrad      = optimizer->amsgrad,
            };
            Tensor* new_p =
                uop_adam_step(tensor, grad, states[i]->exp_avg, states[i]->exp_avg_sq, max_sq, &ap);
            if (!new_p)
                continue;

            adopt_param_data(tensor, new_p);
        }

        group->step_count++;
    }
}

static void rmsprop_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float alpha           = group->beta1; // Reuse beta1 for alpha (decay rate)
        float epsilon         = group->epsilon;

        if (!group->state) {
            group->state = optimizer_alloc_state(group, sizeof(RMSpropState), rmsprop_state_init);
            if (!group->state) {
                LOG_ERROR("Failed to allocate RMSprop state");
                continue;
            }
        }

        RMSpropState** states = (RMSpropState**)group->state;

        for (int i = 0; i < group->num_parameters; i++) {
            float* grad_data;
            size_t numel;
            float* param_data = optim_param_buffers(group, i, &grad_data, &numel);
            if (!param_data || !states[i])
                continue;

            float* square_avg_data = (float*)tensor_data_ptr(states[i]->square_avg);

            if (!square_avg_data)
                continue;

            for (size_t j = 0; j < numel; j++) {
                float grad_val = grad_data[j];

                if (weight_decay > 0.0f) {
                    grad_val += weight_decay * param_data[j];
                }

                square_avg_data[j] =
                    alpha * square_avg_data[j] + (1.0f - alpha) * grad_val * grad_val;

                param_data[j] -= lr * grad_val / (sqrtf(square_avg_data[j]) + epsilon);
            }
        }

        group->step_count++;
    }
}

static void adagrad_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float epsilon         = group->epsilon;

        if (!group->state) {
            group->state = optimizer_alloc_state(group, sizeof(AdagradState), adagrad_state_init);
            if (!group->state) {
                LOG_ERROR("Failed to allocate Adagrad state");
                continue;
            }
        }

        AdagradState** states = (AdagradState**)group->state;

        for (int i = 0; i < group->num_parameters; i++) {
            float* grad_data;
            size_t numel;
            float* param_data = optim_param_buffers(group, i, &grad_data, &numel);
            if (!param_data || !states[i])
                continue;

            float* sum_sq_grad_data = (float*)tensor_data_ptr(states[i]->sum_sq_grad);

            if (!sum_sq_grad_data)
                continue;

            for (size_t j = 0; j < numel; j++) {
                float grad_val = grad_data[j];

                if (weight_decay > 0.0f) {
                    grad_val += weight_decay * param_data[j];
                }

                sum_sq_grad_data[j] += grad_val * grad_val;
                param_data[j] -= lr * grad_val / (sqrtf(sum_sq_grad_data[j]) + epsilon);
            }
        }

        group->step_count++;
    }
}

static void adamw_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float beta1           = group->beta1;
        float beta2           = group->beta2;
        float epsilon         = group->epsilon;

        AdamState** states = adam_states(optimizer, group, "AdamW");
        if (!states)
            continue;
        int step = group->step_count + 1;

        float bias_correction1 = 1.0f - powf(beta1, (float)step);
        float bias_correction2 = 1.0f - powf(beta2, (float)step);

        for (int i = 0; i < group->num_parameters; i++) {
            float* grad_data;
            size_t numel;
            float* param_data = optim_param_buffers(group, i, &grad_data, &numel);
            if (!param_data || !states[i])
                continue;

            float* exp_avg_data    = (float*)tensor_data_ptr(states[i]->exp_avg);
            float* exp_avg_sq_data = (float*)tensor_data_ptr(states[i]->exp_avg_sq);

            if (!exp_avg_data || !exp_avg_sq_data)
                continue;

            for (size_t j = 0; j < numel; j++) {
                float grad_val = grad_data[j];

                if (weight_decay > 0.0f) {
                    param_data[j] -= lr * weight_decay * param_data[j];
                }

                exp_avg_data[j] = beta1 * exp_avg_data[j] + (1.0f - beta1) * grad_val;
                exp_avg_sq_data[j] =
                    beta2 * exp_avg_sq_data[j] + (1.0f - beta2) * grad_val * grad_val;

                float m = exp_avg_data[j] / bias_correction1;
                float v = exp_avg_sq_data[j] / bias_correction2;

                if (optimizer->amsgrad && states[i]->max_exp_avg_sq) {
                    float* max_exp_avg_sq_data = (float*)tensor_data_ptr(states[i]->max_exp_avg_sq);
                    if (max_exp_avg_sq_data) {
                        max_exp_avg_sq_data[j] = fmaxf(max_exp_avg_sq_data[j], v);
                        v                      = max_exp_avg_sq_data[j];
                    }
                }

                param_data[j] -= lr * m / (sqrtf(v) + epsilon);
            }
        }

        group->step_count++;
    }
}

static void adadelta_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float rho             = group->beta1; // Reuse beta1 for rho (decay rate)
        float weight_decay    = group->weight_decay;
        float epsilon         = group->epsilon;

        if (!group->state) {
            group->state = optimizer_alloc_state(group, sizeof(AdaDeltaState), adadelta_state_init);
            if (!group->state) {
                LOG_ERROR("Failed to allocate AdaDelta state");
                continue;
            }
        }

        AdaDeltaState** states = (AdaDeltaState**)group->state;

        for (int i = 0; i < group->num_parameters; i++) {
            float* grad_data;
            size_t numel;
            float* param_data = optim_param_buffers(group, i, &grad_data, &numel);
            if (!param_data || !states[i])
                continue;

            float* acc_grad_data   = (float*)tensor_data_ptr(states[i]->acc_grad);
            float* acc_update_data = (float*)tensor_data_ptr(states[i]->acc_update);

            if (!acc_grad_data || !acc_update_data)
                continue;

            for (size_t j = 0; j < numel; j++) {
                float grad_val = grad_data[j];

                if (weight_decay > 0.0f) {
                    grad_val += weight_decay * param_data[j];
                }

                acc_grad_data[j] = rho * acc_grad_data[j] + (1.0f - rho) * grad_val * grad_val;

                float update = sqrtf(acc_update_data[j] + epsilon) /
                               sqrtf(acc_grad_data[j] + epsilon) * grad_val;

                acc_update_data[j] = rho * acc_update_data[j] + (1.0f - rho) * update * update;

                param_data[j] -= update;
            }
        }

        group->step_count++;
    }
}

/* Create optimizer `name` with a single parameter group at `lr`. Returns the
 * group so the caller can stamp its hyper-parameters; on failure `*out` is
 * already freed and set to NULL. */
static ParameterGroup* optim_new(Optimizer** out, const char* name, StepFn step,
                                 Parameter** parameters, int num_parameters, float lr,
                                 float weight_decay) {
    *out = optimizer_create(name, step, generic_zero_grad);
    if (!*out)
        return NULL;

    if (optimizer_add_param_group(*out, parameters, num_parameters, lr, weight_decay) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to add parameter group to optimizer",
                         __FILE__, __LINE__, __func__);
        optimizer_free(*out);
        *out = NULL;
        return NULL;
    }

    return (*out)->num_param_groups > 0 ? &(*out)->param_groups[0] : NULL;
}

Optimizer* optim_sgd(Parameter** parameters, int num_parameters, float lr, float momentum,
                     float weight_decay) {
    Optimizer* optimizer;
    ParameterGroup* group =
        optim_new(&optimizer, "SGD", sgd_step, parameters, num_parameters, lr, weight_decay);
    if (group) {
        group->momentum = momentum;
    }

    return optimizer;
}

Optimizer* optim_adam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                      float beta1, float beta2, float epsilon) {
    Optimizer* optimizer;
    ParameterGroup* group =
        optim_new(&optimizer, "Adam", adam_step, parameters, num_parameters, lr, weight_decay);
    if (group) {
        group->beta1   = beta1 > 0.0f ? beta1 : 0.9f;
        group->beta2   = beta2 > 0.0f ? beta2 : 0.999f;
        group->epsilon = epsilon > 0.0f ? epsilon : 1e-8f;
    }

    return optimizer;
}

Optimizer* optim_rmsprop(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                         float alpha, float epsilon) {
    Optimizer* optimizer;
    ParameterGroup* group = optim_new(&optimizer, "RMSprop", rmsprop_step, parameters,
                                      num_parameters, lr, weight_decay);
    if (group) {
        group->beta1   = alpha > 0.0f ? alpha : 0.99f; // Reuse beta1 for alpha
        group->epsilon = epsilon > 0.0f ? epsilon : 1e-8f;
    }

    return optimizer;
}

Optimizer* optim_adagrad(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                         float epsilon) {
    Optimizer* optimizer;
    ParameterGroup* group = optim_new(&optimizer, "Adagrad", adagrad_step, parameters,
                                      num_parameters, lr, weight_decay);
    if (group) {
        group->epsilon = epsilon > 0.0f ? epsilon : 1e-8f;
    }

    return optimizer;
}

Optimizer* optim_adamw(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                       float beta1, float beta2, float epsilon) {
    Optimizer* optimizer;
    ParameterGroup* group =
        optim_new(&optimizer, "AdamW", adamw_step, parameters, num_parameters, lr, weight_decay);
    if (group) {
        group->beta1   = beta1 > 0.0f ? beta1 : 0.9f;
        group->beta2   = beta2 > 0.0f ? beta2 : 0.999f;
        group->epsilon = epsilon > 0.0f ? epsilon : 1e-8f;
    }

    return optimizer;
}

Optimizer* optim_adadelta(Parameter** parameters, int num_parameters, float rho, float weight_decay,
                          float epsilon) {
    Optimizer* optimizer;
    ParameterGroup* group = optim_new(&optimizer, "AdaDelta", adadelta_step, parameters,
                                      num_parameters, 1.0f, weight_decay);
    if (group) {
        group->beta1   = rho > 0.0f ? rho : 0.9f; // Reuse beta1 for rho
        group->epsilon = epsilon > 0.0f ? epsilon : 1e-6f;
    }

    return optimizer;
}

Optimizer* optim_adam_for_model(Module* model, float lr, float weight_decay, float beta1,
                                float beta2, float eps) {
    if (!model) {
        return NULL;
    }

    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(model, &params, &num_params, true) != 0) {
        return NULL;
    }

    Optimizer* optimizer = optim_adam(params, num_params, lr, weight_decay, beta1, beta2, eps);
    cml_free(params);

    if (optimizer) {
        extern void cml_track_optimizer(Optimizer*);
        cml_track_optimizer(optimizer);
    }

    return optimizer;
}

static void lamb_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float beta1           = group->beta1;
        float beta2           = group->beta2;
        float epsilon         = group->epsilon;

        if (!group->state) {
            group->state = optimizer_alloc_state(group, sizeof(LAMBState), lamb_state_init);
            if (!group->state) {
                LOG_ERROR("Failed to allocate LAMB state");
                continue;
            }
        }

        LAMBState** states = (LAMBState**)group->state;
        int step           = group->step_count + 1;

        float bias_correction1 = 1.0f - powf(beta1, (float)step);
        float bias_correction2 = 1.0f - powf(beta2, (float)step);

        for (int i = 0; i < group->num_parameters; i++) {
            float* grad_data;
            size_t numel;
            float* param_data = optim_param_buffers(group, i, &grad_data, &numel);
            if (!param_data || !states[i])
                continue;

            float* exp_avg_data    = (float*)tensor_data_ptr(states[i]->exp_avg);
            float* exp_avg_sq_data = (float*)tensor_data_ptr(states[i]->exp_avg_sq);

            if (!exp_avg_data || !exp_avg_sq_data)
                continue;

            float param_norm  = 0.0f;
            float update_norm = 0.0f;

            for (size_t j = 0; j < numel; j++) {
                float grad_val = grad_data[j];

                exp_avg_data[j] = beta1 * exp_avg_data[j] + (1.0f - beta1) * grad_val;
                exp_avg_sq_data[j] =
                    beta2 * exp_avg_sq_data[j] + (1.0f - beta2) * grad_val * grad_val;

                param_norm += param_data[j] * param_data[j];
            }
            param_norm = sqrtf(param_norm);

            for (size_t j = 0; j < numel; j++) {
                float m          = exp_avg_data[j] / bias_correction1;
                float v          = exp_avg_sq_data[j] / bias_correction2;
                float update_val = m / (sqrtf(v) + epsilon);

                if (weight_decay > 0.0f) {
                    update_val += weight_decay * param_data[j];
                }

                update_norm += update_val * update_val;
                grad_data[j] = update_val;
            }
            update_norm = sqrtf(update_norm);

            // Compute trust ratio (LAMB layer-wise scaling)
            float trust_ratio = 1.0f;
            if (param_norm > 0.0f && update_norm > 0.0f) {
                trust_ratio = param_norm / update_norm;
            }

            for (size_t j = 0; j < numel; j++) {
                param_data[j] -= lr * trust_ratio * grad_data[j];
            }
        }

        group->step_count++;
    }
}

static void lars_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float momentum        = group->momentum;
        float trust_coeff     = group->epsilon; // Reuse epsilon for trust coefficient

        if (momentum > 0.0f && !group->state) {
            group->state = optimizer_alloc_state(group, sizeof(LARSState), lars_state_init);
            if (!group->state) {
                LOG_ERROR("Failed to allocate LARS state");
                continue;
            }
        }

        LARSState** states = momentum > 0.0f ? (LARSState**)group->state : NULL;

        for (int i = 0; i < group->num_parameters; i++) {
            float* grad_data;
            size_t numel;
            float* param_data = optim_param_buffers(group, i, &grad_data, &numel);
            if (!param_data || !states[i])
                continue;

            // Compute parameter and gradient norms
            float param_norm = 0.0f;
            float grad_norm  = 0.0f;
            for (size_t j = 0; j < numel; j++) {
                param_norm += param_data[j] * param_data[j];
                float g = grad_data[j];
                if (weight_decay > 0.0f) {
                    g += weight_decay * param_data[j];
                }
                grad_norm += g * g;
            }
            param_norm = sqrtf(param_norm);
            grad_norm  = sqrtf(grad_norm);

            // Compute local learning rate (LARS scaling)
            float local_lr = lr;
            if (param_norm > 0.0f && grad_norm > 0.0f) {
                local_lr = lr * trust_coeff * param_norm / grad_norm;
            }

            if (momentum > 0.0f && states && states[i] && states[i]->momentum_buffer) {
                float* momentum_data = (float*)tensor_data_ptr(states[i]->momentum_buffer);
                if (momentum_data) {
                    for (size_t j = 0; j < numel; j++) {
                        float g = grad_data[j];
                        if (weight_decay > 0.0f) {
                            g += weight_decay * param_data[j];
                        }
                        momentum_data[j] = momentum * momentum_data[j] + local_lr * g;
                        param_data[j] -= momentum_data[j];
                    }
                }
            } else {
                for (size_t j = 0; j < numel; j++) {
                    float g = grad_data[j];
                    if (weight_decay > 0.0f) {
                        g += weight_decay * param_data[j];
                    }
                    param_data[j] -= local_lr * g;
                }
            }
        }

        group->step_count++;
    }
}

Optimizer* optim_lamb(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                      float beta1, float beta2, float epsilon) {
    Optimizer* optimizer;
    ParameterGroup* group =
        optim_new(&optimizer, "LAMB", lamb_step, parameters, num_parameters, lr, weight_decay);
    if (group) {
        group->beta1   = beta1 > 0.0f ? beta1 : 0.9f;
        group->beta2   = beta2 > 0.0f ? beta2 : 0.999f;
        group->epsilon = epsilon > 0.0f ? epsilon : 1e-6f;
    }

    return optimizer;
}

Optimizer* optim_lars(Parameter** parameters, int num_parameters, float lr, float momentum,
                      float weight_decay, float trust_coefficient) {
    Optimizer* optimizer;
    ParameterGroup* group =
        optim_new(&optimizer, "LARS", lars_step, parameters, num_parameters, lr, weight_decay);
    if (group) {
        group->momentum = momentum > 0.0f ? momentum : 0.9f;
        group->epsilon  = trust_coefficient > 0.0f ? trust_coefficient : 0.02f;
    }

    return optimizer;
}

// Newton-Schulz orthogonalization iteration for a square-ish matrix
// Approximates the polar decomposition: U = X * (X^T * X)^{-1/2}
static void newton_schulz_inplace(float* data, size_t numel) {
    // For 1D parameter vectors, just normalize
    float norm = 0.0f;
    for (size_t i = 0; i < numel; i++)
        norm += data[i] * data[i];
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
        for (size_t i = 0; i < numel; i++)
            data[i] /= norm;
    }
}

static void muon_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float momentum_val    = group->momentum;
        bool nesterov         = optimizer->amsgrad; // Reuse amsgrad flag for nesterov

        if (!group->state) {
            group->state = optimizer_alloc_state(group, sizeof(MuonState), muon_state_init);
            if (!group->state)
                continue;
        }

        MuonState** states = (MuonState**)group->state;

        for (int i = 0; i < group->num_parameters; i++) {
            float* grad_data;
            size_t numel;
            float* param_data = optim_param_buffers(group, i, &grad_data, &numel);
            if (!param_data || !states[i])
                continue;

            float* momentum_data = (float*)tensor_data_ptr(states[i]->momentum_buffer);

            if (!momentum_data)
                continue;

            if (weight_decay > 0.0f) {
                for (size_t j = 0; j < numel; j++)
                    grad_data[j] += weight_decay * param_data[j];
            }

            for (size_t j = 0; j < numel; j++)
                momentum_data[j] = momentum_val * momentum_data[j] + grad_data[j];

            float* update = cml_malloc(numel * sizeof(float));
            if (!update)
                continue;

            if (nesterov) {
                for (size_t j = 0; j < numel; j++)
                    update[j] = grad_data[j] + momentum_val * momentum_data[j];
            } else {
                memcpy(update, momentum_data, numel * sizeof(float));
            }

            newton_schulz_inplace(update, numel);

            for (size_t j = 0; j < numel; j++)
                param_data[j] -= lr * update[j];

            cml_free(update);
        }

        group->step_count++;
    }
}

Optimizer* optim_muon(Parameter** parameters, int num_parameters, float lr, float momentum,
                      float weight_decay, bool nesterov) {
    Optimizer* optimizer;
    ParameterGroup* group =
        optim_new(&optimizer, "Muon", muon_step, parameters, num_parameters, lr, weight_decay);
    if (group) {
        group->momentum = momentum > 0.0f ? momentum : 0.95f;
    }

    // Reuse amsgrad flag for nesterov
    if (optimizer)
        optimizer->amsgrad = nesterov;

    return optimizer;
}

static void nadam_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float beta1           = group->beta1;
        float beta2           = group->beta2;
        float epsilon         = group->epsilon;

        AdamState** states = adam_states(optimizer, group, "Nadam");
        if (!states)
            continue;
        int step = group->step_count + 1;

        float bias_correction1 = 1.0f - powf(beta1, (float)step);
        float bias_correction2 = 1.0f - powf(beta2, (float)step);

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor || !param->requires_grad)
                continue;
            Tensor* tensor = param->tensor;
            Tensor* grad   = tensor_get_grad(tensor);
            if (!grad || !states[i])
                continue;

            float* param_data = (float*)tensor_data_ptr(tensor);
            float* grad_data  = (float*)tensor_data_ptr(grad);
            float* m_data     = (float*)tensor_data_ptr(states[i]->exp_avg);
            float* v_data     = (float*)tensor_data_ptr(states[i]->exp_avg_sq);
            if (!param_data || !grad_data || !m_data || !v_data)
                continue;

            for (size_t j = 0; j < tensor->numel; j++) {
                float g = grad_data[j];
                if (weight_decay > 0.0f)
                    g += weight_decay * param_data[j];

                m_data[j] = beta1 * m_data[j] + (1.0f - beta1) * g;
                v_data[j] = beta2 * v_data[j] + (1.0f - beta2) * g * g;

                float m_hat = m_data[j] / bias_correction1;
                float v_hat = v_data[j] / bias_correction2;

                // Nesterov-corrected first moment
                float m_nesterov = (beta1 * m_hat + (1.0f - beta1) * g / bias_correction1);
                param_data[j] -= lr * m_nesterov / (sqrtf(v_hat) + epsilon);
            }
        }
        group->step_count++;
    }
}

Optimizer* optim_nadam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                       float beta1, float beta2, float epsilon) {
    Optimizer* optimizer;
    ParameterGroup* group =
        optim_new(&optimizer, "Nadam", nadam_step, parameters, num_parameters, lr, weight_decay);
    if (group) {
        group->beta1   = beta1 > 0.0f ? beta1 : 0.9f;
        group->beta2   = beta2 > 0.0f ? beta2 : 0.999f;
        group->epsilon = epsilon > 0.0f ? epsilon : 1e-8f;
    }

    return optimizer;
}

static void adamax_step(Optimizer* optimizer) {
    if (!optimizer)
        return;

    for (int g_idx = 0; g_idx < optimizer->num_param_groups; g_idx++) {
        ParameterGroup* group = &optimizer->param_groups[g_idx];
        float lr              = group->lr;
        float weight_decay    = group->weight_decay;
        float beta1           = group->beta1;
        float beta2           = group->beta2;
        float epsilon         = group->epsilon;

        AdamState** states = adam_states(optimizer, group, "AdaMax");
        if (!states)
            continue;
        int step = group->step_count + 1;

        float bias_correction1 = 1.0f - powf(beta1, (float)step);

        for (int i = 0; i < group->num_parameters; i++) {
            Parameter* param = group->parameters[i];
            if (!param || !param->tensor || !param->requires_grad)
                continue;
            Tensor* tensor = param->tensor;
            Tensor* grad   = tensor_get_grad(tensor);
            if (!grad || !states[i])
                continue;

            float* param_data = (float*)tensor_data_ptr(tensor);
            float* grad_data  = (float*)tensor_data_ptr(grad);
            float* m_data     = (float*)tensor_data_ptr(states[i]->exp_avg);
            float* u_data     = (float*)tensor_data_ptr(states[i]->exp_avg_sq); // infinity norm

            if (!param_data || !grad_data || !m_data || !u_data)
                continue;

            for (size_t j = 0; j < tensor->numel; j++) {
                float g = grad_data[j];
                if (weight_decay > 0.0f)
                    g += weight_decay * param_data[j];

                m_data[j] = beta1 * m_data[j] + (1.0f - beta1) * g;
                u_data[j] = fmaxf(beta2 * u_data[j], fabsf(g));
                param_data[j] -= (lr / bias_correction1) * m_data[j] / (u_data[j] + epsilon);
            }
        }
        group->step_count++;
    }
}

Optimizer* optim_adamax(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                        float beta1, float beta2, float epsilon) {
    Optimizer* optimizer;
    ParameterGroup* group =
        optim_new(&optimizer, "AdaMax", adamax_step, parameters, num_parameters, lr, weight_decay);
    if (group) {
        group->beta1   = beta1 > 0.0f ? beta1 : 0.9f;
        group->beta2   = beta2 > 0.0f ? beta2 : 0.999f;
        group->epsilon = epsilon > 0.0f ? epsilon : 1e-8f;
    }

    return optimizer;
}

Optimizer* optim_sgd_for_model(Module* model, float lr, float momentum, float weight_decay) {
    if (!model) {
        return NULL;
    }

    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(model, &params, &num_params, true) != 0) {
        return NULL;
    }

    Optimizer* optimizer = optim_sgd(params, num_params, lr, momentum, weight_decay);
    cml_free(params);

    return optimizer;
}
