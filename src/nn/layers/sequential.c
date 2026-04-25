#include "nn/layers/sequential.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "autograd/autograd.h"
#include "ops/ir/context.h"
#include "ops/ir/graph_cache.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include "core/error_stack.h"
#include "nn/layers/linear.h"
#include "nn/layers/activations.h"
#include "backend/blas.h"
#include "ops/simd_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

static size_t alloc_size_aligned(size_t size, size_t alignment);

/* ---------------------------------------------------------------------------
 * Zero-IR fast path for Sequential inference.
 * Built once from module weights; replays pure BLAS+activation calls with no
 * IR construction, no malloc, and no graph teardown per forward pass.
 * --------------------------------------------------------------------------- */

typedef enum {
    FAST_LINEAR,
    FAST_RELU,
    FAST_SIGMOID,
    FAST_TANH,
} FastOpKind;

typedef struct {
    FastOpKind kind;

    /* LINEAR only */
    float*  weight_data;        /* stable ptr into model's weight tensor (out×in) */
    float*  weight_transposed;  /* pre-transposed copy (in×out) for no-trans BLAS call */
    float*  bias_data;          /* stable ptr into model's bias tensor, or NULL */
    int     in_features;
    int     out_features;

    /* Source of the dynamic input to this op:
     *   src_buf == NULL  →  use the function's input tensor
     *   src_buf != NULL  →  read from this buffer (previous op's output) */
    float*  src_buf;
    size_t  src_numel;

    float*  out_buf;       /* pre-allocated aligned output buffer */
    size_t  out_numel;
} FastOp;

typedef struct SequentialFastPath {
    FastOp*  ops;
    int      num_ops;
    int*     input_shape;
    int      input_ndim;
    size_t   input_numel;
    float*   result_buf;   /* final output (== ops[num_ops-1].out_buf) */
    size_t   result_numel;
    Tensor*  output_tensor; /* reused every call — avoids malloc per forward */
    bool     valid;
} SequentialFastPath;

static void fast_path_free(SequentialFastPath* fp) {
    if (!fp) return;
    if (fp->output_tensor) {
        fp->output_tensor->owns_data = false; /* result_buf freed with ops below */
        tensor_free(fp->output_tensor);
        fp->output_tensor = NULL;
    }
    for (int i = 0; i < fp->num_ops; i++) {
        if (fp->ops[i].weight_transposed)
            free(fp->ops[i].weight_transposed);
        if (fp->ops[i].out_buf &&
            (i == 0 || fp->ops[i].out_buf != fp->ops[i - 1].out_buf)) {
            free(fp->ops[i].out_buf);
        }
    }
    free(fp->ops);
    free(fp->input_shape);
    free(fp);
}

/* Build a SequentialFastPath from the module list and a sample input tensor.
 * Returns NULL if any module type is unsupported (falls back to IR path). */
static SequentialFastPath* fast_path_build(Sequential* seq, Tensor* input) {
    if (!seq || !input || seq->num_modules == 0) return NULL;
    /* Only build when NOT in training mode (weights must be frozen). */
    if (((Module*)seq)->training) return NULL;

    SequentialFastPath* fp = calloc(1, sizeof(SequentialFastPath));
    if (!fp) return NULL;

    fp->ops = calloc((size_t)seq->num_modules, sizeof(FastOp));
    if (!fp->ops) { free(fp); return NULL; }

    fp->input_shape = malloc(sizeof(int) * input->ndim);
    if (!fp->input_shape) { free(fp->ops); free(fp); return NULL; }
    memcpy(fp->input_shape, input->shape, sizeof(int) * input->ndim);
    fp->input_ndim  = input->ndim;
    fp->input_numel = input->numel;

    float* prev_out = NULL; /* NULL = use model input */
    size_t prev_out_numel = input->numel;

    for (int i = 0; i < seq->num_modules; i++) {
        Module* m = seq->modules[i];
        FastOp*  op = &fp->ops[fp->num_ops];

        if (strcmp(m->name, "Linear") == 0) {
            Linear* lin = (Linear*)m;
            if (!lin->weight || !lin->weight->tensor ||
                !lin->weight->tensor->data) {
                fast_path_free(fp); return NULL;
            }
            op->kind         = FAST_LINEAR;
            op->weight_data  = (float*)lin->weight->tensor->data;
            op->bias_data    = (lin->use_bias && lin->bias && lin->bias->tensor)
                               ? (float*)lin->bias->tensor->data : NULL;
            op->in_features  = lin->in_features;
            op->out_features = lin->out_features;
            op->src_buf      = prev_out;
            op->src_numel    = prev_out_numel;
            op->out_numel    = (prev_out_numel / (size_t)lin->in_features)
                               * (size_t)lin->out_features;

            /* Pre-transpose weight from (out×in) → (in×out) so the BLAS call
             * is NoTrans/NoTrans — enables our AVX2 microkernel for small sizes. */
            op->weight_transposed = malloc(
                (size_t)lin->in_features * lin->out_features * sizeof(float));
            if (!op->weight_transposed) { fast_path_free(fp); return NULL; }
            {
                const float* W = op->weight_data;
                float* WT = op->weight_transposed;
                int in = lin->in_features, out = lin->out_features;
                for (int o = 0; o < out; o++)
                    for (int i = 0; i < in; i++)
                        WT[i * out + o] = W[o * in + i];
            }

            size_t nbytes = alloc_size_aligned(op->out_numel * sizeof(float), 64);
            op->out_buf   = aligned_alloc(64, nbytes);
            if (!op->out_buf) { fast_path_free(fp); return NULL; }
            prev_out       = op->out_buf;
            prev_out_numel = op->out_numel;

        } else if (strcmp(m->name, "ReLU") == 0) {
            op->kind      = FAST_RELU;
            op->src_buf   = prev_out;
            op->src_numel = prev_out_numel;
            op->out_buf   = prev_out; /* in-place: reuse previous buffer */
            op->out_numel = prev_out_numel;

        } else if (strcmp(m->name, "Sigmoid") == 0) {
            op->kind      = FAST_SIGMOID;
            op->src_buf   = prev_out;
            op->src_numel = prev_out_numel;
            op->out_buf   = prev_out;
            op->out_numel = prev_out_numel;

        } else if (strcmp(m->name, "Tanh") == 0) {
            op->kind      = FAST_TANH;
            op->src_buf   = prev_out;
            op->src_numel = prev_out_numel;
            op->out_buf   = prev_out;
            op->out_numel = prev_out_numel;

        } else {
            fast_path_free(fp); return NULL;
        }
        fp->num_ops++;
    }

    fp->result_buf   = fp->ops[fp->num_ops - 1].out_buf;
    fp->result_numel = fp->ops[fp->num_ops - 1].out_numel;

    /* Pre-build the output tensor once; reuse it every forward pass */
    int out_shape[8];
    int out_ndim = input->ndim;
    size_t leading = 1;
    for (int d = 0; d < input->ndim - 1; d++) {
        out_shape[d] = input->shape[d];
        leading *= (size_t)input->shape[d];
    }
    out_shape[input->ndim - 1] = (int)(fp->result_numel / leading);
    fp->output_tensor = tensor_empty(out_shape, out_ndim, NULL);
    if (!fp->output_tensor) { fast_path_free(fp); return NULL; }
    if (fp->output_tensor->owns_data && fp->output_tensor->data)
        free(fp->output_tensor->data);
    fp->output_tensor->data      = fp->result_buf;
    fp->output_tensor->owns_data = false;
    fp->output_tensor->is_executed = true;

    fp->valid = true;
    return fp;
}

static Tensor* fast_path_run(SequentialFastPath* fp, Tensor* input) {
    if (!fp || !fp->valid || !input) return NULL;

    CMLBlasContext* blas = cml_blas_get_context();

    for (int i = 0; i < fp->num_ops; i++) {
        FastOp* op = &fp->ops[i];

        float* in_ptr = (op->src_buf == NULL) ? (float*)input->data : op->src_buf;

        switch (op->kind) {
        case FAST_LINEAR: {
            size_t batch = op->src_numel / (size_t)op->in_features;
            if (blas && blas->initialized) {
                /* Use pre-transposed weight (in×out) + no-transpose BLAS call.
                 * This hits the AVX2 microkernel for small matrices. */
                cml_blas_sgemm(blas, in_ptr, op->weight_transposed, op->out_buf,
                               (int)batch, op->out_features, op->in_features,
                               1.0f, 0.0f);
            } else {
                memset(op->out_buf, 0, op->out_numel * sizeof(float));
                for (size_t m = 0; m < batch; m++)
                    for (int n = 0; n < op->out_features; n++) {
                        float s = 0.f;
                        for (int k = 0; k < op->in_features; k++)
                            s += in_ptr[m * op->in_features + k]
                                 * op->weight_data[n * op->in_features + k];
                        op->out_buf[m * op->out_features + n] = s;
                    }
            }
            if (op->bias_data) {
                size_t batch2 = op->out_numel / (size_t)op->out_features;
                int N = op->out_features;
#if defined(__AVX2__)
                for (size_t m = 0; m < batch2; m++) {
                    float* row = op->out_buf + m * N;
                    int n = 0;
                    for (; n + 8 <= N; n += 8)
                        _mm256_storeu_ps(row + n,
                            _mm256_add_ps(_mm256_loadu_ps(row + n),
                                          _mm256_loadu_ps(op->bias_data + n)));
                    for (; n < N; n++) row[n] += op->bias_data[n];
                }
#else
                for (size_t m = 0; m < batch2; m++) {
                    float* row = op->out_buf + m * N;
                    for (int n = 0; n < N; n++) row[n] += op->bias_data[n];
                }
#endif
            }
            break;
        }
        case FAST_RELU: {
#if defined(__AVX2__)
            __m256 zero = _mm256_setzero_ps();
            size_t j = 0;
            for (; j + 8 <= op->out_numel; j += 8)
                _mm256_storeu_ps(op->out_buf + j,
                    _mm256_max_ps(_mm256_loadu_ps(in_ptr + j), zero));
            for (; j < op->out_numel; j++) {
                float v = in_ptr[j]; op->out_buf[j] = v > 0.0f ? v : 0.0f;
            }
#else
            for (size_t j = 0; j < op->out_numel; j++) {
                float v = in_ptr[j]; op->out_buf[j] = v > 0.0f ? v : 0.0f;
            }
#endif
            break;
        }
        case FAST_SIGMOID:
            simd_sigmoid_f32(in_ptr, op->out_buf, op->out_numel);
            break;
        case FAST_TANH:
            simd_tanh_f32(in_ptr, op->out_buf, op->out_numel);
            break;
        }
    }

    /* Return the pre-allocated non-owning output tensor — no malloc per call */
    return fp->output_tensor;
}

static size_t alloc_size_aligned(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

static void free_cached_graph(CachedModelGraph* cache) {
    if (!cache)
        return;

    if (cache->plan) {
        cml_free_execution_plan(cache->plan);
    }
    if (cache->input_shape) {
        free(cache->input_shape);
    }
    if (cache->input_buffer) {
        free(cache->input_buffer);
    }
    if (cache->output_buffer) {
        free(cache->output_buffer);
    }
    free(cache);
}

static bool shapes_match(CachedModelGraph* cache, Tensor* input) {
    if (!cache || !cache->valid || !input)
        return false;
    if (cache->input_ndim != input->ndim)
        return false;

    for (int i = 0; i < input->ndim; i++) {
        if (cache->input_shape[i] != input->shape[i])
            return false;
    }
    return true;
}

static CachedModelGraph* create_cached_graph(Tensor* input, Tensor* output, CMLGraph_t ir) {
    if (!input || !output || !ir)
        return NULL;

    CachedModelGraph* cache = calloc(1, sizeof(CachedModelGraph));
    if (!cache)
        return NULL;

    cache->input_ndim  = input->ndim;
    cache->input_shape = malloc(sizeof(int) * input->ndim);
    if (!cache->input_shape) {
        free(cache);
        return NULL;
    }
    memcpy(cache->input_shape, input->shape, sizeof(int) * input->ndim);
    cache->input_numel = input->numel;

    cache->input_buffer =
        aligned_alloc(32, alloc_size_aligned((size_t)input->numel * sizeof(float), 32));
    if (!cache->input_buffer) {
        free(cache->input_shape);
        free(cache);
        return NULL;
    }

    cache->output_numel  = output->numel;
    cache->output_buffer =
        aligned_alloc(32, alloc_size_aligned((size_t)output->numel * sizeof(float), 32));
    if (!cache->output_buffer) {
        free(cache->input_buffer);
        free(cache->input_shape);
        free(cache);
        return NULL;
    }

    cache->plan = cml_create_execution_plan(ir);
    if (!cache->plan) {
        free(cache->output_buffer);
        free(cache->input_buffer);
        free(cache->input_shape);
        free(cache);
        return NULL;
    }

    cache->valid = true;

    return cache;
}

static Tensor* execute_cached_forward(Sequential* seq, Tensor* input) {
    if (!seq || !seq->cached_graph || !seq->cached_graph->plan) {
        return NULL;
    }

    CachedModelGraph* cache = seq->cached_graph;
    CMLExecutionPlan* plan  = cache->plan;

    if (!plan->valid || plan->num_nodes == 0) {
        return NULL;
    }

    memcpy(cache->input_buffer, input->data, cache->input_numel * sizeof(float));

    for (size_t i = 0; i < plan->num_nodes; i++) {
        struct IRNode* node = plan->nodes[i];
        float* out_buf      = plan->buffers[i];

        if (!node || !out_buf) {
            LOG_ERROR("Invalid node or buffer at index %zu", i);
            return NULL;
        }

        // For each input of this node, we need to determine where to read from:
        // - If it's the original input tensor, read from cache->input_buffer
        // - If it's an intermediate result, read from the appropriate plan->buffers[j]
        for (int inp_idx = 0; inp_idx < node->num_inputs; inp_idx++) {
            if (!node->inputs || !node->inputs[inp_idx])
                continue;

            Tensor* inp_tensor = node->inputs[inp_idx];

            // Check if this input is the original input tensor (compare numel and shape)
            if (inp_tensor->numel == cache->input_numel) {
                // This might be the input tensor - use our cached input buffer
                // Note: we can't compare pointers because the original tensor is gone,
                // but we can check if this tensor's data was never set to a plan buffer
                bool is_plan_buffer = false;
                for (size_t j = 0; j < i; j++) {
                    if (inp_tensor->data == plan->buffers[j]) {
                        is_plan_buffer = true;
                        break;
                    }
                }
                if (!is_plan_buffer) {
                    inp_tensor->data = cache->input_buffer;
                }
            }
            // If data pointer matches a plan buffer, it's already pointing to an intermediate
            // result
        }

        if (cml_execute_node_fast(node, out_buf) != 0)
            return NULL;

        if (node->output) {
            node->output->data = out_buf;
        }
    }

    struct IRNode* last_node = plan->nodes[plan->num_nodes - 1];
    if (last_node && last_node->output) {
        memcpy(cache->output_buffer, plan->buffers[plan->num_nodes - 1],
               cache->output_numel * sizeof(float));
    }
    // The tensor doesn't own the data - we manage it
    Tensor* output = tensor_empty(last_node->output->shape, last_node->output->ndim, NULL);
    if (output) {
        output->data      = cache->output_buffer;
        output->owns_data = false; // We manage the buffer
    }

    return output;
}

static int g_sequential_depth = 0;

static bool fast_path_shapes_match(SequentialFastPath* fp, Tensor* input) {
    if (!fp || !fp->valid || !input) return false;
    if (fp->input_ndim != input->ndim) return false;
    for (int i = 0; i < input->ndim; i++)
        if (fp->input_shape[i] != input->shape[i]) return false;
    return true;
}

static Tensor* sequential_forward(Module* module, Tensor* input) {
    Sequential* seq = (Sequential*)module;

    if (!seq || !input)
        return NULL;

    /* Zero-IR fast path: only in eval mode, only when fully supported */
    if (!((Module*)seq)->training) {
        if (seq->fast_path && fast_path_shapes_match(seq->fast_path, input)) {
            return fast_path_run(seq->fast_path, input);
        }
    }

    if (seq->enable_graph_cache && seq->cached_graph && shapes_match(seq->cached_graph, input)) {
        Tensor* cached_output = execute_cached_forward(seq, input);
        if (cached_output) {
            return cached_output; // Cache hit!
        }
        LOG_WARNING("Cached execution failed, rebuilding graph");
        sequential_invalidate_cache(seq);
    }
    g_sequential_depth++;

    Tensor* output = input;
    for (int i = 0; i < seq->num_modules; i++) {
        if (!seq->modules[i])
            continue;

        Tensor* next_output = module_forward(seq->modules[i], output);

        if (!next_output) {
            LOG_ERROR("Forward pass failed at module %d in Sequential", i);
            if (output != input) {
                tensor_free(output);
            }
            return NULL;
        }

        output = next_output;
    }

    g_sequential_depth--;

    /* Ensure the output data is materialized before returning. */
    if (output && output != input) {
        tensor_ensure_executed(output);
    }

    if (seq->enable_graph_cache && !seq->cached_graph) {
        CMLGraph_t ir = cml_ir_get_or_create_context();
        if (ir) {
            seq->cached_graph = create_cached_graph(input, output, ir);
            if (seq->cached_graph) {
                LOG_DEBUG("Graph cached for Sequential model");
            }
        }
    }

    /* Build the zero-IR fast path after the first successful eval-mode forward */
    if (!((Module*)seq)->training && !seq->fast_path) {
        seq->fast_path = fast_path_build(seq, input);
        if (seq->fast_path)
            fprintf(stderr, "[CML] Zero-IR fast path built (%d ops)\n", seq->fast_path->num_ops);
    }

    return output;
}

static void sequential_free(Module* module) {
    Sequential* seq = (Sequential*)module;
    if (!seq)
        return;
    if (seq->fast_path) {
        fast_path_free(seq->fast_path);
        seq->fast_path = NULL;
    }
    if (seq->cached_graph) {
        free_cached_graph(seq->cached_graph);
        seq->cached_graph = NULL;
    }
    if (seq->modules) {
        for (int i = 0; i < seq->num_modules; i++) {
            if (seq->modules[i]) {
                module_free(seq->modules[i]);
            }
        }
        free(seq->modules);
    }

    free(seq);
}

Sequential* nn_sequential(void) {
    Sequential* seq = malloc(sizeof(Sequential));
    if (!seq) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR,
                         "Failed to allocate memory for Sequential module", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    if (module_init((Module*)seq, "Sequential", sequential_forward, sequential_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize Sequential module", __FILE__,
                         __LINE__, __func__);
        free(seq);
        return NULL;
    }

    seq->modules            = NULL;
    seq->num_modules        = 0;
    seq->capacity           = 0;
    seq->cached_graph       = NULL;
    seq->enable_graph_cache = false;
    seq->fast_path          = NULL;
    extern void cml_track_module(Module*);
    cml_track_module((Module*)seq);
    extern void training_metrics_register_model(Module*);
    training_metrics_register_model((Module*)seq);

    return seq;
}

int sequential_add(Sequential* seq, Module* module) {
    if (!seq || !module)
        return -1;
    if (seq->num_modules >= seq->capacity) {
        int new_capacity     = seq->capacity == 0 ? 8 : seq->capacity * 2;
        Module** new_modules = realloc(seq->modules, (size_t)new_capacity * sizeof(Module*));
        if (!new_modules) {
            LOG_ERROR("Failed to allocate memory for Sequential modules");
            return -1;
        }
        seq->modules  = new_modules;
        seq->capacity = new_capacity;
    }

    seq->modules[seq->num_modules] = module;
    int module_index               = seq->num_modules; // Store index before incrementing
    seq->num_modules++;

    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(module, &params, &num_params, true) == 0) {
        for (int i = 0; i < num_params; i++) {
            if (params[i]) {
                char param_name[256];
                snprintf(param_name, sizeof(param_name), "%d.%s.%s", module_index, module->name,
                         params[i]->name ? params[i]->name : "unnamed");
                Tensor* pt = params[i]->tensor;
                nn_tensor_param_alias(pt);
                int result = module_add_parameter((Module*)seq, pt, param_name,
                                                  params[i]->requires_grad);
                if (result != 0) {
                    pt->ref_count--;
                    LOG_WARNING(
                        "Failed to add parameter '%s' from module '%s' (index %d) to Sequential",
                        param_name, module->name, module_index);
                }
            }
        }
        if (params)
            free(params);
    }

    LOG_DEBUG(
        "Added module '%s' (index %d) to Sequential (total modules: %d, total parameters: %d)",
        module->name, module_index, seq->num_modules, ((Module*)seq)->num_parameters);

    return 0;
}

Module* sequential_get(Sequential* seq, int index) {
    if (!seq || index < 0 || index >= seq->num_modules) {
        return NULL;
    }

    return seq->modules[index];
}

int sequential_get_length(Sequential* seq) { return seq ? seq->num_modules : 0; }

Sequential* sequential_add_chain(Sequential* seq, Module* module) {
    if (!seq || !module) {
        return seq;
    }
    sequential_add(seq, module);
    return seq;
}

Sequential* nn_sequentialv(int num_layers, ...) {
    Sequential* seq = nn_sequential();
    if (!seq) {
        return NULL;
    }

    va_list args;
    va_start(args, num_layers);

    for (int i = 0; i < num_layers; i++) {
        Module* module = va_arg(args, Module*);
        if (module) {
            sequential_add(seq, module);
        }
    }

    va_end(args);
    return seq;
}

void sequential_enable_graph_cache(Sequential* seq, bool enable) {
    if (!seq)
        return;
    seq->enable_graph_cache = enable;
    if (!enable && seq->cached_graph) {
        sequential_invalidate_cache(seq);
    }
    LOG_DEBUG("Graph caching %s for Sequential model", enable ? "enabled" : "disabled");
}

void sequential_invalidate_cache(Sequential* seq) {
    if (!seq || !seq->cached_graph)
        return;
    LOG_DEBUG("Invalidating cached graph for Sequential model");
    free_cached_graph(seq->cached_graph);
    seq->cached_graph = NULL;
}
