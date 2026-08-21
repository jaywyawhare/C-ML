#include "distributed/pipeline_parallel.h"
#include "distributed/distributed.h"
#include "autograd/autograd.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

CMLPipelineParallel* cml_pipeline_create(PipelineStage* stages, int num_stages,
                                          const PipelineConfig* config) {
    if (!stages || num_stages <= 0) {
        LOG_ERROR("Invalid pipeline stages");
        return NULL;
    }

    for (int i = 0; i < num_stages; i++) {
        if (!stages[i].module) {
            LOG_ERROR("Pipeline stage %d has NULL module", i);
            return NULL;
        }
    }

    CMLPipelineParallel* pipeline = cml_calloc(1, sizeof(CMLPipelineParallel));
    if (!pipeline)
        return NULL;

    pipeline->num_stages = num_stages;
    pipeline->stages = cml_malloc(num_stages * sizeof(PipelineStage));
    if (!pipeline->stages) {
        cml_free(pipeline);
        return NULL;
    }
    memcpy(pipeline->stages, stages, num_stages * sizeof(PipelineStage));

    if (config) {
        pipeline->config = *config;
    } else {
        pipeline->config.num_micro_batches = 4;
        pipeline->config.num_stages = num_stages;
        pipeline->config.interleaved = false;
    }

    pipeline->num_micro_batches = pipeline->config.num_micro_batches;
    pipeline->group = cml_dist_get_default_group();

    /* Allocate micro-batch output buffers: [num_stages][num_micro_batches] */
    pipeline->micro_batch_outputs = cml_calloc(num_stages, sizeof(Tensor**));
    if (!pipeline->micro_batch_outputs) {
        cml_free(pipeline->stages);
        cml_free(pipeline);
        return NULL;
    }
    for (int s = 0; s < num_stages; s++) {
        pipeline->micro_batch_outputs[s] = cml_calloc(pipeline->num_micro_batches, sizeof(Tensor*));
        if (!pipeline->micro_batch_outputs[s]) {
            LOG_ERROR("Pipeline: failed to allocate micro-batch buffer for stage %d", s);
            for (int j = 0; j < s; j++)
                cml_free(pipeline->micro_batch_outputs[j]);
            cml_free(pipeline->micro_batch_outputs);
            cml_free(pipeline->stages);
            cml_free(pipeline);
            return NULL;
        }
    }

    LOG_INFO("Pipeline created: %d stages, %d micro-batches",
             num_stages, pipeline->num_micro_batches);
    return pipeline;
}

static Tensor* slice_batch_dim(Tensor* input, int start, int end) {
    if (!input || start < 0 || end <= start || end > input->shape[0])
        return NULL;

    tensor_ensure_executed(input);
    const float* src = (const float*)tensor_data_ptr(input);
    if (!src) return NULL;

    int slice_rows = end - start;
    size_t row_elems = input->numel / (size_t)input->shape[0];
    size_t slice_elems = (size_t)slice_rows * row_elems;
    if (row_elems > 0 && slice_elems / row_elems != (size_t)slice_rows) {
        return NULL; /* overflow */
    }

    float* slice_data = cml_malloc(slice_elems * sizeof(float));
    if (!slice_data) return NULL;

    memcpy(slice_data, src + (size_t)start * row_elems, slice_elems * sizeof(float));

    /* Build shape: same as input but dim 0 = slice_rows */
    int* shape = cml_malloc(input->ndim * sizeof(int));
    if (!shape) { cml_free(slice_data); return NULL; }
    memcpy(shape, input->shape, input->ndim * sizeof(int));
    shape[0] = slice_rows;

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(slice_data, shape, input->ndim, &cfg);
    cml_free(slice_data);
    cml_free(shape);
    return result;
}

static Tensor* concat_batch_dim(Tensor** tensors, int count) {
    if (!tensors || count <= 0 || !tensors[0])
        return NULL;

    /* Compute total batch size */
    size_t total_batch = 0;
    int ndim = tensors[0]->ndim;
    size_t row_elems = tensors[0]->numel / (size_t)tensors[0]->shape[0];

    for (int i = 0; i < count; i++) {
        if (!tensors[i]) return NULL;
        total_batch += (size_t)tensors[i]->shape[0];
    }

    size_t total_elems = total_batch * row_elems;
    if (row_elems > 0 && total_elems / row_elems != total_batch) {
        return NULL; /* overflow */
    }
    float* out_data = cml_malloc(total_elems * sizeof(float));
    if (!out_data) return NULL;

    size_t offset = 0;
    for (int i = 0; i < count; i++) {
        tensor_ensure_executed(tensors[i]);
        const float* src = (const float*)tensor_data_ptr(tensors[i]);
        if (!src) { cml_free(out_data); return NULL; }
        size_t chunk = tensors[i]->numel * sizeof(float);
        memcpy(out_data + offset, src, chunk);
        offset += tensors[i]->numel;
    }

    int* shape = cml_malloc(ndim * sizeof(int));
    if (!shape) { cml_free(out_data); return NULL; }
    memcpy(shape, tensors[0]->shape, ndim * sizeof(int));
    shape[0] = (int)total_batch;

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(out_data, shape, ndim, &cfg);
    cml_free(out_data);
    cml_free(shape);
    return result;
}

Tensor* cml_pipeline_forward(CMLPipelineParallel* pipeline, Tensor* input) {
    if (!pipeline || !input) return NULL;

    int num_mb = pipeline->num_micro_batches;
    int num_stages = pipeline->num_stages;
    int batch_size = input->shape[0];

    if (batch_size <= 0 || num_mb <= 0) {
        LOG_ERROR("Pipeline forward: invalid batch_size=%d or num_micro_batches=%d",
                  batch_size, num_mb);
        return NULL;
    }

    /* Clear any previously cached outputs */
    for (int s = 0; s < num_stages; s++) {
        for (int mb = 0; mb < num_mb; mb++) {
            pipeline->micro_batch_outputs[s][mb] = NULL;
        }
    }

    int mb_size = batch_size / num_mb;
    if (mb_size < 1) mb_size = 1;

    /* Split input into micro-batches along dim 0 */
    Tensor** input_slices = cml_malloc(num_mb * sizeof(Tensor*));
    if (!input_slices) return NULL;

    for (int mb = 0; mb < num_mb; mb++) {
        int start = mb * mb_size;
        int end = (mb == num_mb - 1) ? batch_size : start + mb_size;
        input_slices[mb] = slice_batch_dim(input, start, end);
        if (!input_slices[mb]) {
            LOG_ERROR("Pipeline forward: failed to slice input for micro-batch %d", mb);
            for (int j = 0; j < mb; j++)
                tensor_free(input_slices[j]);
            cml_free(input_slices);
            return NULL;
        }
    }

    /* GPipe schedule: process all micro-batches through stage 0, then stage 1, etc. */
    for (int stage = 0; stage < num_stages; stage++) {
        Module* mod = pipeline->stages[stage].module;

        for (int mb = 0; mb < num_mb; mb++) {
            Tensor* mb_input;

            if (stage == 0) {
                mb_input = input_slices[mb];
            } else {
                mb_input = pipeline->micro_batch_outputs[stage - 1][mb];
            }

            if (!mb_input) {
                LOG_ERROR("Pipeline forward: NULL input at stage %d, micro-batch %d", stage, mb);
                for (int j = 0; j < num_mb; j++)
                    tensor_free(input_slices[j]);
                cml_free(input_slices);
                return NULL;
            }

            Tensor* mb_output = module_forward(mod, mb_input);
            if (!mb_output) {
                LOG_ERROR("Pipeline forward: module_forward failed at stage %d, micro-batch %d",
                          stage, mb);
                for (int j = 0; j < num_mb; j++)
                    tensor_free(input_slices[j]);
                cml_free(input_slices);
                return NULL;
            }

            pipeline->micro_batch_outputs[stage][mb] = mb_output;
        }
    }

    /* Free input slices (not needed after stage 0) */
    for (int mb = 0; mb < num_mb; mb++)
        tensor_free(input_slices[mb]);
    cml_free(input_slices);

    /* Concatenate the final stage's micro-batch outputs along dim 0 */
    Tensor* final_output = concat_batch_dim(
        pipeline->micro_batch_outputs[num_stages - 1], num_mb);

    if (!final_output) {
        LOG_ERROR("Pipeline forward: failed to concatenate final outputs");
    }

    return final_output;
}

int cml_pipeline_backward(CMLPipelineParallel* pipeline, Tensor* grad_output) {
    if (!pipeline || !grad_output) return -1;

    int num_mb = pipeline->num_micro_batches;
    int num_stages = pipeline->num_stages;
    int batch_size = grad_output->shape[0];

    if (batch_size <= 0 || num_mb <= 0) {
        LOG_ERROR("Pipeline backward: invalid batch_size=%d or num_micro_batches=%d",
                  batch_size, num_mb);
        return -1;
    }

    int mb_size = batch_size / num_mb;
    if (mb_size < 1) mb_size = 1;

    /* Split grad_output into micro-batch gradients matching the forward split */
    Tensor** grad_slices = cml_malloc(num_mb * sizeof(Tensor*));
    if (!grad_slices) return -1;

    for (int mb = 0; mb < num_mb; mb++) {
        int start = mb * mb_size;
        int end = (mb == num_mb - 1) ? batch_size : start + mb_size;
        grad_slices[mb] = slice_batch_dim(grad_output, start, end);
        if (!grad_slices[mb]) {
            LOG_ERROR("Pipeline backward: failed to slice grad for micro-batch %d", mb);
            for (int j = 0; j < mb; j++)
                tensor_free(grad_slices[j]);
            cml_free(grad_slices);
            return -1;
        }
    }

    /*
     * GPipe backward: reverse stage order, process each micro-batch.
     * For each micro-batch at each stage, set the gradient on the cached
     * output tensor and call tensor_backward to propagate through autograd.
     */
    for (int stage = num_stages - 1; stage >= 0; stage--) {
        for (int mb = 0; mb < num_mb; mb++) {
            Tensor* mb_output = pipeline->micro_batch_outputs[stage][mb];
            if (!mb_output) {
                LOG_ERROR("Pipeline backward: NULL cached output at stage %d, micro-batch %d",
                          stage, mb);
                continue;
            }

            /* Last stage is seeded with the sliced loss gradient (passed as the
             * backward seed, which tensor_backward CLONES — so we retain
             * ownership of the slice and free it below; the previous code raw-
             * assigned it into mb_output->grad, leaving ownership unmanaged).
             * Intermediate stages get NULL: their gradient already arrived via
             * the autograd graph from the downstream stage. */
            Tensor* seed = (stage == num_stages - 1) ? grad_slices[mb] : NULL;
            tensor_backward(mb_output, seed, false, false);

            LOG_DEBUG("Pipeline backward: completed stage %d, micro-batch %d", stage, mb);
        }
    }

    /* We own the grad slices end-to-end (backward cloned them). */
    for (int mb = 0; mb < num_mb; mb++)
        tensor_free(grad_slices[mb]);
    cml_free(grad_slices);

    LOG_DEBUG("Pipeline backward completed: %d stages, %d micro-batches",
              num_stages, num_mb);
    return 0;
}

/* ── True cross-rank pipeline parallelism ──────────────────────────────────
 * world_size == num_stages; rank r runs ONLY stage r and streams micro-batch
 * activations to r+1 / receives from r-1. Because the stages are separate
 * processes, they execute concurrently — real pipeline overlap.
 *
 * A fixed-size activation-shape header [ndim, dim0..dim7] (as floats) precedes
 * each activation so the receiver can allocate before the data arrives. Meta and
 * data use tags 2*mb and 2*mb+1 so distinct micro-batches never collide. */
#define PIPE_META_LEN 9
#define PIPE_MAX_NDIM 8

static int pipe_send_tensor(Tensor* t, int dst, int mb) {
    tensor_ensure_executed(t);
    if (!t->data) return -1;

    float meta[PIPE_META_LEN];
    memset(meta, 0, sizeof(meta));
    meta[0] = (float)t->ndim;
    for (int i = 0; i < t->ndim && i < PIPE_MAX_NDIM; i++)
        meta[1 + i] = (float)t->shape[i];

    int mshape[1] = { PIPE_META_LEN };
    Tensor mt; memset(&mt, 0, sizeof(mt));
    mt.data = meta; mt.numel = PIPE_META_LEN; mt.ndim = 1; mt.shape = mshape;
    mt.dtype = DTYPE_FLOAT32; mt.device = DEVICE_CPU;

    if (cml_dist_send(&mt, dst, 2 * mb) != 0) return -1;
    return cml_dist_send(t, dst, 2 * mb + 1);
}

static Tensor* pipe_recv_tensor(int src, int mb) {
    float meta[PIPE_META_LEN];
    int mshape[1] = { PIPE_META_LEN };
    Tensor mt; memset(&mt, 0, sizeof(mt));
    mt.data = meta; mt.numel = PIPE_META_LEN; mt.ndim = 1; mt.shape = mshape;
    mt.dtype = DTYPE_FLOAT32; mt.device = DEVICE_CPU;
    if (cml_dist_recv(&mt, src, 2 * mb) != 0) return NULL;

    int ndim = (int)meta[0];
    if (ndim < 1 || ndim > PIPE_MAX_NDIM) return NULL;
    int shape[PIPE_MAX_NDIM];
    size_t numel = 0;
    for (int i = 0; i < ndim; i++) shape[i] = (int)meta[1 + i];
    if (!tensor_numel_checked(shape, ndim, &numel)) return NULL;

    float* data = (float*)cml_malloc(numel * sizeof(float));
    if (!data) return NULL;
    Tensor rt; memset(&rt, 0, sizeof(rt));
    rt.data = data; rt.numel = numel; rt.ndim = ndim; rt.shape = shape;
    rt.dtype = DTYPE_FLOAT32; rt.device = DEVICE_CPU;
    if (cml_dist_recv(&rt, src, 2 * mb + 1) != 0) { cml_free(data); return NULL; }

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* out = tensor_from_data(data, shape, ndim, &cfg);
    cml_free(data);
    return out;
}

static void dist_free_cache(CMLPipelineParallel* p) {
    if (p->dist_stage_inputs) {
        for (int mb = 0; mb < p->num_micro_batches; mb++)
            if (p->dist_stage_inputs[mb]) tensor_free(p->dist_stage_inputs[mb]);
        cml_free(p->dist_stage_inputs);
        p->dist_stage_inputs = NULL;
    }
    if (p->dist_stage_outputs) {
        for (int mb = 0; mb < p->num_micro_batches; mb++)
            if (p->dist_stage_outputs[mb]) tensor_free(p->dist_stage_outputs[mb]);
        cml_free(p->dist_stage_outputs);
        p->dist_stage_outputs = NULL;
    }
}

Tensor* cml_pipeline_dist_forward(CMLPipelineParallel* pipeline, Tensor* input) {
    if (!pipeline) return NULL;
    DistProcessGroup* g = pipeline->group;
    if (!g || g->world_size != pipeline->num_stages) {
        LOG_ERROR("dist pipeline: world_size (%d) must equal num_stages (%d)",
                  g ? g->world_size : -1, pipeline->num_stages);
        return NULL;
    }

    int rank = g->rank;
    int P = pipeline->num_stages;
    int M = pipeline->num_micro_batches;
    Module* stage = pipeline->stages[rank].module;
    if (!stage) { LOG_ERROR("dist pipeline: rank %d has no stage module", rank); return NULL; }

    dist_free_cache(pipeline);
    pipeline->dist_stage_inputs  = cml_calloc((size_t)M, sizeof(Tensor*));
    pipeline->dist_stage_outputs = cml_calloc((size_t)M, sizeof(Tensor*));
    if (!pipeline->dist_stage_inputs || !pipeline->dist_stage_outputs) {
        dist_free_cache(pipeline);
        return NULL;
    }

    int batch = (rank == 0 && input) ? input->shape[0] : 0;
    int mb_size = (batch > 0) ? (batch / M > 0 ? batch / M : 1) : 0;

    Tensor** last_outputs = (rank == P - 1) ? cml_calloc((size_t)M, sizeof(Tensor*)) : NULL;

    for (int mb = 0; mb < M; mb++) {
        Tensor* x;
        if (rank == 0) {
            int start = mb * mb_size;
            int end = (mb == M - 1) ? batch : start + mb_size;
            if (!input || start >= end) { LOG_ERROR("dist pipeline: bad input slice"); goto fail; }
            x = slice_batch_dim(input, start, end);
        } else {
            x = pipe_recv_tensor(rank - 1, mb);
        }
        if (!x) { LOG_ERROR("dist pipeline: rank %d could not obtain input mb %d", rank, mb); goto fail; }

        /* Grad on the stage input so backward can produce the upstream gradient. */
        tensor_set_requires_grad(x, true);
        Tensor* y = module_forward(stage, x);
        if (!y) { tensor_free(x); LOG_ERROR("dist pipeline: stage forward failed"); goto fail; }
        tensor_ensure_executed(y);

        pipeline->dist_stage_inputs[mb]  = x;
        pipeline->dist_stage_outputs[mb] = y;

        if (rank < P - 1) {
            if (pipe_send_tensor(y, rank + 1, mb) != 0) { LOG_ERROR("dist pipeline: send failed"); goto fail; }
        } else {
            last_outputs[mb] = tensor_clone(y);  /* clone so cache stays owned for backward */
        }
    }

    if (rank == P - 1) {
        Tensor* out = concat_batch_dim(last_outputs, M);
        for (int mb = 0; mb < M; mb++)
            if (last_outputs[mb]) tensor_free(last_outputs[mb]);
        cml_free(last_outputs);
        return out;
    }
    return NULL;

fail:
    if (last_outputs) {
        for (int mb = 0; mb < M; mb++)
            if (last_outputs[mb]) tensor_free(last_outputs[mb]);
        cml_free(last_outputs);
    }
    return NULL;
}

int cml_pipeline_dist_backward(CMLPipelineParallel* pipeline, Tensor* grad_output) {
    if (!pipeline) return -1;
    DistProcessGroup* g = pipeline->group;
    if (!g || g->world_size != pipeline->num_stages) return -1;

    int rank = g->rank;
    int P = pipeline->num_stages;
    int M = pipeline->num_micro_batches;
    if (!pipeline->dist_stage_outputs || !pipeline->dist_stage_inputs) {
        LOG_ERROR("dist pipeline backward: run cml_pipeline_dist_forward first");
        return -1;
    }

    int mb_size = 0;
    if (rank == P - 1) {
        if (!grad_output) { LOG_ERROR("dist pipeline backward: last rank needs grad_output"); return -1; }
        int gb = grad_output->shape[0];
        mb_size = gb / M > 0 ? gb / M : 1;
    }

    for (int mb = 0; mb < M; mb++) {
        Tensor* out = pipeline->dist_stage_outputs[mb];
        Tensor* in  = pipeline->dist_stage_inputs[mb];
        if (!out || !in) return -1;

        /* Gradient wrt this stage's output: sliced from the loss on the last
         * rank, received from downstream otherwise. */
        Tensor* gout;
        if (rank == P - 1) {
            int start = mb * mb_size;
            int end = (mb == M - 1) ? grad_output->shape[0] : start + mb_size;
            gout = slice_batch_dim(grad_output, start, end);
        } else {
            gout = pipe_recv_tensor(rank + 1, mb);
        }
        if (!gout) return -1;

        /* Backprop the stage: seeds out->grad with gout, accumulates this
         * stage's weight grads and computes in->grad (across micro-batches the
         * weight grads accumulate, which is the intended behaviour). */
        tensor_backward(out, gout, false, false);
        tensor_free(gout);

        /* Stream the input-gradient upstream (rank 0 has no upstream). */
        if (rank > 0) {
            if (!in->grad) { LOG_ERROR("dist pipeline backward: no input grad at rank %d", rank); return -1; }
            if (pipe_send_tensor(in->grad, rank - 1, mb) != 0) return -1;
        }
    }
    return 0;
}

void cml_pipeline_free(CMLPipelineParallel* pipeline) {
    if (!pipeline) return;

    dist_free_cache(pipeline);


    if (pipeline->micro_batch_outputs) {
        for (int s = 0; s < pipeline->num_stages; s++) {
            if (pipeline->micro_batch_outputs[s]) {
                for (int mb = 0; mb < pipeline->num_micro_batches; mb++) {
                    if (pipeline->micro_batch_outputs[s][mb]) {
                        tensor_free(pipeline->micro_batch_outputs[s][mb]);
                    }
                }
                cml_free(pipeline->micro_batch_outputs[s]);
            }
        }
        cml_free(pipeline->micro_batch_outputs);
    }

    cml_free(pipeline->stages);
    cml_free(pipeline);
}
