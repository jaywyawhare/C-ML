#include "distributed/pipeline_parallel.h"
#include "distributed/distributed.h"
#include "autograd/autograd.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

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

    CMLPipelineParallel* pipeline = calloc(1, sizeof(CMLPipelineParallel));
    if (!pipeline)
        return NULL;

    pipeline->num_stages = num_stages;
    pipeline->stages = malloc(num_stages * sizeof(PipelineStage));
    if (!pipeline->stages) {
        free(pipeline);
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
    pipeline->micro_batch_outputs = calloc(num_stages, sizeof(Tensor**));
    if (!pipeline->micro_batch_outputs) {
        free(pipeline->stages);
        free(pipeline);
        return NULL;
    }
    for (int s = 0; s < num_stages; s++) {
        pipeline->micro_batch_outputs[s] = calloc(pipeline->num_micro_batches, sizeof(Tensor*));
        if (!pipeline->micro_batch_outputs[s]) {
            LOG_ERROR("Pipeline: failed to allocate micro-batch buffer for stage %d", s);
            for (int j = 0; j < s; j++)
                free(pipeline->micro_batch_outputs[j]);
            free(pipeline->micro_batch_outputs);
            free(pipeline->stages);
            free(pipeline);
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

    float* slice_data = malloc(slice_elems * sizeof(float));
    if (!slice_data) return NULL;

    memcpy(slice_data, src + (size_t)start * row_elems, slice_elems * sizeof(float));

    /* Build shape: same as input but dim 0 = slice_rows */
    int* shape = malloc(input->ndim * sizeof(int));
    if (!shape) { free(slice_data); return NULL; }
    memcpy(shape, input->shape, input->ndim * sizeof(int));
    shape[0] = slice_rows;

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(slice_data, shape, input->ndim, &cfg);
    free(slice_data);
    free(shape);
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
    float* out_data = malloc(total_elems * sizeof(float));
    if (!out_data) return NULL;

    size_t offset = 0;
    for (int i = 0; i < count; i++) {
        tensor_ensure_executed(tensors[i]);
        const float* src = (const float*)tensor_data_ptr(tensors[i]);
        if (!src) { free(out_data); return NULL; }
        size_t chunk = tensors[i]->numel * sizeof(float);
        memcpy(out_data + offset, src, chunk);
        offset += tensors[i]->numel;
    }

    int* shape = malloc(ndim * sizeof(int));
    if (!shape) { free(out_data); return NULL; }
    memcpy(shape, tensors[0]->shape, ndim * sizeof(int));
    shape[0] = (int)total_batch;

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(out_data, shape, ndim, &cfg);
    free(out_data);
    free(shape);
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
    Tensor** input_slices = malloc(num_mb * sizeof(Tensor*));
    if (!input_slices) return NULL;

    for (int mb = 0; mb < num_mb; mb++) {
        int start = mb * mb_size;
        int end = (mb == num_mb - 1) ? batch_size : start + mb_size;
        input_slices[mb] = slice_batch_dim(input, start, end);
        if (!input_slices[mb]) {
            LOG_ERROR("Pipeline forward: failed to slice input for micro-batch %d", mb);
            for (int j = 0; j < mb; j++)
                tensor_free(input_slices[j]);
            free(input_slices);
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
                free(input_slices);
                return NULL;
            }

            Tensor* mb_output = module_forward(mod, mb_input);
            if (!mb_output) {
                LOG_ERROR("Pipeline forward: module_forward failed at stage %d, micro-batch %d",
                          stage, mb);
                for (int j = 0; j < num_mb; j++)
                    tensor_free(input_slices[j]);
                free(input_slices);
                return NULL;
            }

            pipeline->micro_batch_outputs[stage][mb] = mb_output;
        }
    }

    /* Free input slices (not needed after stage 0) */
    for (int mb = 0; mb < num_mb; mb++)
        tensor_free(input_slices[mb]);
    free(input_slices);

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
    Tensor** grad_slices = malloc(num_mb * sizeof(Tensor*));
    if (!grad_slices) return -1;

    for (int mb = 0; mb < num_mb; mb++) {
        int start = mb * mb_size;
        int end = (mb == num_mb - 1) ? batch_size : start + mb_size;
        grad_slices[mb] = slice_batch_dim(grad_output, start, end);
        if (!grad_slices[mb]) {
            LOG_ERROR("Pipeline backward: failed to slice grad for micro-batch %d", mb);
            for (int j = 0; j < mb; j++)
                tensor_free(grad_slices[j]);
            free(grad_slices);
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

            if (stage == num_stages - 1) {
                /* Last stage: use the sliced gradient from the loss */
                mb_output->grad = grad_slices[mb];
            }
            /* For intermediate stages, the gradient was already set by
             * tensor_backward of the downstream stage via the autograd graph. */

            tensor_backward(mb_output, NULL, false, false);

            LOG_DEBUG("Pipeline backward: completed stage %d, micro-batch %d", stage, mb);
        }
    }

    free(grad_slices);

    LOG_DEBUG("Pipeline backward completed: %d stages, %d micro-batches",
              num_stages, num_mb);
    return 0;
}

void cml_pipeline_free(CMLPipelineParallel* pipeline) {
    if (!pipeline) return;

    if (pipeline->micro_batch_outputs) {
        for (int s = 0; s < pipeline->num_stages; s++) {
            if (pipeline->micro_batch_outputs[s]) {
                for (int mb = 0; mb < pipeline->num_micro_batches; mb++) {
                    if (pipeline->micro_batch_outputs[s][mb]) {
                        tensor_free(pipeline->micro_batch_outputs[s][mb]);
                    }
                }
                free(pipeline->micro_batch_outputs[s]);
            }
        }
        free(pipeline->micro_batch_outputs);
    }

    free(pipeline->stages);
    free(pipeline);
}
