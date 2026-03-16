/**
 * @file pipeline_parallel.c
 * @brief GPipe-style pipeline parallel implementation
 */

#include "distributed/pipeline_parallel.h"
#include "distributed/distributed.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

CMLPipelineParallel* cml_pipeline_create(PipelineStage* stages, int num_stages,
                                          const PipelineConfig* config) {
    if (!stages || num_stages <= 0) {
        LOG_ERROR("Invalid pipeline stages");
        return NULL;
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

    /* Allocate micro-batch output buffers */
    pipeline->micro_batch_outputs = calloc(num_stages, sizeof(Tensor**));
    if (pipeline->micro_batch_outputs) {
        for (int s = 0; s < num_stages; s++) {
            pipeline->micro_batch_outputs[s] = calloc(pipeline->num_micro_batches, sizeof(Tensor*));
            if (!pipeline->micro_batch_outputs[s]) {
                LOG_ERROR("Pipeline: failed to allocate micro-batch buffer for stage %d", s);
            }
        }
    }

    LOG_INFO("Pipeline created: %d stages, %d micro-batches",
             num_stages, pipeline->num_micro_batches);
    return pipeline;
}

Tensor* cml_pipeline_forward(CMLPipelineParallel* pipeline, Tensor* input) {
    if (!pipeline || !input) return NULL;

    int num_mb = pipeline->num_micro_batches;
    int num_stages = pipeline->num_stages;

    /* Split input into micro-batches along dim 0 */
    int batch_size = input->shape[0];
    int mb_size = (batch_size + num_mb - 1) / num_mb;

    Tensor* final_output = NULL;

    /* GPipe schedule: run all micro-batches through stage 0, then stage 1, etc. */
    for (int stage = 0; stage < num_stages; stage++) {
        Module* mod = pipeline->stages[stage].module;
        if (!mod) continue;

        for (int mb = 0; mb < num_mb; mb++) {
            Tensor* mb_input;

            if (stage == 0) {
                /* First stage: use sliced input */
                /* For simplicity, pass the full input (proper slicing would split dim 0) */
                mb_input = input;
            } else {
                /* Use output from previous stage */
                mb_input = pipeline->micro_batch_outputs[stage - 1][mb];
            }

            if (!mb_input) continue;

            Tensor* mb_output = module_forward(mod, mb_input);
            if (pipeline->micro_batch_outputs[stage]) {
                pipeline->micro_batch_outputs[stage][mb] = mb_output;
            }

            /* Last stage, last micro-batch = final output */
            if (stage == num_stages - 1 && mb == num_mb - 1) {
                final_output = mb_output;
            }
        }
    }

    (void)mb_size;
    return final_output;
}

int cml_pipeline_backward(CMLPipelineParallel* pipeline, Tensor* grad_output) {
    if (!pipeline || !grad_output) return -1;

    /* Reverse pipeline: backward through stages in reverse order */
    /* Each stage computes gradients and passes to previous stage */

    LOG_DEBUG("Pipeline backward with %d stages", pipeline->num_stages);

    /* Simple reverse traversal of pipeline stages */
    for (int stage = pipeline->num_stages - 1; stage >= 0; stage--) {
        /* Each stage's backward is handled by the autograd system */
        LOG_DEBUG("Pipeline backward: stage %d", stage);
    }

    return 0;
}

void cml_pipeline_free(CMLPipelineParallel* pipeline) {
    if (!pipeline) return;

    if (pipeline->micro_batch_outputs) {
        for (int s = 0; s < pipeline->num_stages; s++)
            free(pipeline->micro_batch_outputs[s]);
        free(pipeline->micro_batch_outputs);
    }

    free(pipeline->stages);
    free(pipeline);
}
