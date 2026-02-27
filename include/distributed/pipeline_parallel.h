/**
 * @file pipeline_parallel.h
 * @brief GPipe-style pipeline parallelism
 *
 * Splits a model into stages across different devices, processes
 * micro-batches through the pipeline, and reverses for backward.
 */

#ifndef CML_PIPELINE_PARALLEL_H
#define CML_PIPELINE_PARALLEL_H

#include "distributed/distributed.h"
#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Pipeline stage
 */
typedef struct PipelineStage {
    Module* module;         /* Module for this stage (not owned) */
    int device_id;          /* Device for this stage */
    DeviceType device;      /* Device type */
    int stage_id;           /* Stage index */
} PipelineStage;

/**
 * @brief Pipeline parallel configuration
 */
typedef struct {
    int num_micro_batches;  /* Number of micro-batches (default: 4) */
    int num_stages;         /* Number of pipeline stages */
    bool interleaved;       /* Use interleaved schedule (1F1B) */
} PipelineConfig;

/**
 * @brief Pipeline parallel wrapper
 */
typedef struct CMLPipelineParallel {
    PipelineStage* stages;       /* Array of stages */
    int num_stages;              /* Number of stages */
    PipelineConfig config;       /* Configuration */
    DistProcessGroup* group;     /* Process group */

    /* Micro-batch buffers */
    Tensor*** micro_batch_outputs; /* [stage][micro_batch] */
    int num_micro_batches;
} CMLPipelineParallel;

/**
 * @brief Create pipeline parallel wrapper
 *
 * @param stages Array of pipeline stages
 * @param num_stages Number of stages
 * @param config Configuration (NULL = defaults)
 * @return Pipeline wrapper, or NULL on failure
 */
CMLPipelineParallel* cml_pipeline_create(PipelineStage* stages, int num_stages,
                                          const PipelineConfig* config);

/**
 * @brief Execute pipeline forward pass with micro-batching
 *
 * @param pipeline Pipeline wrapper
 * @param input Input tensor (full batch)
 * @return Output tensor (full batch)
 */
Tensor* cml_pipeline_forward(CMLPipelineParallel* pipeline, Tensor* input);

/**
 * @brief Execute pipeline backward pass
 *
 * @param pipeline Pipeline wrapper
 * @param grad_output Gradient from loss
 * @return 0 on success
 */
int cml_pipeline_backward(CMLPipelineParallel* pipeline, Tensor* grad_output);

/**
 * @brief Free pipeline wrapper
 */
void cml_pipeline_free(CMLPipelineParallel* pipeline);

#ifdef __cplusplus
}
#endif

#endif /* CML_PIPELINE_PARALLEL_H */
