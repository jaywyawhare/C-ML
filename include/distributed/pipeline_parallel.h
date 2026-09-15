#ifndef CML_PIPELINE_PARALLEL_H
#define CML_PIPELINE_PARALLEL_H

#include "distributed/distributed.h"
#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PipelineStage {
    Module* module;         /* Module for this stage (not owned) */
    int device_id;          /* Device for this stage */
    DeviceType device;      /* Device type */
    int stage_id;           /* Stage index */
} PipelineStage;

typedef struct {
    int num_micro_batches;  /* Number of micro-batches (default: 4) */
    int num_stages;         /* Number of pipeline stages */
    bool interleaved;       /* NOT YET HONORED: the schedule is always the
                             * GPipe all-forwards-then-all-backwards order;
                             * setting this has no effect yet. */
} PipelineConfig;

typedef struct CMLPipelineParallel {
    PipelineStage* stages;       /* Array of stages */
    int num_stages;              /* Number of stages */
    PipelineConfig config;       /* Configuration */
    DistProcessGroup* group;     /* Process group */

    /* Micro-batch buffers */
    Tensor*** micro_batch_outputs; /* [stage][micro_batch] */
    int num_micro_batches;

    /* Distributed (cross-rank) mode: this rank owns exactly one stage
     * (stage_id == rank). These cache the per-micro-batch input/output tensors
     * of THIS rank's stage so the distributed backward can back-propagate and
     * stream input-gradients upstream. */
    Tensor** dist_stage_inputs;   /* [micro_batch] — recv'd (or sliced on rank 0) */
    Tensor** dist_stage_outputs;  /* [micro_batch] — this stage's forward output */
} CMLPipelineParallel;

CMLPipelineParallel* cml_pipeline_create(PipelineStage* stages, int num_stages,
                                          const PipelineConfig* config);

Tensor* cml_pipeline_forward(CMLPipelineParallel* pipeline, Tensor* input);

int cml_pipeline_backward(CMLPipelineParallel* pipeline, Tensor* grad_output);

/* True cross-rank pipeline parallelism: world_size == num_stages and this rank
 * runs ONLY stage `rank`, streaming micro-batch activations to rank+1 and
 * receiving from rank-1 over the process group. Because each stage is a separate
 * process, stages run concurrently — real pipeline overlap. `input` is used only
 * on rank 0; every other rank passes NULL and receives from upstream. Returns
 * the assembled output on the LAST rank and NULL on all others (check rank). */
Tensor* cml_pipeline_dist_forward(CMLPipelineParallel* pipeline, Tensor* input);

/* Mirror of the distributed forward: the last rank seeds gradients from
 * `grad_output` (its per-sample loss gradient, same shape as the forward
 * output); every rank back-props its stage and streams the input-gradient
 * upstream so all stages get weight gradients. Must follow a
 * cml_pipeline_dist_forward on the same pipeline. */
int cml_pipeline_dist_backward(CMLPipelineParallel* pipeline, Tensor* grad_output);

void cml_pipeline_free(CMLPipelineParallel* pipeline);

#ifdef __cplusplus
}
#endif

#endif /* CML_PIPELINE_PARALLEL_H */
