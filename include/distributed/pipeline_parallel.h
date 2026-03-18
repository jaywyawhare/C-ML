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
    bool interleaved;       /* Use interleaved schedule (1F1B) */
} PipelineConfig;

typedef struct CMLPipelineParallel {
    PipelineStage* stages;       /* Array of stages */
    int num_stages;              /* Number of stages */
    PipelineConfig config;       /* Configuration */
    DistProcessGroup* group;     /* Process group */

    /* Micro-batch buffers */
    Tensor*** micro_batch_outputs; /* [stage][micro_batch] */
    int num_micro_batches;
} CMLPipelineParallel;

CMLPipelineParallel* cml_pipeline_create(PipelineStage* stages, int num_stages,
                                          const PipelineConfig* config);

Tensor* cml_pipeline_forward(CMLPipelineParallel* pipeline, Tensor* input);

int cml_pipeline_backward(CMLPipelineParallel* pipeline, Tensor* grad_output);

void cml_pipeline_free(CMLPipelineParallel* pipeline);

#ifdef __cplusplus
}
#endif

#endif /* CML_PIPELINE_PARALLEL_H */
