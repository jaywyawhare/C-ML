#ifndef CML_PIPELINE_PARALLEL_H
#define CML_PIPELINE_PARALLEL_H

#include "distributed/distributed.h"
#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PipelineStage {
    Module* module;    /* Module for this stage (not owned) */
    int device_id;     /* Device for this stage */
    DeviceType device; /* Device type */
    int stage_id;      /* Stage index */
} PipelineStage;

typedef struct {
    int num_micro_batches; /* Number of micro-batches (default: 4) */
    int num_stages;        /* Number of pipeline stages */
    /* Honored: selects the execution order (see cml_pipeline_build_schedule).
     * false => GPipe (all forwards, then all backwards); true => 1F1B, which
     * runs each stage's backwards as early as dependencies allow so a
     * micro-batch's cached activations can be released mid-backward. Weight
     * gradients accumulate over micro-batches, so both orders produce the same
     * numbers; 1F1B holds fewer activations at once. */
    bool interleaved;
} PipelineConfig;

/* One unit of pipeline work: the forward or backward of micro-batch
 * `micro_batch` on stage `stage`. */
typedef enum { PIPE_UNIT_FORWARD = 0, PIPE_UNIT_BACKWARD = 1 } PipeUnitKind;

typedef struct {
    int stage;
    int micro_batch;
    PipeUnitKind kind;
} PipeUnit;

typedef struct CMLPipelineParallel {
    PipelineStage* stages;   /* Array of stages */
    int num_stages;          /* Number of stages */
    PipelineConfig config;   /* Configuration */
    DistProcessGroup* group; /* Process group */

    /* Micro-batch buffers */
    Tensor*** micro_batch_outputs; /* [stage][micro_batch] */
    int num_micro_batches;

    /* Distributed (cross-rank) mode: this rank owns exactly one stage
     * (stage_id == rank). These cache the per-micro-batch input/output tensors
     * of THIS rank's stage so the distributed backward can back-propagate and
     * stream input-gradients upstream. */
    Tensor** dist_stage_inputs;  /* [micro_batch] — recv'd (or sliced on rank 0) */
    Tensor** dist_stage_outputs; /* [micro_batch] — this stage's forward output */
} CMLPipelineParallel;

CMLPipelineParallel* cml_pipeline_create(PipelineStage* stages, int num_stages,
                                         const PipelineConfig* config);

/* Build the execution order for `num_stages` x `num_micro_batches` units.
 *
 * Both orders are valid topological orders of the pipeline dependency graph:
 * F(s,m) after F(s-1,m), B(s,m) after F(s,m) and after B(s+1,m). They differ in
 * when backwards run:
 *
 *   GPipe (interleaved=false): every forward, then every backward. All M
 *   micro-batches' activations are live at once at every stage.
 *
 *   1F1B (interleaved=true): stage s runs (num_stages-1-s) warmup forwards, then
 *   alternates one forward with one backward, then drains its remaining
 *   backwards. Each stage holds at most (num_stages-s) micro-batches of
 *   activations, so peak activation memory is bounded by the depth rather than
 *   the micro-batch count.
 *
 * Returns a malloc'd array of 2*num_stages*num_micro_batches units (free with
 * cml_free) and writes its length to *out_num_units. NULL on invalid input. */
PipeUnit* cml_pipeline_build_schedule(int num_stages, int num_micro_batches, bool interleaved,
                                      int* out_num_units);

Tensor* cml_pipeline_forward(CMLPipelineParallel* pipeline, Tensor* input);

/* Back-propagates every (stage, micro-batch) unit in the configured schedule
 * order. Under `interleaved`, a micro-batch's cached stage outputs are released
 * as soon as its backward reaches stage 0 -- that early release is the point of
 * 1F1B -- so an interleaved backward consumes the forward's cache and cannot be
 * run twice against the same forward. */
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
