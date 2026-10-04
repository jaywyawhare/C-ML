#include "distributed/pipeline_parallel.h"
#include "distributed/distributed.h"
#include "autograd/autograd.h"
#include "core/logging.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

/** Create a pipeline over a copy of @p stages, allocating the per-stage,
 * per-micro-batch activation cache. Defaults to 4 micro-batches when @p config
 * is NULL. Returns NULL on bad args or allocation failure. */
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
    pipeline->stages     = cml_malloc(num_stages * sizeof(PipelineStage));
    if (!pipeline->stages) {
        cml_free(pipeline);
        return NULL;
    }
    memcpy(pipeline->stages, stages, num_stages * sizeof(PipelineStage));

    if (config) {
        pipeline->config = *config;
    } else {
        pipeline->config.num_micro_batches = 4;
        pipeline->config.num_stages        = num_stages;
        pipeline->config.interleaved       = false;
    }

    pipeline->num_micro_batches = pipeline->config.num_micro_batches;
    pipeline->group             = cml_dist_get_default_group();

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

    LOG_INFO("Pipeline created: %d stages, %d micro-batches", num_stages,
             pipeline->num_micro_batches);
    return pipeline;
}

/* 1F1B per-stage program: (P-1-s) warmup forwards, then one forward paired with
 * one backward, then the remaining backwards. Deeper stages warm up less, so by
 * the time stage 0 finishes its warmup the last stage is already retiring
 * backwards -- that is what caps the number of simultaneously live
 * activations. */
static void pipe_fill_stage_program(PipeUnit* out, int stage, int P, int M) {
    int k    = 0;
    int warm = P - 1 - stage;
    if (warm > M)
        warm = M;

    for (int m = 0; m < warm; m++)
        out[k++] = (PipeUnit){.stage = stage, .micro_batch = m, .kind = PIPE_UNIT_FORWARD};
    for (int i = 0; i < M - warm; i++) {
        out[k++] = (PipeUnit){.stage = stage, .micro_batch = warm + i, .kind = PIPE_UNIT_FORWARD};
        out[k++] = (PipeUnit){.stage = stage, .micro_batch = i, .kind = PIPE_UNIT_BACKWARD};
    }
    for (int m = M - warm; m < M; m++)
        out[k++] = (PipeUnit){.stage = stage, .micro_batch = m, .kind = PIPE_UNIT_BACKWARD};
}

/** Build the forward/backward execution schedule as a caller-owned array of
 * 2*P*M units: a simple GPipe order, or a 1F1B interleaved order that retires
 * activations as early as dependencies allow. Writes the count to
 * @p out_num_units; returns NULL on bad args, overflow, or a scheduling stall. */
PipeUnit* cml_pipeline_build_schedule(int num_stages, int num_micro_batches, bool interleaved,
                                      int* out_num_units) {
    int P = num_stages, M = num_micro_batches;
    if (P <= 0 || M <= 0 || !out_num_units)
        return NULL;
    if ((long long)2 * P * M > INT_MAX)
        return NULL;

    int total     = 2 * P * M;
    PipeUnit* out = cml_malloc((size_t)total * sizeof(PipeUnit));
    if (!out)
        return NULL;

    int n = 0;
    if (!interleaved) {
        /* GPipe: every forward stage-major, then every backward in reverse
         * stage order. */
        for (int s = 0; s < P; s++)
            for (int m = 0; m < M; m++)
                out[n++] = (PipeUnit){.stage = s, .micro_batch = m, .kind = PIPE_UNIT_FORWARD};
        for (int s = P - 1; s >= 0; s--)
            for (int m = 0; m < M; m++)
                out[n++] = (PipeUnit){.stage = s, .micro_batch = m, .kind = PIPE_UNIT_BACKWARD};
        *out_num_units = n;
        return out;
    }

    /* Interleave by replaying each stage's 1F1B program, emitting whichever
     * stage's next unit has its dependencies met. Backwards are preferred
     * (deepest stage first) so activations retire as early as the dependency
     * graph allows; forwards fill in otherwise. */
    PipeUnit* prog = cml_malloc((size_t)P * (size_t)(2 * M) * sizeof(PipeUnit));
    bool* f_done   = cml_calloc((size_t)P * (size_t)M, sizeof(bool));
    bool* b_done   = cml_calloc((size_t)P * (size_t)M, sizeof(bool));
    int* pc        = cml_calloc((size_t)P, sizeof(int));
    if (!prog || !f_done || !b_done || !pc) {
        cml_free(prog);
        cml_free(f_done);
        cml_free(b_done);
        cml_free(pc);
        cml_free(out);
        return NULL;
    }

    for (int s = 0; s < P; s++)
        pipe_fill_stage_program(prog + (size_t)s * (size_t)(2 * M), s, P, M);

#define PIPE_NEXT(s) (prog[(size_t)(s) * (size_t)(2 * M) + (size_t)pc[s]])
#define PIPE_READY(u)                                                                              \
    ((u).kind == PIPE_UNIT_FORWARD                                                                 \
         ? ((u).stage == 0 ||                                                                      \
            f_done[(size_t)((u).stage - 1) * (size_t)M + (size_t)(u).micro_batch])                 \
         : (f_done[(size_t)(u).stage * (size_t)M + (size_t)(u).micro_batch] &&                     \
            ((u).stage == P - 1 ||                                                                 \
             b_done[(size_t)((u).stage + 1) * (size_t)M + (size_t)(u).micro_batch])))

    while (n < total) {
        bool progress = false;

        for (int s = P - 1; s >= 0 && !progress; s--) {
            if (pc[s] >= 2 * M)
                continue;
            PipeUnit u = PIPE_NEXT(s);
            if (u.kind != PIPE_UNIT_BACKWARD || !PIPE_READY(u))
                continue;
            b_done[(size_t)s * (size_t)M + (size_t)u.micro_batch] = true;
            out[n++]                                              = u;
            pc[s]++;
            progress = true;
        }
        for (int s = 0; s < P && !progress; s++) {
            if (pc[s] >= 2 * M)
                continue;
            PipeUnit u = PIPE_NEXT(s);
            if (u.kind != PIPE_UNIT_FORWARD || !PIPE_READY(u))
                continue;
            f_done[(size_t)s * (size_t)M + (size_t)u.micro_batch] = true;
            out[n++]                                              = u;
            pc[s]++;
            progress = true;
        }

        /* Bounds the loop: every iteration must retire a unit. */
        if (!progress) {
            LOG_ERROR("pipeline: 1F1B schedule stalled at %d/%d units (P=%d, M=%d)", n, total, P,
                      M);
            cml_free(prog);
            cml_free(f_done);
            cml_free(b_done);
            cml_free(pc);
            cml_free(out);
            return NULL;
        }
    }

#undef PIPE_NEXT
#undef PIPE_READY

    cml_free(prog);
    cml_free(f_done);
    cml_free(b_done);
    cml_free(pc);
    *out_num_units = n;
    return out;
}

/** Copy rows [@p start, @p end) of @p input along dim 0 into a fresh
 * caller-owned tensor. Returns NULL on bad range or allocation/overflow. */
static Tensor* slice_batch_dim(Tensor* input, int start, int end) {
    if (!input || start < 0 || end <= start || end > input->shape[0])
        return NULL;

    tensor_ensure_executed(input);
    const float* src = (const float*)tensor_data_ptr(input);
    if (!src)
        return NULL;

    int slice_rows     = end - start;
    size_t row_elems   = input->numel / (size_t)input->shape[0];
    size_t slice_elems = (size_t)slice_rows * row_elems;
    if (row_elems > 0 && slice_elems / row_elems != (size_t)slice_rows) {
        return NULL; /* overflow */
    }

    float* slice_data = cml_malloc(slice_elems * sizeof(float));
    if (!slice_data)
        return NULL;

    memcpy(slice_data, src + (size_t)start * row_elems, slice_elems * sizeof(float));

    /* Build shape: same as input but dim 0 = slice_rows */
    int* shape = cml_malloc(input->ndim * sizeof(int));
    if (!shape) {
        cml_free(slice_data);
        return NULL;
    }
    memcpy(shape, input->shape, input->ndim * sizeof(int));
    shape[0] = slice_rows;

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(slice_data, shape, input->ndim, &cfg);
    cml_free(slice_data);
    cml_free(shape);
    return result;
}

/** Concatenate @p count tensors along dim 0 into a fresh caller-owned tensor
 * (inverse of slice_batch_dim). Returns NULL on bad args or allocation/overflow. */
static Tensor* concat_batch_dim(Tensor** tensors, int count) {
    if (!tensors || count <= 0 || !tensors[0])
        return NULL;

    /* Compute total batch size */
    size_t total_batch = 0;
    int ndim           = tensors[0]->ndim;
    size_t row_elems   = tensors[0]->numel / (size_t)tensors[0]->shape[0];

    for (int i = 0; i < count; i++) {
        if (!tensors[i])
            return NULL;
        total_batch += (size_t)tensors[i]->shape[0];
    }

    size_t total_elems = total_batch * row_elems;
    if (row_elems > 0 && total_elems / row_elems != total_batch) {
        return NULL; /* overflow */
    }
    float* out_data = cml_malloc(total_elems * sizeof(float));
    if (!out_data)
        return NULL;

    size_t offset = 0;
    for (int i = 0; i < count; i++) {
        tensor_ensure_executed(tensors[i]);
        const float* src = (const float*)tensor_data_ptr(tensors[i]);
        if (!src) {
            cml_free(out_data);
            return NULL;
        }
        size_t chunk = tensors[i]->numel * sizeof(float);
        memcpy(out_data + offset, src, chunk);
        offset += tensors[i]->numel;
    }

    int* shape = cml_malloc(ndim * sizeof(int));
    if (!shape) {
        cml_free(out_data);
        return NULL;
    }
    memcpy(shape, tensors[0]->shape, ndim * sizeof(int));
    shape[0] = (int)total_batch;

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(out_data, shape, ndim, &cfg);
    cml_free(out_data);
    cml_free(shape);
    return result;
}

/** Single-process pipeline forward: split @p input into micro-batches, run the
 * scheduled forward units caching each stage's outputs, and concatenate the
 * final stage's micro-batch outputs. Returns the combined output, or NULL. */
Tensor* cml_pipeline_forward(CMLPipelineParallel* pipeline, Tensor* input) {
    if (!pipeline || !input)
        return NULL;

    int num_mb     = pipeline->num_micro_batches;
    int num_stages = pipeline->num_stages;
    int batch_size = input->shape[0];

    if (batch_size <= 0 || num_mb <= 0) {
        LOG_ERROR("Pipeline forward: invalid batch_size=%d or num_micro_batches=%d", batch_size,
                  num_mb);
        return NULL;
    }

    /* Release the previous invocation's cached outputs (owned by the
     * pipeline) before caching the new ones, or repeated forwards leak. */
    for (int s = 0; s < num_stages; s++) {
        for (int mb = 0; mb < num_mb; mb++) {
            if (pipeline->micro_batch_outputs[s][mb])
                tensor_free(pipeline->micro_batch_outputs[s][mb]);
            pipeline->micro_batch_outputs[s][mb] = NULL;
        }
    }

    int mb_size = batch_size / num_mb;
    if (mb_size < 1)
        mb_size = 1;

    /* Split input into micro-batches along dim 0 */
    Tensor** input_slices = cml_malloc(num_mb * sizeof(Tensor*));
    if (!input_slices)
        return NULL;

    for (int mb = 0; mb < num_mb; mb++) {
        int start        = mb * mb_size;
        int end          = (mb == num_mb - 1) ? batch_size : start + mb_size;
        input_slices[mb] = slice_batch_dim(input, start, end);
        if (!input_slices[mb]) {
            LOG_ERROR("Pipeline forward: failed to slice input for micro-batch %d", mb);
            for (int j = 0; j < mb; j++)
                tensor_free(input_slices[j]);
            cml_free(input_slices);
            return NULL;
        }
    }

    /* Run the forward units in the configured schedule order (GPipe or 1F1B).
     * The schedule guarantees F(s-1,mb) precedes F(s,mb), so each unit's input
     * is already cached by the time it runs. */
    int num_units = 0;
    PipeUnit* sched =
        cml_pipeline_build_schedule(num_stages, num_mb, pipeline->config.interleaved, &num_units);
    if (!sched) {
        for (int j = 0; j < num_mb; j++)
            tensor_free(input_slices[j]);
        cml_free(input_slices);
        return NULL;
    }

    for (int u = 0; u < num_units; u++) {
        if (sched[u].kind != PIPE_UNIT_FORWARD)
            continue;

        int stage = sched[u].stage;
        int mb    = sched[u].micro_batch;
        Tensor* mb_input =
            (stage == 0) ? input_slices[mb] : pipeline->micro_batch_outputs[stage - 1][mb];

        if (!mb_input) {
            LOG_ERROR("Pipeline forward: NULL input at stage %d, micro-batch %d", stage, mb);
            goto fwd_fail;
        }

        Tensor* mb_output = module_forward(pipeline->stages[stage].module, mb_input);
        if (!mb_output) {
            LOG_ERROR("Pipeline forward: module_forward failed at stage %d, micro-batch %d", stage,
                      mb);
            goto fwd_fail;
        }

        pipeline->micro_batch_outputs[stage][mb] = mb_output;
    }
    cml_free(sched);

    /* Free input slices (not needed after stage 0) */
    for (int mb = 0; mb < num_mb; mb++)
        tensor_free(input_slices[mb]);
    cml_free(input_slices);

    /* Concatenate the final stage's micro-batch outputs along dim 0 */
    Tensor* final_output = concat_batch_dim(pipeline->micro_batch_outputs[num_stages - 1], num_mb);

    if (!final_output) {
        LOG_ERROR("Pipeline forward: failed to concatenate final outputs");
    }

    return final_output;

fwd_fail:
    cml_free(sched);
    for (int j = 0; j < num_mb; j++)
        tensor_free(input_slices[j]);
    cml_free(input_slices);
    return NULL;
}

/** Single-process pipeline backward: slice @p grad_output per micro-batch and run
 * the scheduled backward units over the cached activations, seeding only the last
 * stage. Under 1F1B, a micro-batch's activations are freed once its backward
 * reaches stage 0. Returns 0 on success, -1 on error. */
int cml_pipeline_backward(CMLPipelineParallel* pipeline, Tensor* grad_output) {
    if (!pipeline || !grad_output)
        return -1;

    int num_mb     = pipeline->num_micro_batches;
    int num_stages = pipeline->num_stages;
    int batch_size = grad_output->shape[0];

    if (batch_size <= 0 || num_mb <= 0) {
        LOG_ERROR("Pipeline backward: invalid batch_size=%d or num_micro_batches=%d", batch_size,
                  num_mb);
        return -1;
    }

    int mb_size = batch_size / num_mb;
    if (mb_size < 1)
        mb_size = 1;

    /* Split grad_output into micro-batch gradients matching the forward split */
    Tensor** grad_slices = cml_malloc(num_mb * sizeof(Tensor*));
    if (!grad_slices)
        return -1;

    for (int mb = 0; mb < num_mb; mb++) {
        int start       = mb * mb_size;
        int end         = (mb == num_mb - 1) ? batch_size : start + mb_size;
        grad_slices[mb] = slice_batch_dim(grad_output, start, end);
        if (!grad_slices[mb]) {
            LOG_ERROR("Pipeline backward: failed to slice grad for micro-batch %d", mb);
            for (int j = 0; j < mb; j++)
                tensor_free(grad_slices[j]);
            cml_free(grad_slices);
            return -1;
        }
    }

    /* Run the backward units in the configured schedule order. Both orders keep
     * B(s+1,mb) before B(s,mb), so a stage's gradient has always arrived from
     * downstream before it runs; they differ only in how early each backward is
     * retired relative to the forwards. */
    bool interleaved = pipeline->config.interleaved;
    int num_units    = 0;
    PipeUnit* sched  = cml_pipeline_build_schedule(num_stages, num_mb, interleaved, &num_units);
    if (!sched) {
        for (int mb = 0; mb < num_mb; mb++)
            tensor_free(grad_slices[mb]);
        cml_free(grad_slices);
        return -1;
    }

    for (int u = 0; u < num_units; u++) {
        if (sched[u].kind != PIPE_UNIT_BACKWARD)
            continue;

        int stage         = sched[u].stage;
        int mb            = sched[u].micro_batch;
        Tensor* mb_output = pipeline->micro_batch_outputs[stage][mb];
        if (!mb_output) {
            LOG_ERROR("Pipeline backward: NULL cached output at stage %d, micro-batch %d", stage,
                      mb);
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

        /* 1F1B retires a micro-batch the moment its backward reaches stage 0:
         * every stage is done with it, so its cached activations are released
         * here instead of being held until the next forward. That early release
         * is the whole point of the schedule -- it also means an interleaved
         * backward consumes the forward's cache (documented in the header). */
        if (interleaved && stage == 0) {
            for (int s = 0; s < num_stages; s++) {
                if (pipeline->micro_batch_outputs[s][mb])
                    tensor_free(pipeline->micro_batch_outputs[s][mb]);
                pipeline->micro_batch_outputs[s][mb] = NULL;
            }
        }
    }
    cml_free(sched);

    /* We own the grad slices end-to-end (backward cloned them). */
    for (int mb = 0; mb < num_mb; mb++)
        tensor_free(grad_slices[mb]);
    cml_free(grad_slices);

    LOG_DEBUG("Pipeline backward completed: %d stages, %d micro-batches", num_stages, num_mb);
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

/** Send activation @p t to @p dst for micro-batch @p mb: a fixed-size shape
 * header on tag 2*mb, then the data on tag 2*mb+1, so the receiver can allocate
 * first and distinct micro-batches never collide. */
static int pipe_send_tensor(Tensor* t, int dst, int mb) {
    tensor_ensure_executed(t);
    if (!t->data)
        return -1;

    float meta[PIPE_META_LEN];
    memset(meta, 0, sizeof(meta));
    meta[0] = (float)t->ndim;
    for (int i = 0; i < t->ndim && i < PIPE_MAX_NDIM; i++)
        meta[1 + i] = (float)t->shape[i];

    int mshape[1] = {PIPE_META_LEN};
    Tensor mt;
    memset(&mt, 0, sizeof(mt));
    mt.data   = meta;
    mt.numel  = PIPE_META_LEN;
    mt.ndim   = 1;
    mt.shape  = mshape;
    mt.dtype  = DTYPE_FLOAT32;
    mt.device = DEVICE_CPU;

    if (cml_dist_send(&mt, dst, 2 * mb) != 0)
        return -1;
    return cml_dist_send(t, dst, 2 * mb + 1);
}

/** Receive an activation from @p src for micro-batch @p mb: read the shape header
 * (tag 2*mb), allocate, then read the data (tag 2*mb+1). Returns a fresh
 * caller-owned tensor, or NULL on error. Counterpart to pipe_send_tensor. */
static Tensor* pipe_recv_tensor(int src, int mb) {
    float meta[PIPE_META_LEN];
    int mshape[1] = {PIPE_META_LEN};
    Tensor mt;
    memset(&mt, 0, sizeof(mt));
    mt.data   = meta;
    mt.numel  = PIPE_META_LEN;
    mt.ndim   = 1;
    mt.shape  = mshape;
    mt.dtype  = DTYPE_FLOAT32;
    mt.device = DEVICE_CPU;
    if (cml_dist_recv(&mt, src, 2 * mb) != 0)
        return NULL;

    int ndim = (int)meta[0];
    if (ndim < 1 || ndim > PIPE_MAX_NDIM)
        return NULL;
    int shape[PIPE_MAX_NDIM];
    size_t numel = 0;
    for (int i = 0; i < ndim; i++)
        shape[i] = (int)meta[1 + i];
    if (!tensor_numel_checked(shape, ndim, &numel))
        return NULL;

    float* data = (float*)cml_malloc(numel * sizeof(float));
    if (!data)
        return NULL;
    Tensor rt;
    memset(&rt, 0, sizeof(rt));
    rt.data   = data;
    rt.numel  = numel;
    rt.ndim   = ndim;
    rt.shape  = shape;
    rt.dtype  = DTYPE_FLOAT32;
    rt.device = DEVICE_CPU;
    if (cml_dist_recv(&rt, src, 2 * mb + 1) != 0) {
        cml_free(data);
        return NULL;
    }

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* out = tensor_from_data(data, shape, ndim, &cfg);
    cml_free(data);
    return out;
}

/** Free the cross-rank forward cache (per-micro-batch stage inputs and outputs)
 * and null the pointers, so a new dist forward can start clean. */
static void dist_free_cache(CMLPipelineParallel* p) {
    if (p->dist_stage_inputs) {
        for (int mb = 0; mb < p->num_micro_batches; mb++)
            if (p->dist_stage_inputs[mb])
                tensor_free(p->dist_stage_inputs[mb]);
        cml_free(p->dist_stage_inputs);
        p->dist_stage_inputs = NULL;
    }
    if (p->dist_stage_outputs) {
        for (int mb = 0; mb < p->num_micro_batches; mb++)
            if (p->dist_stage_outputs[mb])
                tensor_free(p->dist_stage_outputs[mb]);
        cml_free(p->dist_stage_outputs);
        p->dist_stage_outputs = NULL;
    }
}

/** True cross-rank pipeline forward (world_size == num_stages): rank r runs only
 * stage r, receiving activations from r-1 and streaming to r+1. Caches inputs and
 * outputs for backward. Returns the concatenated output on the last rank, NULL
 * elsewhere (or on error). */
Tensor* cml_pipeline_dist_forward(CMLPipelineParallel* pipeline, Tensor* input) {
    if (!pipeline)
        return NULL;
    DistProcessGroup* g = pipeline->group;
    if (!g || g->world_size != pipeline->num_stages) {
        LOG_ERROR("dist pipeline: world_size (%d) must equal num_stages (%d)",
                  g ? g->world_size : -1, pipeline->num_stages);
        return NULL;
    }

    int rank      = g->rank;
    int P         = pipeline->num_stages;
    int M         = pipeline->num_micro_batches;
    Module* stage = pipeline->stages[rank].module;
    if (!stage) {
        LOG_ERROR("dist pipeline: rank %d has no stage module", rank);
        return NULL;
    }

    dist_free_cache(pipeline);
    pipeline->dist_stage_inputs  = cml_calloc((size_t)M, sizeof(Tensor*));
    pipeline->dist_stage_outputs = cml_calloc((size_t)M, sizeof(Tensor*));
    if (!pipeline->dist_stage_inputs || !pipeline->dist_stage_outputs) {
        dist_free_cache(pipeline);
        return NULL;
    }

    int batch   = (rank == 0 && input) ? input->shape[0] : 0;
    int mb_size = (batch > 0) ? (batch / M > 0 ? batch / M : 1) : 0;

    Tensor** last_outputs = (rank == P - 1) ? cml_calloc((size_t)M, sizeof(Tensor*)) : NULL;

    for (int mb = 0; mb < M; mb++) {
        Tensor* x;
        if (rank == 0) {
            int start = mb * mb_size;
            int end   = (mb == M - 1) ? batch : start + mb_size;
            if (!input || start >= end) {
                LOG_ERROR("dist pipeline: bad input slice");
                goto fail;
            }
            x = slice_batch_dim(input, start, end);
        } else {
            x = pipe_recv_tensor(rank - 1, mb);
        }
        if (!x) {
            LOG_ERROR("dist pipeline: rank %d could not obtain input mb %d", rank, mb);
            goto fail;
        }

        /* Grad on the stage input so backward can produce the upstream gradient. */
        tensor_set_requires_grad(x, true);
        Tensor* y = module_forward(stage, x);
        if (!y) {
            tensor_free(x);
            LOG_ERROR("dist pipeline: stage forward failed");
            goto fail;
        }
        tensor_ensure_executed(y);

        pipeline->dist_stage_inputs[mb]  = x;
        pipeline->dist_stage_outputs[mb] = y;

        if (rank < P - 1) {
            if (pipe_send_tensor(y, rank + 1, mb) != 0) {
                LOG_ERROR("dist pipeline: send failed");
                goto fail;
            }
        } else {
            last_outputs[mb] = tensor_clone(y); /* clone so cache stays owned for backward */
        }
    }

    if (rank == P - 1) {
        Tensor* out = concat_batch_dim(last_outputs, M);
        for (int mb = 0; mb < M; mb++)
            if (last_outputs[mb])
                tensor_free(last_outputs[mb]);
        cml_free(last_outputs);
        return out;
    }
    return NULL;

fail:
    if (last_outputs) {
        for (int mb = 0; mb < M; mb++)
            if (last_outputs[mb])
                tensor_free(last_outputs[mb]);
        cml_free(last_outputs);
    }
    return NULL;
}

/** True cross-rank pipeline backward: each rank backprops its cached stage over
 * all micro-batches, seeding the output gradient from @p grad_output on the last
 * rank or receiving it from r+1 otherwise, and streams the input gradient to r-1.
 * Must follow cml_pipeline_dist_forward. Returns 0 on success, -1 on error. */
int cml_pipeline_dist_backward(CMLPipelineParallel* pipeline, Tensor* grad_output) {
    if (!pipeline)
        return -1;
    DistProcessGroup* g = pipeline->group;
    if (!g || g->world_size != pipeline->num_stages)
        return -1;

    int rank = g->rank;
    int P    = pipeline->num_stages;
    int M    = pipeline->num_micro_batches;
    if (!pipeline->dist_stage_outputs || !pipeline->dist_stage_inputs) {
        LOG_ERROR("dist pipeline backward: run cml_pipeline_dist_forward first");
        return -1;
    }

    int mb_size = 0;
    if (rank == P - 1) {
        if (!grad_output) {
            LOG_ERROR("dist pipeline backward: last rank needs grad_output");
            return -1;
        }
        int gb  = grad_output->shape[0];
        mb_size = gb / M > 0 ? gb / M : 1;
    }

    for (int mb = 0; mb < M; mb++) {
        Tensor* out = pipeline->dist_stage_outputs[mb];
        Tensor* in  = pipeline->dist_stage_inputs[mb];
        if (!out || !in)
            return -1;

        /* Gradient wrt this stage's output: sliced from the loss on the last
         * rank, received from downstream otherwise. */
        Tensor* gout;
        if (rank == P - 1) {
            int start = mb * mb_size;
            int end   = (mb == M - 1) ? grad_output->shape[0] : start + mb_size;
            gout      = slice_batch_dim(grad_output, start, end);
        } else {
            gout = pipe_recv_tensor(rank + 1, mb);
        }
        if (!gout)
            return -1;

        /* Backprop the stage: seeds out->grad with gout, accumulates this
         * stage's weight grads and computes in->grad (across micro-batches the
         * weight grads accumulate, which is the intended behaviour). */
        tensor_backward(out, gout, false, false);
        tensor_free(gout);

        /* Stream the input-gradient upstream (rank 0 has no upstream). */
        if (rank > 0) {
            if (!in->grad) {
                LOG_ERROR("dist pipeline backward: no input grad at rank %d", rank);
                return -1;
            }
            if (pipe_send_tensor(in->grad, rank - 1, mb) != 0)
                return -1;
        }
    }
    return 0;
}

/** Free the pipeline: the cross-rank cache, all cached micro-batch activations,
 * the stage copy, and the struct. Does not free the stage modules themselves. */
void cml_pipeline_free(CMLPipelineParallel* pipeline) {
    if (!pipeline)
        return;

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
