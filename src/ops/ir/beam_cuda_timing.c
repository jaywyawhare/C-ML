#ifdef CML_HAS_CUDA

#include "ops/ir/beam_search.h"
#include "ops/ir/gpu/cuda_backend.h"
#include "ops/ir/dispatch.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

double cml_beam_cuda_timing_fn(const CMLBeamVariant* variant, void* user_data)
{
    (void)user_data;

    if (!variant || !variant->source_code) {
        LOG_ERROR("BEAM CUDA timing: NULL variant or source code");
        return -1.0;
    }

    CMLCUDABackend* cuda = cml_dispatch_get_cuda_backend();
    if (!cuda || !cuda->initialized) {
        LOG_ERROR("BEAM CUDA timing: no initialized CUDA backend");
        return -1.0;
    }

    if (!cuda->cuEventCreate || !cuda->cuEventRecord ||
        !cuda->cuEventSynchronize || !cuda->cuEventElapsedTime ||
        !cuda->cuEventDestroy) {
        LOG_ERROR("BEAM CUDA timing: event functions not available");
        return -1.0;
    }

    CMLCUDAKernel* kernel = cml_cuda_compile_source(cuda, variant->source_code,
                                                     "cml_kernel");
    if (!kernel) {
        LOG_DEBUG("BEAM CUDA timing: failed to compile variant");
        return -1.0;
    }

    cml_cuda_kernel_set_launch_config(
        kernel,
        (int)variant->config.grid[0], (int)variant->config.grid[1],
        (int)variant->config.grid[2],
        (int)variant->config.block[0], (int)variant->config.block[1],
        (int)variant->config.block[2]);

    CUevent start = NULL, stop = NULL;
    CUresult err;

    err = cuda->cuEventCreate(&start, 0);
    if (err != 0) {
        cml_cuda_kernel_free(cuda, kernel);
        return -1.0;
    }
    err = cuda->cuEventCreate(&stop, 0);
    if (err != 0) {
        cuda->cuEventDestroy(start);
        cml_cuda_kernel_free(cuda, kernel);
        return -1.0;
    }

    for (int w = 0; w < CML_BEAM_DEFAULT_WARMUP; w++) {
        cuda->cuLaunchKernel(
            kernel->function,
            (unsigned)kernel->grid_dim[0], (unsigned)kernel->grid_dim[1],
            (unsigned)kernel->grid_dim[2],
            (unsigned)kernel->block_dim[0], (unsigned)kernel->block_dim[1],
            (unsigned)kernel->block_dim[2],
            0, cuda->stream, NULL, NULL);
    }
    if (cuda->stream) {
        cuda->cuStreamSynchronize(cuda->stream);
    } else {
        cuda->cuCtxSynchronize();
    }

    cuda->cuEventRecord(start, cuda->stream);

    for (int t = 0; t < CML_BEAM_DEFAULT_TIMING; t++) {
        cuda->cuLaunchKernel(
            kernel->function,
            (unsigned)kernel->grid_dim[0], (unsigned)kernel->grid_dim[1],
            (unsigned)kernel->grid_dim[2],
            (unsigned)kernel->block_dim[0], (unsigned)kernel->block_dim[1],
            (unsigned)kernel->block_dim[2],
            0, cuda->stream, NULL, NULL);
    }

    cuda->cuEventRecord(stop, cuda->stream);
    cuda->cuEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cuda->cuEventElapsedTime(&elapsed_ms, start, stop);

    cuda->cuEventDestroy(start);
    cuda->cuEventDestroy(stop);
    cml_cuda_kernel_free(cuda, kernel);

    double avg_us = (double)(elapsed_ms * 1000.0f) / (double)CML_BEAM_DEFAULT_TIMING;

    LOG_DEBUG("BEAM CUDA timing: %.2f us/launch (block=%d,%d,%d)",
              avg_us,
              variant->config.block_size_x,
              variant->config.block_size_y,
              variant->config.block_size_z);

    return avg_us;
}

#else /* !CML_HAS_CUDA */

#include "ops/ir/beam_search.h"

double cml_beam_cuda_timing_fn(const CMLBeamVariant* variant, void* user_data)
{
    (void)variant;
    (void)user_data;
    return -1.0;
}

#endif /* CML_HAS_CUDA */
