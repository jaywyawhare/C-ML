#include "nn/init.h"
#include "core/threefry.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>

void nn_init_uniform(Tensor* tensor, float low, float high) {
    if (!tensor || low > high)
        return;

    if (tensor_ensure_executed(tensor) != 0)
        return;

    float* data = (float*)tensor_data_ptr(tensor);
    if (!data)
        return;

    CMLRNGState* rng = cml_rng_get_global();
    if (!rng) {
        LOG_ERROR("nn_init_uniform: RNG not initialized");
        return;
    }

    size_t n = tensor->numel;
    if (n == 0)
        return;

    float* samples = (float*)malloc(n * sizeof(float));
    if (!samples)
        return;

    cml_rng_uniform(rng, samples, n);
    float span = high - low;
    for (size_t i = 0; i < n; i++)
        data[i] = low + samples[i] * span;

    free(samples);
}

void nn_init_xavier(Tensor* tensor, int fan_in, int fan_out) {
    if (fan_in <= 0 || fan_out <= 0)
        return;
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    nn_init_uniform(tensor, -scale, scale);
}

void nn_init_kaiming(Tensor* tensor, int fan_in, int kernel_volume) {
    if (fan_in <= 0 || kernel_volume <= 0)
        return;
    float scale = sqrtf(2.0f / (float)(fan_in * kernel_volume));
    nn_init_uniform(tensor, -scale, scale);
}
