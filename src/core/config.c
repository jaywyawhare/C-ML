/**
 * @file config.c
 * @brief Global configuration implementation
 */

#include "core/config.h"
#include "core/logging.h"
#include "backend/device.h"
#include <stdlib.h>
#include <time.h>

static DeviceType g_default_device = DEVICE_CPU;
static DType g_default_dtype       = DTYPE_FLOAT32;
static uint64_t g_rng_seed         = 0;
static bool g_rng_seeded           = false;

void cml_set_default_device(DeviceType device) {
    g_default_device = device;
    device_set_default(device);
}

DeviceType cml_get_default_device(void) { return g_default_device; }

void cml_set_default_dtype(DType dtype) {
    g_default_dtype = dtype;
}

DType cml_get_default_dtype(void) { return g_default_dtype; }

void cml_seed(uint64_t seed) {
    g_rng_seed   = seed;
    g_rng_seeded = true;
    srand((unsigned int)seed);
}

uint64_t cml_random_seed(void) {
    if (!g_rng_seeded) {
        g_rng_seed   = (uint64_t)time(NULL) ^ (uint64_t)clock();
        g_rng_seeded = true;
        srand((unsigned int)g_rng_seed);
    }
    return g_rng_seed;
}
