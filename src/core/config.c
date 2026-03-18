#include "core/config.h"
#include "core/logging.h"
#include "backend/device.h"
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

static DeviceType g_default_device = DEVICE_CPU;
static DType g_default_dtype       = DTYPE_FLOAT32;
static uint64_t g_rng_seed         = 0;
static bool g_rng_seeded           = false;
static pthread_mutex_t g_config_mutex = PTHREAD_MUTEX_INITIALIZER;

void cml_set_default_device(DeviceType device) {
    pthread_mutex_lock(&g_config_mutex);
    g_default_device = device;
    pthread_mutex_unlock(&g_config_mutex);
    device_set_default(device);
}

DeviceType cml_get_default_device(void) {
    pthread_mutex_lock(&g_config_mutex);
    DeviceType d = g_default_device;
    pthread_mutex_unlock(&g_config_mutex);
    return d;
}

void cml_set_default_dtype(DType dtype) {
    pthread_mutex_lock(&g_config_mutex);
    g_default_dtype = dtype;
    pthread_mutex_unlock(&g_config_mutex);
}

DType cml_get_default_dtype(void) {
    pthread_mutex_lock(&g_config_mutex);
    DType d = g_default_dtype;
    pthread_mutex_unlock(&g_config_mutex);
    return d;
}

void cml_seed(uint64_t seed) {
    pthread_mutex_lock(&g_config_mutex);
    g_rng_seed   = seed;
    g_rng_seeded = true;
    srand((unsigned int)seed);
    pthread_mutex_unlock(&g_config_mutex);
}

uint64_t cml_random_seed(void) {
    pthread_mutex_lock(&g_config_mutex);
    if (!g_rng_seeded) {
        g_rng_seed   = (uint64_t)time(NULL) ^ (uint64_t)clock();
        g_rng_seeded = true;
        srand((unsigned int)g_rng_seed);
    }
    uint64_t s = g_rng_seed;
    pthread_mutex_unlock(&g_config_mutex);
    return s;
}
