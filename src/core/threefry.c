#include "core/threefry.h"
#include <math.h>
#include <string.h>

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

static const uint64_t SKEIN_KS_PARITY = 0x1BD11BDAA9FC1A22ULL;

static void threefry2x64(const uint64_t key[2], uint64_t ctr[2], uint64_t out[2]) {
    uint64_t ks[3] = {key[0], key[1], SKEIN_KS_PARITY ^ key[0] ^ key[1]};

    uint64_t x0 = ctr[0] + ks[0];
    uint64_t x1 = ctr[1] + ks[1];

    static const int rotations[8] = {16, 42, 12, 31, 16, 32, 24, 21};

    for (int round = 0; round < 20; round++) {
        x0 += x1;
        x1 = ROTL64(x1, rotations[round % 8]) ^ x0;

        if ((round & 3) == 3) {
            int inject = (round >> 2) + 1;
            x0 += ks[inject % 3];
            x1 += ks[(inject + 1) % 3] + (uint64_t)inject;
        }
    }

    out[0] = x0;
    out[1] = x1;
}

void cml_rng_init(CMLRNGState* state, uint64_t seed) {
    state->key[0] = seed;
    state->key[1] = seed ^ 0x0123456789ABCDEFULL;
    state->counter = 0;
}

void cml_rng_uint32(CMLRNGState* state, uint32_t* out, size_t n) {
    size_t i = 0;
    while (i < n) {
        uint64_t ctr[2] = {state->counter, state->counter >> 1};
        uint64_t result[2];
        threefry2x64(state->key, ctr, result);
        state->counter++;

        out[i++] = (uint32_t)(result[0] & 0xFFFFFFFF);
        if (i < n) out[i++] = (uint32_t)(result[0] >> 32);
        if (i < n) out[i++] = (uint32_t)(result[1] & 0xFFFFFFFF);
        if (i < n) out[i++] = (uint32_t)(result[1] >> 32);
    }
}

void cml_rng_uniform(CMLRNGState* state, float* out, size_t n) {
    static const float SCALE = 1.0f / (float)(1ULL << 24);
    size_t i = 0;
    while (i < n) {
        uint64_t ctr[2] = {state->counter, state->counter >> 1};
        uint64_t result[2];
        threefry2x64(state->key, ctr, result);
        state->counter++;

        out[i++] = (float)(result[0] >> 40) * SCALE;
        if (i < n) out[i++] = (float)(result[1] >> 40) * SCALE;
    }
}

void cml_rng_normal(CMLRNGState* state, float* out, size_t n) {
    static const float TWO_PI = 6.283185307179586f;
    static const float SCALE = 1.0f / (float)(1ULL << 24);

    size_t i = 0;
    while (i < n) {
        uint64_t ctr[2] = {state->counter, state->counter >> 1};
        uint64_t result[2];
        threefry2x64(state->key, ctr, result);
        state->counter++;

        float u1 = (float)(result[0] >> 40) * SCALE;
        float u2 = (float)(result[1] >> 40) * SCALE;
        if (u1 < 1e-10f) u1 = 1e-10f;

        float r = sqrtf(-2.0f * logf(u1));
        float theta = TWO_PI * u2;

        out[i++] = r * cosf(theta);
        if (i < n) out[i++] = r * sinf(theta);
    }
}

static __thread CMLRNGState g_rng = {{0x12345678DEADBEEFULL, 0xFEDCBA9876543210ULL}, 0};
static __thread int g_rng_initialized = 0;

void cml_rng_set_global_seed(uint64_t seed) {
    cml_rng_init(&g_rng, seed);
    g_rng_initialized = 1;
}

CMLRNGState* cml_rng_get_global(void) {
    if (!g_rng_initialized) {
        uint64_t seed = (uint64_t)(uintptr_t)&g_rng ^ 0xDEADBEEFCAFEBABEULL;
        cml_rng_init(&g_rng, seed);
        g_rng_initialized = 1;
    }
    return &g_rng;
}

CMLRNGState cml_rng_fork(CMLRNGState* state) {
    CMLRNGState forked;
    forked.key[0] = state->key[0] ^ (state->counter * 0x9E3779B97F4A7C15ULL);
    forked.key[1] = state->key[1] ^ (state->counter * 0x6C62272E07BB0142ULL);
    forked.counter = 0;
    state->counter++;
    return forked;
}
