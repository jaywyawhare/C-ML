#pragma once
#include <stdint.h>
#include <stddef.h>

typedef struct CMLRNGState {
    uint64_t key[2];
    uint64_t counter;
} CMLRNGState;

void cml_rng_init(CMLRNGState* state, uint64_t seed);

void cml_rng_uniform(CMLRNGState* state, float* out, size_t n);

void cml_rng_normal(CMLRNGState* state, float* out, size_t n);

void cml_rng_uint32(CMLRNGState* state, uint32_t* out, size_t n);

void cml_rng_set_global_seed(uint64_t seed);
CMLRNGState* cml_rng_get_global(void);

CMLRNGState cml_rng_fork(CMLRNGState* state);
