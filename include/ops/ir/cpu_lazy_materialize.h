#ifndef CML_CPU_LAZY_MATERIALIZE_H
#define CML_CPU_LAZY_MATERIALIZE_H

#include "tensor/tensor.h"
#include <stddef.h>

/** @brief Fill `out` with scalar `value` using `out->dtype` element layout (CPU). */
int cml_cpu_lazy_fill(Tensor* out, float value);

/** @brief Copy up to `min(data_size, numel*elem)` bytes from `data` into `out`. */
int cml_cpu_lazy_const(Tensor* out, const void* data, size_t data_size);

/** @brief Uniform random [0,1) per element, cast/stored per `out->dtype`. */
int cml_cpu_lazy_rand_uniform(Tensor* out);

/** @brief Standard normal per element, stored per `out->dtype`. */
int cml_cpu_lazy_rand_normal(Tensor* out);

/** @brief Arange: out[i] = start + i*step in `out->dtype`. */
int cml_cpu_lazy_arange(Tensor* out, float start, float step);

/** @brief Identity matrix n×n into `out` (must be n*n elements). */
int cml_cpu_lazy_eye(Tensor* out, int n);

/** @brief Random integers in [low, high) per element. */
int cml_cpu_lazy_rand_int(Tensor* out, int low, int high);

/** @brief Store float `v` as dtype `dt` at logical index `i` in a dense row-major buffer. */
void cml_cpu_lazy_store_float_elem(void* base, size_t i, DType dt, float v);

#endif /* CML_CPU_LAZY_MATERIALIZE_H */
