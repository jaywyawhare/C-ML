#ifndef CML_CORE_NPY_IO_H
#define CML_CORE_NPY_IO_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Read and write NumPy .npy arrays (format 1.0/2.0, C order, little-endian host). */

/**
 * Save a contiguous tensor as a NumPy .npy file (version 1.0, C order). Returns 0 on success.
 * @param t (Tensor*)
 * @param path (const char*)
 * @return (int)
 */
int cml_tensor_save_npy(Tensor* t, const char* path);

/**
 * Load a NumPy .npy file into a freshly allocated tensor (caller frees). NULL on error or an
 * unsupported dtype / byte order / Fortran ordering.
 * @param path (const char*)
 * @return (Tensor*)
 */
Tensor* cml_tensor_load_npy(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_NPY_IO_H */
