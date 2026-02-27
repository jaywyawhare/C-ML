/**
 * @file comm_backend.h
 * @brief Communication backend interface for distributed training
 *
 * Backend implementations use dlopen to load NCCL/MPI/Gloo at runtime,
 * allowing CML to compile without these dependencies.
 */

#ifndef CML_COMM_BACKEND_H
#define CML_COMM_BACKEND_H

#include "distributed/distributed.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create NCCL backend communication ops
 * @return Comm ops vtable, or NULL if NCCL unavailable
 */
DistCommOps* cml_dist_create_nccl_backend(void);

/**
 * @brief Create MPI backend communication ops
 * @return Comm ops vtable, or NULL if MPI unavailable
 */
DistCommOps* cml_dist_create_mpi_backend(void);

/**
 * @brief Create Gloo backend communication ops (pure CPU fallback)
 * @return Comm ops vtable, or NULL on failure
 */
DistCommOps* cml_dist_create_gloo_backend(void);

/**
 * @brief Free backend ops and associated resources
 */
void cml_dist_free_backend(DistCommOps* ops);

#ifdef __cplusplus
}
#endif

#endif /* CML_COMM_BACKEND_H */
