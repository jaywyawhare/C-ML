/* Backend implementations use dlopen to load NCCL/MPI/Gloo at runtime,
 * allowing CML to compile without these dependencies. */

#ifndef CML_COMM_BACKEND_H
#define CML_COMM_BACKEND_H

#include "distributed/distributed.h"

#ifdef __cplusplus
extern "C" {
#endif

DistCommOps* cml_dist_create_nccl_backend(void);

DistCommOps* cml_dist_create_mpi_backend(void);

DistCommOps* cml_dist_create_gloo_backend(void);

void cml_dist_free_backend(DistCommOps* ops);

/* Tries NCCL first, then MPI, then Gloo. */
DistCommOps* cml_dist_auto_select_backend(void);

const char* cml_dist_backend_name(DistBackendType type);

#ifdef __cplusplus
}
#endif

#endif /* CML_COMM_BACKEND_H */
