/* Windows stubs for the socket-based comm backends. The NCCL/Gloo/IB transport
 * implementations are POSIX-socket-only and excluded from the Windows build;
 * distributed.c/comm_backend.c still reference these factories, so provide
 * stubs that report the backend as unavailable. */
#ifdef _WIN32
#include "distributed/comm_backend.h"

DistCommOps* cml_dist_create_nccl_backend(void) { return NULL; }
DistCommOps* cml_dist_create_gloo_backend(void) { return NULL; }
#endif /* _WIN32 */
