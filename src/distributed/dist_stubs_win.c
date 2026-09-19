/* Windows stubs for the socket-based comm backends. The NCCL/Gloo/IB transport
 * implementations are POSIX-socket-only and excluded from the Windows build;
 * distributed.c/comm_backend.c still reference these factories, so provide
 * stubs that report the backend as unavailable.
 *
 * Only a forward declaration of DistCommOps is used so this TU pulls no heavy
 * headers (comm_backend.h transitively includes windows.h, which trips a MinGW
 * FORCEINLINE/gnu-inline header bug in a minimal translation unit). */
#ifdef _WIN32

typedef struct DistCommOps DistCommOps;

DistCommOps* cml_dist_create_nccl_backend(void) { return (DistCommOps*)0; }
DistCommOps* cml_dist_create_gloo_backend(void) { return (DistCommOps*)0; }

#endif /* _WIN32 */
