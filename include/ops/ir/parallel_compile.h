#ifndef CML_OPS_IR_PARALLEL_COMPILE_H
#define CML_OPS_IR_PARALLEL_COMPILE_H

/* Compile a batch of independent kernels, optionally across the thread pool.
 *
 * Kernel compilation has no data dependencies -- every distinct kernel in a
 * schedule can be built before any of them runs -- so it parallelizes cleanly
 * when the backend's compile path is itself thread-safe. This runs a caller
 * supplied per-kernel compile function over indices [0, n), serially by default
 * and concurrently when `parallel` is set, collecting the first failure.
 *
 * The caller owns thread-safety of `fn`: pass parallel=true only for a backend
 * whose compile is re-entrant. Default-serial keeps existing backends safe. */

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Compile the @p index-th kernel; return 0 on success, non-zero on failure. */
typedef int (*CMLKernelCompileFn)(int index, void* user);

/** Run @p fn over [0, n): concurrently when @p parallel, else serially. Returns
 *  0 if all succeeded, or the first non-zero status otherwise. */
int cml_compile_kernels(int n, CMLKernelCompileFn fn, void* user, bool parallel);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_PARALLEL_COMPILE_H */
