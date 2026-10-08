#include "ops/ir/parallel_compile.h"
#include "backend/threadpool.h"

#include <stdatomic.h>
#include <stddef.h>

typedef struct {
    CMLKernelCompileFn fn;
    void* user;
    atomic_int status; /* first non-zero compile status seen */
} CompileBatch;

/** Thread-pool worker: compile the index range [start, end). */
static void compile_range(void* data, size_t start, size_t end) {
    CompileBatch* b = (CompileBatch*)data;
    for (size_t i = start; i < end; i++) {
        int rc = b->fn((int)i, b->user);
        if (rc != 0) {
            int expected = 0;
            /* Keep the first failure; ignore later ones. */
            atomic_compare_exchange_strong(&b->status, &expected, rc);
        }
    }
}

int cml_compile_kernels(int n, CMLKernelCompileFn fn, void* user, bool parallel) {
    if (n <= 0 || !fn)
        return 0;

    CompileBatch batch = {fn, user, 0};

    if (parallel && n > 1) {
        threadpool_parallel_for(NULL, compile_range, &batch, (size_t)n);
    } else {
        compile_range(&batch, 0, (size_t)n);
    }
    return atomic_load(&batch.status);
}
