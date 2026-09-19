/* Real multi-process distributed collectives: fork() two ranks that talk over
 * the gloo TCP backend on localhost. Every prior distributed test ran with
 * world_size=1 where all collectives are no-ops; this is the first test that
 * exercises the actual cross-rank socket paths (allreduce, broadcast,
 * barrier, ring transfers). */
#include "cml.h"
#include "distributed/distributed.h"
#include "tensor/tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define N 64
#define TEST_PORT "29713"

static Tensor* make_filled(float value) {
    int shape[1]     = {N};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_full(shape, 1, &cfg, value);
    tensor_ensure_executed(t);
    return t;
}

static int check_all(Tensor* t, float expected, const char* what, int rank) {
    const float* d = (const float*)tensor_data_ptr(t);
    if (!d) {
        fprintf(stderr, "[rank %d] %s: no data\n", rank, what);
        return 0;
    }
    for (int i = 0; i < N; i++) {
        if (fabsf(d[i] - expected) > 1e-5f) {
            fprintf(stderr, "[rank %d] %s: elem %d = %f, expected %f\n", rank, what, i, d[i],
                    expected);
            return 0;
        }
    }
    return 1;
}

static int run_rank(int rank, int world) {
    setenv("GLOO_PORT", TEST_PORT, 1);
    setenv("MASTER_ADDR", "127.0.0.1", 1);

    if (cml_dist_init(DIST_BACKEND_GLOO, world, rank) != 0) {
        fprintf(stderr, "[rank %d] cml_dist_init failed\n", rank);
        return 1;
    }
    if (cml_dist_get_world_size() != world || cml_dist_get_rank() != rank) {
        fprintf(stderr, "[rank %d] wrong rank/world after init\n", rank);
        return 1;
    }

    /* allreduce SUM: rank r contributes (r+1) everywhere -> sum = 1+2 = 3 */
    Tensor* t = make_filled((float)(rank + 1));
    if (cml_dist_allreduce(t, DIST_REDUCE_SUM) != 0) {
        fprintf(stderr, "[rank %d] allreduce SUM failed\n", rank);
        return 1;
    }
    if (!check_all(t, 3.0f, "allreduce SUM", rank))
        return 1;

    /* allreduce AVG: rank r contributes r -> avg = (0+1)/2 = 0.5 */
    Tensor* a = make_filled((float)rank);
    if (cml_dist_allreduce(a, DIST_REDUCE_AVG) != 0) {
        fprintf(stderr, "[rank %d] allreduce AVG failed\n", rank);
        return 1;
    }
    if (!check_all(a, 0.5f, "allreduce AVG", rank))
        return 1;

    /* broadcast from rank 0: only rank 0 starts with 42 */
    Tensor* b = make_filled(rank == 0 ? 42.0f : 0.0f);
    if (cml_dist_broadcast(b, 0) != 0) {
        fprintf(stderr, "[rank %d] broadcast failed\n", rank);
        return 1;
    }
    if (!check_all(b, 42.0f, "broadcast", rank))
        return 1;

    if (cml_dist_barrier() != 0) {
        fprintf(stderr, "[rank %d] barrier failed\n", rank);
        return 1;
    }

    cml_dist_destroy();
    return 0;
}

int main(void) {
    const int world = 2;

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }
    if (pid == 0) {
        /* Child = rank 1. _exit avoids flushing the parent's stdio twice. */
        int rc = run_rank(1, world);
        _exit(rc);
    }

    int rc0 = run_rank(0, world);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return 1;
    }
    int rc1 = (WIFEXITED(status)) ? WEXITSTATUS(status) : 1;

    if (rc0 == 0 && rc1 == 0) {
        printf("test_multirank: all multi-process collectives passed (world=2)\n");
        return 0;
    }
    fprintf(stderr, "test_multirank: FAILED (rank0=%d rank1=%d)\n", rc0, rc1);
    return 1;
}
