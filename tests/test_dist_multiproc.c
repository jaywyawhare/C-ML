/* Real multi-process test of the Gloo (TCP) collectives.
 *
 * Unlike the other distributed tests (which run world_size=1 and hit the no-op
 * fast paths), this forks a real process group and drives the sockets:
 * all-reduce (SUM/AVG, count NOT divisible by world_size to exercise the ring
 * chunk clamping), broadcast, barrier, all-gather, and reduce-scatter.
 *
 * Each rank runs in its own process; a rank exits 0 on success, 1 on any failed
 * check. The parent (rank 0) waits for all children and fails if any rank did. */

#include "distributed/distributed.h"
#include "tensor/tensor.h"
#include "alloc/cml_allocator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>

#define WS 4          /* world size */
#define N  7          /* elements per tensor (7 % 4 != 0 → tests chunk clamp) */
#define EPS 1e-4f

/* Build a heap-backed 1-D float tensor (avoids the lazy graph entirely). */
static Tensor make_vec(float* buf, int n) {
    static int shape_store[64]; /* per-process; fine for a sequential test */
    (void)shape_store;
    Tensor t;
    memset(&t, 0, sizeof(t));
    t.data = buf;
    t.numel = (size_t)n;
    t.ndim = 1;
    t.shape = (int*)cml_malloc(sizeof(int));
    t.shape[0] = n;
    t.dtype = DTYPE_FLOAT32;
    t.device = DEVICE_CPU;
    t.owns_data = false;
    return t;
}

static int approx(float a, float b) { return fabsf(a - b) <= EPS; }

/* Returns 0 on success, non-zero on the first failed check. */
static int run_rank(int rank) {
    if (cml_dist_init(DIST_BACKEND_GLOO, WS, rank) != 0) {
        fprintf(stderr, "[rank %d] dist_init failed\n", rank);
        return 1;
    }
    if (cml_dist_get_world_size() != WS) return 2;

    int fails = 0;

    /* ---- all-reduce SUM: rank r contributes (r+1); expect sum_{r}(r+1) ---- */
    {
        float buf[N];
        for (int i = 0; i < N; i++) buf[i] = (float)(rank + 1);
        Tensor t = make_vec(buf, N);
        if (cml_dist_allreduce(&t, DIST_REDUCE_SUM) != 0) { fails |= 4; }
        float expect = (float)(WS * (WS + 1) / 2);   /* 1+2+3+4 = 10 */
        for (int i = 0; i < N; i++)
            if (!approx(buf[i], expect)) { fprintf(stderr, "[rank %d] allreduce SUM buf[%d]=%f want %f\n", rank, i, buf[i], expect); fails |= 4; break; }
        cml_free(t.shape);
    }

    /* ---- all-reduce AVG: expect mean of (r+1) = (WS+1)/2 ---- */
    {
        float buf[N];
        for (int i = 0; i < N; i++) buf[i] = (float)(rank + 1);
        Tensor t = make_vec(buf, N);
        if (cml_dist_allreduce(&t, DIST_REDUCE_AVG) != 0) { fails |= 8; }
        float expect = (float)(WS + 1) / 2.0f;       /* 2.5 */
        for (int i = 0; i < N; i++)
            if (!approx(buf[i], expect)) { fprintf(stderr, "[rank %d] allreduce AVG buf[%d]=%f want %f\n", rank, i, buf[i], expect); fails |= 8; break; }
        cml_free(t.shape);
    }

    /* ---- broadcast from rank 2: only rank 2 has real data ---- */
    {
        float buf[N];
        for (int i = 0; i < N; i++) buf[i] = (rank == 2) ? (float)(100 + i) : -1.0f;
        Tensor t = make_vec(buf, N);
        if (cml_dist_broadcast(&t, 2) != 0) { fails |= 16; }
        for (int i = 0; i < N; i++)
            if (!approx(buf[i], (float)(100 + i))) { fprintf(stderr, "[rank %d] broadcast buf[%d]=%f want %d\n", rank, i, buf[i], 100 + i); fails |= 16; break; }
        cml_free(t.shape);
    }

    /* ---- barrier: must not hang ---- */
    if (cml_dist_barrier() != 0) fails |= 32;

    /* ---- all-gather: rank r contributes r; output[r] must be all-r ---- */
    {
        float in[N];
        for (int i = 0; i < N; i++) in[i] = (float)rank;
        Tensor tin = make_vec(in, N);

        float* obuf[WS];
        Tensor* outv[WS];
        Tensor outs[WS];
        for (int r = 0; r < WS; r++) {
            obuf[r] = (float*)cml_malloc(sizeof(float) * N);
            for (int i = 0; i < N; i++) obuf[r][i] = -1.0f;
            outs[r] = make_vec(obuf[r], N);
            outv[r] = &outs[r];
        }
        if (cml_dist_allgather(outv, &tin) != 0) fails |= 64;
        for (int r = 0; r < WS && !(fails & 64); r++)
            for (int i = 0; i < N; i++)
                if (!approx(obuf[r][i], (float)r)) { fprintf(stderr, "[rank %d] allgather out[%d][%d]=%f want %d\n", rank, r, i, obuf[r][i], r); fails |= 64; break; }
        for (int r = 0; r < WS; r++) { cml_free(obuf[r]); cml_free(outs[r].shape); }
        cml_free(tin.shape);
    }

    cml_dist_barrier();
    cml_dist_destroy();
    return fails;
}

int main(void) {
    /* Uncommon fixed port base, below the ephemeral range (from 32768 here) so
     * an unrelated outgoing connection can't be holding it; SO_REUSEADDR on the
     * listener lets us rebind across quick re-runs. Kept distinct from the other
     * distributed tests so a parallel ctest doesn't make them collide. */
    setenv("MASTER_ADDR", "127.0.0.1", 1);
    setenv("GLOO_PORT", "29731", 1);

    pid_t pids[WS];
    for (int r = 1; r < WS; r++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return 1; }
        if (pid == 0) {
            int rc = run_rank(r);
            _exit(rc == 0 ? 0 : 1);
        }
        pids[r] = pid;
    }

    /* Parent is rank 0. */
    int rc0 = run_rank(0);

    int all_ok = (rc0 == 0);
    if (rc0 != 0) fprintf(stderr, "[rank 0] FAILED (mask=%d)\n", rc0);
    for (int r = 1; r < WS; r++) {
        int status = 0;
        waitpid(pids[r], &status, 0);
        int ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
        if (!ok) { fprintf(stderr, "[rank %d] child FAILED (status=%d)\n", r, status); all_ok = 0; }
    }

    if (all_ok) {
        printf("test_dist_multiproc: PASSED (world_size=%d, all collectives over TCP)\n", WS);
        return 0;
    }
    printf("test_dist_multiproc: FAILED\n");
    return 1;
}
