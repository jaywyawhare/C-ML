/* Exercises the InfiniBand transport's RDMA data path via the built-in mock
 * verbs layer (IB_MOCK=1) — a TCP-loopback emulation with no HCA. This validates
 * the fixes that were previously untestable: the ring all-reduce with
 * parity-ordered send/recv, and the MR lkey handling (the mock post_send FAILS
 * the completion if the SGE lkey isn't a real registered-MR key, i.e. it would
 * fail loudly if lkey regressed to 0). Runs a real 4-rank forked process group. */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>

#include "distributed/ib_transport.h"

#define WS 4
#define N  7   /* not divisible by WS → exercises ring chunk clamping */

static int run_rank(int rank) {
    setenv("IB_MOCK", "1", 1);

    CMLIBTransport* ib = cml_ib_create(rank, WS);
    if (!ib) { fprintf(stderr, "[rank %d] cml_ib_create failed\n", rank); return 1; }

    int fails = 0;

    /* Ring all-reduce SUM: rank r contributes (r+1); expect sum_{r}(r+1). */
    float buf[N];
    for (int i = 0; i < N; i++) buf[i] = (float)(rank + 1);
    if (cml_ib_allreduce(ib, buf, sizeof(buf), (int)sizeof(float)) != 0) {
        fprintf(stderr, "[rank %d] cml_ib_allreduce failed\n", rank);
        fails |= 1;
    }
    float expect = (float)(WS * (WS + 1) / 2);   /* 1+2+3+4 = 10 */
    for (int i = 0; i < N && !(fails & 1); i++)
        if (fabsf(buf[i] - expect) > 1e-4f) {
            fprintf(stderr, "[rank %d] allreduce buf[%d]=%f want %f\n", rank, i, buf[i], expect);
            fails |= 1;
        }

    if (cml_ib_barrier(ib) != 0) { fprintf(stderr, "[rank %d] barrier failed\n", rank); fails |= 2; }

    cml_ib_free(ib);
    return fails;
}

int main(void) {
    setenv("IB_MOCK", "1", 1);
    setenv("IB_MOCK_PORT", "39590", 1);

    pid_t pids[WS];
    for (int r = 1; r < WS; r++) {
        pid_t p = fork();
        if (p < 0) { perror("fork"); return 1; }
        if (p == 0) { _exit(run_rank(r) ? 1 : 0); }
        pids[r] = p;
    }

    int rc0 = run_rank(0);
    int ok = (rc0 == 0);
    if (rc0) fprintf(stderr, "[rank 0] FAILED (mask=%d)\n", rc0);
    for (int r = 1; r < WS; r++) {
        int st = 0;
        waitpid(pids[r], &st, 0);
        if (!(WIFEXITED(st) && WEXITSTATUS(st) == 0)) {
            fprintf(stderr, "[rank %d] child FAILED (status=%d)\n", r, st);
            ok = 0;
        }
    }

    if (ok) { printf("test_ib_mock: PASSED (mock-verbs ring all-reduce + barrier, %d ranks)\n", WS); return 0; }
    printf("test_ib_mock: FAILED\n");
    return 1;
}
