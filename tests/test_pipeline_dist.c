/* True cross-rank pipeline parallelism over a real (forked) process group.
 *
 * world_size == num_stages == 4. Rank r owns stage r only; activations stream
 * r -> r+1 over TCP. Each stage is a Linear [D,D] whose weight is set to
 * (r+1)*I, so the whole pipeline computes  out = in * (1*2*3*4) = in * 24.
 * The last rank assembles the output and checks it; then a backward pass streams
 * gradients upstream and every rank confirms its stage weight received a grad.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>

#include "cml.h"
#include "nn.h"
#include "autograd/autograd.h"
#include "distributed/distributed.h"
#include "distributed/pipeline_parallel.h"

#define WS 4
#define D  3
#define B  8
#define M  4
#define EPS 2e-3f

/* Linear [D,D] with weight = scale*I, no bias. */
static Module* make_scale_stage(float scale) {
    Linear* lin = nn_linear(D, D, DTYPE_FLOAT32, DEVICE_CPU, false);
    if (!lin) return NULL;
    Parameter** ps = NULL; int np = 0;
    module_collect_parameters((Module*)lin, &ps, &np, true);
    if (np < 1 || !ps || !ps[0] || !ps[0]->tensor) return NULL;
    Tensor* w = ps[0]->tensor;                 /* [D,D] */
    tensor_ensure_executed(w);
    float* wd = (float*)tensor_data_ptr(w);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            wd[i * D + j] = (i == j) ? scale : 0.0f;
    /* ps array intentionally leaked (short-lived test process). */
    return (Module*)lin;
}

static int run_rank(int rank) {
    if (cml_dist_init(DIST_BACKEND_GLOO, WS, rank) != 0) return 1;

    /* Every rank builds all stages (deterministic weights → identical across
     * ranks); each rank only executes its own in the distributed forward. */
    PipelineStage stages[WS];
    for (int s = 0; s < WS; s++) {
        stages[s].module = make_scale_stage((float)(s + 1));
        stages[s].device_id = 0;
        stages[s].device = DEVICE_CPU;
        stages[s].stage_id = s;
        if (!stages[s].module) return 2;
    }

    PipelineConfig cfg = { .num_micro_batches = M, .num_stages = WS, .interleaved = false };
    CMLPipelineParallel* pipe = cml_pipeline_create(stages, WS, &cfg);
    if (!pipe) return 3;

    /* Rank 0 provides the global input [B,D]; other ranks pass NULL. */
    Tensor* input = NULL;
    float in_data[B * D];
    for (int i = 0; i < B * D; i++) in_data[i] = (float)i;
    if (rank == 0) {
        int shape[2] = {B, D};
        TensorConfig tc = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};
        input = tensor_from_data(in_data, shape, 2, &tc);
    }

    int fails = 0;

    Tensor* out = cml_pipeline_dist_forward(pipe, input);

    if (rank == WS - 1) {
        /* out = in * 24 */
        if (!out) { fprintf(stderr, "[rank %d] dist_forward returned NULL on last stage\n", rank); fails |= 1; }
        else {
            tensor_ensure_executed(out);
            const float* od = (const float*)tensor_data_ptr(out);
            for (int i = 0; i < B * D && !(fails & 1); i++) {
                float want = in_data[i] * 24.0f;
                if (fabsf(od[i] - want) > EPS * (1.0f + fabsf(want))) {
                    fprintf(stderr, "[rank %d] out[%d]=%f want %f\n", rank, i, od[i], want);
                    fails |= 1;
                }
            }
        }
    } else if (out != NULL) {
        fprintf(stderr, "[rank %d] non-last rank should return NULL\n", rank); fails |= 2;
    }

    /* Backward: last rank seeds dL/dout = ones; grads stream upstream. */
    Tensor* grad = NULL;
    float ones[B * D];
    for (int i = 0; i < B * D; i++) ones[i] = 1.0f;
    if (rank == WS - 1) {
        int shape[2] = {B, D};
        TensorConfig tc = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};
        grad = tensor_from_data(ones, shape, 2, &tc);
    }
    if (cml_pipeline_dist_backward(pipe, grad) != 0) { fprintf(stderr, "[rank %d] dist_backward failed\n", rank); fails |= 4; }

    /* This rank's stage weight must have received a gradient. */
    Parameter** ps = NULL; int np = 0;
    module_collect_parameters(stages[rank].module, &ps, &np, true);
    if (np < 1 || !ps[0]->tensor->grad) { fprintf(stderr, "[rank %d] stage weight has no grad\n", rank); fails |= 8; }

    cml_dist_barrier();
    if (out) tensor_free(out);
    if (input) tensor_free(input);
    if (grad) tensor_free(grad);
    cml_pipeline_free(pipe);
    cml_dist_destroy();
    return fails;
}

int main(void) {
    setenv("MASTER_ADDR", "127.0.0.1", 1);
    /* Below the ephemeral range (/proc/sys/net/ipv4/ip_local_port_range, from
     * 32768 here): the ranks bind port_base+rank as fixed listeners, so a base
     * inside that range can be taken at any moment by some unrelated outgoing
     * connection on the machine, and the run fails with "Address already in
     * use". Kept distinct from the other distributed tests so a parallel ctest
     * doesn't make them collide with each other. */
    setenv("GLOO_PORT", "29761", 1);

    pid_t pids[WS];
    for (int r = 1; r < WS; r++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return 1; }
        if (pid == 0) { int rc = run_rank(r); _exit(rc == 0 ? 0 : 1); }
        pids[r] = pid;
    }

    int rc0 = run_rank(0);
    int all_ok = (rc0 == 0);
    if (rc0 != 0) fprintf(stderr, "[rank 0] FAILED (mask=%d)\n", rc0);
    for (int r = 1; r < WS; r++) {
        int status = 0;
        waitpid(pids[r], &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            fprintf(stderr, "[rank %d] child FAILED (status=%d)\n", r, status);
            all_ok = 0;
        }
    }

    if (all_ok) { printf("test_pipeline_dist: PASSED (4-stage cross-rank pipeline, fwd+bwd)\n"); return 0; }
    printf("test_pipeline_dist: FAILED\n");
    return 1;
}
