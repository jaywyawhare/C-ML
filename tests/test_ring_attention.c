/* Real multi-process test of ring (sequence-parallel) attention over Gloo (TCP).
 *
 * Forks a world_size=WS process group on loopback. Each rank holds its local Q/K/V
 * shard for sequence positions [rank*L, rank*L+L); every rank also regenerates the
 * FULL deterministic sequence and computes a single-process FlashAttention reference,
 * then checks that:
 *   - prefill: ring forward (causal) over the sharded KV equals the reference rows for
 *     this rank's query shard, and
 *   - decode: ring decode of one new query over the sharded KV cache equals the
 *     reference attention of that query over the whole (non-causal) sequence.
 *
 * The world_size==1 fallback (which defers to the local flash kernel) is checked in the
 * parent before forking. A rank exits 0 on success, 1 on any failed check. */

#include "distributed/distributed.h"
#include "distributed/ring_attention.h"
#include "nn/llm_ops.h"
#include "tensor/tensor.h"
#include "alloc/cml_allocator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>

#define WS 4 /* world size */
#define L 3  /* tokens per shard */
#define SEQ (WS * L)
#define NUM_HEADS 4
#define NUM_KV_HEADS 2
#define HEAD_DIM 8
#define QDIM (NUM_HEADS * HEAD_DIM)
#define KVDIM (NUM_KV_HEADS * HEAD_DIM)
#define EPS 2e-3f

static float gen_q(int pos, int c) { return sinf(0.10f * pos + 0.03f * c + 1.0f) * 0.5f; }
static float gen_k(int pos, int c) { return cosf(0.07f * pos + 0.05f * c) * 0.5f; }
static float gen_v(int pos, int c) { return sinf(0.05f * pos - 0.02f * c + 0.5f) * 0.5f; }

/**
 * Build a [1, rows, dim] CPU tensor filled by `gen` over positions [pos0, pos0+rows).
 * @return (Tensor*)
 */
static Tensor* build(int pos0, int rows, int dim, float (*gen)(int, int)) {
    float* buf = (float*)cml_malloc((size_t)rows * dim * sizeof(float));
    for (int i = 0; i < rows; i++)
        for (int c = 0; c < dim; c++)
            buf[i * dim + c] = gen(pos0 + i, c);
    int shape[]      = {1, rows, dim};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_from_data(buf, shape, 3, &cfg);
    cml_free(buf);
    return t;
}

static int close_rows(const float* got, const float* ref, int rows, int dim, const char* what,
                      int rank) {
    for (int i = 0; i < rows; i++)
        for (int c = 0; c < dim; c++) {
            float a = got[i * dim + c];
            float b = ref[i * dim + c];
            if (fabsf(a - b) > EPS) {
                fprintf(stderr, "[rank %d] %s mismatch at row %d col %d: %f vs %f\n", rank, what, i,
                        c, a, b);
                return 0;
            }
        }
    return 1;
}

static CMLGQAConfig gqa_cfg(bool causal) {
    CMLGQAConfig c = {.num_heads    = NUM_HEADS,
                      .num_kv_heads = NUM_KV_HEADS,
                      .head_dim     = HEAD_DIM,
                      .scale        = 0.0f,
                      .causal       = causal,
                      .window_size  = 0};
    return c;
}

/**
 * Returns 0 on success, non-zero on the first failed check.
 * @param rank (int)
 * @return (int)
 */
static int run_rank(int rank) {
    if (cml_dist_init(DIST_BACKEND_GLOO, WS, rank) != 0) {
        fprintf(stderr, "[rank %d] dist_init failed\n", rank);
        return 1;
    }

    int fails                     = 0;
    CMLFlashAttentionConfig flash = cml_flash_attention_default_config();

    /* ---- prefill: ring causal attention == reference over the full sequence ---- */
    {
        Tensor* fullQ   = build(0, SEQ, QDIM, gen_q);
        Tensor* fullK   = build(0, SEQ, KVDIM, gen_k);
        Tensor* fullV   = build(0, SEQ, KVDIM, gen_v);
        CMLGQAConfig cc = gqa_cfg(true);
        Tensor* ref     = cml_gqa_flash_forward(fullQ, fullK, fullV, &cc, &flash);

        Tensor* Q   = build(rank * L, L, QDIM, gen_q);
        Tensor* K   = build(rank * L, L, KVDIM, gen_k);
        Tensor* V   = build(rank * L, L, KVDIM, gen_v);
        Tensor* out = cml_ring_attention_forward(Q, K, V, &cc);

        if (!ref || !out) {
            fails |= 1;
        } else {
            const float* rp = (const float*)tensor_data_ptr(ref) + (size_t)rank * L * QDIM;
            const float* op = (const float*)tensor_data_ptr(out);
            if (!close_rows(op, rp, L, QDIM, "prefill", rank))
                fails |= 1;
        }
        tensor_free(fullQ);
        tensor_free(fullK);
        tensor_free(fullV);
        tensor_free(ref);
        tensor_free(Q);
        tensor_free(K);
        tensor_free(V);
        tensor_free(out);
    }

    /* ---- decode: ring decode of one new query == reference attention over all KV ---- */
    {
        Tensor* q       = build(SEQ, 1, QDIM, gen_q); /* the "next" token, same on every rank */
        Tensor* fullK   = build(0, SEQ, KVDIM, gen_k);
        Tensor* fullV   = build(0, SEQ, KVDIM, gen_v);
        CMLGQAConfig cc = gqa_cfg(false);
        Tensor* ref     = cml_gqa_flash_forward(q, fullK, fullV, &cc, &flash);

        Tensor* K   = build(rank * L, L, KVDIM, gen_k);
        Tensor* V   = build(rank * L, L, KVDIM, gen_v);
        Tensor* out = cml_ring_attention_decode(q, K, V, &cc);

        if (!ref || !out) {
            fails |= 2;
        } else {
            const float* rp = (const float*)tensor_data_ptr(ref);
            const float* op = (const float*)tensor_data_ptr(out);
            if (!close_rows(op, rp, 1, QDIM, "decode", rank))
                fails |= 2;
        }
        tensor_free(q);
        tensor_free(fullK);
        tensor_free(fullV);
        tensor_free(ref);
        tensor_free(K);
        tensor_free(V);
        tensor_free(out);
    }

    cml_dist_barrier();
    cml_dist_destroy();
    return fails;
}

/**
 * world_size==1 must equal the local flash kernel (no process group needed).
 * @return (int)
 */
static int check_single_rank(void) {
    Tensor* Q                     = build(0, SEQ, QDIM, gen_q);
    Tensor* K                     = build(0, SEQ, KVDIM, gen_k);
    Tensor* V                     = build(0, SEQ, KVDIM, gen_v);
    CMLGQAConfig cc               = gqa_cfg(true);
    CMLFlashAttentionConfig flash = cml_flash_attention_default_config();
    Tensor* ref                   = cml_gqa_flash_forward(Q, K, V, &cc, &flash);
    Tensor* out                   = cml_ring_attention_forward(Q, K, V, &cc);
    int ok                        = ref && out &&
             close_rows((const float*)tensor_data_ptr(out), (const float*)tensor_data_ptr(ref), SEQ,
                        QDIM, "single-rank", 0);
    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    tensor_free(ref);
    tensor_free(out);
    return ok ? 0 : 1;
}

int main(void) {
    if (check_single_rank() != 0) {
        printf("test_ring_attention: FAILED (world_size==1 fallback)\n");
        return 1;
    }

    /* Distinct port from the other distributed tests so a parallel ctest can't collide. */
    setenv("MASTER_ADDR", "127.0.0.1", 1);
    setenv("GLOO_PORT", "29741", 1);

    pid_t pids[WS];
    for (int r = 1; r < WS; r++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0)
            _exit(run_rank(r) == 0 ? 0 : 1);
        pids[r] = pid;
    }

    int rc0    = run_rank(0);
    int all_ok = (rc0 == 0);
    if (rc0 != 0)
        fprintf(stderr, "[rank 0] FAILED (mask=%d)\n", rc0);
    for (int r = 1; r < WS; r++) {
        int status = 0;
        waitpid(pids[r], &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            fprintf(stderr, "[rank %d] child FAILED (status=%d)\n", r, status);
            all_ok = 0;
        }
    }

    if (all_ok) {
        printf("test_ring_attention: PASSED (world_size=%d, prefill + decode over TCP)\n", WS);
        return 0;
    }
    printf("test_ring_attention: FAILED\n");
    return 1;
}
