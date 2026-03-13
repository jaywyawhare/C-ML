/**
 * @file test_tensor_parallel.c
 * @brief Tests for tensor parallelism (column-parallel, row-parallel, all-reduce)
 *
 * All tests use tp_size=2 with small matrices so the expected results can be
 * computed by hand.  No GPU or multi-process setup is required -- the
 * all-reduce is simulated in-process.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "distributed/tensor_parallel.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-55s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)

#define EPSILON 1e-4f

static bool float_eq(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

/* ===== Helper: reference matmul (C = A @ B^T) ==========================
 * A is [M, K], B is [N, K], C is [M, N]
 * C[i][j] = sum_k A[i][k] * B[j][k]
 */
static void ref_matmul_transposed(const float* A, int M, int K,
                                  const float* B, int N,
                                  float* C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) {
                s += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = s;
        }
    }
}

/* ===== Helper: create a Tensor from a float array ====================== */
static Tensor* make_tensor_2d(const float* data, int rows, int cols) {
    int shape[2] = {rows, cols};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    return tensor_from_data(data, shape, 2, &cfg);
}

static Tensor* make_tensor_1d(const float* data, int len) {
    int shape[1] = {len};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    return tensor_from_data(data, shape, 1, &cfg);
}

/* ========================================================================
 * Test 1: Weight sharding utility
 * ======================================================================== */

static bool test_weight_shard_dim0(void) {
    /*
     * Weight [4, 2]:
     *   row0: 1  2
     *   row1: 3  4
     *   row2: 5  6
     *   row3: 7  8
     *
     * tp_size=2:
     *   rank0 -> rows 0-1: [[1,2],[3,4]]
     *   rank1 -> rows 2-3: [[5,6],[7,8]]
     */
    float w_data[] = {1,2, 3,4, 5,6, 7,8};
    Tensor* weight = make_tensor_2d(w_data, 4, 2);
    if (!weight) return false;

    Tensor* shard0 = cml_tp_shard_weight(weight, 0, 2, 0);
    Tensor* shard1 = cml_tp_shard_weight(weight, 0, 2, 1);
    if (!shard0 || !shard1) {
        tensor_free(weight);
        if (shard0) tensor_free(shard0);
        if (shard1) tensor_free(shard1);
        return false;
    }

    /* Check shapes */
    if (shard0->shape[0] != 2 || shard0->shape[1] != 2) { tensor_free(weight); tensor_free(shard0); tensor_free(shard1); return false; }
    if (shard1->shape[0] != 2 || shard1->shape[1] != 2) { tensor_free(weight); tensor_free(shard0); tensor_free(shard1); return false; }

    tensor_ensure_executed(shard0);
    tensor_ensure_executed(shard1);
    const float* s0 = (const float*)tensor_data_ptr(shard0);
    const float* s1 = (const float*)tensor_data_ptr(shard1);

    bool ok = float_eq(s0[0], 1) && float_eq(s0[1], 2) &&
              float_eq(s0[2], 3) && float_eq(s0[3], 4) &&
              float_eq(s1[0], 5) && float_eq(s1[1], 6) &&
              float_eq(s1[2], 7) && float_eq(s1[3], 8);

    tensor_free(weight);
    tensor_free(shard0);
    tensor_free(shard1);
    return ok;
}

static bool test_weight_shard_dim1(void) {
    /*
     * Weight [2, 4]:
     *   row0: 1  2  3  4
     *   row1: 5  6  7  8
     *
     * tp_size=2, dim=1:
     *   rank0 -> cols 0-1: [[1,2],[5,6]]
     *   rank1 -> cols 2-3: [[3,4],[7,8]]
     */
    float w_data[] = {1,2,3,4, 5,6,7,8};
    Tensor* weight = make_tensor_2d(w_data, 2, 4);
    if (!weight) return false;

    Tensor* shard0 = cml_tp_shard_weight(weight, 1, 2, 0);
    Tensor* shard1 = cml_tp_shard_weight(weight, 1, 2, 1);
    if (!shard0 || !shard1) {
        tensor_free(weight);
        if (shard0) tensor_free(shard0);
        if (shard1) tensor_free(shard1);
        return false;
    }

    if (shard0->shape[0] != 2 || shard0->shape[1] != 2) { tensor_free(weight); tensor_free(shard0); tensor_free(shard1); return false; }
    if (shard1->shape[0] != 2 || shard1->shape[1] != 2) { tensor_free(weight); tensor_free(shard0); tensor_free(shard1); return false; }

    tensor_ensure_executed(shard0);
    tensor_ensure_executed(shard1);
    const float* s0 = (const float*)tensor_data_ptr(shard0);
    const float* s1 = (const float*)tensor_data_ptr(shard1);

    bool ok = float_eq(s0[0], 1) && float_eq(s0[1], 2) &&
              float_eq(s0[2], 5) && float_eq(s0[3], 6) &&
              float_eq(s1[0], 3) && float_eq(s1[1], 4) &&
              float_eq(s1[2], 7) && float_eq(s1[3], 8);

    tensor_free(weight);
    tensor_free(shard0);
    tensor_free(shard1);
    return ok;
}

/* ========================================================================
 * Test 2: Column-parallel forward
 *
 * Full weight [4, 2] (out=4, in=2), input [1, 2].
 * Column-parallel shards output dim -> each rank gets [2, 2].
 * Full output = input @ W^T -> [1, 4].
 * Rank 0 output = first 2 cols of full output.
 * Rank 1 output = last  2 cols of full output.
 * Concatenating rank outputs should equal the full matmul.
 * ======================================================================== */

static bool test_column_parallel_forward(void) {
    float w_data[] = {
        1, 0,    /* row 0 */
        0, 1,    /* row 1 */
        2, 0,    /* row 2 */
        0, 2     /* row 3 */
    };
    float x_data[] = {3, 4};  /* [1, 2] */

    Tensor* weight = make_tensor_2d(w_data, 4, 2);
    Tensor* input  = make_tensor_2d(x_data, 1, 2);
    if (!weight || !input) return false;

    /* Expected full output = [3, 4, 6, 8] */

    CMLColumnParallelLinear* cp0 = cml_column_parallel_create(weight, NULL, 2, 0);
    CMLColumnParallelLinear* cp1 = cml_column_parallel_create(weight, NULL, 2, 1);
    if (!cp0 || !cp1) {
        tensor_free(weight); tensor_free(input);
        if (cp0) cml_column_parallel_free(cp0);
        if (cp1) cml_column_parallel_free(cp1);
        return false;
    }

    /* Each rank's weight should be [2, 2] */
    if (cp0->weight->shape[0] != 2 || cp0->weight->shape[1] != 2) {
        tensor_free(weight); tensor_free(input);
        cml_column_parallel_free(cp0); cml_column_parallel_free(cp1);
        return false;
    }

    Tensor* out0 = cml_column_parallel_forward(cp0, input);
    Tensor* out1 = cml_column_parallel_forward(cp1, input);
    if (!out0 || !out1) {
        tensor_free(weight); tensor_free(input);
        cml_column_parallel_free(cp0); cml_column_parallel_free(cp1);
        if (out0) tensor_free(out0);
        if (out1) tensor_free(out1);
        return false;
    }

    /* out0 shape should be [1, 2], out1 shape should be [1, 2] */
    if (out0->shape[0] != 1 || out0->shape[1] != 2 ||
        out1->shape[0] != 1 || out1->shape[1] != 2) {
        tensor_free(weight); tensor_free(input);
        cml_column_parallel_free(cp0); cml_column_parallel_free(cp1);
        tensor_free(out0); tensor_free(out1);
        return false;
    }

    tensor_ensure_executed(out0);
    tensor_ensure_executed(out1);
    const float* r0 = (const float*)tensor_data_ptr(out0);
    const float* r1 = (const float*)tensor_data_ptr(out1);

    /* Rank 0: weight rows 0,1 -> [1,0;0,1] -> out = [3, 4] */
    /* Rank 1: weight rows 2,3 -> [2,0;0,2] -> out = [6, 8] */
    bool ok = float_eq(r0[0], 3.0f) && float_eq(r0[1], 4.0f) &&
              float_eq(r1[0], 6.0f) && float_eq(r1[1], 8.0f);

    tensor_free(weight);
    tensor_free(input);
    cml_column_parallel_free(cp0);
    cml_column_parallel_free(cp1);
    tensor_free(out0);
    tensor_free(out1);
    return ok;
}

/* ========================================================================
 * Test 3: Row-parallel forward + all-reduce
 *
 * Full weight [2, 4] (out=2, in=4), input [1, 4].
 * Row-parallel shards input dim -> each rank gets [2, 2].
 * Each rank computes a partial [1, 2], then all-reduce sum gives full output.
 * ======================================================================== */

static bool test_row_parallel_forward_allreduce(void) {
    float w_data[] = {
        1, 2, 3, 4,  /* row 0 of W */
        5, 6, 7, 8   /* row 1 of W */
    };
    float x_data[] = {1, 1, 1, 1};  /* [1, 4] */

    Tensor* weight = make_tensor_2d(w_data, 2, 4);
    Tensor* input  = make_tensor_2d(x_data, 1, 4);
    if (!weight || !input) return false;

    /* Full output = input @ W^T = [1+2+3+4, 5+6+7+8] = [10, 26] */

    CMLRowParallelLinear* rp0 = cml_row_parallel_create(weight, NULL, 2, 0);
    CMLRowParallelLinear* rp1 = cml_row_parallel_create(weight, NULL, 2, 1);
    if (!rp0 || !rp1) {
        tensor_free(weight); tensor_free(input);
        if (rp0) cml_row_parallel_free(rp0);
        if (rp1) cml_row_parallel_free(rp1);
        return false;
    }

    /* Rank 0 gets cols 0-1: W_shard0 = [[1,2],[5,6]], input_shard0 = [1,1] */
    /* Rank 1 gets cols 2-3: W_shard1 = [[3,4],[7,8]], input_shard1 = [1,1] */

    /* Create input shards for each rank */
    float x0_data[] = {1, 1};  /* first 2 features */
    float x1_data[] = {1, 1};  /* last 2 features */
    Tensor* in0 = make_tensor_2d(x0_data, 1, 2);
    Tensor* in1 = make_tensor_2d(x1_data, 1, 2);
    if (!in0 || !in1) {
        tensor_free(weight); tensor_free(input);
        cml_row_parallel_free(rp0); cml_row_parallel_free(rp1);
        if (in0) tensor_free(in0);
        if (in1) tensor_free(in1);
        return false;
    }

    Tensor* out0 = cml_row_parallel_forward(rp0, in0);
    Tensor* out1 = cml_row_parallel_forward(rp1, in1);
    if (!out0 || !out1) {
        tensor_free(weight); tensor_free(input); tensor_free(in0); tensor_free(in1);
        cml_row_parallel_free(rp0); cml_row_parallel_free(rp1);
        if (out0) tensor_free(out0);
        if (out1) tensor_free(out1);
        return false;
    }

    /* Rank 0 partial: [1*1+1*2, 1*5+1*6] = [3, 11] */
    /* Rank 1 partial: [1*3+1*4, 1*7+1*8] = [7, 15] */

    tensor_ensure_executed(out0);
    tensor_ensure_executed(out1);
    const float* p0 = (const float*)tensor_data_ptr(out0);
    const float* p1 = (const float*)tensor_data_ptr(out1);

    bool partial_ok = float_eq(p0[0], 3.0f) && float_eq(p0[1], 11.0f) &&
                      float_eq(p1[0], 7.0f) && float_eq(p1[1], 15.0f);
    if (!partial_ok) {
        printf("\n    Partial mismatch: rank0=[%.1f,%.1f] rank1=[%.1f,%.1f]\n",
               p0[0], p0[1], p1[0], p1[1]);
    }

    /* All-reduce sum: [3+7, 11+15] = [10, 26] */
    Tensor* partials[] = {out0, out1};
    Tensor* reduced = cml_tp_all_reduce_sum(partials, 2);
    if (!reduced) {
        tensor_free(weight); tensor_free(input); tensor_free(in0); tensor_free(in1);
        cml_row_parallel_free(rp0); cml_row_parallel_free(rp1);
        tensor_free(out0); tensor_free(out1);
        return false;
    }

    tensor_ensure_executed(reduced);
    const float* rdata = (const float*)tensor_data_ptr(reduced);
    bool reduce_ok = float_eq(rdata[0], 10.0f) && float_eq(rdata[1], 26.0f);
    if (!reduce_ok) {
        printf("\n    Reduced mismatch: [%.1f, %.1f] expected [10.0, 26.0]\n",
               rdata[0], rdata[1]);
    }

    tensor_free(weight);
    tensor_free(input);
    tensor_free(in0);
    tensor_free(in1);
    cml_row_parallel_free(rp0);
    cml_row_parallel_free(rp1);
    tensor_free(out0);
    tensor_free(out1);
    tensor_free(reduced);
    return partial_ok && reduce_ok;
}

/* ========================================================================
 * Test 4: Full TP simulation (column -> row -> all-reduce)
 *
 * Simulate a two-layer MLP with tensor parallelism:
 *   hidden = input @ W1^T   (column-parallel, split output)
 *   output = hidden @ W2^T  (row-parallel, split input, all-reduce)
 *
 * Verify the result matches the full (non-parallel) matmul chain.
 * ======================================================================== */

static bool test_full_tp_simulation(void) {
    /* W1: [4, 2] (out=4, in=2) -- column-parallel splits output to 2 per rank */
    float w1_data[] = {
        1, 0,
        0, 1,
        1, 1,
        2, -1
    };

    /* W2: [2, 4] (out=2, in=4) -- row-parallel splits input to 2 per rank */
    float w2_data[] = {
        1, 0, 1, 0,
        0, 1, 0, 1
    };

    /* Input: [1, 2] */
    float x_data[] = {2, 3};

    Tensor* W1 = make_tensor_2d(w1_data, 4, 2);
    Tensor* W2 = make_tensor_2d(w2_data, 2, 4);
    Tensor* input = make_tensor_2d(x_data, 1, 2);
    if (!W1 || !W2 || !input) {
        if (W1) tensor_free(W1);
        if (W2) tensor_free(W2);
        if (input) tensor_free(input);
        return false;
    }

    /* ---- Reference: full matmul ---- */
    /* hidden = input @ W1^T = [2,3] @ [[1,0],[0,1],[1,1],[2,-1]]^T
     *        = [2*1+3*0, 2*0+3*1, 2*1+3*1, 2*2+3*(-1)]
     *        = [2, 3, 5, 1]
     */
    float ref_hidden[4];
    tensor_ensure_executed(W1);
    tensor_ensure_executed(input);
    ref_matmul_transposed(x_data, 1, 2, w1_data, 4, ref_hidden);

    /* output = hidden @ W2^T = [2,3,5,1] @ [[1,0,1,0],[0,1,0,1]]^T
     *        = [2*1+3*0+5*1+1*0, 2*0+3*1+5*0+1*1]
     *        = [7, 4]
     */
    float ref_output[2];
    ref_matmul_transposed(ref_hidden, 1, 4, w2_data, 2, ref_output);

    /* ---- Tensor-parallel: rank 0 and rank 1 ---- */

    /* Column-parallel on W1 */
    CMLColumnParallelLinear* cp0 = cml_column_parallel_create(W1, NULL, 2, 0);
    CMLColumnParallelLinear* cp1 = cml_column_parallel_create(W1, NULL, 2, 1);
    if (!cp0 || !cp1) {
        tensor_free(W1); tensor_free(W2); tensor_free(input);
        if (cp0) cml_column_parallel_free(cp0);
        if (cp1) cml_column_parallel_free(cp1);
        return false;
    }

    /* Column-parallel forward: each rank gets [1, 2] */
    Tensor* h0 = cml_column_parallel_forward(cp0, input);
    Tensor* h1 = cml_column_parallel_forward(cp1, input);
    if (!h0 || !h1) {
        tensor_free(W1); tensor_free(W2); tensor_free(input);
        cml_column_parallel_free(cp0); cml_column_parallel_free(cp1);
        if (h0) tensor_free(h0);
        if (h1) tensor_free(h1);
        return false;
    }

    /* Row-parallel on W2: splits input dim (4) into 2 per rank */
    CMLRowParallelLinear* rp0 = cml_row_parallel_create(W2, NULL, 2, 0);
    CMLRowParallelLinear* rp1 = cml_row_parallel_create(W2, NULL, 2, 1);
    if (!rp0 || !rp1) {
        tensor_free(W1); tensor_free(W2); tensor_free(input);
        cml_column_parallel_free(cp0); cml_column_parallel_free(cp1);
        tensor_free(h0); tensor_free(h1);
        if (rp0) cml_row_parallel_free(rp0);
        if (rp1) cml_row_parallel_free(rp1);
        return false;
    }

    /* Row-parallel forward: each rank's hidden [1, 2] feeds into its W2 shard */
    Tensor* o0 = cml_row_parallel_forward(rp0, h0);
    Tensor* o1 = cml_row_parallel_forward(rp1, h1);
    if (!o0 || !o1) {
        tensor_free(W1); tensor_free(W2); tensor_free(input);
        cml_column_parallel_free(cp0); cml_column_parallel_free(cp1);
        tensor_free(h0); tensor_free(h1);
        cml_row_parallel_free(rp0); cml_row_parallel_free(rp1);
        if (o0) tensor_free(o0);
        if (o1) tensor_free(o1);
        return false;
    }

    /* All-reduce sum the two partial outputs */
    Tensor* partials[] = {o0, o1};
    Tensor* final_output = cml_tp_all_reduce_sum(partials, 2);
    if (!final_output) {
        tensor_free(W1); tensor_free(W2); tensor_free(input);
        cml_column_parallel_free(cp0); cml_column_parallel_free(cp1);
        tensor_free(h0); tensor_free(h1);
        cml_row_parallel_free(rp0); cml_row_parallel_free(rp1);
        tensor_free(o0); tensor_free(o1);
        return false;
    }

    tensor_ensure_executed(final_output);
    const float* fdata = (const float*)tensor_data_ptr(final_output);

    bool ok = float_eq(fdata[0], ref_output[0]) && float_eq(fdata[1], ref_output[1]);
    if (!ok) {
        printf("\n    Full TP mismatch: got [%.2f, %.2f] expected [%.2f, %.2f]\n",
               fdata[0], fdata[1], ref_output[0], ref_output[1]);
    }

    tensor_free(W1);
    tensor_free(W2);
    tensor_free(input);
    cml_column_parallel_free(cp0);
    cml_column_parallel_free(cp1);
    tensor_free(h0);
    tensor_free(h1);
    cml_row_parallel_free(rp0);
    cml_row_parallel_free(rp1);
    tensor_free(o0);
    tensor_free(o1);
    tensor_free(final_output);
    return ok;
}

/* ========================================================================
 * Test 5: Column-parallel with bias
 * ======================================================================== */

static bool test_column_parallel_with_bias(void) {
    float w_data[] = {
        1, 0,
        0, 1,
        2, 0,
        0, 2
    };
    float b_data[] = {10, 20, 30, 40};
    float x_data[] = {1, 1};

    Tensor* weight = make_tensor_2d(w_data, 4, 2);
    Tensor* bias   = make_tensor_1d(b_data, 4);
    Tensor* input  = make_tensor_2d(x_data, 1, 2);
    if (!weight || !bias || !input) {
        if (weight) tensor_free(weight);
        if (bias) tensor_free(bias);
        if (input) tensor_free(input);
        return false;
    }

    CMLColumnParallelLinear* cp0 = cml_column_parallel_create(weight, bias, 2, 0);
    CMLColumnParallelLinear* cp1 = cml_column_parallel_create(weight, bias, 2, 1);
    if (!cp0 || !cp1) {
        tensor_free(weight); tensor_free(bias); tensor_free(input);
        if (cp0) cml_column_parallel_free(cp0);
        if (cp1) cml_column_parallel_free(cp1);
        return false;
    }

    Tensor* out0 = cml_column_parallel_forward(cp0, input);
    Tensor* out1 = cml_column_parallel_forward(cp1, input);
    if (!out0 || !out1) {
        tensor_free(weight); tensor_free(bias); tensor_free(input);
        cml_column_parallel_free(cp0); cml_column_parallel_free(cp1);
        if (out0) tensor_free(out0);
        if (out1) tensor_free(out1);
        return false;
    }

    tensor_ensure_executed(out0);
    tensor_ensure_executed(out1);
    const float* r0 = (const float*)tensor_data_ptr(out0);
    const float* r1 = (const float*)tensor_data_ptr(out1);

    /* Rank 0: W=[[1,0],[0,1]], b=[10,20], input=[1,1]
     * matmul: [1, 1] + [10, 20] = [11, 21] */
    /* Rank 1: W=[[2,0],[0,2]], b=[30,40], input=[1,1]
     * matmul: [2, 2] + [30, 40] = [32, 42] */
    bool ok = float_eq(r0[0], 11.0f) && float_eq(r0[1], 21.0f) &&
              float_eq(r1[0], 32.0f) && float_eq(r1[1], 42.0f);

    tensor_free(weight);
    tensor_free(bias);
    tensor_free(input);
    cml_column_parallel_free(cp0);
    cml_column_parallel_free(cp1);
    tensor_free(out0);
    tensor_free(out1);
    return ok;
}

/* ========================================================================
 * Test 6: All-reduce sum correctness
 * ======================================================================== */

static bool test_all_reduce_sum(void) {
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {10, 20, 30, 40, 50, 60};
    float c_data[] = {100, 200, 300, 400, 500, 600};

    Tensor* a = make_tensor_2d(a_data, 2, 3);
    Tensor* b = make_tensor_2d(b_data, 2, 3);
    Tensor* c = make_tensor_2d(c_data, 2, 3);
    if (!a || !b || !c) {
        if (a) tensor_free(a);
        if (b) tensor_free(b);
        if (c) tensor_free(c);
        return false;
    }

    Tensor* partials[] = {a, b, c};
    Tensor* result = cml_tp_all_reduce_sum(partials, 3);
    if (!result) {
        tensor_free(a); tensor_free(b); tensor_free(c);
        return false;
    }

    tensor_ensure_executed(result);
    const float* rdata = (const float*)tensor_data_ptr(result);

    bool ok = float_eq(rdata[0], 111.0f) && float_eq(rdata[1], 222.0f) &&
              float_eq(rdata[2], 333.0f) && float_eq(rdata[3], 444.0f) &&
              float_eq(rdata[4], 555.0f) && float_eq(rdata[5], 666.0f);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(result);
    return ok;
}

/* ========================================================================
 * Test 7: Larger batch size
 * ======================================================================== */

static bool test_column_parallel_batch(void) {
    /* Weight [4, 2], batch of 3 inputs [3, 2] */
    float w_data[] = {1,0, 0,1, 1,1, -1,1};
    float x_data[] = {
        1, 0,
        0, 1,
        2, 3
    };

    Tensor* weight = make_tensor_2d(w_data, 4, 2);
    Tensor* input  = make_tensor_2d(x_data, 3, 2);
    if (!weight || !input) {
        if (weight) tensor_free(weight);
        if (input) tensor_free(input);
        return false;
    }

    /* Full output = input @ W^T:
     * [1,0] @ [[1,0],[0,1],[1,1],[-1,1]]^T = [1, 0, 1, -1]
     * [0,1] @ ...                           = [0, 1, 1,  1]
     * [2,3] @ ...                           = [2, 3, 5,  1]
     */

    CMLColumnParallelLinear* cp0 = cml_column_parallel_create(weight, NULL, 2, 0);
    CMLColumnParallelLinear* cp1 = cml_column_parallel_create(weight, NULL, 2, 1);
    if (!cp0 || !cp1) {
        tensor_free(weight); tensor_free(input);
        if (cp0) cml_column_parallel_free(cp0);
        if (cp1) cml_column_parallel_free(cp1);
        return false;
    }

    Tensor* out0 = cml_column_parallel_forward(cp0, input);
    Tensor* out1 = cml_column_parallel_forward(cp1, input);
    if (!out0 || !out1) {
        tensor_free(weight); tensor_free(input);
        cml_column_parallel_free(cp0); cml_column_parallel_free(cp1);
        if (out0) tensor_free(out0);
        if (out1) tensor_free(out1);
        return false;
    }

    /* out0 [3,2]: rank0 gets W rows 0,1 -> [[1,0],[0,1]]
     * [1,0], [0,1], [2,3] */
    /* out1 [3,2]: rank1 gets W rows 2,3 -> [[1,1],[-1,1]]
     * [1,-1], [1,1], [5,1] */

    if (out0->shape[0] != 3 || out0->shape[1] != 2 ||
        out1->shape[0] != 3 || out1->shape[1] != 2) {
        tensor_free(weight); tensor_free(input);
        cml_column_parallel_free(cp0); cml_column_parallel_free(cp1);
        tensor_free(out0); tensor_free(out1);
        return false;
    }

    tensor_ensure_executed(out0);
    tensor_ensure_executed(out1);
    const float* r0 = (const float*)tensor_data_ptr(out0);
    const float* r1 = (const float*)tensor_data_ptr(out1);

    bool ok = /* row 0 */
              float_eq(r0[0], 1.0f) && float_eq(r0[1], 0.0f) &&
              float_eq(r1[0], 1.0f) && float_eq(r1[1], -1.0f) &&
              /* row 1 */
              float_eq(r0[2], 0.0f) && float_eq(r0[3], 1.0f) &&
              float_eq(r1[2], 1.0f) && float_eq(r1[3], 1.0f) &&
              /* row 2 */
              float_eq(r0[4], 2.0f) && float_eq(r0[5], 3.0f) &&
              float_eq(r1[4], 5.0f) && float_eq(r1[5], 1.0f);

    tensor_free(weight);
    tensor_free(input);
    cml_column_parallel_free(cp0);
    cml_column_parallel_free(cp1);
    tensor_free(out0);
    tensor_free(out1);
    return ok;
}

/* ========================================================================
 * main
 * ======================================================================== */

int main(void) {
    printf("=== Tensor Parallel Tests ===\n\n");

    printf("Weight sharding:\n");
    TEST(weight_shard_dim0);
    TEST(weight_shard_dim1);

    printf("\nColumn-parallel:\n");
    TEST(column_parallel_forward);
    TEST(column_parallel_with_bias);
    TEST(column_parallel_batch);

    printf("\nRow-parallel + all-reduce:\n");
    TEST(row_parallel_forward_allreduce);

    printf("\nAll-reduce:\n");
    TEST(all_reduce_sum);

    printf("\nFull TP simulation:\n");
    TEST(full_tp_simulation);

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
