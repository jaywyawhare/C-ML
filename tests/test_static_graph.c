/* Zero-rebuild static graph: build a graph once, then re-run it on new input /
 * weight buffers via cml_ir_reexecute — no per-iteration rebuild (constant node
 * count), results identical to a fresh dynamic rebuild.
 *
 *  - inference : swap the input buffer, re-run, compare to a dynamic rebuild.
 *  - training  : re-run fwd+bwd each step + in-place SGD; loss converges, and it
 *                stays correct under the default TinyJit (cml_ir_reexecute
 *                bypasses TinyJit's replay). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "test_harness.h"

static int node_count(CMLGraph_t ir) {
    int n = 0;
    for (struct IRNode* p = ir->head; p; p = p->next)
        n++;
    return n;
}

static float W1[16], B1[4], W2[16], B2[4];
static void init_weights(void) {
    for (int i = 0; i < 16; i++) {
        W1[i] = 0.1f * (i % 5 - 2);
        W2[i] = 0.1f * (i % 7 - 3);
    }
    for (int i = 0; i < 4; i++) {
        B1[i] = 0.05f * i;
        B2[i] = -0.1f * i;
    }
}
/* out = relu(X@W1 + B1) @ W2 + B2, built around the given input tensor X */
static Tensor* build_mlp(Tensor* X) {
    Tensor *w1 = cml_tensor_2d(W1, 4, 4), *b1 = cml_tensor_1d(B1, 4);
    Tensor *w2 = cml_tensor_2d(W2, 4, 4), *b2 = cml_tensor_1d(B2, 4);
    Tensor* h = cml_relu(cml_add(cml_matmul(X, w1), b1));
    return cml_add(cml_matmul(h, w2), b2);
}

static void test_static_inference(void) {
    printf("Test: static inference graph (build once, swap input, zero rebuild)\n");
    init_weights();
    float inA[16], inB[16];
    for (int i = 0; i < 16; i++) {
        inA[i] = 0.1f * i - 0.5f;
        inB[i] = 0.2f * (i % 6) - 0.3f;
    }

    cml_reset_ir_context();
    CMLGraph_t ir = cml_ir_get_or_create_context();
    float xbuf[16];
    memcpy(xbuf, inA, sizeof inA);
    Tensor* X   = cml_tensor_2d(xbuf, 4, 4);
    Tensor* out = build_mlp(X);
    float outA_s[4], outB_s[4], outA2_s[4];
    memcpy(outA_s, tensor_data_ptr(out), sizeof outA_s);
    int n1 = node_count(ir);
    memcpy(X->data, inB, sizeof inB);
    cml_ir_reexecute(ir);
    memcpy(outB_s, tensor_data_ptr(out), sizeof outB_s);
    int n2 = node_count(ir);
    memcpy(X->data, inA, sizeof inA);
    cml_ir_reexecute(ir);
    memcpy(outA2_s, tensor_data_ptr(out), sizeof outA2_s);
    int n3 = node_count(ir);

    cml_reset_ir_context();
    float xb2[16];
    memcpy(xb2, inA, sizeof inA);
    float outA_d[4];
    memcpy(outA_d, tensor_data_ptr(build_mlp(cml_tensor_2d(xb2, 4, 4))), sizeof outA_d);
    cml_reset_ir_context();
    float xb3[16];
    memcpy(xb3, inB, sizeof inB);
    float outB_d[4];
    memcpy(outB_d, tensor_data_ptr(build_mlp(cml_tensor_2d(xb3, 4, 4))), sizeof outB_d);

    CHECK("node count constant across re-executions (zero rebuild)", n1 == n2 && n2 == n3);
    float eA = 0, eB = 0, eA2 = 0;
    for (int i = 0; i < 4; i++) {
        eA += fabsf(outA_s[i] - outA_d[i]);
        eB += fabsf(outB_s[i] - outB_d[i]);
        eA2 += fabsf(outA2_s[i] - outA_d[i]);
    }
    CHECK("static output A matches dynamic rebuild", eA < 1e-5f);
    CHECK("static output B matches dynamic rebuild", eB < 1e-5f);
    CHECK("re-run of A reproduces A (no state leakage)", eA2 < 1e-5f);
    cml_reset_ir_context();
}

static void test_static_training(void) {
    printf("Test: static training step (build once, re-run fwd+bwd + in-place SGD)\n");
    int D = 4;
    float xb[16], yb[16], w[16], bb[4];
    for (int i = 0; i < 16; i++) {
        xb[i] = 0.1f * (i % 5) - 0.2f;
        yb[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < 16; i++)
        w[i] = 0.01f * (i % 3 - 1);
    for (int i = 0; i < 4; i++)
        bb[i] = 0.0f;

    cml_reset_ir_context();
    CMLGraph_t ir = cml_ir_get_or_create_context();
    Tensor* X     = cml_tensor_2d(xb, 4, D);
    Tensor* Y     = cml_tensor_2d(yb, 4, D);
    Tensor* W     = cml_tensor_2d(w, D, D);
    cml_set_requires_grad(W, true);
    Tensor* Bs = cml_tensor_1d(bb, D);
    cml_set_requires_grad(Bs, true);
    Tensor* pred = cml_add(cml_matmul(X, W), Bs);
    Tensor* d    = cml_sub(pred, Y);
    Tensor* loss = cml_sum(cml_mul(d, d), -1, false);

    float lr = 0.05f, loss0 = 0, lossN = 0;
    int n_after_first = 0, n_last = 0;
    for (int it = 0; it < 50; it++) {
        if (it == 0)
            tensor_backward(loss, NULL, false, false);
        else
            cml_ir_reexecute(ir);
        float l = ((float*)tensor_data_ptr(loss))[0];
        if (it == 0) {
            loss0 = l;
        }
        if (it == 1) {
            n_after_first = node_count(ir);
        }
        if (it == 49) {
            lossN  = l;
            n_last = node_count(ir);
        }
        float* gw = W->grad ? (float*)W->grad->data : NULL;
        float* gb = Bs->grad ? (float*)Bs->grad->data : NULL;
        if (gw)
            for (int i = 0; i < 16; i++)
                ((float*)W->data)[i] -= lr * gw[i];
        if (gb)
            for (int i = 0; i < 4; i++)
                ((float*)Bs->data)[i] -= lr * gb[i];
    }
    printf("    loss %.4f -> %.4f, nodes %d -> %d\n", loss0, lossN, n_after_first, n_last);
    CHECK("training node count constant (zero rebuild)", n_after_first == n_last);
    CHECK("loss decreases across static training steps", lossN < loss0 * 0.25f);
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("=== Static graph (zero-rebuild) tests ===\n");
    test_static_inference();
    test_static_training();
    return TEST_SUMMARY();
}
