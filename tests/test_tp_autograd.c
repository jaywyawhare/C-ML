/* Tensor-parallel autograd: verifies that a column-parallel linear layer now
 * produces a gradient for its (sharded) weight after backward.
 *
 * Before the fix, the TP forward did a raw matmul into a fresh buffer with no
 * autograd edge, so tensor_backward produced NO weight gradient — TP layers
 * were inference-only. This test would have failed (grad == NULL). */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"
#include "distributed/tensor_parallel.h"
#include "autograd/autograd.h"

#define EPS 1e-4f

static Tensor* mk2d(const float* d, int r, int c) {
    int shape[2]     = {r, c};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    return tensor_from_data(d, shape, 2, &cfg);
}

int main(void) {
    /* Full weight [4,2]; tp_size=1 so the shard is the whole weight (out=4). */
    float w_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float x_data[2] = {0.5f, -1.5f};

    Tensor* weight = mk2d(w_data, 4, 2);
    Tensor* input  = mk2d(x_data, 1, 2);

    CMLColumnParallelLinear* cp = cml_column_parallel_create(weight, NULL, 1, 0);
    if (!cp) {
        printf("test_tp_autograd: create failed\n[FAIL]\n");
        return 1;
    }

    /* The sharded weight must be a leaf that accumulates gradients. */
    tensor_set_requires_grad(cp->weight, true);

    Tensor* out = cml_column_parallel_forward(cp, input); /* [1,4] */
    if (!out) {
        printf("test_tp_autograd: forward returned NULL\n[FAIL]\n");
        return 1;
    }

    /* Forward value check: out[0,j] = sum_k x[k]*W[j,k]. */
    tensor_ensure_executed(out);
    const float* od = (const float*)tensor_data_ptr(out);
    int ok          = (od != NULL);
    for (int j = 0; j < 4 && ok; j++) {
        float ref = x_data[0] * w_data[j * 2 + 0] + x_data[1] * w_data[j * 2 + 1];
        if (fabsf(od[j] - ref) > EPS) {
            printf("forward mismatch out[%d]=%f want %f\n", j, od[j], ref);
            ok = 0;
        }
    }

    /* loss = sum over the 4 outputs (batch=1) → [1]; dL/dW[j,k] = x[k]. */
    Tensor* loss = tensor_sum(out, 1, false);
    tensor_ensure_executed(loss);
    cml_backward(loss, NULL, false, false);

    Tensor* g = cp->weight->grad;
    if (!g) {
        printf("test_tp_autograd: weight grad is NULL (TP still inference-only)\n[FAIL]\n");
        return 1;
    }
    tensor_ensure_executed(g);
    const float* gd = (const float*)tensor_data_ptr(g);
    if (!gd) {
        printf("test_tp_autograd: weight grad has no data\n[FAIL]\n");
        return 1;
    }

    for (int j = 0; j < 4 && ok; j++) {
        for (int k = 0; k < 2; k++) {
            float ref = x_data[k];
            if (fabsf(gd[j * 2 + k] - ref) > EPS) {
                printf("grad mismatch dW[%d,%d]=%f want %f\n", j, k, gd[j * 2 + k], ref);
                ok = 0;
            }
        }
    }

    cml_column_parallel_free(cp);
    tensor_free(weight);
    tensor_free(input);

    if (ok) {
        printf("test_tp_autograd: PASSED (weight gradient flows through TP forward)\n");
        return 0;
    }
    printf("test_tp_autograd: FAILED\n");
    return 1;
}
