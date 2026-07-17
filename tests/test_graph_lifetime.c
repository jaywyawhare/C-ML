/* Regression guard for the training-loop graph-accumulation fix.
 *
 * A hand-written loop that reuses the same parameters (as the Python bindings
 * do) must not pile every step's autograd graph onto them — otherwise each
 * backward re-traverses all prior steps (O(n^2) blow-up + compounding gradients
 * -> divergence/hang). The C training loop avoids this by freeing the loss each
 * step; cml_autograd_step_end() gives the same guarantee without freeing the
 * loss object (it detaches it and resets the accumulated graph), which is what
 * language bindings that keep the loss alive rely on.
 *
 * This trains Linear(1,1) SGD on y = 2x+1 WITHOUT freeing the loss each step,
 * using cml_autograd_step_end() as the per-step boundary, and asserts it
 * converges (it diverges/hangs without the boundary reset).
 */

#include "cml.h"
#include "nn.h"
#include "autograd/autograd.h"
#include "optim.h"
#include "tensor/realize.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float get_scalar(Tensor* t) {
    tensor_ensure_executed(t);
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[0] : NAN;
}

int main(void) {
    cml_init();

    Linear* layer = nn_linear(1, 1, DTYPE_FLOAT32, DEVICE_CPU, true);
    Module* model = (Module*)layer;
    module_set_training(model, true);
    Optimizer* opt = cml_optim_sgd_for_model(model, 0.05f, 0.0f, 0.0f);

    enum { N = 20 };
    float xd[N], yd[N];
    for (int i = 0; i < N; i++) { xd[i] = (float)i / (N - 1) * 2.0f - 1.0f; yd[i] = 2.0f * xd[i] + 1.0f; }
    Tensor* X = cml_tensor_2d(xd, N, 1);
    Tensor* Y = cml_tensor_2d(yd, N, 1);
    tensor_realize(X);
    tensor_realize(Y);

    float final_loss = INFINITY;
    for (int e = 0; e < 300; e++) {
        optimizer_zero_grad(opt);
        Tensor* out  = module_forward(model, X);   /* reused X + params, loss NOT freed */
        Tensor* loss = tensor_mse_loss(out, Y);
        tensor_backward(loss, NULL, false, false);
        optimizer_step(opt);
        final_loss = get_scalar(loss);
        cml_autograd_step_end(loss);               /* detach loss + reset the step graph */
    }

    optimizer_free(opt);
    module_free(model);
    cml_cleanup();

    printf("final loss (no per-step free, with step_end) = %.6f\n", (double)final_loss);
    if (isfinite(final_loss) && final_loss < 0.05f) { printf("PASS\n"); return 0; }
    printf("FAIL\n");
    return 1;
}
