#include "cml.h"
#include <stdio.h>

int main(void) {
    if (cml_init() != 0) {
        fprintf(stderr, "C-ML init failed\n");
        return 1;
    }

    autograd_init();
    autograd_set_grad_mode(true);

    // Build a tiny compute graph: y = ReLU((X @ W^T) + b)
    int in = 4, out = 3, batch = 2;
    int x_shape[] = {batch, in};
    int w_shape[] = {out, in};
    int b_shape[] = {out};
    Tensor* X     = tensor_empty(x_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    Tensor* W     = tensor_empty(w_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    Tensor* b     = tensor_empty(b_shape, 1, DTYPE_FLOAT32, DEVICE_CPU);

    tensor_set_requires_grad(X, true);
    tensor_set_requires_grad(W, true);
    tensor_set_requires_grad(b, true);

    // simple deterministic init
    for (int i = 0; i < batch * in; i++)
        ((float*)tensor_data_ptr(X))[i] = (float)((i % 7) - 3) * 0.1f;
    for (int i = 0; i < out * in; i++)
        ((float*)tensor_data_ptr(W))[i] = (float)((i % 5) - 2) * 0.2f;
    for (int i = 0; i < out; i++)
        ((float*)tensor_data_ptr(b))[i] = 0.1f * (float)i;

    Tensor* WT     = tensor_transpose(W, 0, 1);
    Tensor* Z      = tensor_matmul(X, WT);
    int b2_shape[] = {1, out};
    Tensor* b2     = tensor_reshape(b, b2_shape, 2);
    Tensor* Zb     = tensor_add(Z, b2);
    Tensor* Y      = tensor_relu(Zb);

    Tensor* target = tensor_zeros(b2_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    Tensor* loss   = tensor_mse_loss(Y, target);
    tensor_backward(loss, NULL, false, false);

    const char* out_path = "viz-ui/public/graph.json";
    int rc               = autograd_export_json(Y, out_path);
    printf("Export graph -> %s (rc=%d)\n", out_path, rc);

    if (loss)
        tensor_free(loss);
    if (target)
        tensor_free(target);
    if (Y)
        tensor_free(Y);
    if (Zb)
        tensor_free(Zb);
    if (b2)
        tensor_free(b2);
    if (Z)
        tensor_free(Z);
    if (WT)
        tensor_free(WT);
    if (b)
        tensor_free(b);
    if (W)
        tensor_free(W);
    if (X)
        tensor_free(X);
    autograd_shutdown();
    cml_cleanup();
    return rc == 0 ? 0 : 2;
}
