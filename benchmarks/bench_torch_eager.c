/*
 * bench_torch_eager.c — Eager (zero-IR) vs lazy-IR overhead for hot ops.
 *
 * Runs a small MLP-style inference forward many times and compares:
 *   1. Lazy IR path (matmul+add+relu) with full context reset per iteration
 *   2. Lazy IR path with soft reset (keeps plan/buffer caches warm)
 *   3. Zero-IR eager fused linear (torch_linear_relu / torch_linear), no reset
 *
 * Weights are realized once (detached from IR) so they survive resets.
 */

#include "torch/torch_c.h"
#include <stdio.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static Tensor* randn_f32(int* shape, int ndim) {
    TorchTensorOptions o = torch_options();
    o = torch_options_dtype(o, DTYPE_FLOAT32);
    o = torch_options_device(o, DEVICE_CPU);
    Tensor* t = torch_randn(shape, ndim, &o);
    if (!t || torch_realize(t) != 0) {
        fprintf(stderr, "bench_torch_eager: tensor setup failed\n");
        exit(1);
    }
    return t;
}

int main(void) {
    torch_init();
    torch_manual_seed(0);

    const int B = 64, D1 = 256, D2 = 256, D3 = 64;
    const int ITERS = 1000;

    int xs[]  = {B, D1};
    int w1s[] = {D2, D1};
    int b1s[] = {D2};
    int w2s[] = {D3, D2};
    int b2s[] = {D3};

    Tensor* x  = randn_f32(xs, 2);
    Tensor* w1 = randn_f32(w1s, 2);
    Tensor* b1 = randn_f32(b1s, 1);
    Tensor* w2 = randn_f32(w2s, 2);
    Tensor* b2 = randn_f32(b2s, 1);

    printf("MLP inference: [%d,%d] -> %d -> %d, %d iters\n\n", B, D1, D2, D3, ITERS);

    /* ---- Lazy MLP via generic ops (matmul + bias-broadcast add + relu) ---- */
    /* eager mode OFF: torch_matmul/add/relu build IR nodes; reset owns outputs */
    torch_set_eager_mode(false);
    torch_no_grad();

    double t0 = now_ms();
    for (int i = 0; i < ITERS; i++) {
        Tensor* w1t = torch_transpose(w1, 0, 1);
        Tensor* h   = torch_matmul(x, w1t);
        Tensor* hb  = torch_add(h, b1);
        Tensor* a   = torch_relu(hb);
        Tensor* w2t = torch_transpose(w2, 0, 1);
        Tensor* h2  = torch_matmul(a, w2t);
        Tensor* y   = torch_add(h2, b2);
        torch_tensor_data_ptr(y);
        torch_reset_ir(); /* full reset frees all graph nodes + outputs */
    }
    double t_hard = now_ms() - t0;

    t0 = now_ms();
    for (int i = 0; i < ITERS; i++) {
        Tensor* w1t = torch_transpose(w1, 0, 1);
        Tensor* h   = torch_matmul(x, w1t);
        Tensor* hb  = torch_add(h, b1);
        Tensor* a   = torch_relu(hb);
        Tensor* w2t = torch_transpose(w2, 0, 1);
        Tensor* h2  = torch_matmul(a, w2t);
        Tensor* y   = torch_add(h2, b2);
        torch_tensor_data_ptr(y);
        torch_reset_ir_soft(); /* keep plan/buffer caches warm */
    }
    double t_soft = now_ms() - t0;

    /* ---- Zero-IR eager fused linear, no reset ---- */
    torch_inference_mode(true); /* eager + no_grad */
    t0 = now_ms();
    for (int i = 0; i < ITERS; i++) {
        Tensor* h = torch_linear_relu(x, w1, b1); /* fused GEMM+bias+relu */
        Tensor* y = torch_linear(h, w2, b2);      /* fused GEMM+bias      */
        torch_tensor_data_ptr(y);
        torch_tensor_free(h); /* eager outputs are standalone leaves */
        torch_tensor_free(y);
    }
    double t_eager = now_ms() - t0;
    torch_inference_mode(false);

    printf("1. lazy + hard reset : %8.2f ms  (%6.2f us/iter)\n", t_hard, t_hard * 1e3 / ITERS);
    printf("2. lazy + soft reset : %8.2f ms  (%6.2f us/iter)\n", t_soft, t_soft * 1e3 / ITERS);
    printf("3. eager (zero-IR)   : %8.2f ms  (%6.2f us/iter)\n", t_eager, t_eager * 1e3 / ITERS);
    printf("\nspeedup eager vs hard reset: %.2fx\n", t_hard / t_eager);
    printf("speedup eager vs soft reset: %.2fx\n", t_soft / t_eager);

    torch_tensor_free(x);
    torch_tensor_free(w1);
    torch_tensor_free(b1);
    torch_tensor_free(w2);
    torch_tensor_free(b2);
    torch_cleanup();
    return 0;
}
