#include "torch/torch_c.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static float* data(Tensor* t) {
    float* p = torch_tensor_data_ptr_f32(t);
    assert(p != NULL);
    return p;
}

static Tensor* make(float* vals, int* shape, int ndim) {
    TorchTensorOptions o = torch_options();
    o = torch_options_dtype(o, DTYPE_FLOAT32);
    o = torch_options_device(o, DEVICE_CPU);
    Tensor* t = torch_empty(shape, ndim, &o);
    memcpy(torch_tensor_data_ptr_f32(t), vals, (size_t)torch_tensor_numel(t) * sizeof(float));
    return t;
}

static void test_eager_elementwise(void) {
    printf("  test_eager_elementwise...");
    torch_set_eager_mode(true);
    torch_no_grad();

    float av[] = {1, 2, 3, 4};
    float bv[] = {10, 20, 30, 40};
    int shape[] = {2, 2};
    Tensor* a = make(av, shape, 2);
    Tensor* b = make(bv, shape, 2);

    Tensor* s = torch_add(a, b);
    float* sd = data(s);
    assert(sd[0] == 11 && sd[1] == 22 && sd[2] == 33 && sd[3] == 44);
    assert(torch_tensor_is_materialized(s));

    Tensor* m = torch_mul(a, b);
    float* md = data(m);
    assert(md[0] == 10 && md[3] == 160);

    Tensor* d = torch_sub(b, a);
    float* dd = data(d);
    assert(dd[0] == 9 && dd[3] == 36);

    torch_tensor_free(a);
    torch_tensor_free(b);
    torch_tensor_free(s);
    torch_tensor_free(m);
    torch_tensor_free(d);
    torch_set_eager_mode(false);
    printf(" PASSED\n");
}

static void test_eager_matmul(void) {
    printf("  test_eager_matmul...");
    torch_set_eager_mode(true);
    torch_no_grad();

    float av[] = {1, 2, 3, 4, 5, 6};   /* [2,3] */
    float bv[] = {1, 0, 0, 1, 1, 1};   /* [3,2] */
    int as[] = {2, 3};
    int bs[] = {3, 2};
    Tensor* a = make(av, as, 2);
    Tensor* b = make(bv, bs, 2);

    Tensor* c = torch_matmul(a, b); /* [2,2] */
    float* cd = data(c);
    /* row0: [1*1+2*0+3*0, 1*0+2*1+3*1] = [1,5]; row1: [4,11] */
    assert(cd[0] == 1 && cd[1] == 5 && cd[2] == 4 && cd[3] == 11);
    assert(torch_tensor_numel(c) == 4 && torch_tensor_is_materialized(c));

    torch_tensor_free(a);
    torch_tensor_free(b);
    torch_tensor_free(c);
    torch_set_eager_mode(false);
    printf(" PASSED\n");
}

static void test_eager_activations(void) {
    printf("  test_eager_activations...");
    torch_set_eager_mode(true);
    torch_no_grad();

    float xv[] = {-2, -0.5f, 0, 3};
    int shape[] = {4};
    Tensor* x = make(xv, shape, 1);

    Tensor* r = torch_relu(x);
    float* rd = data(r);
    assert(rd[0] == 0 && rd[1] == 0 && rd[2] == 0 && rd[3] == 3);

    Tensor* s = torch_sigmoid(x);
    float* sd = data(s);
    assert(fabsf(sd[2] - 0.5f) < 1e-5f);

    torch_tensor_free(x);
    torch_tensor_free(r);
    torch_tensor_free(s);
    torch_set_eager_mode(false);
    printf(" PASSED\n");
}

static void test_fused_linear(void) {
    printf("  test_fused_linear...");
    torch_set_eager_mode(true);
    torch_no_grad();

    /* input [2,3], weight [2,3] (out=2,in=3), bias [2] */
    float xv[] = {1, 2, 3, 4, 5, 6};
    float wv[] = {1, 0, 0, 0, 1, 0}; /* row0 picks x0, row1 picks x1 */
    float bvv[] = {100, 200};
    int xs[] = {2, 3};
    int ws[] = {2, 3};
    int bs[] = {2};
    Tensor* x = make(xv, xs, 2);
    Tensor* w = make(wv, ws, 2);
    Tensor* bias = make(bvv, bs, 1);

    Tensor* y = torch_linear(x, w, bias); /* [2,2] */
    float* yd = data(y);
    /* row0: x@w^T=[1,2]; +bias=[101,202]. row1: [4,5]; +bias=[104,205] */
    assert(yd[0] == 101 && yd[1] == 202 && yd[2] == 104 && yd[3] == 205);

    float xneg[] = {-1, -2, -3, -4, -5, -6};
    Tensor* xn = make(xneg, xs, 2);
    Tensor* z = torch_linear_relu(xn, w, NULL); /* negatives -> relu 0 */
    float* zd = data(z);
    assert(zd[0] == 0 && zd[1] == 0 && zd[2] == 0 && zd[3] == 0);

    torch_tensor_free(x);
    torch_tensor_free(w);
    torch_tensor_free(bias);
    torch_tensor_free(y);
    torch_tensor_free(xn);
    torch_tensor_free(z);
    torch_set_eager_mode(false);
    printf(" PASSED\n");
}

static void test_grad_safe_fallback(void) {
    printf("  test_grad_safe_fallback...");
    torch_set_eager_mode(true);
    torch_enable_grad(); /* grad ON */

    float av[] = {1, 2, 3, 4};
    int shape[] = {2, 2};
    Tensor* a = make(av, shape, 2);
    Tensor* b = make(av, shape, 2);
    torch_tensor_set_requires_grad(a, true);

    /* requires_grad input + grad enabled => must use lazy IR path (builds graph) */
    Tensor* c = torch_add(a, b);
    assert(c != NULL);
    assert(torch_tensor_has_lazy_ir(c));
    float* cd = data(c);
    assert(cd[0] == 2 && cd[3] == 8);

    torch_tensor_free(a);
    torch_tensor_free(b);
    torch_tensor_free(c);
    torch_reset_ir();
    torch_set_eager_mode(false);
    printf(" PASSED\n");
}

static void test_thread_control(void) {
    printf("  test_thread_control...");
    torch_set_num_threads(2);
    int n = torch_get_num_threads();
    assert(n >= 1);
    torch_set_num_threads(4);
    assert(torch_get_num_threads() >= 1);
    printf(" PASSED\n");
}

int main(void) {
    torch_init();
    printf("Running torch eager tests:\n");

    test_eager_elementwise();
    test_eager_matmul();
    test_eager_activations();
    test_fused_linear();
    test_grad_safe_fallback();
    test_thread_control();

    torch_cleanup();
    printf("All torch eager tests passed.\n");
    return 0;
}
