/* In-place elementwise ops: a op= b mutates a's buffer directly (no allocation),
 * for same-shape, scalar, and trailing-broadcast operands. */
#include <stdio.h>
#include <math.h>
#include "cml.h"
#include "test_harness.h"

static int approx(const float* d, const float* want, int n) {
    for (int i = 0; i < n; i++)
        if (fabsf(d[i] - want[i]) > 1e-5f)
            return 0;
    return 1;
}

int main(void) {
    cml_init();
    printf("=== In-place ops ===\n");

    /* same-shape add: returns a, mutates in place, no new tensor */
    {
        float a[4] = {1, 2, 3, 4}, b[4] = {10, 20, 30, 40}, w[4] = {11, 22, 33, 44};
        Tensor* ta = cml_tensor_1d(a, 4);
        Tensor* tb = cml_tensor_1d(b, 4);
        Tensor* r  = cml_add_(ta, tb);
        CHECK("add_ returns a (in-place identity)", r == ta);
        CHECK("add_ same-shape correct", approx((float*)tensor_data_ptr(ta), w, 4));
    }

    /* scalar broadcast mul */
    {
        float a[4] = {1, 2, 3, 4}, s[1] = {3}, w[4] = {3, 6, 9, 12};
        Tensor* ta = cml_tensor_1d(a, 4);
        Tensor* ts = cml_tensor_1d(s, 1);
        cml_mul_(ta, ts);
        CHECK("mul_ scalar correct", approx((float*)tensor_data_ptr(ta), w, 4));
    }

    /* scalar broadcast div */
    {
        float a[4] = {2, 4, 6, 8}, s[1] = {2}, w[4] = {1, 2, 3, 4};
        Tensor* ta = cml_tensor_1d(a, 4);
        Tensor* ts = cml_tensor_1d(s, 1);
        cml_div_(ta, ts);
        CHECK("div_ scalar correct", approx((float*)tensor_data_ptr(ta), w, 4));
    }

    /* sub_ same shape */
    {
        float a[3] = {5, 7, 9}, b[3] = {1, 2, 3}, w[3] = {4, 5, 6};
        Tensor* ta = cml_tensor_1d(a, 3);
        Tensor* tb = cml_tensor_1d(b, 3);
        cml_sub_(ta, tb);
        CHECK("sub_ same-shape correct", approx((float*)tensor_data_ptr(ta), w, 3));
    }

    /* trailing broadcast: [2,3] += [3] (bias-style) */
    {
        float a[6] = {0, 0, 0, 1, 1, 1}, b[3] = {10, 20, 30}, w[6] = {10, 20, 30, 11, 21, 31};
        Tensor* ta = cml_tensor_2d(a, 2, 3);
        Tensor* tb = cml_tensor_1d(b, 3);
        cml_add_(ta, tb);
        CHECK("add_ trailing broadcast [2,3]+=[3]", approx((float*)tensor_data_ptr(ta), w, 6));
    }

    return TEST_SUMMARY();
}
