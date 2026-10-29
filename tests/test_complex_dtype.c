#include <math.h>
#include <stdio.h>

#include "tensor/tensor.h"
#include "tensor/complex.h"
#include "test_require.h"
#include "test_harness.h"

static Tensor* real_vec(const float* vals, int n) {
    int shape[1]     = {n};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .has_dtype = true};
    return tensor_from_data(vals, shape, 1, &cfg);
}

static float cre(Tensor* c, size_t i) {
    Tensor* r = tensor_real(c);
    float v   = tensor_get_float(r, i);
    tensor_free(r);
    return v;
}
static float cim(Tensor* c, size_t i) {
    Tensor* im = tensor_imag(c);
    float v    = tensor_get_float(im, i);
    tensor_free(im);
    return v;
}
static int close(float a, float b) { return fabsf(a - b) < 1e-4f; }

/* dtype plumbing: sizes and names. */
static int test_dtype_plumbing(void) {
    REQUIRE(cml_dtype_size(DTYPE_COMPLEX64) == 8);
    REQUIRE(cml_dtype_size(DTYPE_COMPLEX128) == 16);
    REQUIRE(cml_dtype_is_complex(DTYPE_COMPLEX64));
    REQUIRE(cml_dtype_is_complex(DTYPE_COMPLEX128));
    REQUIRE(!cml_dtype_is_complex(DTYPE_FLOAT32));
    REQUIRE(cml_complex_component_dtype(DTYPE_COMPLEX64) == DTYPE_FLOAT32);
    REQUIRE(cml_complex_component_dtype(DTYPE_COMPLEX128) == DTYPE_FLOAT64);
    return 1;
}

/* tensor_complex + real/imag extraction. */
static int test_construct_extract(void) {
    float r[3] = {1.0f, 2.0f, 3.0f};
    float i[3] = {4.0f, 5.0f, 6.0f};
    Tensor* re = real_vec(r, 3);
    Tensor* im = real_vec(i, 3);
    Tensor* c  = tensor_complex(re, im);
    REQUIRE(c != NULL);
    REQUIRE(c->dtype == DTYPE_COMPLEX64);
    REQUIRE(c->numel == 3);
    for (size_t k = 0; k < 3; k++) {
        REQUIRE(close(cre(c, k), r[k]));
        REQUIRE(close(cim(c, k), i[k]));
    }
    tensor_free(re);
    tensor_free(im);
    tensor_free(c);
    return 1;
}

/* conj, abs, angle. */
static int test_conj_abs_angle(void) {
    float r[2] = {3.0f, 0.0f};
    float i[2] = {4.0f, 2.0f};
    Tensor* re = real_vec(r, 2);
    Tensor* im = real_vec(i, 2);
    Tensor* c  = tensor_complex(re, im);

    Tensor* cj = tensor_conj(c);
    REQUIRE(close(cim(cj, 0), -4.0f));
    REQUIRE(close(cre(cj, 0), 3.0f));

    Tensor* ab = tensor_abs(c);
    REQUIRE(ab->dtype == DTYPE_FLOAT32);
    REQUIRE(close(tensor_get_float(ab, 0), 5.0f)); /* |3+4i| = 5 */
    REQUIRE(close(tensor_get_float(ab, 1), 2.0f)); /* |0+2i| = 2 */

    Tensor* an = tensor_angle(c);
    REQUIRE(close(tensor_get_float(an, 1), (float)(M_PI / 2.0))); /* angle(2i) = pi/2 */

    tensor_free(re);
    tensor_free(im);
    tensor_free(c);
    tensor_free(cj);
    tensor_free(ab);
    tensor_free(an);
    return 1;
}

/* complex multiply vs hand-computed (a+bi)(c+di). */
static int test_complex_mul(void) {
    /* (1+2i) * (3+4i) = (3-8) + (4+6)i = -5 + 10i */
    float ar[1] = {1.0f}, ai[1] = {2.0f}, br[1] = {3.0f}, bi[1] = {4.0f};
    Tensor* a = tensor_complex(real_vec(ar, 1), real_vec(ai, 1));
    Tensor* b = tensor_complex(real_vec(br, 1), real_vec(bi, 1));
    Tensor* p = cml_complex_mul(a, b);
    REQUIRE(close(cre(p, 0), -5.0f));
    REQUIRE(close(cim(p, 0), 10.0f));

    /* division: p / b should recover a */
    Tensor* q = cml_complex_div(p, b);
    REQUIRE(close(cre(q, 0), 1.0f));
    REQUIRE(close(cim(q, 0), 2.0f));

    /* add / sub */
    Tensor* s = cml_complex_add(a, b);
    REQUIRE(close(cre(s, 0), 4.0f) && close(cim(s, 0), 6.0f));
    Tensor* d = cml_complex_sub(a, b);
    REQUIRE(close(cre(d, 0), -2.0f) && close(cim(d, 0), -2.0f));
    return 1;
}

/* complex exp: exp(i*pi) = -1 + 0i. */
static int test_complex_exp(void) {
    float r[1] = {0.0f}, i[1] = {(float)M_PI};
    Tensor* z = tensor_complex(real_vec(r, 1), real_vec(i, 1));
    Tensor* e = cml_complex_exp(z);
    REQUIRE(close(cre(e, 0), -1.0f));
    REQUIRE(close(cim(e, 0), 0.0f));
    return 1;
}

/* complex matmul against a known 2x2 result.
 * A = [[1+i, 2],[0, 1-i]], B = [[1, i],[2, 0]]
 * AB = [[5+i, -1+i], [2-2i, 0]] */
static int test_complex_matmul(void) {
    float ar[4] = {1, 2, 0, 1};
    float ai[4] = {1, 0, 0, -1};
    float br[4] = {1, 0, 2, 0};
    float bi[4] = {0, 1, 0, 0};
    int sh[2]   = {2, 2};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .has_dtype = true};
    Tensor* a = tensor_complex(tensor_from_data(ar, sh, 2, &cfg), tensor_from_data(ai, sh, 2, &cfg));
    Tensor* b = tensor_complex(tensor_from_data(br, sh, 2, &cfg), tensor_from_data(bi, sh, 2, &cfg));
    Tensor* m = cml_complex_matmul(a, b);
    REQUIRE(m != NULL && m->numel == 4);
    REQUIRE(close(cre(m, 0), 5.0f) && close(cim(m, 0), 1.0f));  /* 5+i */
    REQUIRE(close(cre(m, 1), -1.0f) && close(cim(m, 1), 1.0f)); /* -1+i */
    REQUIRE(close(cre(m, 2), 2.0f) && close(cim(m, 2), -2.0f)); /* 2-2i */
    REQUIRE(close(cre(m, 3), 0.0f) && close(cim(m, 3), 0.0f));  /* 0 */
    return 1;
}

/* view_as_real / view_as_complex round-trip. */
static int test_view_roundtrip(void) {
    float r[3] = {1, 2, 3}, i[3] = {-1, -2, -3};
    Tensor* c  = tensor_complex(real_vec(r, 3), real_vec(i, 3));
    Tensor* rr = view_as_real(c); /* [3,2] */
    REQUIRE(rr->ndim == 2 && rr->shape[0] == 3 && rr->shape[1] == 2);
    Tensor* back = view_as_complex(rr);
    REQUIRE(back->numel == 3);
    for (size_t k = 0; k < 3; k++) {
        REQUIRE(close(cre(back, k), r[k]));
        REQUIRE(close(cim(back, k), i[k]));
    }
    return 1;
}

/* sum / mean reductions. */
static int test_reductions(void) {
    float r[3] = {1, 2, 3}, i[3] = {4, 5, 6};
    Tensor* c = tensor_complex(real_vec(r, 3), real_vec(i, 3));
    Tensor* s = cml_complex_sum(c);
    REQUIRE(close(cre(s, 0), 6.0f) && close(cim(s, 0), 15.0f));
    Tensor* mn = cml_complex_mean(c);
    REQUIRE(close(cre(mn, 0), 2.0f) && close(cim(mn, 0), 5.0f));
    return 1;
}

/* fft -> ifft round-trip returns the input within tolerance. */
static int test_fft_roundtrip(void) {
    float r[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float i[4] = {0.0f, -1.0f, 0.5f, 2.0f};
    Tensor* x  = tensor_complex(real_vec(r, 4), real_vec(i, 4));
    Tensor* F  = cml_fft_complex(x, 0);
    REQUIRE(F != NULL && F->numel == 4);
    Tensor* back = cml_ifft_complex(F);
    REQUIRE(back != NULL);
    for (size_t k = 0; k < 4; k++) {
        REQUIRE(close(cre(back, k), r[k]));
        REQUIRE(close(cim(back, k), i[k]));
    }
    return 1;
}

int main(void) {
    printf("=== Complex Dtype Tests ===\n");
    TEST(dtype_plumbing);
    TEST(construct_extract);
    TEST(conj_abs_angle);
    TEST(complex_mul);
    TEST(complex_exp);
    TEST(complex_matmul);
    TEST(view_roundtrip);
    TEST(reductions);
    TEST(fft_roundtrip);
    return TEST_SUMMARY();
}
