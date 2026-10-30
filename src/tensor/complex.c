#include "tensor/complex.h"
#include "tensor/realize.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"
#include <math.h>
#include <string.h>

/* Non-differentiable: builders refuse requires_grad and every op returns an
 * executed leaf, so Wirtinger autograd stays out of scope. */

/**
 * Real component dtype of a complex dtype (complex64->float32, complex128->float64).
 * @param d (DType)
 * @return (DType)
 */
DType cml_complex_component_dtype(DType d) {
    switch (d) {
    case DTYPE_COMPLEX64:
        return DTYPE_FLOAT32;
    case DTYPE_COMPLEX128:
        return DTYPE_FLOAT64;
    default:
        return DTYPE_FLOAT32;
    }
}

/**
 * Complex dtype whose component is `d` (float64->complex128, else complex64).
 * @param d (DType)
 * @return (DType)
 */
DType cml_real_to_complex_dtype(DType d) {
    return (d == DTYPE_FLOAT64) ? DTYPE_COMPLEX128 : DTYPE_COMPLEX64;
}

/**
 * Load complex element `i` of `p` (dtype `d`) into (*re, *im) as doubles.
 * @param p (const void*)
 * @param i (size_t)
 * @param d (DType)
 * @param re (double*)
 * @param im (double*)
 * @return (void)
 */
static void cload(const void* p, size_t i, DType d, double* re, double* im) {
    if (d == DTYPE_COMPLEX128) {
        const double* dp = (const double*)p;
        *re              = dp[2 * i];
        *im              = dp[2 * i + 1];
    } else {
        const float* fp = (const float*)p;
        *re             = (double)fp[2 * i];
        *im             = (double)fp[2 * i + 1];
    }
}

/**
 * Store (re, im) into complex element `i` of `p` (dtype `d`).
 * @param p (void*)
 * @param i (size_t)
 * @param d (DType)
 * @param re (double)
 * @param im (double)
 * @return (void)
 */
static void cstore(void* p, size_t i, DType d, double re, double im) {
    if (d == DTYPE_COMPLEX128) {
        double* dp    = (double*)p;
        dp[2 * i]     = re;
        dp[2 * i + 1] = im;
    } else {
        float* fp     = (float*)p;
        fp[2 * i]     = (float)re;
        fp[2 * i + 1] = (float)im;
    }
}

/**
 * Allocate an executed complex leaf of `dtype` shaped like `ref`.
 * @param ref (const Tensor*)
 * @param dtype (DType)
 * @return (Tensor*)
 */
static Tensor* complex_like(const Tensor* ref, DType dtype) {
    Tensor* out = tensor_create(dtype, ref->device, ref->ndim, ref->shape, false);
    if (out)
        out->is_executed = true;
    return out;
}

/**
 * Realize `t` and verify it is complex, readying it for element access.
 * @param t (Tensor*)
 * @return (bool)
 */
static bool complex_ready(Tensor* t) {
    if (!t)
        return false;
    if (!cml_dtype_is_complex(t->dtype)) {
        LOG_ERROR("expected a complex tensor, got %s", cml_dtype_to_string(t->dtype));
        return false;
    }
    if (tensor_realize(t) != 0 || !t->data)
        return false;
    return true;
}

/**
 * True when `a` and `b` have identical rank and extents.
 * @param a (const Tensor*)
 * @param b (const Tensor*)
 * @return (bool)
 */
static bool same_shape(const Tensor* a, const Tensor* b) {
    if (a->ndim != b->ndim)
        return false;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i])
            return false;
    return true;
}

/**
 * Build a complex tensor from matching-shape real tensors `real` and `imag`.
 * @param real (Tensor*)
 * @param imag (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_complex(Tensor* real, Tensor* imag) {
    if (!real || !imag)
        return NULL;
    if (tensor_realize(real) != 0 || tensor_realize(imag) != 0)
        return NULL;
    if (!real->data || !imag->data)
        return NULL;
    if (!same_shape(real, imag)) {
        LOG_ERROR("tensor_complex: real and imag shapes differ");
        return NULL;
    }
    if (cml_dtype_is_complex(real->dtype) || cml_dtype_is_complex(imag->dtype)) {
        LOG_ERROR("tensor_complex: inputs must be real tensors");
        return NULL;
    }
    if (real->requires_grad || imag->requires_grad)
        LOG_WARNING("tensor_complex: complex tensors are non-differentiable; grad will not flow");

    DType cdtype = (real->dtype == DTYPE_FLOAT64 || imag->dtype == DTYPE_FLOAT64) ? DTYPE_COMPLEX128
                                                                                  : DTYPE_COMPLEX64;
    Tensor* out  = complex_like(real, cdtype);
    if (!out)
        return NULL;
    for (size_t i = 0; i < out->numel; i++)
        cstore(out->data, i, cdtype, tensor_get_float(real, i), tensor_get_float(imag, i));
    return out;
}

/**
 * Map each complex element to a real scalar into a new component-dtype tensor.
 * op: 0 real, 1 imag, 2 abs, 3 angle.
 * @param t (Tensor*)
 * @param op (int)
 * @return (Tensor*)
 */
static Tensor* complex_to_real(Tensor* t, int op) {
    if (!complex_ready(t))
        return NULL;
    DType rdtype = cml_complex_component_dtype(t->dtype);
    Tensor* out  = tensor_create(rdtype, t->device, t->ndim, t->shape, false);
    if (!out)
        return NULL;
    for (size_t i = 0; i < t->numel; i++) {
        double re, im, v = 0.0;
        cload(t->data, i, t->dtype, &re, &im);
        switch (op) {
        case 0:
            v = re;
            break;
        case 1:
            v = im;
            break;
        case 2:
            v = hypot(re, im);
            break;
        case 3:
            v = atan2(im, re);
            break;
        default:
            break;
        }
        tensor_set_float(out, i, (float)v);
    }
    return out;
}

Tensor* tensor_real(Tensor* t) { return complex_to_real(t, 0); }
Tensor* tensor_imag(Tensor* t) { return complex_to_real(t, 1); }
Tensor* tensor_abs(Tensor* t) { return complex_to_real(t, 2); }
Tensor* tensor_angle(Tensor* t) { return complex_to_real(t, 3); }

/**
 * Map each complex element to a new complex element of the same dtype.
 * op: 0 conj, 1 exp.
 * @param a (Tensor*)
 * @param op (int)
 * @return (Tensor*)
 */
static Tensor* complex_unary(Tensor* a, int op) {
    if (!complex_ready(a))
        return NULL;
    Tensor* out = complex_like(a, a->dtype);
    if (!out)
        return NULL;
    for (size_t i = 0; i < a->numel; i++) {
        double re, im, rr = 0.0, ri = 0.0;
        cload(a->data, i, a->dtype, &re, &im);
        switch (op) {
        case 0:
            rr = re;
            ri = -im;
            break;
        case 1: {
            double e = exp(re);
            rr       = e * cos(im);
            ri       = e * sin(im);
            break;
        }
        default:
            break;
        }
        cstore(out->data, i, a->dtype, rr, ri);
    }
    return out;
}

Tensor* tensor_conj(Tensor* t) { return complex_unary(t, 0); }
Tensor* cml_complex_exp(Tensor* a) { return complex_unary(a, 1); }

/**
 * View a real [..., 2] tensor as a complex tensor of the enclosing shape (copy).
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* view_as_complex(Tensor* t) {
    if (!t)
        return NULL;
    if (tensor_realize(t) != 0 || !t->data)
        return NULL;
    if (cml_dtype_is_complex(t->dtype)) {
        LOG_ERROR("view_as_complex: input is already complex");
        return NULL;
    }
    if (t->ndim < 1 || t->shape[t->ndim - 1] != 2) {
        LOG_ERROR("view_as_complex: last dim must be 2");
        return NULL;
    }
    DType cdtype = cml_real_to_complex_dtype(t->dtype);
    int ndim     = t->ndim - 1;
    int one      = 1;
    /* tensor_create needs a non-NULL shape; a scalar complex becomes a [1]. */
    int rank    = ndim > 0 ? ndim : 1;
    int* shape  = ndim > 0 ? t->shape : &one;
    Tensor* out = tensor_create(cdtype, t->device, rank, shape, false);
    if (!out)
        return NULL;
    out->is_executed = true;
    for (size_t i = 0; i < out->numel; i++)
        cstore(out->data, i, cdtype, tensor_get_float(t, 2 * i), tensor_get_float(t, 2 * i + 1));
    return out;
}

/**
 * View a complex tensor as a real [..., 2] tensor (copy).
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* view_as_real(Tensor* t) {
    if (!complex_ready(t))
        return NULL;
    DType rdtype = cml_complex_component_dtype(t->dtype);
    int ndim     = t->ndim + 1;
    int* shape   = (int*)cml_malloc((size_t)ndim * sizeof(int));
    if (!shape)
        return NULL;
    for (int i = 0; i < t->ndim; i++)
        shape[i] = t->shape[i];
    shape[ndim - 1] = 2;
    Tensor* out     = tensor_create(rdtype, t->device, ndim, shape, false);
    cml_free(shape);
    if (!out)
        return NULL;
    for (size_t i = 0; i < t->numel; i++) {
        double re, im;
        cload(t->data, i, t->dtype, &re, &im);
        tensor_set_float(out, 2 * i, (float)re);
        tensor_set_float(out, 2 * i + 1, (float)im);
    }
    return out;
}

/**
 * Element-wise binary op into a new complex tensor. op: 0 add, 1 sub, 2 mul, 3 div.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @param op (int)
 * @return (Tensor*)
 */
static Tensor* complex_binary(Tensor* a, Tensor* b, int op) {
    if (!complex_ready(a) || !complex_ready(b))
        return NULL;
    if (!same_shape(a, b)) {
        LOG_ERROR("complex binary op: shape mismatch");
        return NULL;
    }
    DType cdtype = (a->dtype == DTYPE_COMPLEX128 || b->dtype == DTYPE_COMPLEX128) ? DTYPE_COMPLEX128
                                                                                  : DTYPE_COMPLEX64;
    Tensor* out  = complex_like(a, cdtype);
    if (!out)
        return NULL;
    for (size_t i = 0; i < a->numel; i++) {
        double ar, ai, br, bi, rr = 0.0, ri = 0.0;
        cload(a->data, i, a->dtype, &ar, &ai);
        cload(b->data, i, b->dtype, &br, &bi);
        switch (op) {
        case 0:
            rr = ar + br;
            ri = ai + bi;
            break;
        case 1:
            rr = ar - br;
            ri = ai - bi;
            break;
        case 2:
            rr = ar * br - ai * bi;
            ri = ar * bi + ai * br;
            break;
        case 3: {
            double den = br * br + bi * bi;
            rr         = (ar * br + ai * bi) / den;
            ri         = (ai * br - ar * bi) / den;
            break;
        }
        default:
            break;
        }
        cstore(out->data, i, cdtype, rr, ri);
    }
    return out;
}

Tensor* cml_complex_add(Tensor* a, Tensor* b) { return complex_binary(a, b, 0); }
Tensor* cml_complex_sub(Tensor* a, Tensor* b) { return complex_binary(a, b, 1); }
Tensor* cml_complex_mul(Tensor* a, Tensor* b) { return complex_binary(a, b, 2); }
Tensor* cml_complex_div(Tensor* a, Tensor* b) { return complex_binary(a, b, 3); }

/**
 * Complex matrix multiply of 2-D complex tensors [M,K] x [K,N] -> [M,N].
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_complex_matmul(Tensor* a, Tensor* b) {
    if (!complex_ready(a) || !complex_ready(b))
        return NULL;
    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) {
        LOG_ERROR("cml_complex_matmul: expected [M,K] x [K,N] 2-D complex tensors");
        return NULL;
    }
    int M = a->shape[0], K = a->shape[1], N = b->shape[1];
    DType cdtype = (a->dtype == DTYPE_COMPLEX128 || b->dtype == DTYPE_COMPLEX128) ? DTYPE_COMPLEX128
                                                                                  : DTYPE_COMPLEX64;
    int out_shape[2] = {M, N};
    Tensor* out      = tensor_create(cdtype, a->device, 2, out_shape, false);
    if (!out)
        return NULL;
    out->is_executed = true;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double accr = 0.0, acci = 0.0;
            for (int k = 0; k < K; k++) {
                double ar, ai, br, bi;
                cload(a->data, (size_t)m * K + k, a->dtype, &ar, &ai);
                cload(b->data, (size_t)k * N + n, b->dtype, &br, &bi);
                accr += ar * br - ai * bi;
                acci += ar * bi + ai * br;
            }
            cstore(out->data, (size_t)m * N + n, cdtype, accr, acci);
        }
    }
    return out;
}

/**
 * Reduce all complex elements to a scalar complex tensor; divides by numel when `mean`.
 * @param a (Tensor*)
 * @param mean (bool)
 * @return (Tensor*)
 */
static Tensor* complex_reduce(Tensor* a, bool mean) {
    if (!complex_ready(a))
        return NULL;
    int one     = 1;
    Tensor* out = tensor_create(a->dtype, a->device, 0, NULL, false);
    if (!out)
        out = tensor_create(a->dtype, a->device, 1, &one, false);
    if (!out)
        return NULL;
    out->is_executed = true;
    double sr = 0.0, si = 0.0;
    for (size_t i = 0; i < a->numel; i++) {
        double re, im;
        cload(a->data, i, a->dtype, &re, &im);
        sr += re;
        si += im;
    }
    if (mean && a->numel > 0) {
        sr /= (double)a->numel;
        si /= (double)a->numel;
    }
    cstore(out->data, 0, a->dtype, sr, si);
    return out;
}

Tensor* cml_complex_sum(Tensor* a) { return complex_reduce(a, false); }
Tensor* cml_complex_mean(Tensor* a) { return complex_reduce(a, true); }

/**
 * 1-D FFT over the last axis of a native complex tensor; inverse!=0 inverts.
 * @param x (Tensor*)
 * @param inverse (int)
 * @return (Tensor*)
 */
Tensor* cml_fft_complex(Tensor* x, int inverse) {
    if (!complex_ready(x))
        return NULL;
    if (x->ndim < 1) {
        LOG_ERROR("cml_fft_complex: need at least 1 dimension");
        return NULL;
    }
    extern int cml_fft_1d(float* re, float* im, int n, int inverse);
    int n        = x->shape[x->ndim - 1];
    size_t batch = x->numel / (size_t)n;
    Tensor* out  = complex_like(x, x->dtype);
    if (!out)
        return NULL;
    float* re = (float*)cml_malloc((size_t)n * sizeof(float));
    float* im = (float*)cml_malloc((size_t)n * sizeof(float));
    if (!re || !im) {
        cml_free(re);
        cml_free(im);
        tensor_free(out);
        return NULL;
    }
    for (size_t b = 0; b < batch; b++) {
        size_t base = b * (size_t)n;
        for (int i = 0; i < n; i++) {
            double r, m;
            cload(x->data, base + (size_t)i, x->dtype, &r, &m);
            re[i] = (float)r;
            im[i] = (float)m;
        }
        if (cml_fft_1d(re, im, n, inverse) != 0) {
            cml_free(re);
            cml_free(im);
            tensor_free(out);
            return NULL;
        }
        for (int i = 0; i < n; i++)
            cstore(out->data, base + (size_t)i, out->dtype, re[i], im[i]);
    }
    cml_free(re);
    cml_free(im);
    return out;
}

Tensor* cml_ifft_complex(Tensor* x) { return cml_fft_complex(x, 1); }
