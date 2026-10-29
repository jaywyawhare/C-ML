#ifndef CML_TENSOR_COMPLEX_H
#define CML_TENSOR_COMPLEX_H

#include "tensor/tensor.h"
#include "tensor/dtype_access.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Complex dtypes store [re, im] interleaved per element (complex64 = 2x float32,
 * complex128 = 2x float64); numel counts complex elements. Non-differentiable:
 * the builders refuse requires_grad and every op produces a leaf tensor. */

/**
 * Real float dtype backing a complex dtype (complex64 -> float32,
 * complex128 -> float64); returns DTYPE_FLOAT32 for non-complex input.
 * @param d (DType)
 * @return (DType)
 */
DType cml_complex_component_dtype(DType d);

/**
 * Complex dtype whose component is `d` (float32 -> complex64,
 * float64 -> complex128); returns DTYPE_COMPLEX64 for other inputs.
 * @param d (DType)
 * @return (DType)
 */
DType cml_real_to_complex_dtype(DType d);

/**
 * Build a complex tensor from matching-shape real tensors `real` and `imag`.
 * complex64 if the inputs are float32, else complex128 (inputs cast to double).
 * Returns NULL on NULL/shape/dtype mismatch.
 * @param real (Tensor*)
 * @param imag (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_complex(Tensor* real, Tensor* imag);

/**
 * Real part of a complex tensor as a new real tensor (float32/float64).
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_real(Tensor* t);

/**
 * Imaginary part of a complex tensor as a new real tensor (float32/float64).
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_imag(Tensor* t);

/**
 * Complex conjugate (negated imaginary part) as a new complex tensor.
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_conj(Tensor* t);

/**
 * Element-wise phase angle atan2(im, re) as a new real tensor.
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_angle(Tensor* t);

/**
 * Element-wise magnitude hypot(re, im) of a complex tensor as a new real tensor.
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_abs(Tensor* t);

/**
 * View a real [..., 2] tensor as a complex tensor of the enclosing shape
 * (copy; the trailing size-2 axis becomes the {re, im} pair).
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* view_as_complex(Tensor* t);

/**
 * View a complex tensor as a real [..., 2] tensor (copy).
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* view_as_real(Tensor* t);

Tensor* cml_complex_add(Tensor* a, Tensor* b);
Tensor* cml_complex_sub(Tensor* a, Tensor* b);
Tensor* cml_complex_mul(Tensor* a, Tensor* b);
Tensor* cml_complex_div(Tensor* a, Tensor* b);

/**
 * Element-wise complex exp: exp(re)*(cos(im) + i*sin(im)).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_complex_exp(Tensor* a);

/**
 * Complex matrix multiply of 2-D complex tensors [M,K] x [K,N] -> [M,N].
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_complex_matmul(Tensor* a, Tensor* b);

/**
 * Sum of all elements of a complex tensor as a scalar complex tensor.
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_complex_sum(Tensor* a);

/**
 * Mean of all elements of a complex tensor as a scalar complex tensor.
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_complex_mean(Tensor* a);

/**
 * 1-D FFT of a native complex tensor (last axis transformed). Returns a new
 * complex tensor of the same shape; inverse!=0 does the inverse transform.
 * @param x (Tensor*)
 * @param inverse (int)
 * @return (Tensor*)
 */
Tensor* cml_fft_complex(Tensor* x, int inverse);

/**
 * Inverse of cml_fft_complex (shorthand for cml_fft_complex(x, 1)).
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_ifft_complex(Tensor* x);

#ifdef __cplusplus
}
#endif

#endif /* CML_TENSOR_COMPLEX_H */
