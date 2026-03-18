#ifndef CML_AUTOGRAD_FORWARD_OPS_H
#define CML_AUTOGRAD_FORWARD_OPS_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_div(Tensor* a, Tensor* b);
Tensor* tensor_exp(Tensor* a);
Tensor* tensor_log(Tensor* a);
Tensor* tensor_sqrt(Tensor* a);
Tensor* tensor_relu(Tensor* a);
Tensor* tensor_sigmoid(Tensor* a);
Tensor* tensor_tanh(Tensor* a);
Tensor* tensor_neg(Tensor* a);
Tensor* tensor_leaky_relu(Tensor* a, float negative_slope);
Tensor* tensor_pow(Tensor* a, Tensor* b);
Tensor* tensor_sin(Tensor* a);
Tensor* tensor_cos(Tensor* a);
Tensor* tensor_tan(Tensor* a);
Tensor* tensor_softmax(Tensor* a, int dim);

/* dim=-1 reduces all dimensions */
Tensor* tensor_sum(Tensor* a, int dim, bool keepdim);
Tensor* tensor_mean(Tensor* a, int dim, bool keepdim);
Tensor* tensor_max(Tensor* a, int dim, bool keepdim);
Tensor* tensor_min(Tensor* a, int dim, bool keepdim);

Tensor* tensor_matmul(Tensor* a, Tensor* b);
Tensor* tensor_transpose(Tensor* a, int dim1, int dim2);

/* unbiased=true uses Bessel's correction (N-1) */
Tensor* tensor_std(Tensor* a, int dim, bool unbiased, bool keepdim);
Tensor* tensor_var(Tensor* a, int dim, bool unbiased, bool keepdim);

Tensor* tensor_argmax(Tensor* a, int dim);
Tensor* tensor_argmin(Tensor* a, int dim);
bool tensor_has_grad(Tensor* a);

/* ELU: x > 0 ? x : alpha*(exp(x)-1) */
Tensor* tensor_elu(Tensor* a, float alpha);
Tensor* tensor_selu(Tensor* a);
/* x * tanh(softplus(x)) */
Tensor* tensor_mish(Tensor* a);
/* x * sigmoid(x) */
Tensor* tensor_silu(Tensor* a);
Tensor* tensor_hardswish(Tensor* a);

Tensor* tensor_sort(Tensor* a, int dim, bool descending);
Tensor* tensor_topk(Tensor* a, int k, int dim, bool largest, bool sorted);
Tensor* tensor_masked_select(Tensor* a, Tensor* mask);
Tensor** tensor_meshgrid(Tensor** tensors, int num_tensors, int* num_outputs);
Tensor* tensor_diagonal(Tensor* a, int offset, int dim1, int dim2);
/* a + weight*(b-a) */
Tensor* tensor_lerp(Tensor* a, Tensor* b, float weight);
Tensor* tensor_idiv(Tensor* a, Tensor* b);
Tensor* tensor_mod(Tensor* a, Tensor* b);

#ifdef __cplusplus
}
#endif

#endif // CML_AUTOGRAD_FORWARD_OPS_H
