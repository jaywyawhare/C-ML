#include "tensor/tensor_compare.h"
#include "ops/uops.h"
#include "tensor/tensor.h"

/**
 * Elementwise a < b (0/1 mask, lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_lt(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_cmplt(a, b);
}

/**
 * Elementwise a <= b (0/1 mask, lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_le(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_cmple(a, b);
}

/**
 * Elementwise a > b (0/1 mask, lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_gt(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_cmpgt(a, b);
}

/**
 * Elementwise a >= b (0/1 mask, lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_ge(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_cmpge(a, b);
}

/**
 * Elementwise a == b (0/1 mask, lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_eq(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_cmpeq(a, b);
}

/**
 * Elementwise a != b (0/1 mask, lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_ne(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_cmpne(a, b);
}

/**
 * Elementwise logical AND of the operands' truth values (0/1 mask, lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_logical_and(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_logical_and(a, b);
}

/**
 * Elementwise logical OR of the operands' truth values (0/1 mask, lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_logical_or(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_logical_or(a, b);
}

/**
 * Elementwise logical NOT: 1 where a == 0, else 0 (lazy).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_logical_not(Tensor* a) {
    if (!a)
        return NULL;
    return uop_logical_not(a);
}

/**
 * Elementwise logical XOR of the operands' truth values (0/1 mask, lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_logical_xor(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;

    /* No UOP_LOGICAL_XOR primitive, so compose it as (a || b) && !(a && b). */
    Tensor* or_ab = uop_logical_or(a, b);
    if (!or_ab)
        return NULL;
    Tensor* and_ab = uop_logical_and(a, b);
    if (!and_ab) {
        tensor_free(or_ab);
        return NULL;
    }
    Tensor* nand_ab = uop_logical_not(and_ab);
    if (!nand_ab) {
        tensor_free(and_ab);
        tensor_free(or_ab);
        return NULL;
    }

    Tensor* result = uop_logical_and(or_ab, nand_ab);

    tensor_free(nand_ab);
    tensor_free(and_ab);
    tensor_free(or_ab);
    return result;
}

/**
 * Elementwise bitwise AND of integer operands (lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_bitwise_and(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_bitwise_and(a, b);
}

/**
 * Elementwise bitwise OR of integer operands (lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_bitwise_or(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_bitwise_or(a, b);
}

/**
 * Elementwise bitwise XOR of integer operands (lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_bitwise_xor(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_bitwise_xor(a, b);
}

/**
 * Elementwise bitwise NOT of an integer operand (lazy).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_bitwise_not(Tensor* a) {
    if (!a)
        return NULL;
    return uop_bitwise_not(a);
}

/**
 * Elementwise minimum of two tensors, differentiable (lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_minimum(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_minimum(a, b);
}

/**
 * Elementwise maximum of two tensors, differentiable (lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_maximum(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_max(a, b);
}
