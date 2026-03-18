#include "cml.h"
#include <stdio.h>

static void print_tensor(const char* name, Tensor* t, int rows, int cols) {
    printf("%s [%dx%d]:\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++)
            printf("%8.4f ", tensor_get_float(t, i * cols + j));
        printf("\n");
    }
}

int main(void) {
    cml_init();
    printf("Example 01: Basic Tensor Operations\n\n");

    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {7, 8, 9, 10, 11, 12};
    int shape23[] = {2, 3};

    Tensor* A = cml_tensor(a_data, shape23, 2, NULL);
    Tensor* B = cml_tensor(b_data, shape23, 2, NULL);

    print_tensor("A", A, 2, 3);
    print_tensor("B", B, 2, 3);

    Tensor* C = cml_add(A, B);
    tensor_ensure_executed(C);
    print_tensor("A + B", C, 2, 3);

    Tensor* D = cml_sub(A, B);
    tensor_ensure_executed(D);
    print_tensor("A - B", D, 2, 3);

    Tensor* E = cml_mul(A, B);
    tensor_ensure_executed(E);
    print_tensor("A * B (element-wise)", E, 2, 3);

    Tensor* F = cml_relu(A);
    tensor_ensure_executed(F);
    print_tensor("relu(A)", F, 2, 3);

    float neg_data[] = {-1, 2, -3, 4, -5, 6};
    Tensor* G = cml_tensor(neg_data, shape23, 2, NULL);
    Tensor* H = cml_relu(G);
    tensor_ensure_executed(H);
    print_tensor("relu([-1,2,-3,4,-5,6])", H, 2, 3);

    float m1[] = {1, 2, 3, 4};
    float m2[] = {5, 6, 7, 8};
    int shape22[] = {2, 2};
    Tensor* M1 = cml_tensor(m1, shape22, 2, NULL);
    Tensor* M2 = cml_tensor(m2, shape22, 2, NULL);
    Tensor* MM = cml_matmul(M1, M2);
    tensor_ensure_executed(MM);
    print_tensor("M1", M1, 2, 2);
    print_tensor("M2", M2, 2, 2);
    print_tensor("M1 @ M2", MM, 2, 2);

    printf("\nAll tensor operations completed successfully.\n");
    cml_cleanup();
    return 0;
}
