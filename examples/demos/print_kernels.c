#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cml.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/context.h"
#include "ops/uops.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"

static void print_separator(const char* title) {
    printf("\n");
    printf("  %s\n", title);
    printf("\n");
}

static void print_cpu_fallback_pseudocode(CMLGraph_t ir) {
    print_separator("CPU Fallback (Interpreter Pseudocode)");

    if (!ir || !ir->head) {
        printf("  (No IR nodes)\n");
        return;
    }

    printf("// CPU Interpreter execution plan:\n\n");

    struct IRNode* node = ir->head;
    int op_num          = 0;
    while (node) {
        printf("Op %d: ", op_num++);

        switch (node->type) {
        case UOP_ADD:
            printf("ADD");
            break;
        case UOP_SUB:
            printf("SUB");
            break;
        case UOP_MUL:
            printf("MUL");
            break;
        case UOP_DIV:
            printf("DIV");
            break;
        case UOP_MATMUL:
            printf("MATMUL");
            break;
        case UOP_EXP:
            printf("EXP");
            break;
        case UOP_LOG:
            printf("LOG");
            break;
        case UOP_SQRT:
            printf("SQRT");
            break;
        case UOP_NEG:
            printf("NEG");
            break;
        case UOP_SIGMOID:
            printf("SIGMOID");
            break;
        case UOP_TANH:
            printf("TANH");
            break;
        case UOP_SUM:
            printf("SUM");
            break;
        case UOP_MEAN:
            printf("MEAN");
            break;
        case UOP_MAX:
            printf("MAX");
            break;
        case UOP_MAX_REDUCE:
            printf("MAX_REDUCE");
            break;
        default:
            printf("OP_%d", node->type);
            break;
        }

        if (node->output) {
            printf(" -> [");
            for (int i = 0; i < node->output->ndim; i++) {
                printf("%d", node->output->shape[i]);
                if (i < node->output->ndim - 1)
                    printf(", ");
            }
            printf("]");
        }

        printf("\n");

        switch (node->type) {
        case UOP_ADD:
            printf("    for i in 0..n: out[i] = in0[i] + in1[i]\n");
            break;
        case UOP_MUL:
            printf("    for i in 0..n: out[i] = in0[i] * in1[i]\n");
            break;
        case UOP_MATMUL:
            printf("    for i in 0..M:\n");
            printf("      for j in 0..N:\n");
            printf("        sum = 0\n");
            printf("        for k in 0..K: sum += A[i,k] * B[k,j]\n");
            printf("        C[i,j] = sum\n");
            break;
        case UOP_EXP:
            printf("    for i in 0..n: out[i] = exp(in[i])\n");
            break;
        case UOP_SIGMOID:
            printf("    for i in 0..n: out[i] = 1 / (1 + exp(-in[i]))\n");
            break;
        default:
            break;
        }

        printf("\n");
        node = node->next;
    }
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("     CML Kernel Code Generator\n");
    printf("\n");

    int size = 4;
    if (argc > 1) {
        size = atoi(argv[1]);
    }

    printf("\nGenerating kernels for %dx%d matrix multiplication...\n", size, size);

    Tensor* A = tensor_empty_2d(size, size);
    Tensor* B = tensor_empty_2d(size, size);

    if (!A || !B) {
        printf("Failed to allocate tensors\n");
        return 1;
    }

    float* a_data = (float*)A->data;
    float* b_data = (float*)B->data;
    for (int i = 0; i < size * size; i++) {
        a_data[i] = (float)(i % 10) / 10.0f;
        b_data[i] = (float)((i + 1) % 10) / 10.0f;
    }

    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    cml_ir_set_global_context(ir);

    tensor_matmul(A, B);

    print_cpu_fallback_pseudocode(ir);

    cml_ir_free(ir);
    tensor_free(A);
    tensor_free(B);

    printf("\n");
    printf("           Generation Complete\n");
    printf("\n");

    return 0;
}
