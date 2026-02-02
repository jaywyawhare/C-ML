/**
 * @file print_kernels.c
 * @brief Print compiled/optimized kernels for all backends
 */

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
#include "ops/ir/mlir/mlir_codegen.h"
#include "ops/ir/mlir/mlir_backend.h"
#include "ops/ir/mlir/mlir_context.h"
#include "ops/ir/mlir/mlir_convert.h"

// Forward declaration for dump function (implemented in C++ bridge)
#ifdef CML_HAS_MLIR
char* cml_mlir_dump_module(void* mlir_module);
#endif

static void print_separator(const char* title) {
    printf("\n");
    printf("================================================================================\n");
    printf("  %s\n", title);
    printf("================================================================================\n\n");
}

static void print_mlir_module(CMLIR_t ir) {
    print_separator("MLIR Module (High-Level IR)");

#ifdef CML_HAS_MLIR
    if (ir && ir->mlir_ctx) {
        // Access the module from the context
        typedef struct {
            void* context;
            void* location;
            void* module;
            // ... other fields
        } MLIRCtxInternal;

        MLIRCtxInternal* ctx = (MLIRCtxInternal*)ir->mlir_ctx;
        char* mlir_str       = cml_mlir_dump_module(ctx->module);
        if (mlir_str) {
            printf("%s\n", mlir_str);
            free(mlir_str);
        } else {
            printf("  (Failed to dump MLIR module)\n");
        }
    } else {
        printf("  (No MLIR context available)\n");
    }
#else
    printf("  (MLIR not compiled in)\n");
    (void)ir;
#endif
}

static void print_llvm_ir(CMLIR_t ir) {
    print_separator("LLVM IR (CPU Backend)");

#ifdef CML_HAS_MLIR
    if (ir && ir->mlir_ctx) {
        typedef struct {
            void* context;
            void* location;
            void* module;
        } MLIRCtxInternal;

        MLIRCtxInternal* ctx = (MLIRCtxInternal*)ir->mlir_ctx;
        char* llvm_ir        = cml_mlir_gen_llvm_ir(ctx->module);
        if (llvm_ir) {
            printf("%s\n", llvm_ir);
            free(llvm_ir);
        } else {
            printf("  (Failed to generate LLVM IR - lowering may have failed)\n");
        }
    } else {
        printf("  (No MLIR context available)\n");
    }
#else
    printf("  (MLIR not compiled in)\n");
    (void)ir;
#endif
}

static void print_cuda_ptx(CMLIR_t ir) {
    print_separator("CUDA PTX (NVIDIA GPU Backend)");

#ifdef CML_HAS_MLIR
    if (ir && ir->mlir_ctx) {
        typedef struct {
            void* context;
            void* location;
            void* module;
        } MLIRCtxInternal;

        MLIRCtxInternal* ctx = (MLIRCtxInternal*)ir->mlir_ctx;
        char* ptx            = cml_mlir_gen_ptx(ctx->module);
        if (ptx) {
            printf("%s\n", ptx);
            free(ptx);
        } else {
            printf("  (PTX generation not implemented or GPU dialect not used)\n");
        }
    } else {
        printf("  (No MLIR context available)\n");
    }
#else
    printf("  (MLIR not compiled in)\n");
    (void)ir;
#endif
}

static void print_spirv(CMLIR_t ir) {
    print_separator("SPIR-V (Vulkan Backend)");

#ifdef CML_HAS_MLIR
    if (ir && ir->mlir_ctx) {
        typedef struct {
            void* context;
            void* location;
            void* module;
        } MLIRCtxInternal;

        MLIRCtxInternal* ctx = (MLIRCtxInternal*)ir->mlir_ctx;
        size_t size          = 0;
        uint32_t* spirv      = cml_mlir_gen_spirv(ctx->module, &size);
        if (spirv && size > 0) {
            printf("  SPIR-V Binary (%zu bytes, %zu words):\n", size, size / 4);
            printf("  Magic: 0x%08X\n", spirv[0]);
            printf("  Version: %d.%d\n", (spirv[1] >> 16) & 0xFF, (spirv[1] >> 8) & 0xFF);
            printf("  Generator: 0x%08X\n", spirv[2]);
            printf("  Bound: %u\n", spirv[3]);
            printf("  ...\n");
            printf("  (Full binary: %zu words)\n", size / 4);
            free(spirv);
        } else {
            printf("  (SPIR-V generation not implemented or SPIRV dialect not used)\n");
        }
    } else {
        printf("  (No MLIR context available)\n");
    }
#else
    printf("  (MLIR not compiled in)\n");
    (void)ir;
#endif
}

static void print_metal_msl(CMLIR_t ir) {
    print_separator("Metal Shading Language (Apple GPU Backend)");

#ifdef CML_HAS_MLIR
    if (ir && ir->mlir_ctx) {
        typedef struct {
            void* context;
            void* location;
            void* module;
        } MLIRCtxInternal;

        MLIRCtxInternal* ctx = (MLIRCtxInternal*)ir->mlir_ctx;
        char* msl            = cml_mlir_gen_metal(ctx->module);
        if (msl) {
            printf("%s\n", msl);
            free(msl);
        } else {
            printf("  (MSL generation not implemented)\n");
        }
    } else {
        printf("  (No MLIR context available)\n");
    }
#else
    printf("  (MLIR not compiled in)\n");
    (void)ir;
#endif
}

static void print_wgsl(CMLIR_t ir) {
    print_separator("WGSL (WebGPU Shading Language)");

#ifdef CML_HAS_MLIR
    if (ir && ir->mlir_ctx) {
        typedef struct {
            void* context;
            void* location;
            void* module;
        } MLIRCtxInternal;

        MLIRCtxInternal* ctx = (MLIRCtxInternal*)ir->mlir_ctx;
        char* wgsl           = cml_mlir_gen_wgsl(ctx->module);
        if (wgsl) {
            printf("%s\n", wgsl);
            free(wgsl);
        } else {
            printf("  (WGSL generation not implemented)\n");
        }
    } else {
        printf("  (No MLIR context available)\n");
    }
#else
    printf("  (MLIR not compiled in)\n");
    (void)ir;
#endif
}

static void print_cpu_fallback_pseudocode(CMLIR_t ir) {
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

        // Print operation type
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

        // Print shapes
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

        // Print pseudocode for the operation
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
    printf("========================================\n");
    printf("     CML Kernel Code Generator\n");
    printf("========================================\n");

    // Parse args
    int size = 4; // Small size for readable output
    if (argc > 1) {
        size = atoi(argv[1]);
    }

    printf("\nGenerating kernels for %dx%d matrix multiplication...\n", size, size);

    // Create tensors
    Tensor* A = tensor_empty_2d(size, size);
    Tensor* B = tensor_empty_2d(size, size);

    if (!A || !B) {
        printf("Failed to allocate tensors\n");
        return 1;
    }

    // Fill with sample data
    float* a_data = (float*)A->data;
    float* b_data = (float*)B->data;
    for (int i = 0; i < size * size; i++) {
        a_data[i] = (float)(i % 10) / 10.0f;
        b_data[i] = (float)((i + 1) % 10) / 10.0f;
    }

    // Create IR and add matmul operation
    CMLIR_t ir = cml_ir_new(IR_TARGET_C);
    cml_ir_set_global_context(ir);

    // Add matmul to IR
    tensor_matmul(A, B);

#ifdef CML_HAS_MLIR
    // Initialize MLIR context and convert IR
    printf("Initializing MLIR and converting IR...\n");
    CMLMLIRContext* mlir_ctx = cml_mlir_init();
    if (mlir_ctx) {
        ir->mlir_ctx = mlir_ctx;
        if (cml_ir_to_mlir(mlir_ctx, ir)) {
            printf("MLIR conversion complete.\n");
        } else {
            printf("MLIR conversion failed.\n");
        }
    } else {
        printf("Failed to initialize MLIR context.\n");
    }
#endif

    // Print all kernel representations
    print_cpu_fallback_pseudocode(ir);
    print_mlir_module(ir);
    print_llvm_ir(ir);
    print_cuda_ptx(ir);
    print_spirv(ir);
    print_metal_msl(ir);
    print_wgsl(ir);

    // Cleanup
    cml_ir_free(ir);
    tensor_free(A);
    tensor_free(B);

    printf("\n========================================\n");
    printf("           Generation Complete\n");
    printf("========================================\n\n");

    return 0;
}
