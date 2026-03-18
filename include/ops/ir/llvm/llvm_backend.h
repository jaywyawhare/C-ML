/*
 * Direct LLVM IR backend for CML.
 * Builds LLVM IR from UOps using the LLVM C API, optimizes with
 * the new PassBuilder (O2), and JIT-compiles via ORC LLJIT.
 * Uses a raw float* pointer ABI (no memref descriptors).
 */

#ifndef CML_OPS_IR_LLVM_BACKEND_H
#define CML_OPS_IR_LLVM_BACKEND_H

#include "ops/ir/ir.h"
#include "tensor/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLLLVMBackend CMLLLVMBackend;

CMLLLVMBackend* cml_llvm_backend_init(void);
void cml_llvm_backend_destroy(CMLLLVMBackend* backend);

/* Walks the IR node list, builds LLVM IR for un-executed nodes,
   optimizes, JIT-compiles, and executes. */
int cml_llvm_execute(CMLLLVMBackend* backend, CMLGraph_t ir);
int cml_llvm_execute_up_to(CMLLLVMBackend* backend, CMLGraph_t ir,
                           struct IRNode* target_node);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_LLVM_BACKEND_H
