/**
 * @file llvm_backend.h
 * @brief Direct LLVM IR backend for CML
 *
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

/** Opaque handle to the LLVM JIT backend context */
typedef struct CMLLLVMBackend CMLLLVMBackend;

/**
 * @brief Initialize the LLVM JIT backend (call once at startup)
 * @return Backend context, or NULL on failure
 */
CMLLLVMBackend* cml_llvm_backend_init(void);

/**
 * @brief Destroy the LLVM JIT backend and free all resources
 * @param backend The backend context
 */
void cml_llvm_backend_destroy(CMLLLVMBackend* backend);

/**
 * @brief Compile and execute an IR graph via LLVM JIT
 *
 * Walks the IR node list, builds LLVM IR for un-executed nodes,
 * optimizes, JIT-compiles, and executes. Output tensors are
 * populated with results.
 *
 * @param backend The LLVM backend context
 * @param ir The IR graph to execute
 * @return 0 on success, -1 on failure
 */
int cml_llvm_execute(CMLLLVMBackend* backend, CMLGraph_t ir);

/**
 * @brief Execute IR graph up to a specific target node
 *
 * @param backend The LLVM backend context
 * @param ir The IR graph
 * @param target_node Execute up to and including this node
 * @return 0 on success, -1 on failure
 */
int cml_llvm_execute_up_to(CMLLLVMBackend* backend, CMLGraph_t ir,
                           struct IRNode* target_node);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_LLVM_BACKEND_H
