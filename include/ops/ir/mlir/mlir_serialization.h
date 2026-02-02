#ifndef CML_OPS_IR_MLIR_SERIALIZATION_H
#define CML_OPS_IR_MLIR_SERIALIZATION_H

#include "ops/ir/mlir/mlir_context.h"
#include "ops/ir/mlir/mlir_multi_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Serialize MLIR module to text file
 * @param mlir_module MLIR module to serialize
 * @param filepath Output file path
 * @return 0 on success, -1 on failure
 */
int cml_mlir_serialize_to_file(const void* mlir_module, const char* filepath);

/**
 * @brief Deserialize MLIR module from text file
 * @param ctx MLIR context
 * @param filepath Input file path
 * @return MLIR module on success, NULL on failure
 */
const void* cml_mlir_deserialize_from_file(CMLMLIRContext* ctx, const char* filepath);

/**
 * @brief Serialize MLIR module to bytecode
 * @param mlir_module MLIR module to serialize
 * @param filepath Output file path
 * @return 0 on success, -1 on failure
 */
int cml_mlir_serialize_bytecode(const void* mlir_module, const char* filepath);

/**
 * @brief Export model for deployment with all necessary files
 * @param mlir_module MLIR module
 * @param output_dir Output directory for deployment files
 * @param target Target backend
 * @return 0 on success, -1 on failure
 *
 * Creates deployment package with:
 * - MLIR text representation
 * - Target-specific code (LLVM IR, PTX, SPIR-V, etc.)
 * - Compiled binaries (for CPU target)
 */
int cml_mlir_export_for_deployment(const void* mlir_module, const char* output_dir,
                                   MLIRTargetBackend target);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_MLIR_SERIALIZATION_H
