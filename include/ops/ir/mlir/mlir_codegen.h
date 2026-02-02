#ifndef CML_OPS_IR_MLIR_CODEGEN_H
#define CML_OPS_IR_MLIR_CODEGEN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Translate MLIR module to LLVM IR
 * @param module MLIR module (read-only)
 * @return Allocated string containing LLVM IR (caller must free)
 */
char* cml_mlir_gen_llvm_ir(const void* module);

/**
 * @brief Translate MLIR module to CUDA PTX
 * @param module MLIR module (read-only)
 * @return Allocated string containing PTX code (caller must free)
 */
char* cml_mlir_gen_ptx(const void* module);

/**
 * @brief Translate MLIR module to SPIR-V binary
 * @param module MLIR module (read-only)
 * @param size Output: Size of the binary in bytes
 * @return Allocated buffer containing SPIR-V binary (caller must free)
 */
uint32_t* cml_mlir_gen_spirv(const void* module, size_t* size);

/**
 * @brief Translate MLIR module to Metal Shading Language
 * @param module MLIR module (read-only)
 * @return Allocated string containing MSL code (caller must free)
 */
char* cml_mlir_gen_metal(const void* module);

/**
 * @brief Translate MLIR module to WGSL (WebGPU)
 * @param module MLIR module (read-only)
 * @return Allocated string containing WGSL code (caller must free)
 */
char* cml_mlir_gen_wgsl(const void* module);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_MLIR_CODEGEN_H
