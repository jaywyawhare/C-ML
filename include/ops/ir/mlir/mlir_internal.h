#ifndef CML_OPS_IR_MLIR_INTERNAL_H
#define CML_OPS_IR_MLIR_INTERNAL_H

#ifdef CML_HAS_MLIR
#include <mlir-c/IR.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Math.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/ExecutionEngine.h>
#include <mlir-c/Pass.h>

#include "ops/ir/mlir/mlir_backend.h"
#include "ops/ir/ir.h"
#include "core/logging.h"

// ============================================================================
// Internal Structures
// ============================================================================

struct CMLMLIRContext {
    MlirContext context;
    MlirLocation location;
    MlirModule module;
    bool initialized;
    MLIRTargetBackend target;
    // Input/Output tracking for execution
    Tensor** inputs;
    int num_inputs;
    Tensor** outputs;
    int num_outputs;
    // All tensors in the IR (for MLIR execution)
    Tensor** tensors;
    int num_tensors;
    // Cached execution engine
    void* cached_engine; // MlirExecutionEngine stored as void*
};

// Symbol Table Entry
typedef struct {
    char* key;
    MlirValue value;
} SymbolEntry;

// Symbol Table
typedef struct {
    SymbolEntry* entries;
    size_t count;
    size_t capacity;
} SymbolTable;

// ============================================================================
// Internal Helper Functions
// ============================================================================

void symbol_table_init(SymbolTable* table);
void symbol_table_free(SymbolTable* table);
void symbol_table_insert(SymbolTable* table, const char* key, MlirValue value);
MlirValue symbol_table_lookup(SymbolTable* table, const char* key);

// MLIR Construction Helpers
MlirType get_f32_type(MlirContext ctx);
MlirType get_tensor_type(MlirContext ctx, int64_t* shape, int rank);
MlirValue create_constant_f32(CMLMLIRContext* ctx, MlirBlock block, float value);
MlirValue create_memref_alloc(CMLMLIRContext* ctx, MlirBlock block, int64_t* shape, int rank);
MlirValue create_memref_alloc_from_type(CMLMLIRContext* ctx, MlirBlock block, MlirType memref_type);
MlirValue create_linalg_fill(CMLMLIRContext* ctx, MlirBlock block, MlirValue input,
                             MlirValue value);
void create_linalg_matmul(CMLMLIRContext* ctx, MlirBlock block, MlirValue a, MlirValue b,
                          MlirValue output);
MlirValue emit_binary_op(CMLMLIRContext* ctx, MlirBlock block, const char* op_name, MlirValue lhs,
                         MlirValue rhs);
MlirValue emit_unary_op(CMLMLIRContext* ctx, MlirBlock block, const char* op_name,
                        MlirValue operand);
MlirValue create_transpose(CMLMLIRContext* ctx, MlirBlock block, MlirValue input,
                           int64_t* out_shape, int rank);
MlirValue create_reduction(CMLMLIRContext* ctx, MlirBlock block, MlirValue input, int* reduced_dims,
                           int num_reduced_dims, int64_t* out_shape, int out_rank,
                           const char* reduce_op);

// MLIR Module Utilities
void mlir_print_module(MlirModule module);
char* mlir_module_to_string(MlirModule module);
bool mlir_verify_module(MlirModule module);
#endif

#endif // CML_OPS_IR_MLIR_INTERNAL_H
