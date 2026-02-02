/**
 * @file mlir_utils.c
 * @brief MLIR utility functions for type creation, operation emission, etc.
 */

#include "ops/ir/mlir/mlir_internal.h"
#include "ops/ir/mlir/mlir_cpp_bridge.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Symbol Table Implementation
// ============================================================================

void symbol_table_init(SymbolTable* table) {
    table->entries  = NULL;
    table->count    = 0;
    table->capacity = 0;
}

void symbol_table_free(SymbolTable* table) {
    if (!table)
        return;

    for (size_t i = 0; i < table->count; i++) {
        if (table->entries[i].key) {
            free(table->entries[i].key);
        }
    }

    if (table->entries) {
        free(table->entries);
    }

    table->entries  = NULL;
    table->count    = 0;
    table->capacity = 0;
}

#ifdef CML_HAS_MLIR

void symbol_table_insert(SymbolTable* table, const char* key, MlirValue value) {
    if (!table || !key)
        return;

    // Check if key already exists
    for (size_t i = 0; i < table->count; i++) {
        if (strcmp(table->entries[i].key, key) == 0) {
            table->entries[i].value = value;
            return;
        }
    }

    // Expand capacity if needed
    if (table->count >= table->capacity) {
        size_t new_capacity = table->capacity == 0 ? 16 : table->capacity * 2;
        SymbolEntry* new_entries =
            (SymbolEntry*)realloc(table->entries, new_capacity * sizeof(SymbolEntry));
        if (!new_entries)
            return;

        table->entries  = new_entries;
        table->capacity = new_capacity;
    }

    // Add new entry
    table->entries[table->count].key   = strdup(key);
    table->entries[table->count].value = value;
    table->count++;
}

MlirValue symbol_table_lookup(SymbolTable* table, const char* key) {
    if (!table || !key) {
        MlirValue null_val = {NULL};
        return null_val;
    }

    for (size_t i = 0; i < table->count; i++) {
        if (strcmp(table->entries[i].key, key) == 0) {
            return table->entries[i].value;
        }
    }

    MlirValue null_val = {NULL};
    return null_val;
}

// ============================================================================
// Type Creation Helpers
// ============================================================================

MlirType get_f32_type(MlirContext ctx) { return mlirF32TypeGet(ctx); }

MlirType get_tensor_type(MlirContext context, int64_t* shape, int rank) {
    MlirType element_type = get_f32_type(context);

    // Use memref instead of tensor to bypass bufferization
    // This allows direct lowering to LLVM without the bufferization pass
    return mlirMemRefTypeGet(element_type, rank, shape, mlirAttributeGetNull(),
                             mlirAttributeGetNull());
}

// ============================================================================
// Operation Emission Helpers
// ============================================================================

MlirValue create_constant_f32(CMLMLIRContext* ctx, MlirBlock block, float value) {
    MlirType f32 = get_f32_type(ctx->context);

    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("arith.constant"), ctx->location);

    MlirAttribute value_attr      = mlirFloatAttrDoubleGet(ctx->context, f32, (double)value);
    MlirNamedAttribute named_attr = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("value")), value_attr);

    mlirOperationStateAddAttributes(&state, 1, &named_attr);
    mlirOperationStateAddResults(&state, 1, &f32);

    MlirOperation op = mlirOperationCreate(&state);
    mlirBlockAppendOwnedOperation(block, op);

    return mlirOperationGetResult(op, 0);
}

MlirValue create_memref_alloc(CMLMLIRContext* ctx, MlirBlock block, int64_t* shape, int rank) {
    // Create the memref type with the given shape
    MlirType memref_type = mlirMemRefTypeGet(get_f32_type(ctx->context), rank, shape,
                                             mlirAttributeGetNull(), mlirAttributeGetNull());

    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("memref.alloc"), ctx->location);

    mlirOperationStateAddResults(&state, 1, &memref_type);

    MlirOperation op = mlirOperationCreate(&state);
    mlirBlockAppendOwnedOperation(block, op);

    return mlirOperationGetResult(op, 0);
}

// Create allocated memref from an existing memref type (for dynamic shapes)
MlirValue create_memref_alloc_from_type(CMLMLIRContext* ctx, MlirBlock block,
                                        MlirType memref_type) {
    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("memref.alloc"), ctx->location);

    mlirOperationStateAddResults(&state, 1, &memref_type);

    MlirOperation op = mlirOperationCreate(&state);
    mlirBlockAppendOwnedOperation(block, op);

    return mlirOperationGetResult(op, 0);
}

MlirValue create_linalg_fill(CMLMLIRContext* ctx, MlirBlock block, MlirValue input,
                             MlirValue value) {
    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.fill"), ctx->location);

    mlirOperationStateAddOperands(&state, 1, &value);
    mlirOperationStateAddOperands(&state, 1, &input);

    // operandSegmentSizes: [num_inputs, num_outputs] = [1, 1] for fill
    int32_t fill_sizes[]                  = {1, 1};
    MlirAttribute fill_segment_attr       = mlirDenseI32ArrayGet(ctx->context, 2, fill_sizes);
    MlirNamedAttribute fill_segment_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("operandSegmentSizes")),
        fill_segment_attr);
    mlirOperationStateAddAttributes(&state, 1, &fill_segment_named);

    // linalg.fill requires a region with a yield
    MlirRegion body_region    = mlirRegionCreate();
    MlirType f32_type         = get_f32_type(ctx->context);
    MlirType body_arg_types[] = {f32_type, f32_type};
    MlirBlock body_block      = mlirBlockCreate(2, body_arg_types, &ctx->location);
    mlirRegionAppendOwnedBlock(body_region, body_block);

    // Yield the fill value (first argument)
    MlirValue fill_val = mlirBlockGetArgument(body_block, 0);
    MlirOperationState yield_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.yield"), ctx->location);
    mlirOperationStateAddOperands(&yield_state, 1, &fill_val);
    MlirOperation yield_op = mlirOperationCreate(&yield_state);
    mlirBlockAppendOwnedOperation(body_block, yield_op);

    mlirOperationStateAddOwnedRegions(&state, 1, &body_region);

    MlirOperation op = mlirOperationCreate(&state);
    mlirBlockAppendOwnedOperation(block, op);

    // Return the input memref (which is now filled)
    return input;
}

void create_linalg_matmul(CMLMLIRContext* ctx, MlirBlock block, MlirValue a, MlirValue b,
                          MlirValue output) {
    MlirOperationState matmul_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.matmul"), ctx->location);

    mlirOperationStateAddOperands(&matmul_state, 1, &a);
    mlirOperationStateAddOperands(&matmul_state, 1, &b);
    mlirOperationStateAddOperands(&matmul_state, 1, &output);

    // operandSegmentSizes: [num_inputs, num_outputs] = [2, 1] for matmul
    int32_t matmul_sizes[]                  = {2, 1};
    MlirAttribute matmul_segment_attr       = mlirDenseI32ArrayGet(ctx->context, 2, matmul_sizes);
    MlirNamedAttribute matmul_segment_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("operandSegmentSizes")),
        matmul_segment_attr);
    mlirOperationStateAddAttributes(&matmul_state, 1, &matmul_segment_named);

    // linalg.matmul requires a region with body: acc + lhs * rhs
    MlirRegion body_region    = mlirRegionCreate();
    MlirType f32_type         = get_f32_type(ctx->context);
    MlirType body_arg_types[] = {f32_type, f32_type, f32_type};
    MlirLocation locs[]       = {ctx->location, ctx->location, ctx->location};
    MlirBlock body_block      = mlirBlockCreate(3, body_arg_types, locs);
    mlirRegionAppendOwnedBlock(body_region, body_block);

    MlirValue lhs_arg = mlirBlockGetArgument(body_block, 0);
    MlirValue rhs_arg = mlirBlockGetArgument(body_block, 1);
    MlirValue acc_arg = mlirBlockGetArgument(body_block, 2);

    // mul: lhs * rhs
    MlirOperationState mul_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("arith.mulf"), ctx->location);
    mlirOperationStateAddOperands(&mul_state, 1, &lhs_arg);
    mlirOperationStateAddOperands(&mul_state, 1, &rhs_arg);
    mlirOperationStateAddResults(&mul_state, 1, &f32_type);
    MlirOperation mul_op = mlirOperationCreate(&mul_state);
    mlirBlockAppendOwnedOperation(body_block, mul_op);
    MlirValue mul_result = mlirOperationGetResult(mul_op, 0);

    // add: acc + (lhs * rhs)
    MlirOperationState add_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("arith.addf"), ctx->location);
    mlirOperationStateAddOperands(&add_state, 1, &acc_arg);
    mlirOperationStateAddOperands(&add_state, 1, &mul_result);
    mlirOperationStateAddResults(&add_state, 1, &f32_type);
    MlirOperation add_op = mlirOperationCreate(&add_state);
    mlirBlockAppendOwnedOperation(body_block, add_op);
    MlirValue add_result = mlirOperationGetResult(add_op, 0);

    // yield
    MlirOperationState yield_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.yield"), ctx->location);
    mlirOperationStateAddOperands(&yield_state, 1, &add_result);
    MlirOperation yield_op = mlirOperationCreate(&yield_state);
    mlirBlockAppendOwnedOperation(body_block, yield_op);

    mlirOperationStateAddOwnedRegions(&matmul_state, 1, &body_region);

    MlirOperation matmul_op = mlirOperationCreate(&matmul_state);
    mlirBlockAppendOwnedOperation(block, matmul_op);
}

// Helper to check if a type is a tensor or memref (shaped type)
static bool is_tensor_type(MlirType type) {
    if (mlirTypeIsNull(type))
        return false;

    // Check if it's a shaped type (tensor or memref)
    return mlirTypeIsAShaped(type);
}

// Create linalg.generic for elementwise binary operation
static MlirValue create_linalg_generic_binary(CMLMLIRContext* ctx, MlirBlock block,
                                              const char* body_op, MlirValue lhs, MlirValue rhs) {
    MlirType lhs_type    = mlirValueGetType(lhs);
    MlirType rhs_type    = mlirValueGetType(rhs);
    int rank_lhs         = is_tensor_type(lhs_type) ? mlirShapedTypeGetRank(lhs_type) : 0;
    int rank_rhs         = is_tensor_type(rhs_type) ? mlirShapedTypeGetRank(rhs_type) : 0;
    MlirType result_type = lhs_type; // Assume lhs is result shape

    // Swap if rank_lhs < rank_rhs to make lhs the larger one
    if (rank_lhs < rank_rhs) {
        MlirValue temp_val = lhs;
        lhs                = rhs;
        rhs                = temp_val;
        MlirType temp_type = lhs_type;
        lhs_type           = rhs_type;
        rhs_type           = temp_type;
        int temp_rank      = rank_lhs;
        rank_lhs           = rank_rhs;
        rank_rhs           = temp_rank;
        result_type        = lhs_type;
    }

    // Create output memref with the same type as the result
    MlirValue output = create_memref_alloc_from_type(ctx, block, result_type);

    // Create linalg.generic operation
    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.generic"), ctx->location);

    // Add inputs and output (output is also an operand for memrefs)
    MlirValue operands[] = {lhs, rhs, output};
    mlirOperationStateAddOperands(&state, 3, operands);

    // For memrefs, linalg.generic has NO results - it writes to the output operand
    // mlirOperationStateAddResults(&state, 1, &result_type); // REMOVED

    // Add indexing maps
    MlirAttribute indexing_maps_attr;
    int num_iterators = rank_lhs;
    if (rank_lhs == rank_rhs) {
        // Same rank, identity maps
        indexing_maps_attr = cml_mlir_create_indexing_maps_attr(ctx->context, 3, rank_lhs);
    } else if (rank_rhs == 0) {
        // Scalar rhs, broadcast
        indexing_maps_attr = cml_mlir_create_scalar_broadcast_maps_attr(ctx->context, rank_lhs);
    } else if (rank_rhs == 1 && rank_lhs > 1) {
        // Broadcast rhs on last dim
        indexing_maps_attr = cml_mlir_create_broadcast_maps_attr(ctx->context, rank_lhs);
    } else {
        LOG_ERROR(
            "Unsupported rank combination for linalg.generic binary: lhs rank %d, rhs rank %d",
            rank_lhs, rank_rhs);
        MlirValue null_val = {NULL};
        return null_val;
    }
    if (mlirAttributeIsNull(indexing_maps_attr)) {
        LOG_ERROR("Failed to create indexing maps for linalg.generic");
        MlirValue null_val = {NULL};
        return null_val;
    }
    MlirNamedAttribute indexing_maps_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("indexing_maps")),
        indexing_maps_attr);
    mlirOperationStateAddAttributes(&state, 1, &indexing_maps_named);

    // Add iterator types: parallel for all (using linalg iterator type enum)
    MlirAttribute* iterator_attrs = (MlirAttribute*)malloc(sizeof(MlirAttribute) * num_iterators);
    MlirAttribute parallel_attr   = mlirAttributeParseGet(
        ctx->context, mlirStringRefCreateFromCString("#linalg.iterator_type<parallel>"));
    for (int i = 0; i < num_iterators; i++) {
        iterator_attrs[i] = parallel_attr;
    }
    MlirAttribute iterator_types_attr =
        mlirArrayAttrGet(ctx->context, num_iterators, iterator_attrs);
    free(iterator_attrs);
    MlirNamedAttribute iterator_attr = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("iterator_types")),
        iterator_types_attr);
    mlirOperationStateAddAttributes(&state, 1, &iterator_attr);

    // operandSegmentSizes: [num_inputs, num_outputs] = [2, 1] for binary ops
    int32_t binary_sizes[]                  = {2, 1};
    MlirAttribute binary_segment_attr       = mlirDenseI32ArrayGet(ctx->context, 2, binary_sizes);
    MlirNamedAttribute binary_segment_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("operandSegmentSizes")),
        binary_segment_attr);
    mlirOperationStateAddAttributes(&state, 1, &binary_segment_named);

    // Create body region with a block
    MlirRegion body_region = mlirRegionCreate();
    MlirType f32_type      = get_f32_type(ctx->context);
    // For memrefs, the body block arguments match the operands (lhs, rhs, out)
    MlirType body_arg_types[] = {f32_type, f32_type, f32_type};
    MlirBlock body_block      = mlirBlockCreate(3, body_arg_types, &ctx->location);
    mlirRegionAppendOwnedBlock(body_region, body_block);

    // Add body operation: the actual arithmetic
    MlirValue arg0 = mlirBlockGetArgument(body_block, 0);
    MlirValue arg1 = mlirBlockGetArgument(body_block, 1);

    MlirOperationState body_op_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString(body_op), ctx->location);
    mlirOperationStateAddOperands(&body_op_state, 1, &arg0);
    mlirOperationStateAddOperands(&body_op_state, 1, &arg1);
    mlirOperationStateAddResults(&body_op_state, 1, &f32_type);
    MlirOperation body_op_inst = mlirOperationCreate(&body_op_state);
    mlirBlockAppendOwnedOperation(body_block, body_op_inst);
    MlirValue body_result = mlirOperationGetResult(body_op_inst, 0);

    // Add linalg.yield
    MlirOperationState yield_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.yield"), ctx->location);
    mlirOperationStateAddOperands(&yield_state, 1, &body_result);
    MlirOperation yield_op = mlirOperationCreate(&yield_state);
    mlirBlockAppendOwnedOperation(body_block, yield_op);

    mlirOperationStateAddOwnedRegions(&state, 1, &body_region);

    MlirOperation op = mlirOperationCreate(&state);
    mlirBlockAppendOwnedOperation(block, op);

    // Return the output memref (which now contains the result)
    return output;
}

// Broadcast scalar to memref shape using linalg.fill
static MlirValue broadcast_scalar_to_tensor(CMLMLIRContext* ctx, MlirBlock block, MlirValue scalar,
                                            MlirType memref_type) {
    // Allocate output memref with the given shape
    MlirValue output = create_memref_alloc_from_type(ctx, block, memref_type);

    // Use linalg.fill to fill the memref with the scalar value
    return create_linalg_fill(ctx, block, output, scalar);
}

MlirValue emit_binary_op(CMLMLIRContext* ctx, MlirBlock block, const char* op_name, MlirValue lhs,
                         MlirValue rhs) {
    // Check for null values first
    if (mlirValueIsNull(lhs) || mlirValueIsNull(rhs)) {
        LOG_ERROR("emit_binary_op: null MLIR value(s) - lhs: %s, rhs: %s",
                  mlirValueIsNull(lhs) ? "null" : "valid", mlirValueIsNull(rhs) ? "null" : "valid");
        MlirValue null_val = {NULL};
        return null_val;
    }

    // Check if operands are tensors
    MlirType lhs_type  = mlirValueGetType(lhs);
    MlirType rhs_type  = mlirValueGetType(rhs);
    bool lhs_is_tensor = is_tensor_type(lhs_type);
    bool rhs_is_tensor = is_tensor_type(rhs_type);

    if (lhs_is_tensor && rhs_is_tensor) {
        // Both tensors
        return create_linalg_generic_binary(ctx, block, op_name, lhs, rhs);
    } else if (lhs_is_tensor && !rhs_is_tensor) {
        // lhs tensor, rhs scalar: broadcast rhs
        MlirValue broadcasted_rhs = broadcast_scalar_to_tensor(ctx, block, rhs, lhs_type);
        return create_linalg_generic_binary(ctx, block, op_name, lhs, broadcasted_rhs);
    } else if (!lhs_is_tensor && rhs_is_tensor) {
        // lhs scalar, rhs tensor: broadcast lhs
        MlirValue broadcasted_lhs = broadcast_scalar_to_tensor(ctx, block, lhs, rhs_type);
        return create_linalg_generic_binary(ctx, block, op_name, broadcasted_lhs, rhs);
    } else {
        // Both scalars: use arith directly
        MlirOperationState state =
            mlirOperationStateGet(mlirStringRefCreateFromCString(op_name), ctx->location);

        mlirOperationStateAddOperands(&state, 1, &lhs);
        mlirOperationStateAddOperands(&state, 1, &rhs);

        MlirType result_type = mlirValueGetType(lhs);
        mlirOperationStateAddResults(&state, 1, &result_type);

        MlirOperation op = mlirOperationCreate(&state);
        mlirBlockAppendOwnedOperation(block, op);

        return mlirOperationGetResult(op, 0);
    }
}
static MlirValue create_linalg_generic_unary(CMLMLIRContext* ctx, MlirBlock block,
                                             const char* body_op, MlirValue operand) {
    MlirType result_type = mlirValueGetType(operand);

    // Create output memref with the same type as the result
    MlirValue output = create_memref_alloc_from_type(ctx, block, result_type);

    // Create linalg.generic operation
    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.generic"), ctx->location);

    // Add inputs and output (output is also an operand for memrefs)
    MlirValue operands[] = {operand, output};
    mlirOperationStateAddOperands(&state, 2, operands);

    // For memrefs, linalg.generic has NO results - it writes to the output operand
    // mlirOperationStateAddResults(&state, 1, &result_type); // REMOVED

    // Get rank from the result type
    int rank = mlirShapedTypeGetRank(result_type);

    // Add indexing maps: identity for all operands (2 maps: input, output)
    MlirAttribute indexing_maps_attr = cml_mlir_create_indexing_maps_attr(ctx->context, 2, rank);
    if (mlirAttributeIsNull(indexing_maps_attr)) {
        LOG_ERROR("Failed to create indexing maps for linalg.generic");
        MlirValue null_val = {NULL};
        return null_val;
    }
    MlirNamedAttribute indexing_maps_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("indexing_maps")),
        indexing_maps_attr);
    mlirOperationStateAddAttributes(&state, 1, &indexing_maps_named);

    // Add iterator types: parallel for all (based on rank, using linalg enum)
    MlirAttribute* iter_attrs   = (MlirAttribute*)malloc(sizeof(MlirAttribute) * rank);
    MlirAttribute parallel_attr = mlirAttributeParseGet(
        ctx->context, mlirStringRefCreateFromCString("#linalg.iterator_type<parallel>"));
    for (int i = 0; i < rank; i++) {
        iter_attrs[i] = parallel_attr;
    }
    MlirAttribute iterator_types_attr = mlirArrayAttrGet(ctx->context, rank, iter_attrs);
    free(iter_attrs);
    MlirNamedAttribute iterator_attr = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("iterator_types")),
        iterator_types_attr);
    mlirOperationStateAddAttributes(&state, 1, &iterator_attr);

    // operandSegmentSizes: [num_inputs, num_outputs] = [1, 1] for unary ops
    int32_t unary_sizes[]                  = {1, 1};
    MlirAttribute unary_segment_attr       = mlirDenseI32ArrayGet(ctx->context, 2, unary_sizes);
    MlirNamedAttribute unary_segment_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("operandSegmentSizes")),
        unary_segment_attr);
    mlirOperationStateAddAttributes(&state, 1, &unary_segment_named);

    // Create body region with a block
    MlirRegion body_region = mlirRegionCreate();
    MlirType f32_type      = get_f32_type(ctx->context);
    // For memrefs, the body block arguments match the operands (input, out)
    MlirType body_arg_types[] = {f32_type, f32_type};
    MlirBlock body_block      = mlirBlockCreate(2, body_arg_types, &ctx->location);
    mlirRegionAppendOwnedBlock(body_region, body_block);

    // Add body operation: the actual arithmetic
    MlirValue arg0 = mlirBlockGetArgument(body_block, 0);

    MlirOperationState body_op_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString(body_op), ctx->location);
    mlirOperationStateAddOperands(&body_op_state, 1, &arg0);
    mlirOperationStateAddResults(&body_op_state, 1, &f32_type);
    MlirOperation body_op_inst = mlirOperationCreate(&body_op_state);
    mlirBlockAppendOwnedOperation(body_block, body_op_inst);
    MlirValue body_result = mlirOperationGetResult(body_op_inst, 0);

    // Add linalg.yield
    MlirOperationState yield_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.yield"), ctx->location);
    mlirOperationStateAddOperands(&yield_state, 1, &body_result);
    MlirOperation yield_op = mlirOperationCreate(&yield_state);
    mlirBlockAppendOwnedOperation(body_block, yield_op);

    mlirOperationStateAddOwnedRegions(&state, 1, &body_region);

    MlirOperation op = mlirOperationCreate(&state);
    mlirBlockAppendOwnedOperation(block, op);

    // Return the output memref (which now contains the result)
    return output;
}

MlirValue emit_unary_op(CMLMLIRContext* ctx, MlirBlock block, const char* op_name,
                        MlirValue operand) {
    // Check if operand is a tensor - if so, use linalg.generic
    MlirType operand_type = mlirValueGetType(operand);

    if (is_tensor_type(operand_type)) {
        // Use linalg.generic for tensor operations
        return create_linalg_generic_unary(ctx, block, op_name, operand);
    }

    // For scalar operations, use math/arith directly
    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString(op_name), ctx->location);

    mlirOperationStateAddOperands(&state, 1, &operand);

    MlirType result_type = mlirValueGetType(operand);
    mlirOperationStateAddResults(&state, 1, &result_type);

    MlirOperation op = mlirOperationCreate(&state);
    mlirBlockAppendOwnedOperation(block, op);

    return mlirOperationGetResult(op, 0);
}

MlirValue create_transpose(CMLMLIRContext* ctx, MlirBlock block, MlirValue input,
                           int64_t* out_shape, int rank) {
    // Allocate output memref
    MlirValue output = create_memref_alloc(ctx, block, out_shape, rank);
    if (mlirValueIsNull(output)) {
        LOG_ERROR("Failed to allocate output memref for transpose");
        MlirValue null_val = {NULL};
        return null_val;
    }

    // Create linalg.generic
    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.generic"), ctx->location);

    MlirValue operands[] = {input, output};
    mlirOperationStateAddOperands(&state, 2, operands);

    // Indexing maps
    MlirAttribute maps_attr = cml_mlir_create_transpose_maps_attr(ctx->context);
    if (mlirAttributeIsNull(maps_attr)) {
        LOG_ERROR("Failed to create transpose maps");
        MlirValue null_val = {NULL};
        return null_val;
    }

    MlirNamedAttribute maps_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("indexing_maps")),
        maps_attr);
    mlirOperationStateAddAttributes(&state, 1, &maps_named);

    // Iterator types (using linalg iterator type enum)
    MlirAttribute parallel_attr = mlirAttributeParseGet(
        ctx->context, mlirStringRefCreateFromCString("#linalg.iterator_type<parallel>"));
    MlirAttribute iter_attrs[2]       = {parallel_attr, parallel_attr};
    MlirAttribute iterator_types_attr = mlirArrayAttrGet(ctx->context, 2, iter_attrs);
    MlirNamedAttribute iterator_attr  = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("iterator_types")),
        iterator_types_attr);
    mlirOperationStateAddAttributes(&state, 1, &iterator_attr);

    // operandSegmentSizes: [num_inputs, num_outputs] = [1, 1] for transpose
    int32_t transpose_sizes[]            = {1, 1};
    MlirAttribute transpose_segment_attr = mlirDenseI32ArrayGet(ctx->context, 2, transpose_sizes);
    MlirNamedAttribute transpose_segment_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("operandSegmentSizes")),
        transpose_segment_attr);
    mlirOperationStateAddAttributes(&state, 1, &transpose_segment_named);

    // Body
    MlirRegion body_region = mlirRegionCreate();
    MlirType f32           = get_f32_type(ctx->context);
    MlirType args[]        = {f32, f32};
    MlirLocation locs[]    = {ctx->location, ctx->location};
    MlirBlock body_block   = mlirBlockCreate(2, args, locs);
    mlirRegionAppendOwnedBlock(body_region, body_block);

    MlirValue yield_val = mlirBlockGetArgument(body_block, 0); // Yield input (transpose)

    MlirOperationState yield_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.yield"), ctx->location);
    mlirOperationStateAddOperands(&yield_state, 1, &yield_val);
    mlirBlockAppendOwnedOperation(body_block, mlirOperationCreate(&yield_state));

    mlirOperationStateAddOwnedRegions(&state, 1, &body_region);

    mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&state));

    return output;
}

#include <math.h>
#include <stdio.h>

MlirValue create_reduction(CMLMLIRContext* ctx, MlirBlock block, MlirValue input, int* reduced_dims,
                           int num_reduced_dims, int64_t* out_shape, int out_rank,
                           const char* reduce_op) {

    // Allocate output memref
    MlirValue output = create_memref_alloc(ctx, block, out_shape, out_rank);

    // Initialize output with identity value
    float init_val = 123.0f;
    if (strcmp(reduce_op, "arith.maximumf") == 0)
        init_val = -INFINITY;
    // Add other ops if needed

    MlirValue init_const = create_constant_f32(ctx, block, init_val);
    create_linalg_fill(ctx, block, output, init_const);

    // Create linalg.generic
    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.generic"), ctx->location);

    MlirValue operands[] = {input, output};
    mlirOperationStateAddOperands(&state, 2, operands);

    // Indexing maps
    MlirType input_type = mlirValueGetType(input);
    int rank            = mlirShapedTypeGetRank(input_type);

    MlirAttribute maps_attr = cml_mlir_create_reduction_maps_attr(ctx->context, rank, reduced_dims,
                                                                  num_reduced_dims, out_rank);
    if (mlirAttributeIsNull(maps_attr)) {
        LOG_ERROR("Failed to create reduction maps");
        MlirValue null_val = {NULL};
        return null_val;
    }

    MlirNamedAttribute maps_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("indexing_maps")),
        maps_attr);
    mlirOperationStateAddAttributes(&state, 1, &maps_named);

    // Iterator types (using linalg iterator type enum)
    MlirAttribute* iterators    = (MlirAttribute*)malloc(sizeof(MlirAttribute) * rank);
    MlirAttribute parallel_attr = mlirAttributeParseGet(
        ctx->context, mlirStringRefCreateFromCString("#linalg.iterator_type<parallel>"));
    MlirAttribute reduction_attr = mlirAttributeParseGet(
        ctx->context, mlirStringRefCreateFromCString("#linalg.iterator_type<reduction>"));
    for (int i = 0; i < rank; i++) {
        bool is_reduced = false;
        for (int j = 0; j < num_reduced_dims; j++) {
            if (reduced_dims[j] == i) {
                is_reduced = true;
                break;
            }
        }
        if (is_reduced) {
            iterators[i] = reduction_attr;
        } else {
            iterators[i] = parallel_attr;
        }
    }
    MlirAttribute iterator_types_attr = mlirArrayAttrGet(ctx->context, rank, iterators);
    free(iterators);

    MlirNamedAttribute iterator_attr = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("iterator_types")),
        iterator_types_attr);
    mlirOperationStateAddAttributes(&state, 1, &iterator_attr);

    // operandSegmentSizes: [num_inputs, num_outputs] = [1, 1] for reduction
    int32_t reduce_sizes[]                  = {1, 1};
    MlirAttribute reduce_segment_attr       = mlirDenseI32ArrayGet(ctx->context, 2, reduce_sizes);
    MlirNamedAttribute reduce_segment_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("operandSegmentSizes")),
        reduce_segment_attr);
    mlirOperationStateAddAttributes(&state, 1, &reduce_segment_named);

    // Body
    MlirRegion body_region = mlirRegionCreate();
    MlirType f32           = get_f32_type(ctx->context);
    MlirType args[]        = {f32, f32};
    MlirLocation locs[]    = {ctx->location, ctx->location};
    MlirBlock body_block   = mlirBlockCreate(2, args, locs);
    mlirRegionAppendOwnedBlock(body_region, body_block);

    MlirValue lhs = mlirBlockGetArgument(body_block, 0); // Input element
    MlirValue rhs = mlirBlockGetArgument(body_block, 1); // Accumulator value

    MlirValue res = emit_binary_op(ctx, body_block, reduce_op, lhs, rhs);

    MlirOperationState yield_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.yield"), ctx->location);
    mlirOperationStateAddOperands(&yield_state, 1, &res);
    mlirBlockAppendOwnedOperation(body_block, mlirOperationCreate(&yield_state));

    mlirOperationStateAddOwnedRegions(&state, 1, &body_region);

    mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&state));

    return output;
}

// ============================================================================
// MLIR Module Utilities
// ============================================================================

void mlir_print_module(MlirModule module) {
    MlirOperation op = mlirModuleGetOperation(module);
    mlirOperationPrint(op, NULL, NULL);
}

// Internal helper for string buffer
typedef struct {
    char* data;
    size_t size;
    size_t capacity;
} StringBuffer;

static void append_callback(MlirStringRef str, void* userdata) {
    StringBuffer* buf = (StringBuffer*)userdata;
    size_t needed     = buf->size + str.length + 1;

    if (needed > buf->capacity) {
        size_t new_cap = buf->capacity == 0 ? 1024 : buf->capacity * 2;
        while (new_cap < needed)
            new_cap *= 2;

        char* new_data = (char*)realloc(buf->data, new_cap);
        if (!new_data)
            return;

        buf->data     = new_data;
        buf->capacity = new_cap;
    }

    memcpy(buf->data + buf->size, str.data, str.length);
    buf->size += str.length;
    buf->data[buf->size] = '\0';
}

char* mlir_module_to_string(MlirModule module) {
    StringBuffer buffer = {NULL, 0, 0};

    MlirOperation op = mlirModuleGetOperation(module);
    mlirOperationPrint(op, append_callback, &buffer);

    return buffer.data;
}

// ============================================================================
// Verification and Validation
// ============================================================================

bool mlir_verify_module(MlirModule module) {
    // Run MLIR verification - AffineExpr bug has been fixed
    MlirOperation op = mlirModuleGetOperation(module);

    // mlirOperationVerify returns bool in newer MLIR versions
    bool verified = mlirOperationVerify(op);

    if (!verified) {
        LOG_ERROR("MLIR module verification failed");
        return false;
    }

    LOG_DEBUG("MLIR module verification passed");
    return true;
}

#endif // CML_HAS_MLIR
