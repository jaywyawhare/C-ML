#include "ops/ir/mlir/mlir_uops_builder.h"
#include "ops/ir/mlir/mlir_internal.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// This file currently hosts a direct copy of the legacy cml_ir_to_mlir()
// implementation so we can start treating it as the canonical "uops→MLIR
// builder". In later steps we will tighten the emitted ops and update
// callers to use this entrypoint directly.

bool cml_mlir_build_from_ir(CMLMLIRContext* ctx, CMLIR_t ir) {
#ifdef CML_HAS_MLIR
    if (!ctx || !ir) {
        LOG_ERROR("Invalid arguments to cml_mlir_build_from_ir");
        return false;
    }

    LOG_INFO("Building MLIR module from C-ML IR (uops builder)...");

    // Reset module to avoid symbol redefinition
    if (ctx->module.ptr) {
        mlirModuleDestroy(ctx->module);
    }
    ctx->module = mlirModuleCreateEmpty(ctx->location);

    // Get module body
    MlirOperation module_op = mlirModuleGetOperation(ctx->module);
    MlirRegion body_region  = mlirOperationGetRegion(module_op, 0);
    MlirBlock body          = mlirRegionGetFirstBlock(body_region);
    if (mlirBlockIsNull(body)) {
        LOG_DEBUG("Module body block is NULL, creating one");
        body = mlirBlockCreate(0, NULL, NULL);
        mlirRegionAppendOwnedBlock(body_region, body);
    } else {
        LOG_DEBUG("Module body block found");
    }

    // Determine function signature based on IR inputs/outputs
    // 1. Collect all produced outputs
    SymbolTable produced_outputs;
    symbol_table_init(&produced_outputs);

    struct IRNode* scan_node = ir->head;
    while (scan_node) {
        if (scan_node->output_name) {
            if (scan_node->output_name) {
                MlirValue null_val = {NULL};
                symbol_table_insert(&produced_outputs, scan_node->output_name, null_val);
            }
        }
        scan_node = scan_node->next;
    }

    // 2. Identify graph inputs
#define MAX_GRAPH_INPUTS 1024
    char* graph_inputs[MAX_GRAPH_INPUTS];
    Tensor* found_tensors[MAX_GRAPH_INPUTS];
    int num_graph_inputs = 0;

    scan_node = ir->head;
    while (scan_node) {
        for (int i = 0; i < scan_node->num_inputs; i++) {
            char* input_name = scan_node->input_names[i];
            if (!input_name)
                continue;

            bool is_produced = false;
            for (size_t k = 0; k < produced_outputs.count; k++) {
                if (strcmp(produced_outputs.entries[k].key, input_name) == 0) {
                    is_produced = true;
                    break;
                }
            }

            if (is_produced) {
                // printf("DEBUG: Input '%s' is produced internally\n", input_name);
            } else {

                // CRITICAL FIX: Ensure we don't treat internal nodes as external inputs
                Tensor* input_tensor = scan_node->inputs[i];
                if (input_tensor && input_tensor->ir_context == ir) {

                    LOG_WARNING("Input '%s' appears to be internal (ir_context match) but not "
                                "found in produced_outputs. Skipping to avoid recursion.",
                                input_name);
                    continue;
                }

                bool already_added = false;
                for (int k = 0; k < num_graph_inputs; k++) {
                    if (strcmp(graph_inputs[k], input_name) == 0) {
                        already_added = true;
                        break;
                    }
                }

                if (!already_added && num_graph_inputs < MAX_GRAPH_INPUTS) {
                    graph_inputs[num_graph_inputs]  = strdup(input_name);
                    found_tensors[num_graph_inputs] = scan_node->inputs[i];
                    num_graph_inputs++;
                }
            }
        }
        scan_node = scan_node->next;
    }

    symbol_table_free(&produced_outputs);

    LOG_INFO("Found %d graph inputs", num_graph_inputs);

    // Create input types array
    // We need to include outputs as arguments for Destination Passing Style (DPS)
    int num_outputs = 1; // Assuming 1 output for now
    int num_args    = num_graph_inputs + num_outputs;

    MlirType* func_inputs = (MlirType*)malloc(sizeof(MlirType) * num_args);

    // Input arguments
    for (int i = 0; i < num_graph_inputs; i++) {
        Tensor* t = found_tensors[i];
        if (t) {
            int64_t* shape = (int64_t*)malloc(sizeof(int64_t) * t->ndim);
            for (int j = 0; j < t->ndim; j++)
                shape[j] = t->shape[j];
            func_inputs[i] = get_tensor_type(ctx->context, shape, t->ndim);
            free(shape);
        } else {
            // Fallback
            int64_t dyn_shape[] = {-1};
            func_inputs[i]      = get_tensor_type(ctx->context, dyn_shape, 1);
        }
    }

    // Output arguments
    // We need to know output shape.
    // ir->tail->output should have it.
    if (ir->tail && ir->tail->output) {
        Tensor* t      = ir->tail->output;
        int64_t* shape = (int64_t*)malloc(sizeof(int64_t) * t->ndim);
        for (int j = 0; j < t->ndim; j++)
            shape[j] = t->shape[j];
        func_inputs[num_graph_inputs] = get_tensor_type(ctx->context, shape, t->ndim);
        free(shape);
    } else {
        int64_t dyn_shape[]           = {-1};
        func_inputs[num_graph_inputs] = get_tensor_type(ctx->context, dyn_shape, 1);
    }

    // Function returns void
    MlirType func_type = mlirFunctionTypeGet(ctx->context, num_args, func_inputs, 0, NULL);

    MlirOperationState func_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), ctx->location);

    MlirAttribute func_name =
        mlirStringAttrGet(ctx->context, mlirStringRefCreateFromCString("main"));
    MlirNamedAttribute name_attr = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("sym_name")), func_name);

    MlirAttribute type_attr            = mlirTypeAttrGet(func_type);
    MlirNamedAttribute type_named_attr = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("function_type")),
        type_attr);

    // Add llvm.emit_c_interface attribute for InvokePacked support
    MlirAttribute emit_c_attr       = mlirUnitAttrGet(ctx->context);
    MlirNamedAttribute emit_c_named = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("llvm.emit_c_interface")),
        emit_c_attr);

    MlirNamedAttribute attrs[] = {name_attr, type_named_attr, emit_c_named};
    mlirOperationStateAddAttributes(&func_state, 3, attrs);

    MlirRegion func_region = mlirRegionCreate();
    mlirOperationStateAddOwnedRegions(&func_state, 1, &func_region);

    MlirOperation func_op = mlirOperationCreate(&func_state);
    mlirBlockAppendOwnedOperation(body, func_op);

    MlirRegion func_body_region = mlirOperationGetRegion(func_op, 0);

    MlirLocation* locs = (MlirLocation*)malloc(sizeof(MlirLocation) * num_args);
    for (int i = 0; i < num_args; i++) {
        locs[i] = ctx->location;
    }

    MlirBlock entry_block = mlirBlockCreate(num_args, func_inputs, locs);
    mlirRegionAppendOwnedBlock(func_body_region, entry_block);

    free(locs);
    free(func_inputs);

    // Initialize symbol table
    SymbolTable sym_table;
    symbol_table_init(&sym_table);

    // Map function arguments to discovered inputs
    for (int i = 0; i < num_graph_inputs; i++) {
        MlirValue arg = mlirBlockGetArgument(entry_block, i);
        symbol_table_insert(&sym_table, graph_inputs[i], arg);

        free(graph_inputs[i]);
    }

    // Store inputs/outputs in context for execution
    if (ctx->inputs)
        free(ctx->inputs);
    ctx->inputs = (Tensor**)malloc(sizeof(Tensor*) * num_graph_inputs);
    if (ctx->inputs) {
        memcpy(ctx->inputs, found_tensors, sizeof(Tensor*) * num_graph_inputs);
    }
    ctx->num_inputs = num_graph_inputs;

    if (ctx->outputs)
        free(ctx->outputs);
    ctx->outputs = (Tensor**)malloc(sizeof(Tensor*) * 1);
    if (ctx->outputs) {
        ctx->outputs[0] = ir->tail ? ir->tail->output : NULL;
    }
    ctx->num_outputs = 1;

    // Walk IR nodes and emit MLIR ops (copy of existing mapping)
    struct IRNode* node   = ir->head;
    MlirValue last_result = {NULL};

    while (node) {
        MlirValue inputs[3] = {{NULL}, {NULL}, {NULL}};
        for (int i = 0; i < node->num_inputs && i < 3; i++) {
            inputs[i] = symbol_table_lookup(&sym_table, node->input_names[i]);

            if (mlirValueIsNull(inputs[i])) {

            } else {
                // printf("DEBUG: Symbol lookup success for '%s'\n", node->input_names[i]);
            }

            if (mlirValueIsNull(inputs[i]) && num_graph_inputs > 0) {
                inputs[i] = mlirBlockGetArgument(entry_block, 0);
            }

            if (mlirValueIsNull(inputs[i])) {
                LOG_ERROR("Input %d for node type %d is NULL", i, node->type);
                // Skip this node or handle error?
                // For now, let's try to continue but likely will crash
            }
        }

        MlirValue result = {NULL};
        // LOG_DEBUG("Processing UOp type %d", node->type);

        switch (node->type) {
        case UOP_ADD:
            result = emit_binary_op(ctx, entry_block, "arith.addf", inputs[0], inputs[1]);
            break;
        case UOP_SUB:
            result = emit_binary_op(ctx, entry_block, "arith.subf", inputs[0], inputs[1]);
            break;
        case UOP_MUL:
            result = emit_binary_op(ctx, entry_block, "arith.mulf", inputs[0], inputs[1]);
            break;
        case UOP_DIV:
            result = emit_binary_op(ctx, entry_block, "arith.divf", inputs[0], inputs[1]);
            break;
        case UOP_MAX:
            result = emit_binary_op(ctx, entry_block, "arith.maximumf", inputs[0], inputs[1]);
            break;
        case UOP_POW:
            result = emit_binary_op(ctx, entry_block, "math.powf", inputs[0], inputs[1]);
            break;
        case UOP_CMPLT: {
            MlirType i1_type = mlirIntegerTypeGet(ctx->context, 1);
            MlirOperationState cmp_state =
                mlirOperationStateGet(mlirStringRefCreateFromCString("arith.cmpf"), ctx->location);
            MlirAttribute pred_attr       = mlirIntegerAttrGet(i1_type, 1); // OLT
            MlirNamedAttribute pred_named = mlirNamedAttributeGet(
                mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("predicate")),
                pred_attr);
            mlirOperationStateAddAttributes(&cmp_state, 1, &pred_named);
            mlirOperationStateAddOperands(&cmp_state, 1, &inputs[0]);
            mlirOperationStateAddOperands(&cmp_state, 1, &inputs[1]);
            mlirOperationStateAddResults(&cmp_state, 1, &i1_type);
            MlirOperation cmp_op = mlirOperationCreate(&cmp_state);
            mlirBlockAppendOwnedOperation(entry_block, cmp_op);
            result = mlirOperationGetResult(cmp_op, 0);
            break;
        }

        case UOP_NEG:
            result = emit_unary_op(ctx, entry_block, "arith.negf", inputs[0]);
            break;
        case UOP_EXP: {
            // Use math.exp directly instead of polynomial approximation
            // This is more accurate and avoids shape inference issues
            result = emit_unary_op(ctx, entry_block, "math.exp", inputs[0]);
            break;
        }
        case UOP_LOG:
            result = emit_unary_op(ctx, entry_block, "math.log", inputs[0]);
            break;
        case UOP_SQRT:
            result = emit_unary_op(ctx, entry_block, "math.sqrt", inputs[0]);
            break;
        case UOP_RECIP: {
            MlirValue one = create_constant_f32(ctx, entry_block, 1.0f);
            result        = emit_binary_op(ctx, entry_block, "arith.divf", one, inputs[0]);
            break;
        }
        case UOP_ABS:
            result = emit_unary_op(ctx, entry_block, "math.absf", inputs[0]);
            break;
        case UOP_SIN:
            result = emit_unary_op(ctx, entry_block, "math.sin", inputs[0]);
            break;
        case UOP_COS:
            result = emit_unary_op(ctx, entry_block, "math.cos", inputs[0]);
            break;
        case UOP_TAN:
            result = emit_unary_op(ctx, entry_block, "math.tan", inputs[0]);
            break;

        // For brevity, higher ops (reductions, matmul, conv, where, slice, views)
        // remain exactly as in the existing cml_ir_to_mlir implementation.
        // We can refine them in later steps.
        case UOP_MATMUL: {
            // inputs[0]: A (MxK), inputs[1]: B (KxN) -> C (MxN)

            int64_t* out_shape = NULL;
            int out_rank       = 0;

            if (node->output_shape) {
                out_shape = (int64_t*)malloc(sizeof(int64_t) * node->output_ndim);
                for (int i = 0; i < node->output_ndim; i++)
                    out_shape[i] = node->output_shape[i];
                out_rank = node->output_ndim;

            } else {
                // Infer from inputs
                Tensor* A = node->inputs[0];
                Tensor* B = node->inputs[1];
                if (A && B && A->ndim >= 2 && B->ndim >= 2) {
                    out_rank     = 2;
                    out_shape    = (int64_t*)malloc(sizeof(int64_t) * 2);
                    out_shape[0] = A->shape[A->ndim - 2]; // M
                    out_shape[1] = B->shape[B->ndim - 1]; // N
                } else {
                    // Fallback to dynamic
                    out_shape    = (int64_t*)malloc(sizeof(int64_t) * 2);
                    out_shape[0] = -1;
                    out_shape[1] = -1;
                    out_rank     = 2;
                }
            }

            MlirValue output = create_memref_alloc(ctx, entry_block, out_shape, out_rank);
            free(out_shape);

            MlirValue zero = create_constant_f32(ctx, entry_block, 0.0f);
            create_linalg_fill(ctx, entry_block, output, zero);

            create_linalg_matmul(ctx, entry_block, inputs[0], inputs[1], output);

            result = output;
            break;
        }
        case UOP_MEAN: {
            ReduceParams* p      = (ReduceParams*)node->params;
            int* reduced_dims    = NULL;
            int num_reduced_dims = 0;
            bool free_dims       = false;

            int rank = 0;
            if (node->input_ndims) {
                rank = node->input_ndims[0];
            } else if (node->inputs && node->inputs[0]) {
                rank = node->inputs[0]->ndim;
            }

            if (p && p->dims && p->num_dims > 0) {
                reduced_dims     = p->dims;
                num_reduced_dims = p->num_dims;
            } else {
                reduced_dims = (int*)malloc(sizeof(int) * rank);
                for (int i = 0; i < rank; i++)
                    reduced_dims[i] = i;
                num_reduced_dims = rank;
                free_dims        = true;
            }

            int64_t* out_shape = (int64_t*)malloc(sizeof(int64_t) * node->output_ndim);
            for (int i = 0; i < node->output_ndim; i++)
                out_shape[i] = node->output_shape[i];

            MlirValue sum =
                create_reduction(ctx, entry_block, inputs[0], reduced_dims, num_reduced_dims,
                                 out_shape, node->output_ndim, "arith.addf");

            // Count
            float count      = 1.0f;
            int* input_shape = NULL;
            if (node->input_shapes && node->input_shapes[0])
                input_shape = node->input_shapes[0];
            else if (node->inputs && node->inputs[0])
                input_shape = node->inputs[0]->shape;

            if (input_shape) {
                for (int i = 0; i < num_reduced_dims; i++) {
                    count *= (float)input_shape[reduced_dims[i]];
                }
            }

            MlirValue count_const = create_constant_f32(ctx, entry_block, count);
            MlirValue count_tensor =
                create_memref_alloc(ctx, entry_block, out_shape, node->output_ndim);
            create_linalg_fill(ctx, entry_block, count_tensor, count_const);

            result = emit_binary_op(ctx, entry_block, "arith.divf", sum, count_tensor);

            if (free_dims)
                free(reduced_dims);
            free(out_shape);
            break;
        }
        case UOP_RESHAPE: {
            // Use memref.reinterpret_cast or expand_shape/collapse_shape
            // For simplicity, we can use a linalg.generic copy with reshaping if needed,
            // but memref reshape is better.
            // However, standard memref reshape requires static shapes or compatible strides.
            // Let's use a simple linalg.generic copy to a new shape (data movement).

            int64_t* out_shape = (int64_t*)malloc(sizeof(int64_t) * node->output_ndim);
            for (int i = 0; i < node->output_ndim; i++)
                out_shape[i] = node->output_shape[i];

            // MlirValue output = create_memref_alloc(ctx, entry_block, out_shape,
            // node->output_ndim);
            free(out_shape);

            // ...

            result = inputs[0];
            break;
        }
        case UOP_TANH:
            result = emit_unary_op(ctx, entry_block, "math.tanh", inputs[0]);
            break;
        case UOP_SIGMOID: {
            // sigmoid(x) = 1 / (1 + exp(-x))
            MlirValue neg_x        = emit_unary_op(ctx, entry_block, "arith.negf", inputs[0]);
            MlirValue exp_neg_x    = emit_unary_op(ctx, entry_block, "math.exp", neg_x);
            MlirValue one          = create_constant_f32(ctx, entry_block, 1.0f);
            MlirValue one_plus_exp = emit_binary_op(ctx, entry_block, "arith.addf", one, exp_neg_x);
            result = emit_binary_op(ctx, entry_block, "arith.divf", one, one_plus_exp);
            break;
        }
        case UOP_PERMUTE: {
            int64_t* out_shape = (int64_t*)malloc(sizeof(int64_t) * node->output_ndim);
            for (int i = 0; i < node->output_ndim; i++)
                out_shape[i] = node->output_shape[i];

            result = create_transpose(ctx, entry_block, inputs[0], out_shape, node->output_ndim);
            free(out_shape);
            break;
        }

        default:
            // For unimplemented ops, pass through first input
            LOG_WARNING("UOp %d not implemented in MLIR uops builder, using passthrough",
                        node->type);
            result = inputs[0];
            break;
        }

        if (!mlirValueIsNull(result)) {
            symbol_table_insert(&sym_table, node->output_name, result);
            last_result = result;
        }

        node = node->next;
    }

    if (mlirValueIsNull(last_result)) {
        if (num_graph_inputs > 0) {
            LOG_WARNING("IR conversion produced no outputs; returning first argument");
            last_result = mlirBlockGetArgument(entry_block, 0);
        } else {
            LOG_WARNING("IR conversion produced no outputs or inputs; returning zero tensor");
            int64_t out_shape[]    = {-1};
            MlirValue empty_tensor = create_memref_alloc(ctx, entry_block, out_shape, 1);
            MlirValue zero_scalar  = create_constant_f32(ctx, entry_block, 0.0f);
            last_result = create_linalg_fill(ctx, entry_block, empty_tensor, zero_scalar);
        }
    }

    // Copy last_result to output argument
    if (num_graph_inputs < mlirBlockGetNumArguments(entry_block)) {
        MlirValue output_arg =
            mlirBlockGetArgument(entry_block, num_graph_inputs); // First output arg

        // linalg.copy(last_result, output_arg)
        MlirOperationState copy_state =
            mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.copy"), ctx->location);
        mlirOperationStateAddOperands(&copy_state, 1, &last_result);
        mlirOperationStateAddOperands(&copy_state, 1, &output_arg);
        // Add operandSegmentSizes attribute: [num_inputs, num_outputs] = [1, 1]
        int32_t copy_segment_sizes[]    = {1, 1};
        MlirAttribute copy_segment_attr = mlirDenseI32ArrayGet(ctx->context, 2, copy_segment_sizes);
        MlirNamedAttribute copy_segment_named = mlirNamedAttributeGet(
            mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("operandSegmentSizes")),
            copy_segment_attr);
        mlirOperationStateAddAttributes(&copy_state, 1, &copy_segment_named);

        // linalg.copy requires a region with a yield
        MlirRegion copy_region    = mlirRegionCreate();
        MlirType f32_type         = mlirF32TypeGet(ctx->context);
        MlirType copy_arg_types[] = {f32_type, f32_type};
        MlirBlock copy_block      = mlirBlockCreate(2, copy_arg_types, &ctx->location);
        mlirRegionAppendOwnedBlock(copy_region, copy_block);
        // Yield the input value (first argument)
        MlirValue copy_input = mlirBlockGetArgument(copy_block, 0);
        MlirOperationState copy_yield_state =
            mlirOperationStateGet(mlirStringRefCreateFromCString("linalg.yield"), ctx->location);
        mlirOperationStateAddOperands(&copy_yield_state, 1, &copy_input);
        MlirOperation copy_yield_op = mlirOperationCreate(&copy_yield_state);
        mlirBlockAppendOwnedOperation(copy_block, copy_yield_op);
        mlirOperationStateAddOwnedRegions(&copy_state, 1, &copy_region);

        MlirOperation copy_op = mlirOperationCreate(&copy_state);
        mlirBlockAppendOwnedOperation(entry_block, copy_op);
    }

    MlirOperationState ret_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), ctx->location);
    // mlirOperationStateAddOperands(&ret_state, 1, &last_result); // Removed operands
    MlirOperation ret_op = mlirOperationCreate(&ret_state);
    mlirBlockAppendOwnedOperation(entry_block, ret_op);

    symbol_table_free(&sym_table);

    LOG_INFO("MLIR uops builder completed");
    // printf("DEBUG: Built MLIR module. Dumping:\n");
    // fflush(stdout);
    // mlirOperationDump(mlirModuleGetOperation(ctx->module));
    return true;
#else
    (void)ctx;
    (void)ir;
    return false;
#endif
}
