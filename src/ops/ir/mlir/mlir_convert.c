#include "ops/ir/mlir/mlir_internal.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// IR to MLIR Conversion
// ============================================================================

bool cml_ir_to_mlir(CMLMLIRContext* ctx, CMLIR_t ir) {
#ifdef CML_HAS_MLIR
    if (!ctx || !ir) {
        LOG_ERROR("Invalid arguments to cml_ir_to_mlir");
        return false;
    }

    LOG_INFO("Converting C-ML IR to MLIR (with chaining)...");

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

    // Create function
    // MlirType f32 = get_f32_type(ctx->context); // Unused

    // Determine function signature based on IR inputs/outputs
    // Scan IR to find actual graph inputs (values consumed but not produced within the graph)

    // 1. Collect all produced outputs
    SymbolTable produced_outputs;
    symbol_table_init(&produced_outputs);

    struct IRNode* scan_node = ir->head;
    while (scan_node) {
        if (scan_node->output_name) {
            // Value is dummy, we just care about keys
            MlirValue null_val = {NULL};
            symbol_table_insert(&produced_outputs, scan_node->output_name, null_val);
        }
        scan_node = scan_node->next;
    }

// 2. Identify graph inputs
#define MAX_GRAPH_INPUTS 32
    char* graph_inputs[MAX_GRAPH_INPUTS];
    Tensor* found_tensors[MAX_GRAPH_INPUTS];
    int num_graph_inputs = 0;

    scan_node = ir->head;
    while (scan_node) {
        for (int i = 0; i < scan_node->num_inputs; i++) {
            char* input_name = scan_node->input_names[i];
            if (!input_name)
                continue;

            // Check if produced internally
            // Re-implement check: iterate produced_outputs
            bool is_produced = false;
            for (size_t k = 0; k < produced_outputs.count; k++) {
                if (strcmp(produced_outputs.entries[k].key, input_name) == 0) {
                    is_produced = true;
                    break;
                }
            }

            if (!is_produced) {
                // It's an external input. Check if already added.
                // CRITICAL FIX: Ensure we don't treat internal nodes as external inputs
                // This prevents infinite recursion in execution (tensor_data_ptr -> execute ->
                // tensor_data_ptr)
                Tensor* input_tensor = scan_node->inputs[i];
                if (input_tensor) {
                    if (input_tensor->ir_context == ir) {
                        LOG_WARNING("Input '%s' appears to be internal (ir_context match) but not "
                                    "found in produced_outputs. Skipping to avoid recursion.",
                                    input_name);
                        continue;
                    } else {
                        // Debug mismatch
                        printf(
                            "DEBUG: Input '%s' context mismatch. Tensor ctx: %p, Current ir: %p\n",
                            input_name, (void*)input_tensor->ir_context, (void*)ir);
                    }
                } else {
                    printf("DEBUG: Input '%s' tensor is NULL\n", input_name);
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

    // If no inputs found (e.g. constant graph), ensure at least 0 inputs
    LOG_INFO("Found %d graph inputs", num_graph_inputs);

    // Create input types array
    // We need to include outputs as arguments for Destination Passing Style (DPS)
    int num_outputs = 1; // Assuming 1 output for now
    int num_args    = num_graph_inputs + num_outputs;

    MlirType* func_inputs = (MlirType*)malloc(sizeof(MlirType) * num_args);

    // Input arguments with static shapes
    for (int i = 0; i < num_graph_inputs; i++) {
        Tensor* t = found_tensors[i];
        if (t) {
            int64_t* shape = (int64_t*)malloc(sizeof(int64_t) * t->ndim);
            for (int j = 0; j < t->ndim; j++)
                shape[j] = t->shape[j];
            func_inputs[i] = get_tensor_type(ctx->context, shape, t->ndim);
            free(shape);
        } else {
            // Fallback to dynamic shape (shouldn't happen)
            int64_t dyn_shape[] = {-1};
            func_inputs[i]      = get_tensor_type(ctx->context, dyn_shape, 1);
        }
    }

    // Output arguments with static shapes
    // The output corresponds to the tail of the graph
    struct IRNode* output_node = ir->tail;
    if (output_node) {
        int64_t* shape = (int64_t*)malloc(sizeof(int64_t) * output_node->output_ndim);
        for (int j = 0; j < output_node->output_ndim; j++)
            shape[j] = output_node->output_shape[j];
        func_inputs[num_graph_inputs] =
            get_tensor_type(ctx->context, shape, output_node->output_ndim);
        free(shape);
    } else {
        int64_t dyn_shape[]           = {-1};
        func_inputs[num_graph_inputs] = get_tensor_type(ctx->context, dyn_shape, 1);
    }

    // Function returns void
    MlirType func_type = mlirFunctionTypeGet(ctx->context, num_args, func_inputs, 0, NULL);

    // Create function operation
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

    // Create one region for function body
    MlirRegion func_region = mlirRegionCreate();
    mlirOperationStateAddOwnedRegions(&func_state, 1, &func_region);

    MlirOperation func_op = mlirOperationCreate(&func_state);
    mlirBlockAppendOwnedOperation(body, func_op);

    // Get region from op to be safe
    MlirRegion func_body_region = mlirOperationGetRegion(func_op, 0);

    // Create entry block with arguments
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
        free(graph_inputs[i]); // Free duplicated string
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

    // Process IR nodes
    struct IRNode* node   = ir->head;
    MlirValue last_result = {NULL};

    while (node) {
        // Resolve inputs
        MlirValue inputs[3] = {{NULL}, {NULL}, {NULL}};

        for (int i = 0; i < node->num_inputs && i < 3; i++) {
            inputs[i] = symbol_table_lookup(&sym_table, node->input_names[i]);

            // Fallback for testing if input not found
            if (mlirValueIsNull(inputs[i]) && num_graph_inputs > 0) {
                inputs[i] = mlirBlockGetArgument(entry_block, 0);
            }
        }

        MlirValue result    = {NULL};
        SliceParams* params = NULL;

        switch (node->type) {
        // Binary elementwise operations
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
            MlirAttribute pred_attr       = mlirIntegerAttrGet(i1_type, 1); // 1 = OLT
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

        // Unary elementwise operations
        case UOP_NEG:
            result = emit_unary_op(ctx, entry_block, "arith.negf", inputs[0]);
            break;
        case UOP_EXP:
            result = emit_unary_op(ctx, entry_block, "math.exp", inputs[0]);
            break;
        case UOP_LOG:
            result = emit_unary_op(ctx, entry_block, "math.log", inputs[0]);
            break;
        case UOP_SQRT:
            result = emit_unary_op(ctx, entry_block, "math.sqrt", inputs[0]);
            break;
        case UOP_RECIP: {
            // 1.0 / x
            // Create memref of 1s with same shape/type as input
            MlirType input_type = mlirValueGetType(inputs[0]);
            MlirValue ones      = create_memref_alloc_from_type(ctx, entry_block, input_type);
            MlirValue one_const = create_constant_f32(ctx, entry_block, 1.0f);
            create_linalg_fill(ctx, entry_block, ones, one_const);

            result = emit_binary_op(ctx, entry_block, "arith.divf", ones, inputs[0]);
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

        // Reduction operations
        case UOP_SUM:
        case UOP_MEAN:
        case UOP_MAX_REDUCE: {
            // 1. Prepare Init Tensor (Accumulator)
            // Use static output shape
            int64_t* out_shape = (int64_t*)malloc(sizeof(int64_t) * node->output_ndim);
            for (int i = 0; i < node->output_ndim; i++) {
                out_shape[i] = node->output_shape[i];
            }
            MlirValue init_empty =
                create_memref_alloc(ctx, entry_block, out_shape, node->output_ndim);
            free(out_shape);

            // 2. Fill with neutral element
            float neutral_val       = (node->type == UOP_MAX_REDUCE) ? -3.40282347e+38F : 0.0f;
            MlirValue neutral_const = create_constant_f32(ctx, entry_block, neutral_val);
            MlirValue init_tensor = create_linalg_fill(ctx, entry_block, init_empty, neutral_const);

            // 3. Create linalg.reduce
            MlirOperationState red_state = mlirOperationStateGet(
                mlirStringRefCreateFromCString("linalg.reduce"), ctx->location);

            mlirOperationStateAddOperands(&red_state, 1, &inputs[0]);   // Input
            mlirOperationStateAddOperands(&red_state, 1, &init_tensor); // Init

            // Dimensions: Extract from params
            ReduceParams* op_params = (ReduceParams*)node->params;
            int64_t* dims           = NULL;
            int num_dims            = 0;

            if (op_params && op_params->dims && op_params->num_dims > 0) {
                num_dims = op_params->num_dims;
                dims     = (int64_t*)malloc(sizeof(int64_t) * num_dims);
                for (int k = 0; k < num_dims; k++) {
                    dims[k] = (int64_t)op_params->dims[k];
                }
            } else {
                // Default: Reduce all dimensions
                if (node->inputs[0]) {
                    num_dims = node->inputs[0]->ndim;
                    dims     = (int64_t*)malloc(sizeof(int64_t) * num_dims);
                    for (int k = 0; k < num_dims; k++) {
                        dims[k] = (int64_t)k;
                    }
                } else {
                    num_dims = 0;
                }
            }

            MlirAttribute dim_attr = mlirDenseI64ArrayGet(ctx->context, num_dims, dims);
            free(dims);

            MlirNamedAttribute dim_named = mlirNamedAttributeGet(
                mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("dimensions")),
                dim_attr);
            mlirOperationStateAddAttributes(&red_state, 1, &dim_named);

            // 4. Create Region & Block
            MlirRegion region = mlirRegionCreate();
            mlirOperationStateAddOwnedRegions(&red_state, 1, &region);

            MlirType f32_type         = get_f32_type(ctx->context);
            MlirType block_args[]     = {f32_type, f32_type};
            MlirLocation block_locs[] = {ctx->location, ctx->location};
            MlirBlock block           = mlirBlockCreate(2, block_args, block_locs);
            mlirRegionAppendOwnedBlock(region, block);

            // 5. Block Body
            MlirValue arg0 = mlirBlockGetArgument(block, 0);
            MlirValue arg1 = mlirBlockGetArgument(block, 1);

            MlirValue red_res;
            if (node->type == UOP_MAX_REDUCE) {
                red_res = emit_binary_op(ctx, block, "arith.maximumf", arg0, arg1);
            } else {
                red_res = emit_binary_op(ctx, block, "arith.addf", arg0, arg1);
            }

            MlirOperationState yield_state = mlirOperationStateGet(
                mlirStringRefCreateFromCString("linalg.yield"), ctx->location);
            mlirOperationStateAddOperands(&yield_state, 1, &red_res);
            MlirOperation yield_op = mlirOperationCreate(&yield_state);
            mlirBlockAppendOwnedOperation(block, yield_op);

            // Finalize Reduce Op (No results for memref)
            MlirOperation red_op = mlirOperationCreate(&red_state);
            mlirBlockAppendOwnedOperation(entry_block, red_op);

            result = init_tensor;

            // Handle MEAN
            if (node->type == UOP_MEAN) {
                ReduceParams* mean_params = (ReduceParams*)node->params;
                int64_t count             = 1;
                if (node->inputs[0]) {
                    if (mean_params && mean_params->dims && mean_params->num_dims > 0) {
                        for (int k = 0; k < mean_params->num_dims; k++) {
                            int d = mean_params->dims[k];
                            if (d < node->inputs[0]->ndim)
                                count *= node->inputs[0]->shape[d];
                        }
                    } else {
                        count = node->inputs[0]->numel;
                    }
                }

                // Create memref of counts
                MlirType res_type     = mlirValueGetType(result);
                MlirValue counts      = create_memref_alloc_from_type(ctx, entry_block, res_type);
                MlirValue count_const = create_constant_f32(ctx, entry_block, (float)count);
                create_linalg_fill(ctx, entry_block, counts, count_const);

                // Divide result by counts
                result = emit_binary_op(ctx, entry_block, "arith.divf", result, counts);
            }
            break;
        }

        // Matrix operations
        case UOP_MATMUL: {
            // linalg.matmul inputs: A, B, C(init)
            int64_t* out_shape = (int64_t*)malloc(sizeof(int64_t) * node->output_ndim);
            for (int i = 0; i < node->output_ndim; i++) {
                out_shape[i] = node->output_shape[i];
            }

            MlirValue init_empty =
                create_memref_alloc(ctx, entry_block, out_shape, node->output_ndim);
            free(out_shape);

            MlirValue zero_const  = create_constant_f32(ctx, entry_block, 0.0f);
            MlirValue init_tensor = create_linalg_fill(ctx, entry_block, init_empty, zero_const);

            create_linalg_matmul(ctx, entry_block, inputs[0], inputs[1], init_tensor);

            result = init_tensor;
            break;
        }

        case UOP_CONV2D: {
            // linalg.conv_2d_nhwc_hwcf inputs: Input, Filter, Output(init)
            int64_t out_shape[]   = {-1, -1, -1, -1};
            MlirValue init_empty  = create_memref_alloc(ctx, entry_block, out_shape, 4);
            MlirValue zero_const  = create_constant_f32(ctx, entry_block, 0.0f);
            MlirValue init_tensor = create_linalg_fill(ctx, entry_block, init_empty, zero_const);

            MlirOperationState conv_state = mlirOperationStateGet(
                mlirStringRefCreateFromCString("linalg.conv_2d_nhwc_hwcf"), ctx->location);

            mlirOperationStateAddOperands(&conv_state, 1, &inputs[0]);   // Input
            mlirOperationStateAddOperands(&conv_state, 1, &inputs[1]);   // Filter
            mlirOperationStateAddOperands(&conv_state, 1, &init_tensor); // Output

            // Attributes: strides, dilations
            // Extract from params
            Conv2DParams* op_params = (Conv2DParams*)node->params;
            int64_t stride_h = 1, stride_w = 1;
            int64_t dilation_h = 1, dilation_w = 1;

            if (op_params) {
                stride_h   = op_params->stride[0];
                stride_w   = op_params->stride[1];
                dilation_h = op_params->dilation[0];
                dilation_w = op_params->dilation[1];
            }

            int64_t strides[]   = {stride_h, stride_w};
            int64_t dilations[] = {dilation_h, dilation_w};

            MlirAttribute strides_attr       = mlirDenseI64ArrayGet(ctx->context, 2, strides);
            MlirNamedAttribute strides_named = mlirNamedAttributeGet(
                mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("strides")),
                strides_attr);
            mlirOperationStateAddAttributes(&conv_state, 1, &strides_named);

            MlirAttribute dilations_attr       = mlirDenseI64ArrayGet(ctx->context, 2, dilations);
            MlirNamedAttribute dilations_named = mlirNamedAttributeGet(
                mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("dilations")),
                dilations_attr);
            mlirOperationStateAddAttributes(&conv_state, 1, &dilations_named);

            MlirType result_type = mlirValueGetType(init_tensor);
            mlirOperationStateAddResults(&conv_state, 1, &result_type);

            MlirOperation conv_op = mlirOperationCreate(&conv_state);
            mlirBlockAppendOwnedOperation(entry_block, conv_op);
            result = mlirOperationGetResult(conv_op, 0);
            break;
        }

        case UOP_WHERE: {
            // arith.select condition, true_val, false_val
            // Input 0: Condition (float)
            // Input 1: True value
            // Input 2: False value

            // 1. Create Zero Tensor for comparison
            // We need a zero tensor of the same shape as the condition (Input 0)
            // Use tensor.dim to get shape? Or just tensor.empty with dynamic shape?
            // Assuming dynamic rank-1 or rank-2 for now, or use the helper.
            // Better: Use linalg.fill on an empty tensor of same shape.
            // But we don't know the shape statically.
            // We can use `tensor.empty` with dynamic sizes extracted from input 0.
            // For simplicity in this phase, we'll assume rank-compatibility and use a
            // generic dynamic tensor creation or rely on the fact that we can't easily
            // do element-wise select without linalg.generic in pure C-API without
            // verbose boilerplate.

            // However, let's try the linalg.fill approach which is standard.
            // We need to get the shape of input[0] to create the empty tensor.
            // This requires a loop over dimensions with tensor.dim.
            // Too much boilerplate for this snippet.

            // ALTERNATIVE: Use `arith.cmpf` with a Splat Constant?
            // MLIR C API doesn't easily support creating splat constants for dynamic shapes.

            // SHORTCUT: We will implement WHERE using `linalg.generic` (map)
            // which takes 3 inputs and produces 1 output.
            // The region will do:
            // ^bb0(%cond: f32, %true: f32, %false: f32):
            //   %c0 = arith.constant 0.0 : f32
            //   %pred = arith.cmpf "one", %cond, %c0 : f32
            //   %res = arith.select %pred, %true, %false : f32
            //   linalg.yield %res : f32

            // 1. Create Init Tensor (Output)
            // Shape should match inputs (broadcasting rules apply, but assuming same shape)
            // We'll use input[1] (true_val) to determine output shape/type
            // int64_t out_shape[] = {-1, -1}; // 2D dynamic
            // MlirValue init_empty = create_memref_alloc(ctx, entry_block, out_shape, 2);
            // We don't need to fill it, linalg.generic will overwrite.

            // 2. Create linalg.generic
            // MlirOperationState gen_state = mlirOperationStateGet(
            //     mlirStringRefCreateFromCString("linalg.generic"),
            //     ctx->location
            // );
            // WHERE(condition, true_val, false_val)
            // Emit: arith.cmpf(condition, 0.0) -> i1, then arith.select

            if (node->num_inputs < 3) {
                LOG_ERROR("UOP_WHERE requires 3 inputs");
                result = inputs[0];
                break;
            }

            // MlirValue condition = inputs[0];
            // MlirValue true_val = inputs[1];
            // MlirValue false_val = inputs[2];

            // Create constant 0.0 for comparison
            MlirValue zero = create_constant_f32(ctx, entry_block, 0.0f);

            // Compare condition != 0.0
            MlirOperationState cmp_state =
                mlirOperationStateGet(mlirStringRefCreateFromCString("arith.cmpf"), ctx->location);
            MlirAttribute pred_attr =
                mlirIntegerAttrGet(mlirIntegerTypeGet(ctx->context, 64), 4); // ONE
            MlirNamedAttribute pred_named = mlirNamedAttributeGet(
                mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("predicate")),
                pred_attr);
            mlirOperationStateAddAttributes(&cmp_state, 1, &pred_named);
            mlirOperationStateAddOperands(&cmp_state, 1, &inputs[0]);
            mlirOperationStateAddOperands(&cmp_state, 1, &zero);

            // Result type: memref<?x?xi1>
            MlirType i1            = mlirIntegerTypeGet(ctx->context, 1);
            int64_t dyn_shape_2d[] = {-1, -1};
            MlirType res_i1_type   = mlirMemRefTypeGet(i1, 2, dyn_shape_2d, mlirAttributeGetNull(),
                                                       mlirAttributeGetNull());
            mlirOperationStateAddResults(&cmp_state, 1, &res_i1_type);

            MlirOperation cmp_op = mlirOperationCreate(&cmp_state);
            mlirBlockAppendOwnedOperation(entry_block, cmp_op);
            MlirValue cond_i1 = mlirOperationGetResult(cmp_op, 0);

            // Now select
            MlirOperationState sel_state = mlirOperationStateGet(
                mlirStringRefCreateFromCString("arith.select"), ctx->location);
            mlirOperationStateAddOperands(&sel_state, 1, &cond_i1);
            mlirOperationStateAddOperands(&sel_state, 1, &inputs[1]);
            mlirOperationStateAddOperands(&sel_state, 1, &inputs[2]);

            MlirType res_type = mlirValueGetType(inputs[1]); // Assume type matches true branch
            mlirOperationStateAddResults(&sel_state, 1, &res_type);

            MlirOperation sel_op = mlirOperationCreate(&sel_state);
            mlirBlockAppendOwnedOperation(entry_block, sel_op);
            result = mlirOperationGetResult(sel_op, 0);
            break;
        }

        // View operations (no computation, just metadata change)
        case UOP_RESHAPE:
        case UOP_EXPAND:
        case UOP_STRIDE: {
            MlirOperationState cast_state =
                mlirOperationStateGet(mlirStringRefCreateFromCString("tensor.cast"), ctx->location);

            mlirOperationStateAddOperands(&cast_state, 1, &inputs[0]);

            int64_t shape[]   = {-1};
            MlirType res_type = get_tensor_type(ctx->context, shape, 1);
            mlirOperationStateAddResults(&cast_state, 1, &res_type);

            MlirOperation cast_op = mlirOperationCreate(&cast_state);
            mlirBlockAppendOwnedOperation(entry_block, cast_op);
            result = mlirOperationGetResult(cast_op, 0);
            break;
        }

        // Permute (transpose) - requires actual data movement
        case UOP_PERMUTE: {
            // Only support 2D transpose for now
            if (node->output_ndim != 2) {
                LOG_WARNING(
                    "UOP_PERMUTE only supported for 2D tensors in cml_ir_to_mlir (got %d dims)",
                    node->output_ndim);
                // Fall back to no-op (pass through input)
                result = inputs[0];
                break;
            }

            // Get output shape from the node
            int64_t out_shape[2];
            out_shape[0] = node->output_shape[0];
            out_shape[1] = node->output_shape[1];

            // Create actual transpose using create_transpose helper
            result = create_transpose(ctx, entry_block, inputs[0], out_shape, 2);
            if (mlirValueIsNull(result)) {
                LOG_ERROR("Failed to create transpose in cml_ir_to_mlir");
                result = inputs[0]; // Fallback
            }
            break;
        }

        case UOP_SLICE: {
            // tensor.extract_slice input[offsets...][sizes...][strides...]
            params = (SliceParams*)node->params;
            if (!params) {
                LOG_ERROR("Slice params are NULL");
                result = inputs[0]; // Fallback to input
                break;
            }

            {
                // Determine rank from input tensor
                int rank = 2; // Default
                if (node->num_inputs > 0 && node->inputs[0]) {
                    rank = node->inputs[0]->ndim;
                }

                // Build offsets, sizes, strides arrays
                int64_t* offsets = (int64_t*)malloc(sizeof(int64_t) * rank);
                int64_t* sizes   = (int64_t*)malloc(sizeof(int64_t) * rank);
                int64_t* strides = (int64_t*)malloc(sizeof(int64_t) * rank);

                for (int i = 0; i < rank; i++) {
                    int64_t start = (int64_t)params->start[i];
                    int64_t end   = (int64_t)params->end[i];
                    int64_t step  = params->step ? (int64_t)params->step[i] : 1;

                    offsets[i] = start;
                    strides[i] = step;

                    // Calculate size: ceil((end - start) / step)
                    int64_t size = (end - start + step - 1) / step;
                    if (size < 0)
                        size = 0;
                    sizes[i] = size;
                }

                MlirAttribute off_attr    = mlirDenseI64ArrayGet(ctx->context, rank, offsets);
                MlirAttribute size_attr   = mlirDenseI64ArrayGet(ctx->context, rank, sizes);
                MlirAttribute stride_attr = mlirDenseI64ArrayGet(ctx->context, rank, strides);

                free(offsets);
                free(sizes);
                free(strides);

                MlirNamedAttribute off_named = mlirNamedAttributeGet(
                    mlirIdentifierGet(ctx->context,
                                      mlirStringRefCreateFromCString("static_offsets")),
                    off_attr);
                MlirNamedAttribute size_named = mlirNamedAttributeGet(
                    mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("static_sizes")),
                    size_attr);
                MlirNamedAttribute stride_named = mlirNamedAttributeGet(
                    mlirIdentifierGet(ctx->context,
                                      mlirStringRefCreateFromCString("static_strides")),
                    stride_attr);

                MlirNamedAttribute slice_attrs[] = {off_named, size_named, stride_named};

                MlirOperationState slice_state = mlirOperationStateGet(
                    mlirStringRefCreateFromCString("tensor.extract_slice"), ctx->location);

                mlirOperationStateAddOperands(&slice_state, 1, &inputs[0]);
                mlirOperationStateAddAttributes(&slice_state, 3, slice_attrs);

                // Result type
                int64_t* result_shape = (int64_t*)malloc(sizeof(int64_t) * rank);
                for (int i = 0; i < rank; i++) {
                    int64_t start = (int64_t)params->start[i];
                    int64_t end   = (int64_t)params->end[i];
                    int64_t step  = params->step ? (int64_t)params->step[i] : 1;
                    int64_t size  = (end - start + step - 1) / step;
                    if (size < 0)
                        size = 0;
                    result_shape[i] = size;
                }
                MlirType slice_result_type = get_tensor_type(ctx->context, result_shape, rank);
                free(result_shape);

                mlirOperationStateAddResults(&slice_state, 1, &slice_result_type);

                MlirOperation slice_op = mlirOperationCreate(&slice_state);
                mlirBlockAppendOwnedOperation(entry_block, slice_op);

                result = mlirOperationGetResult(slice_op, 0);
            }
            break;
        }

        default:
            result = inputs[0];
            break;
        }

        // Register result
        if (!mlirValueIsNull(result)) {
            symbol_table_insert(&sym_table, node->output_name, result);
            last_result = result;
        }

        node = node->next;
    }

    // Ensure we always have something to return that matches the function signature.
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

    // Return void
    MlirOperationState ret_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), ctx->location);
    // mlirOperationStateAddOperands(&ret_state, 1, &last_result); // Removed operands

    MlirOperation ret_op = mlirOperationCreate(&ret_state);
    mlirBlockAppendOwnedOperation(entry_block, ret_op);

    symbol_table_free(&sym_table);

    LOG_INFO("MLIR generation complete");
    return true;
#else
    (void)ctx;
    (void)ir;
    return false;
#endif
}

// ============================================================================
// Backward IR to MLIR Conversion
// ============================================================================

// Helper to convert pointer to string key
static void ptr_to_key(void* ptr, char* buffer) { snprintf(buffer, 32, "%p", ptr); }

// Helper to emit forward node (duplicated from cml_ir_to_mlir for now to avoid refactoring
// risk)
static MlirValue emit_forward_node(CMLMLIRContext* ctx, MlirBlock block, struct IRNode* node,
                                   SymbolTable* sym_table) {
    MlirValue inputs[3] = {{NULL}, {NULL}, {NULL}};
    char key[32];

    for (int i = 0; i < node->num_inputs && i < 3; i++) {
        // Use Tensor pointer as key
        if (node->inputs && node->inputs[i]) {
            ptr_to_key(node->inputs[i], key);
            inputs[i] = symbol_table_lookup(sym_table, key);
            if (mlirValueIsNull(inputs[i])) {
                printf("DEBUG: emit_forward_node: Input tensor %p (name '%s') not found for node "
                       "type %d\n",
                       (void*)node->inputs[i], node->input_names[i], node->type);
            }
        } else {
            // Fallback to name if tensor ptr is missing (shouldn't happen for valid IR)
            inputs[i] = symbol_table_lookup(sym_table, node->input_names[i]);
            if (mlirValueIsNull(inputs[i])) {
                printf("DEBUG: emit_forward_node: Input '%s' not found for node type %d\n",
                       node->input_names[i], node->type);
            }
        }
    }

    switch (node->type) {
    case UOP_ADD:
        return emit_binary_op(ctx, block, "arith.addf", inputs[0], inputs[1]);
    case UOP_SUB:
        return emit_binary_op(ctx, block, "arith.subf", inputs[0], inputs[1]);
    case UOP_MUL:
        return emit_binary_op(ctx, block, "arith.mulf", inputs[0], inputs[1]);
    case UOP_DIV:
        return emit_binary_op(ctx, block, "arith.divf", inputs[0], inputs[1]);
    case UOP_MAX:
        return emit_binary_op(ctx, block, "arith.maximumf", inputs[0], inputs[1]);
    case UOP_POW:
        return emit_binary_op(ctx, block, "math.powf", inputs[0], inputs[1]);
    case UOP_NEG:
        return emit_unary_op(ctx, block, "arith.negf", inputs[0]);
    case UOP_EXP:
        return emit_unary_op(ctx, block, "math.exp", inputs[0]);
    case UOP_LOG:
        return emit_unary_op(ctx, block, "math.log", inputs[0]);
    case UOP_RECIP: {
        MlirType input_type = mlirValueGetType(inputs[0]);
        MlirValue ones      = create_memref_alloc_from_type(ctx, block, input_type);
        MlirValue one_const = create_constant_f32(ctx, block, 1.0f);
        create_linalg_fill(ctx, block, ones, one_const);
        return emit_binary_op(ctx, block, "arith.divf", ones, inputs[0]);
    }
    case UOP_PERMUTE: {
        // Only support 2D transpose for now
        if (node->output_ndim != 2) {
            printf("DEBUG: UOP_PERMUTE only supported for 2D tensors (got %d)\n",
                   node->output_ndim);
            return (MlirValue){NULL};
        }
        int64_t* out_shape = (int64_t*)malloc(sizeof(int64_t) * node->output_ndim);
        for (int i = 0; i < node->output_ndim; i++)
            out_shape[i] = node->output_shape[i];
        MlirValue res = create_transpose(ctx, block, inputs[0], out_shape, node->output_ndim);
        free(out_shape);
        return res;
    }
    case UOP_MATMUL: {
        int64_t* out_shape = (int64_t*)malloc(sizeof(int64_t) * node->output_ndim);
        for (int i = 0; i < node->output_ndim; i++)
            out_shape[i] = node->output_shape[i];
        MlirValue init_empty = create_memref_alloc(ctx, block, out_shape, node->output_ndim);
        free(out_shape);
        MlirValue zero_const  = create_constant_f32(ctx, block, 0.0f);
        MlirValue init_tensor = create_linalg_fill(ctx, block, init_empty, zero_const);

        create_linalg_matmul(ctx, block, inputs[0], inputs[1], init_tensor);
        return init_tensor;
    }
    case UOP_SUM: {
        ReduceParams* p      = (ReduceParams*)node->params;
        int* reduced_dims    = NULL;
        int num_reduced_dims = 0;
        bool free_dims       = false;

        int rank = 0;
        if (node->input_ndims) {
            rank = node->input_ndims[0];
        } else if (node->inputs && node->inputs[0]) {
            rank = node->inputs[0]->ndim;
        } else {
            printf("DEBUG: Cannot determine input rank for UOP_SUM\n");
            fflush(stdout);
            return (MlirValue){NULL};
        }

        if (p && p->dims && p->num_dims > 0) {
            reduced_dims     = p->dims;
            num_reduced_dims = p->num_dims;
        } else {
            // Reduce all dims
            reduced_dims = (int*)malloc(sizeof(int) * rank);
            for (int i = 0; i < rank; i++)
                reduced_dims[i] = i;
            num_reduced_dims = rank;
            free_dims        = true;
        }

        int64_t* out_shape = (int64_t*)malloc(sizeof(int64_t) * node->output_ndim);
        for (int i = 0; i < node->output_ndim; i++)
            out_shape[i] = node->output_shape[i];

        MlirValue res = create_reduction(ctx, block, inputs[0], reduced_dims, num_reduced_dims,
                                         out_shape, node->output_ndim, "arith.addf");

        if (free_dims)
            free(reduced_dims);
        free(out_shape);
        return res;
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
        } else {
            printf("DEBUG: Cannot determine input rank for UOP_MEAN\n");
            fflush(stdout);
            return (MlirValue){NULL};
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

        // Calculate count
        float count      = 1.0f;
        int* input_shape = NULL;
        if (node->input_shapes && node->input_shapes[0]) {
            input_shape = node->input_shapes[0];
        } else if (node->inputs && node->inputs[0]) {
            input_shape = node->inputs[0]->shape;
        }

        if (!input_shape) {
            printf("DEBUG: Cannot determine input shape for UOP_MEAN count calculation\n");
            fflush(stdout);
            return (MlirValue){NULL};
        }

        for (int i = 0; i < num_reduced_dims; i++) {
            count *= (float)input_shape[reduced_dims[i]];
        }

        int64_t* out_shape = (int64_t*)malloc(sizeof(int64_t) * node->output_ndim);
        for (int i = 0; i < node->output_ndim; i++)
            out_shape[i] = node->output_shape[i];

        // Sum
        MlirValue sum = create_reduction(ctx, block, inputs[0], reduced_dims, num_reduced_dims,
                                         out_shape, node->output_ndim, "arith.addf");

        // Div by count
        MlirValue count_const = create_constant_f32(ctx, block, count);

        // Workaround: Create a tensor full of count with same shape as sum.
        MlirValue count_tensor = create_memref_alloc(ctx, block, out_shape, node->output_ndim);
        create_linalg_fill(ctx, block, count_tensor, count_const);

        MlirValue res = emit_binary_op(ctx, block, "arith.divf", sum, count_tensor);

        if (free_dims)
            free(reduced_dims);
        free(out_shape);
        return res;
    }
    default:
        return (MlirValue){NULL};
    }
}

// Helper to accumulate gradient
static void accumulate_grad(CMLMLIRContext* ctx, MlirBlock block, SymbolTable* table,
                            const char* name, MlirValue grad) {
    MlirValue existing = symbol_table_lookup(table, name);
    if (existing.ptr) {
        MlirValue sum = emit_binary_op(ctx, block, "arith.addf", existing, grad);
        symbol_table_insert(table, name, sum);
    } else {
        symbol_table_insert(table, name, grad);
    }
}

bool cml_ir_backward_to_mlir(CMLMLIRContext* ctx, CMLIR_t ir) {
#ifdef CML_HAS_MLIR
    if (!ctx || !ir)
        return false;

    LOG_INFO("Converting Backward IR to MLIR with automatic differentiation...");

    // Create a new module for backward pass
    MlirModule backward_module = mlirModuleCreateEmpty(ctx->location);
    ctx->module                = backward_module;
    MlirOperation module_op    = mlirModuleGetOperation(backward_module);
    MlirRegion body_region     = mlirOperationGetRegion(module_op, 0);
    MlirBlock body             = mlirRegionGetFirstBlock(body_region);

    // Create backward function
    // MlirType f32 = get_f32_type(ctx->context); // Unused
    // Create backward function
    // Use static shape from the tail node (output) to ensure ABI compatibility
    MlirType tensor_type;
    if (ir->tail && ir->tail->output_ndim > 0) {
        int64_t* shape = (int64_t*)malloc(sizeof(int64_t) * ir->tail->output_ndim);
        for (int i = 0; i < ir->tail->output_ndim; i++) {
            shape[i] = ir->tail->output_shape[i];
        }
        tensor_type = get_tensor_type(ctx->context, shape, ir->tail->output_ndim);
        free(shape);
    } else {
        // Fallback to dynamic shape (risky with bare ptr ABI)
        int64_t dyn_shape[] = {-1};
        tensor_type         = get_tensor_type(ctx->context, dyn_shape, 1);
    }

    // Backward function takes gradients of outputs as inputs
    // For simplicity, we assume 1 output gradient for the whole graph (loss gradient)
    // or we match the number of outputs.
    // Let's assume the IR tail is the loss/output.
    // 1. Identify produced outputs to find external inputs
    // We use a pointer array for efficiency and to avoid string key issues
    Tensor** produced_ptrs =
        (Tensor**)malloc(sizeof(Tensor*) * ir->node_count * 2); // Safe upper bound
    int produced_count = 0;

    struct IRNode* scan = ir->head;
    while (scan) {
        if (scan->output) {
            produced_ptrs[produced_count++] = scan->output;
        }
        scan = scan->next;
    }

    // 2. Identify graph inputs (leafs)
#define MAX_GRAPH_TENSORS 1024
    // We store Tensor* directly
    Tensor* graph_inputs[MAX_GRAPH_TENSORS];
    int num_graph_inputs = 0;
    char key[32];

    scan = ir->head;
    while (scan) {
        for (int i = 0; i < scan->num_inputs; i++) {
            Tensor* input_tensor = scan->inputs[i];
            if (!input_tensor)
                continue;

            bool is_produced = false;
            for (int k = 0; k < produced_count; k++) {
                if (produced_ptrs[k] == input_tensor) {
                    is_produced = true;
                    break;
                }
            }

            if (!is_produced) {
                bool already_added = false;
                for (int k = 0; k < num_graph_inputs; k++) {
                    if (graph_inputs[k] == input_tensor) {
                        already_added = true;
                        break;
                    }
                }

                if (!already_added && num_graph_inputs < MAX_GRAPH_INPUTS) {
                    graph_inputs[num_graph_inputs] = input_tensor;
                    num_graph_inputs++;
                }
            }
        }
        scan = scan->next;
    }
    free(produced_ptrs);

    // Backward function signature: (loss_grad, leaf_inputs...) -> (leaf_grads...)

    int num_loss_grads = 1; // Assuming scalar loss or single output
    int num_leafs      = num_graph_inputs;

    printf("DEBUG: Found %d leaf inputs for backward pass:\n", num_leafs);
    for (int i = 0; i < num_leafs; i++) {
        ptr_to_key(graph_inputs[i], key);
        printf("DEBUG:   Leaf %d: %p (key: %s)\n", i, (void*)graph_inputs[i], key);
    }

    int num_args = num_loss_grads + num_leafs + num_leafs; // loss_grad + inputs + outputs

    MlirType* func_inputs = (MlirType*)malloc(sizeof(MlirType) * num_args);
    for (int i = 0; i < num_args; i++) {
        func_inputs[i] = tensor_type;
    }

    // Function signature
    MlirType func_type = mlirFunctionTypeGet(ctx->context, num_args, func_inputs, 0, NULL);

    // Create function operation
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
    MlirAttribute emit_c_attr2       = mlirUnitAttrGet(ctx->context);
    MlirNamedAttribute emit_c_named2 = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx->context, mlirStringRefCreateFromCString("llvm.emit_c_interface")),
        emit_c_attr2);

    MlirNamedAttribute attrs[] = {name_attr, type_named_attr, emit_c_named2};
    mlirOperationStateAddAttributes(&func_state, 3, attrs);

    // Create region and add to operation state
    MlirRegion func_region = mlirRegionCreate();
    mlirOperationStateAddOwnedRegions(&func_state, 1, &func_region);

    MlirOperation func_op = mlirOperationCreate(&func_state);
    mlirBlockAppendOwnedOperation(body, func_op);

    // Get the region back from the operation to append block
    MlirRegion func_body_region = mlirOperationGetRegion(func_op, 0);

    // Create entry block with proper locations
    // Create entry block with proper locations
    MlirLocation* block_locs = (MlirLocation*)malloc(sizeof(MlirLocation) * num_args);
    for (int i = 0; i < num_args; i++) {
        block_locs[i] = ctx->location;
    }
    MlirBlock entry_block = mlirBlockCreate(num_args, func_inputs, block_locs);
    mlirRegionAppendOwnedBlock(func_body_region, entry_block);

    free(block_locs);
    free(func_inputs);

    // Initialize symbol tables for gradients and forward values
    SymbolTable grad_table;
    symbol_table_init(&grad_table);
    SymbolTable forward_table;
    symbol_table_init(&forward_table);

    // First pass: map arguments
    // Arg 0: loss_grad
    MlirValue loss_grad_val = mlirBlockGetArgument(entry_block, 0);
    // Map to output gradient (assuming tail is output)
    if (ir->tail && ir->tail->output) {
        ptr_to_key(ir->tail->output, key);
        symbol_table_insert(&grad_table, key, loss_grad_val);
    }

    // Args 1..N: leaf inputs (forward values)
    for (int i = 0; i < num_leafs; i++) {
        MlirValue arg = mlirBlockGetArgument(entry_block, 1 + i);
        ptr_to_key(graph_inputs[i], key);
        symbol_table_insert(&forward_table, key, arg);
    }

    // Args N+1..2N: leaf gradients (outputs)
    // We don't map them here, we use them at the end to copy results.

    // Update ctx inputs/outputs for execution
    if (ctx->inputs)
        free(ctx->inputs);
    ctx->num_inputs = num_loss_grads + num_leafs;
    ctx->inputs     = (Tensor**)malloc(sizeof(Tensor*) * ctx->num_inputs);

    // Input 0: Loss gradient tensor
    Tensor* loss_grad_tensor = tensor_ones_2d(1, 1); // Assuming scalar
    ctx->inputs[0]           = loss_grad_tensor;

    // Inputs 1..N: Leaf tensors
    for (int i = 0; i < num_leafs; i++) {
        ctx->inputs[1 + i] = graph_inputs[i];
    }

    if (ctx->outputs)
        free(ctx->outputs);
    ctx->num_outputs = num_leafs;
    ctx->outputs     = (Tensor**)malloc(sizeof(Tensor*) * ctx->num_outputs);
    for (int i = 0; i < num_leafs; i++) {
        // We need the GRADIENT tensor of the leaf
        Tensor* leaf = graph_inputs[i];
        if (!leaf->grad) {
            // Allocate grad tensor if missing
            TensorConfig config = {.dtype      = leaf->dtype,
                                   .device     = leaf->device,
                                   .has_dtype  = true,
                                   .has_device = true};
            leaf->grad          = tensor_zeros(leaf->shape, leaf->ndim, &config);
        }
        ctx->outputs[i] = leaf->grad;
    }

    // Re-emit forward pass to populate forward_table with intermediate values
    scan = ir->head;
    while (scan) {
        // Skip if output already exists (e.g. input)
        if (scan->output) {
            ptr_to_key(scan->output, key);
            if (!mlirValueIsNull(symbol_table_lookup(&forward_table, key))) {
                // printf("DEBUG: Skipping node producing %p (already exists)\n",
                // (void*)scan->output);
                scan = scan->next;
                continue;
            }
        }

        // printf("DEBUG: Emitting node type %d producing %p\n", scan->type,
        // (void*)scan->output);

        // Get the entry block from the region
        MlirBlock block  = mlirRegionGetFirstBlock(func_body_region);
        MlirValue result = emit_forward_node(ctx, block, scan, &forward_table);

        if (!mlirValueIsNull(result) && scan->output) {
            ptr_to_key(scan->output, key);
            symbol_table_insert(&forward_table, key, result);
            // printf("DEBUG: Registered %p in forward_table\n", (void*)scan->output);
        } else if (scan->output) {
            printf("DEBUG: Failed to emit node type %d producing %p (result is NULL)\n", scan->type,
                   (void*)scan->output);
        }

        scan = scan->next;
    }

    // Register loss gradient
    if (ir->tail && ir->tail->output) {
        MlirValue loss_grad = mlirBlockGetArgument(entry_block, 0);
        ptr_to_key(ir->tail->output, key);
        symbol_table_insert(&grad_table, key, loss_grad);
    }

    // Linearize nodes for reverse iteration
    int node_count = 0;
    scan           = ir->head;
    while (scan) {
        node_count++;
        scan = scan->next;
    }

    struct IRNode** nodes = (struct IRNode**)malloc(sizeof(struct IRNode*) * node_count);
    scan                  = ir->head;
    for (int i = 0; i < node_count; i++) {
        nodes[i] = scan;
        scan     = scan->next;
    }

    // Iterate backwards
    for (int i = node_count - 1; i >= 0; i--) {
        struct IRNode* node = nodes[i];
        if (!node->requires_grad || !node->output_name)
            continue;

        MlirValue grad_out = symbol_table_lookup(&grad_table, node->output_name);
        if (!grad_out.ptr)
            continue; // No gradient flowing here

        // Compute input gradients
        switch (node->type) {
        case UOP_ADD:
            // z = x + y => dx = dz, dy = dz
            if (node->num_inputs >= 1) {
                accumulate_grad(ctx, entry_block, &grad_table, node->input_names[0], grad_out);
            }
            if (node->num_inputs >= 2) {
                accumulate_grad(ctx, entry_block, &grad_table, node->input_names[1], grad_out);
            }
            break;

        case UOP_SUB:
            // z = x - y => dx = dz, dy = -dz
            if (node->num_inputs >= 1) {
                accumulate_grad(ctx, entry_block, &grad_table, node->input_names[0], grad_out);
            }
            if (node->num_inputs >= 2) {
                MlirValue neg_grad = emit_unary_op(ctx, entry_block, "arith.negf", grad_out);
                accumulate_grad(ctx, entry_block, &grad_table, node->input_names[1], neg_grad);
            }
            break;

        case UOP_MUL:
            // z = x * y => dx = dz * y, dy = dz * x
            if (node->num_inputs >= 2) {
                // Get forward values of inputs
                MlirValue x_forward = symbol_table_lookup(&forward_table, node->input_names[0]);
                MlirValue y_forward = symbol_table_lookup(&forward_table, node->input_names[1]);

                // If forward values not available, try to get from grad table (may have been
                // computed)
                if (mlirValueIsNull(x_forward)) {
                    x_forward = symbol_table_lookup(&grad_table, node->input_names[0]);
                }
                if (mlirValueIsNull(y_forward)) {
                    y_forward = symbol_table_lookup(&grad_table, node->input_names[1]);
                }

                // Compute gradients: dx = dz * y, dy = dz * x
                if (!mlirValueIsNull(y_forward)) {
                    MlirValue dx =
                        emit_binary_op(ctx, entry_block, "arith.mulf", grad_out, y_forward);
                    accumulate_grad(ctx, entry_block, &grad_table, node->input_names[0], dx);
                } else {
                    // Fallback: propagate gradient directly
                    accumulate_grad(ctx, entry_block, &grad_table, node->input_names[0], grad_out);
                }

                if (!mlirValueIsNull(x_forward)) {
                    MlirValue dy =
                        emit_binary_op(ctx, entry_block, "arith.mulf", grad_out, x_forward);
                    accumulate_grad(ctx, entry_block, &grad_table, node->input_names[1], dy);
                } else {
                    // Fallback: propagate gradient directly
                    accumulate_grad(ctx, entry_block, &grad_table, node->input_names[1], grad_out);
                }
            }
            break;

        case UOP_MATMUL:
            // MatMul backward: if z = x @ y, then:
            // dx = dz @ y^T, dy = x^T @ dz
            if (node->num_inputs >= 2) {
                MlirValue x_forward = symbol_table_lookup(&forward_table, node->input_names[0]);
                MlirValue y_forward = symbol_table_lookup(&forward_table, node->input_names[1]);

                if (mlirValueIsNull(x_forward) || mlirValueIsNull(y_forward)) {
                    LOG_WARNING("Missing forward values for MATMUL backward");
                    break;
                }

                // dx = dz @ y^T
                if (node->num_inputs >= 1) {
                    // Transpose y (KxN -> NxK)
                    int64_t y_shape[] = {node->inputs[1]->shape[1], node->inputs[1]->shape[0]};
                    MlirValue y_T     = create_transpose(ctx, entry_block, y_forward, y_shape, 2);

                    // MatMul: dz @ y^T
                    // Output shape: MxK (same as x)
                    int64_t dx_shape[] = {node->inputs[0]->shape[0], node->inputs[0]->shape[1]};
                    MlirValue dx_empty = create_memref_alloc(ctx, entry_block, dx_shape, 2);
                    MlirValue zero     = create_constant_f32(ctx, entry_block, 0.0f);
                    MlirValue dx_init  = create_linalg_fill(ctx, entry_block, dx_empty, zero);

                    create_linalg_matmul(ctx, entry_block, grad_out, y_T, dx_init);

                    accumulate_grad(ctx, entry_block, &grad_table, node->input_names[0], dx_init);
                }

                // dy = x^T @ dz
                if (node->num_inputs >= 2) {
                    // Transpose x (MxK -> KxM)
                    int64_t x_shape[] = {node->inputs[0]->shape[1], node->inputs[0]->shape[0]};
                    MlirValue x_T     = create_transpose(ctx, entry_block, x_forward, x_shape, 2);

                    // MatMul: x^T @ dz
                    // Output shape: KxN (same as y)
                    int64_t dy_shape[] = {node->inputs[1]->shape[0], node->inputs[1]->shape[1]};
                    MlirValue dy_empty = create_memref_alloc(ctx, entry_block, dy_shape, 2);
                    MlirValue zero     = create_constant_f32(ctx, entry_block, 0.0f);
                    MlirValue dy_init  = create_linalg_fill(ctx, entry_block, dy_empty, zero);

                    create_linalg_matmul(ctx, entry_block, x_T, grad_out, dy_init);

                    accumulate_grad(ctx, entry_block, &grad_table, node->input_names[1], dy_init);
                }
            }
            break;

        default:
            break;
        }
    }

    free(nodes);

    // Return void
    MlirOperationState ret_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), ctx->location);
    MlirOperation ret_op = mlirOperationCreate(&ret_state);
    mlirBlockAppendOwnedOperation(entry_block, ret_op);

    symbol_table_free(&grad_table);
    symbol_table_free(&forward_table);

    LOG_INFO("Backward MLIR generation complete");
    return true;
#else
    (void)ctx;
    (void)ir;
    return false;
#endif
}
