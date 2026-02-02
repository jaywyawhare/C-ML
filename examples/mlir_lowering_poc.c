#include "core/logging.h"
#include "ops/ir/mlir/mlir_cpp_bridge.h"

#include <stdio.h>

#ifdef CML_HAS_MLIR
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

static void print_callback(MlirStringRef string, void* user_data) {
    FILE* out = (FILE*)user_data;
    fwrite(string.data, 1, string.length, out);
}

static void dump_module(const char* header, MlirModule module) {
    fprintf(stdout, "\n=== %s ===\n", header);
    MlirOperation op = mlirModuleGetOperation(module);
    mlirOperationPrint(op, print_callback, stdout);
    fprintf(stdout, "\n");
}

static int setup_context(MlirContext* out_ctx, MlirLocation* out_loc) {
    MlirContext ctx  = mlirContextCreate();
    MlirLocation loc = mlirLocationUnknownGet(ctx);

    MlirDialectRegistry registry = mlirDialectRegistryCreate();
    if (cml_mlir_register_all_dialects_cpp(registry) != 0 ||
        cml_mlir_register_all_extensions(registry) != 0 ||
        cml_mlir_register_func_to_llvm_interface(registry) != 0 ||
        cml_mlir_register_all_llvm_translations(registry) != 0) {
        mlirDialectRegistryDestroy(registry);
        mlirContextDestroy(ctx);
        return -1;
    }

    mlirContextAppendDialectRegistry(ctx, registry);
    mlirDialectRegistryDestroy(registry);
    mlirContextLoadAllAvailableDialects(ctx);
    mlirContextSetAllowUnregisteredDialects(ctx, true);

    if (cml_mlir_register_all_passes_cpp() != 0) {
        mlirContextDestroy(ctx);
        return -1;
    }

    *out_ctx = ctx;
    *out_loc = loc;
    return 0;
}

static MlirModule build_passthrough_module(MlirContext ctx, MlirLocation loc) {
    MlirModule module       = mlirModuleCreateEmpty(loc);
    MlirOperation module_op = mlirModuleGetOperation(module);
    MlirRegion body         = mlirOperationGetRegion(module_op, 0);
    MlirBlock body_block    = mlirRegionGetFirstBlock(body);
    if (mlirBlockIsNull(body_block)) {
        body_block = mlirBlockCreate(0, NULL, NULL);
        mlirRegionAppendOwnedBlock(body, body_block);
    }

    int64_t dyn_shape[]   = {-1};
    MlirType element_type = mlirF32TypeGet(ctx);
    MlirType tensor_type =
        mlirRankedTensorTypeGet(1, dyn_shape, element_type, mlirAttributeGetNull());

    MlirType func_inputs[]  = {tensor_type};
    MlirType func_results[] = {tensor_type};
    MlirType func_type      = mlirFunctionTypeGet(ctx, 1, func_inputs, 1, func_results);

    MlirOperationState func_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);

    MlirNamedAttribute attrs[3];
    attrs[0] =
        mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")),
                              mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("main")));
    attrs[1] = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")),
        mlirTypeAttrGet(func_type));
    attrs[2] = mlirNamedAttributeGet(
        mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_visibility")),
        mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("private")));
    mlirOperationStateAddAttributes(&func_state, 3, attrs);

    MlirRegion func_region = mlirRegionCreate();
    mlirOperationStateAddOwnedRegions(&func_state, 1, &func_region);
    MlirOperation func_op = mlirOperationCreate(&func_state);
    mlirBlockAppendOwnedOperation(body_block, func_op);

    MlirLocation arg_locs[] = {loc};
    MlirBlock entry_block   = mlirBlockCreate(1, func_inputs, arg_locs);
    mlirRegionAppendOwnedBlock(func_region, entry_block);

    MlirValue arg0 = mlirBlockGetArgument(entry_block, 0);
    MlirOperationState ret_state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
    mlirOperationStateAddOperands(&ret_state, 1, &arg0);
    MlirOperation ret_op = mlirOperationCreate(&ret_state);
    mlirBlockAppendOwnedOperation(entry_block, ret_op);

    return module;
}

int main(void) {
    MlirContext ctx  = {0};
    MlirLocation loc = {0};
    if (setup_context(&ctx, &loc) != 0) {
        fprintf(stderr, "Failed to initialize MLIR context\n");
        return 1;
    }

    MlirModule module = build_passthrough_module(ctx, loc);
    dump_module("Original MLIR module", module);

    if (cml_mlir_lower_module_to_llvm(module) != 0) {
        fprintf(stderr, "Lowering pipeline failed\n");
        mlirModuleDestroy(module);
        mlirContextDestroy(ctx);
        return 1;
    }

    dump_module("Lowered LLVM dialect module", module);

    mlirModuleDestroy(module);
    mlirContextDestroy(ctx);
    return 0;
}

#else

int main(void) {
    fprintf(stderr, "MLIR support not available in this build.\n");
    return 1;
}

#endif // CML_HAS_MLIR
