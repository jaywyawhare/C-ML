#include "ops/ir/mlir/mlir_context.h"
#include "ops/ir/mlir/mlir_internal.h"
#include "ops/ir/mlir/mlir_cpp_bridge.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef CML_HAS_MLIR
#include <mlir-c/IR.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Math.h>
#include <mlir-c/Dialect/MemRef.h>
#include <mlir-c/Dialect/Linalg.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/LLVM.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Target/LLVMIR.h>
#endif

bool cml_mlir_is_available(void) {
#ifdef CML_HAS_MLIR
    return true;
#else
    return false;
#endif
}

CMLMLIRStatus cml_mlir_get_status(void) {
#ifdef CML_HAS_MLIR
    return CML_MLIR_AVAILABLE;
#else
    return CML_MLIR_NOT_AVAILABLE;
#endif
}

CMLMLIRContext* cml_mlir_init(void) {
#ifdef CML_HAS_MLIR
    CMLMLIRContext* ctx = (CMLMLIRContext*)malloc(sizeof(CMLMLIRContext));
    if (!ctx)
        return NULL;

    // Create dialect registry and register all dialects
    MlirDialectRegistry registry = mlirDialectRegistryCreate();
    mlirRegisterAllDialects(registry);

    // Initialize MLIR context with the registry
    ctx->context = mlirContextCreate();
    if (mlirContextIsNull(ctx->context)) {
        mlirDialectRegistryDestroy(registry);
        free(ctx);
        return NULL;
    }

    // Append the registry to the context
    mlirContextAppendDialectRegistry(ctx->context, registry);

    // Load all available dialects
    mlirContextLoadAllAvailableDialects(ctx->context);

    // Register all LLVM translations for lowering
    mlirRegisterAllLLVMTranslations(ctx->context);

    // Register all passes
    cml_mlir_register_all_passes_cpp();

    // Clean up registry (context has taken ownership of what it needs)
    mlirDialectRegistryDestroy(registry);

    // Create location
    ctx->location = mlirLocationUnknownGet(ctx->context);

    // Create empty module
    ctx->module = mlirModuleCreateEmpty(ctx->location);

    ctx->initialized = true;
    ctx->target      = MLIR_TARGET_CPU;

    // Initialize tensor tracking
    ctx->inputs        = NULL;
    ctx->num_inputs    = 0;
    ctx->outputs       = NULL;
    ctx->num_outputs   = 0;
    ctx->tensors       = NULL;
    ctx->num_tensors   = 0;
    ctx->cached_engine = NULL;

    return ctx;
#else
    return NULL;
#endif
}

void cml_mlir_destroy(CMLMLIRContext* ctx) {
#ifdef CML_HAS_MLIR
    if (!ctx)
        return;

    // Free cached execution engine
    if (ctx->cached_engine) {
        MlirExecutionEngine engine = {ctx->cached_engine};
        mlirExecutionEngineDestroy(engine);
        ctx->cached_engine = NULL;
    }

    // Free tensor tracking arrays (we don't own the tensors, just the arrays)
    if (ctx->inputs) {
        free(ctx->inputs);
        ctx->inputs = NULL;
    }
    if (ctx->outputs) {
        free(ctx->outputs);
        ctx->outputs = NULL;
    }
    if (ctx->tensors) {
        free(ctx->tensors);
        ctx->tensors = NULL;
    }

    // Destroy MLIR module and context
    if (!mlirModuleIsNull(ctx->module)) {
        mlirModuleDestroy(ctx->module);
    }

    if (!mlirContextIsNull(ctx->context)) {
        mlirContextDestroy(ctx->context);
    }

    free(ctx);
#endif
}

void cml_set_execution_mode(CMLExecutionMode mode) { (void)mode; }

CMLExecutionMode cml_get_execution_mode(void) { return CML_EXEC_INTERPRETED; }

void cml_enable_jit(bool enable) { (void)enable; }
