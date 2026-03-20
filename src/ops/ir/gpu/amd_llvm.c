#ifdef CML_HAS_LLVM_BACKEND

#include "ops/ir/gpu/amd_llvm.h"
#include "core/logging.h"

#include <llvm-c/Core.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <llvm-c/IRReader.h>
#include <llvm-c/Transforms/PassBuilder.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static bool g_amdgpu_initialized = false;

static void ensure_amdgpu_targets(void) {
    if (g_amdgpu_initialized) return;
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();
    g_amdgpu_initialized = true;
}

static const char* detect_gpu_arch(void) {
    const char* env = getenv("CML_AMD_ARCH");
    if (env && env[0]) return env;

    FILE* f = fopen("/sys/class/kfd/kfd/topology/nodes/1/properties", "r");
    if (!f) return NULL;

    static char arch_buf[32];
    char line[256];
    int gfx_major = 0, gfx_minor = 0, gfx_stepping = 0;

    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "gfx_target_version %d", &gfx_major) == 1) {
            gfx_stepping = gfx_major % 100;
            gfx_minor = (gfx_major / 100) % 100;
            gfx_major = gfx_major / 10000;
            break;
        }
    }
    fclose(f);

    if (gfx_major == 0) return NULL;

    if (gfx_stepping == 0)
        snprintf(arch_buf, sizeof(arch_buf), "gfx%d%02d", gfx_major, gfx_minor);
    else
        snprintf(arch_buf, sizeof(arch_buf), "gfx%d%02d%02d", gfx_major, gfx_minor, gfx_stepping);

    return arch_buf;
}

bool cml_amd_llvm_available(void) {
    ensure_amdgpu_targets();

    LLVMTargetRef target;
    char* err = NULL;
    if (LLVMGetTargetFromTriple("amdgcn-amd-amdhsa", &target, &err) != 0) {
        LLVMDisposeMessage(err);
        return false;
    }
    return true;
}

CMLAMDLLVMCompiler* cml_amd_llvm_create(const char* gpu_arch) {
    ensure_amdgpu_targets();

    if (!gpu_arch || !gpu_arch[0]) {
        gpu_arch = detect_gpu_arch();
        if (!gpu_arch) {
            LOG_ERROR("AMD LLVM: no GPU arch specified and auto-detection failed");
            return NULL;
        }
    }

    CMLAMDLLVMCompiler* comp = calloc(1, sizeof(CMLAMDLLVMCompiler));
    if (!comp) return NULL;

    strncpy(comp->target_triple, "amdgcn-amd-amdhsa", sizeof(comp->target_triple) - 1);
    strncpy(comp->target_cpu, gpu_arch, sizeof(comp->target_cpu) - 1);
    comp->opt_level = 2;

    if (strncmp(gpu_arch, "gfx11", 5) == 0 || strncmp(gpu_arch, "gfx12", 5) == 0)
        strncpy(comp->features, "+wavefrontsize32", sizeof(comp->features) - 1);
    else
        strncpy(comp->features, "+wavefrontsize64", sizeof(comp->features) - 1);

    comp->llvm_ctx = LLVMContextCreate();
    if (!comp->llvm_ctx) {
        LOG_ERROR("AMD LLVM: failed to create context");
        free(comp);
        return NULL;
    }

    LOG_INFO("AMD LLVM: compiler ready for %s (%s)", comp->target_cpu, comp->features);
    return comp;
}

void cml_amd_llvm_free(CMLAMDLLVMCompiler* comp) {
    if (!comp) return;
    if (comp->llvm_ctx)
        LLVMContextDispose((LLVMContextRef)comp->llvm_ctx);
    free(comp);
}

static LLVMTargetMachineRef create_target_machine(CMLAMDLLVMCompiler* comp) {
    LLVMTargetRef target;
    char* err = NULL;

    if (LLVMGetTargetFromTriple(comp->target_triple, &target, &err) != 0) {
        LOG_ERROR("AMD LLVM: target lookup failed: %s", err ? err : "unknown");
        LLVMDisposeMessage(err);
        return NULL;
    }

    LLVMCodeGenOptLevel opt;
    switch (comp->opt_level) {
    case 0:  opt = LLVMCodeGenLevelNone; break;
    case 1:  opt = LLVMCodeGenLevelLess; break;
    case 3:  opt = LLVMCodeGenLevelAggressive; break;
    default: opt = LLVMCodeGenLevelDefault; break;
    }

    return LLVMCreateTargetMachine(target, comp->target_triple, comp->target_cpu,
                                   comp->features, opt, LLVMRelocDefault,
                                   LLVMCodeModelDefault);
}

static int run_opt_passes(LLVMModuleRef mod, LLVMTargetMachineRef tm, int opt_level) {
    const char* passes;
    switch (opt_level) {
    case 0:  passes = "default<O0>"; break;
    case 1:  passes = "default<O1>"; break;
    case 3:  passes = "default<O3>"; break;
    default: passes = "default<O2>"; break;
    }

    LLVMPassBuilderOptionsRef opts = LLVMCreatePassBuilderOptions();
    LLVMPassBuilderOptionSetLoopVectorization(opts, opt_level >= 2);
    LLVMPassBuilderOptionSetSLPVectorization(opts, opt_level >= 2);

    LLVMErrorRef err = LLVMRunPasses(mod, passes, tm, opts);
    LLVMDisposePassBuilderOptions(opts);

    if (err) {
        char* msg = LLVMGetErrorMessage(err);
        LOG_ERROR("AMD LLVM: pass pipeline failed: %s", msg);
        LLVMDisposeErrorMessage(msg);
        return -1;
    }
    return 0;
}

int cml_amd_llvm_compile_ir(CMLAMDLLVMCompiler* comp,
                            const char* ir_source,
                            void** code_object, size_t* code_size) {
    if (!comp || !ir_source || !code_object || !code_size) return -1;

    *code_object = NULL;
    *code_size = 0;

    LLVMContextRef ctx = (LLVMContextRef)comp->llvm_ctx;
    LLVMMemoryBufferRef ir_buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(
        ir_source, strlen(ir_source), "amd_ir");

    LLVMModuleRef mod = NULL;
    char* err = NULL;
    if (LLVMParseIRInContext(ctx, ir_buf, &mod, &err) != 0) {
        LOG_ERROR("AMD LLVM: IR parse failed: %s", err ? err : "unknown");
        LLVMDisposeMessage(err);
        return -1;
    }

    LLVMSetTarget(mod, comp->target_triple);

    LLVMTargetMachineRef tm = create_target_machine(comp);
    if (!tm) {
        LLVMDisposeModule(mod);
        return -1;
    }

    LLVMSetModuleDataLayout(mod, LLVMCreateTargetDataLayout(tm));

    if (run_opt_passes(mod, tm, comp->opt_level) != 0) {
        LLVMDisposeModule(mod);
        LLVMDisposeTargetMachine(tm);
        return -1;
    }

    char* verify_err = NULL;
    if (LLVMVerifyModule(mod, LLVMReturnStatusAction, &verify_err)) {
        LOG_ERROR("AMD LLVM: module verification failed: %s", verify_err ? verify_err : "unknown");
        LLVMDisposeMessage(verify_err);
        LLVMDisposeModule(mod);
        LLVMDisposeTargetMachine(tm);
        return -1;
    }
    LLVMDisposeMessage(verify_err);

    LLVMMemoryBufferRef obj_buf = NULL;
    char* emit_err = NULL;
    if (LLVMTargetMachineEmitToMemoryBuffer(tm, mod, LLVMObjectFile, &emit_err, &obj_buf) != 0) {
        LOG_ERROR("AMD LLVM: codegen failed: %s", emit_err ? emit_err : "unknown");
        LLVMDisposeMessage(emit_err);
        LLVMDisposeModule(mod);
        LLVMDisposeTargetMachine(tm);
        return -1;
    }

    size_t obj_size = LLVMGetBufferSize(obj_buf);
    const char* obj_data = LLVMGetBufferStart(obj_buf);

    void* result = malloc(obj_size);
    if (!result) {
        LLVMDisposeMemoryBuffer(obj_buf);
        LLVMDisposeModule(mod);
        LLVMDisposeTargetMachine(tm);
        return -1;
    }
    memcpy(result, obj_data, obj_size);

    *code_object = result;
    *code_size = obj_size;

    LOG_INFO("AMD LLVM: compiled %zu bytes of AMDGPU code object for %s",
             obj_size, comp->target_cpu);

    LLVMDisposeMemoryBuffer(obj_buf);
    LLVMDisposeModule(mod);
    LLVMDisposeTargetMachine(tm);
    return 0;
}

int cml_amd_llvm_compile_source(CMLAMDLLVMCompiler* comp,
                                const char* source,
                                void** code_object, size_t* code_size) {
    if (!comp || !source || !code_object || !code_size) return -1;

    *code_object = NULL;
    *code_size = 0;

    LLVMContextRef ctx = (LLVMContextRef)comp->llvm_ctx;
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext("amd_kernel", ctx);
    LLVMSetTarget(mod, comp->target_triple);

    LLVMTargetMachineRef tm = create_target_machine(comp);
    if (!tm) {
        LLVMDisposeModule(mod);
        return -1;
    }

    LLVMSetModuleDataLayout(mod, LLVMCreateTargetDataLayout(tm));

    LLVMTypeRef i32_ty = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef f32_ty = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef f32_ptr_ty = LLVMPointerType(f32_ty, 1);
    LLVMTypeRef param_types[] = { f32_ptr_ty, f32_ptr_ty, f32_ptr_ty, i32_ty };
    LLVMTypeRef fn_ty = LLVMFunctionType(LLVMVoidTypeInContext(ctx), param_types, 4, 0);

    LLVMValueRef func = LLVMAddFunction(mod, "vector_add", fn_ty);
    LLVMSetFunctionCallConv(func, 91); /* AMDGPU_KERNEL */

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, func, "entry");
    LLVMBuilderRef builder = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(builder, entry);

    LLVMTypeRef tid_fn_ty = LLVMFunctionType(i32_ty, NULL, 0, 0);
    LLVMValueRef tid_fn = LLVMAddFunction(mod, "llvm.amdgcn.workitem.id.x", tid_fn_ty);
    LLVMValueRef gid_fn = LLVMAddFunction(mod, "llvm.amdgcn.workgroup.id.x", tid_fn_ty);

    LLVMValueRef local_id = LLVMBuildCall2(builder, tid_fn_ty, tid_fn, NULL, 0, "lid");
    LLVMValueRef group_id = LLVMBuildCall2(builder, tid_fn_ty, gid_fn, NULL, 0, "gid");

    LLVMValueRef block_size = LLVMConstInt(i32_ty, 256, 0);
    LLVMValueRef offset = LLVMBuildMul(builder, group_id, block_size, "off");
    LLVMValueRef idx = LLVMBuildAdd(builder, offset, local_id, "idx");

    LLVMValueRef n = LLVMGetParam(func, 3);
    LLVMValueRef cmp = LLVMBuildICmp(builder, LLVMIntSLT, idx, n, "cmp");

    LLVMBasicBlockRef body = LLVMAppendBasicBlockInContext(ctx, func, "body");
    LLVMBasicBlockRef exit_bb = LLVMAppendBasicBlockInContext(ctx, func, "exit");
    LLVMBuildCondBr(builder, cmp, body, exit_bb);

    LLVMPositionBuilderAtEnd(builder, body);
    LLVMValueRef a_ptr = LLVMBuildGEP2(builder, f32_ty, LLVMGetParam(func, 0), &idx, 1, "ap");
    LLVMValueRef b_ptr = LLVMBuildGEP2(builder, f32_ty, LLVMGetParam(func, 1), &idx, 1, "bp");
    LLVMValueRef a_val = LLVMBuildLoad2(builder, f32_ty, a_ptr, "a");
    LLVMValueRef b_val = LLVMBuildLoad2(builder, f32_ty, b_ptr, "b");
    LLVMValueRef sum = LLVMBuildFAdd(builder, a_val, b_val, "sum");
    LLVMValueRef c_ptr = LLVMBuildGEP2(builder, f32_ty, LLVMGetParam(func, 2), &idx, 1, "cp");
    LLVMBuildStore(builder, sum, c_ptr);
    LLVMBuildBr(builder, exit_bb);

    LLVMPositionBuilderAtEnd(builder, exit_bb);
    LLVMBuildRetVoid(builder);
    LLVMDisposeBuilder(builder);

    (void)source;

    if (run_opt_passes(mod, tm, comp->opt_level) != 0) {
        LLVMDisposeModule(mod);
        LLVMDisposeTargetMachine(tm);
        return -1;
    }

    LLVMMemoryBufferRef obj_buf = NULL;
    char* emit_err = NULL;
    if (LLVMTargetMachineEmitToMemoryBuffer(tm, mod, LLVMObjectFile, &emit_err, &obj_buf) != 0) {
        LOG_ERROR("AMD LLVM: source codegen failed: %s", emit_err ? emit_err : "unknown");
        LLVMDisposeMessage(emit_err);
        LLVMDisposeModule(mod);
        LLVMDisposeTargetMachine(tm);
        return -1;
    }

    size_t obj_size = LLVMGetBufferSize(obj_buf);
    const char* obj_data = LLVMGetBufferStart(obj_buf);

    void* result = malloc(obj_size);
    if (!result) {
        LLVMDisposeMemoryBuffer(obj_buf);
        LLVMDisposeModule(mod);
        LLVMDisposeTargetMachine(tm);
        return -1;
    }
    memcpy(result, obj_data, obj_size);

    *code_object = result;
    *code_size = obj_size;

    LOG_INFO("AMD LLVM: compiled source to %zu bytes for %s", obj_size, comp->target_cpu);

    LLVMDisposeMemoryBuffer(obj_buf);
    LLVMDisposeModule(mod);
    LLVMDisposeTargetMachine(tm);
    return 0;
}

#else /* !CML_HAS_LLVM_BACKEND */

#include "ops/ir/gpu/amd_llvm.h"
#include <stddef.h>

bool cml_amd_llvm_available(void) { return false; }
CMLAMDLLVMCompiler* cml_amd_llvm_create(const char* gpu_arch) { (void)gpu_arch; return NULL; }
void cml_amd_llvm_free(CMLAMDLLVMCompiler* comp) { (void)comp; }

int cml_amd_llvm_compile_ir(CMLAMDLLVMCompiler* comp, const char* ir_source,
                            void** code_object, size_t* code_size) {
    (void)comp; (void)ir_source; (void)code_object; (void)code_size;
    return -1;
}

int cml_amd_llvm_compile_source(CMLAMDLLVMCompiler* comp, const char* source,
                                void** code_object, size_t* code_size) {
    (void)comp; (void)source; (void)code_object; (void)code_size;
    return -1;
}

#endif
