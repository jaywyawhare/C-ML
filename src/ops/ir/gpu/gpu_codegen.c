/**
 * @file gpu_codegen.c
 * @brief GPU kernel code generation via LLVM NVPTX/AMDGPU targets
 *
 * Mirrors the CPU LLVM backend (llvm_backend.c) but replaces loops with
 * GPU thread indexing. Emits PTX (CUDA) or HSACO (ROCm) via LLVM target
 * machines, then uses the existing runtime backends for execution.
 *
 * ABI: All kernels use addrspace(1) float* pointers (global memory).
 * Sizes are i32 (adequate for practical tensor dimensions).
 */

#ifdef CML_HAS_LLVM_BACKEND

#include "ops/ir/gpu/gpu_codegen.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "core/logging.h"

#include <llvm-c/Core.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

static bool g_nvptx_initialized = false;
static bool g_amdgpu_initialized = false;

static void init_nvptx_target(void) {
    if (g_nvptx_initialized) return;
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    g_nvptx_initialized = true;
}

static void init_amdgpu_target(void) {
    if (g_amdgpu_initialized) return;
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();
    g_amdgpu_initialized = true;
}

CMLGPUCodegen* cml_gpu_codegen_create(GPUTarget target, void* backend) {
    if (!backend) return NULL;

    CMLGPUCodegen* cg = calloc(1, sizeof(CMLGPUCodegen));
    if (!cg) return NULL;

    cg->target = target;
    cg->default_block_size = 256;

    if (target == GPU_TARGET_CUDA) {
        init_nvptx_target();
        cg->cuda = (CMLCUDABackend*)backend;
        strncpy(cg->target_triple, "nvptx64-nvidia-cuda", sizeof(cg->target_triple) - 1);
        snprintf(cg->target_cpu, sizeof(cg->target_cpu), "sm_%d%d",
                 cg->cuda->compute_capability_major,
                 cg->cuda->compute_capability_minor);
        LOG_INFO("GPU codegen: CUDA target %s", cg->target_cpu);
    } else {
        init_amdgpu_target();
        cg->rocm = (CMLROCmBackend*)backend;
        strncpy(cg->target_triple, "amdgcn-amd-amdhsa", sizeof(cg->target_triple) - 1);
        strncpy(cg->target_cpu, "gfx900", sizeof(cg->target_cpu) - 1);
        LOG_INFO("GPU codegen: ROCm target %s", cg->target_cpu);
    }

    cg->initialized = true;
    cg->kernel_count = 0;
    return cg;
}

void cml_gpu_codegen_destroy(CMLGPUCodegen* cg) {
    free(cg);
}

// NVPTX calling convention for kernels
#define CC_PTX_KERNEL 71
// AMDGPU calling convention for kernels
#define CC_AMDGPU_KERNEL 91

// Get global thread ID: gid = blockIdx.x * blockDim.x + threadIdx.x
static LLVMValueRef emit_global_thread_id(LLVMBuilderRef bld, LLVMModuleRef mod,
                                           LLVMContextRef ctx, GPUTarget target) {
    LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx);

    if (target == GPU_TARGET_CUDA) {
        // NVPTX intrinsics
        LLVMTypeRef intr_type = LLVMFunctionType(i32, NULL, 0, 0);

        LLVMValueRef tid_fn = LLVMGetNamedFunction(mod, "llvm.nvvm.read.ptx.sreg.tid.x");
        if (!tid_fn) tid_fn = LLVMAddFunction(mod, "llvm.nvvm.read.ptx.sreg.tid.x", intr_type);

        LLVMValueRef ctaid_fn = LLVMGetNamedFunction(mod, "llvm.nvvm.read.ptx.sreg.ctaid.x");
        if (!ctaid_fn) ctaid_fn = LLVMAddFunction(mod, "llvm.nvvm.read.ptx.sreg.ctaid.x", intr_type);

        LLVMValueRef ntid_fn = LLVMGetNamedFunction(mod, "llvm.nvvm.read.ptx.sreg.ntid.x");
        if (!ntid_fn) ntid_fn = LLVMAddFunction(mod, "llvm.nvvm.read.ptx.sreg.ntid.x", intr_type);

        LLVMValueRef tid = LLVMBuildCall2(bld, intr_type, tid_fn, NULL, 0, "tid");
        LLVMValueRef ctaid = LLVMBuildCall2(bld, intr_type, ctaid_fn, NULL, 0, "ctaid");
        LLVMValueRef ntid = LLVMBuildCall2(bld, intr_type, ntid_fn, NULL, 0, "ntid");

        LLVMValueRef block_off = LLVMBuildMul(bld, ctaid, ntid, "block_off");
        return LLVMBuildAdd(bld, block_off, tid, "gid");
    } else {
        // AMDGPU intrinsics
        LLVMTypeRef intr_type = LLVMFunctionType(i32, NULL, 0, 0);

        LLVMValueRef wid_fn = LLVMGetNamedFunction(mod, "llvm.amdgcn.workitem.id.x");
        if (!wid_fn) wid_fn = LLVMAddFunction(mod, "llvm.amdgcn.workitem.id.x", intr_type);

        LLVMValueRef gid_fn = LLVMGetNamedFunction(mod, "llvm.amdgcn.workgroup.id.x");
        if (!gid_fn) gid_fn = LLVMAddFunction(mod, "llvm.amdgcn.workgroup.id.x", intr_type);

        LLVMValueRef wid = LLVMBuildCall2(bld, intr_type, wid_fn, NULL, 0, "wid");
        LLVMValueRef wgid = LLVMBuildCall2(bld, intr_type, gid_fn, NULL, 0, "wgid");

        // blockDim is a compile-time constant for AMDGPU
        LLVMValueRef block_size = LLVMConstInt(i32, 256, 0);
        LLVMValueRef block_off = LLVMBuildMul(bld, wgid, block_size, "block_off");
        return LLVMBuildAdd(bld, block_off, wid, "gid");
    }
}

// Set up module with target triple and data layout, configure function as kernel
static void configure_gpu_module(LLVMModuleRef mod, LLVMValueRef fn,
                                  CMLGPUCodegen* cg) {
    LLVMSetTarget(mod, cg->target_triple);

    if (cg->target == GPU_TARGET_CUDA) {
        LLVMSetDataLayout(mod, "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                               "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
                               "v64:64:64-v128:128:128-n16:32:64");
        LLVMSetFunctionCallConv(fn, CC_PTX_KERNEL);

        // Add nvvm.annotations metadata: {fn, "kernel", i32 1}
        LLVMContextRef ctx = LLVMGetModuleContext(mod);
        LLVMValueRef md_vals[3];
        md_vals[0] = fn;
        md_vals[1] = LLVMMDStringInContext(ctx, "kernel", 6);
        md_vals[2] = LLVMConstInt(LLVMInt32TypeInContext(ctx), 1, 0);
        LLVMValueRef md_node = LLVMMDNodeInContext(ctx, md_vals, 3);

        LLVMAddNamedMetadataOperand(mod, "nvvm.annotations", md_node);
    } else {
        LLVMSetDataLayout(mod, "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-"
                               "p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-"
                               "v48:64-v96:128-v192:256-v256:256-v512:512-"
                               "v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7");
        LLVMSetFunctionCallConv(fn, CC_AMDGPU_KERNEL);
    }
}

// Emit bounds check: if (gid >= n) return;
static LLVMBasicBlockRef emit_bounds_check(LLVMBuilderRef bld, LLVMContextRef ctx,
                                            LLVMValueRef fn, LLVMValueRef gid,
                                            LLVMValueRef n) {
    LLVMBasicBlockRef body = LLVMAppendBasicBlockInContext(ctx, fn, "body");
    LLVMBasicBlockRef exit_bb = LLVMAppendBasicBlockInContext(ctx, fn, "exit");

    LLVMValueRef cond = LLVMBuildICmp(bld, LLVMIntULT, gid, n, "in_bounds");
    LLVMBuildCondBr(bld, cond, body, exit_bb);

    // Exit block: just return
    LLVMPositionBuilderAtEnd(bld, exit_bb);
    LLVMBuildRetVoid(bld);

    // Position at body for caller to emit kernel logic
    LLVMPositionBuilderAtEnd(bld, body);
    return body;
}

// Binary: void kernel(ptr(1) in0, ptr(1) in1, ptr(1) out, i32 n, i32 n0, i32 n1)
static LLVMModuleRef gpu_build_binary_op(LLVMContextRef ctx, UOpType type,
                                          const char* fn_name, CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1); // addrspace(1)
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr1, ptr1, ptr1, i32, i32, i32 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 6, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef in0   = LLVMGetParam(fn, 0);
    LLVMValueRef in1   = LLVMGetParam(fn, 1);
    LLVMValueRef out   = LLVMGetParam(fn, 2);
    LLVMValueRef n     = LLVMGetParam(fn, 3);
    LLVMValueRef n0    = LLVMGetParam(fn, 4);
    LLVMValueRef n1    = LLVMGetParam(fn, 5);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);
    emit_bounds_check(bld, ctx, fn, gid, n);

    // Broadcast indices: i0 = gid % n0, i1 = gid % n1
    LLVMValueRef i0 = LLVMBuildURem(bld, gid, n0, "i0");
    LLVMValueRef i1 = LLVMBuildURem(bld, gid, n1, "i1");

    // Extend to i64 for GEP
    LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx);
    LLVMValueRef i0_64 = LLVMBuildZExt(bld, i0, i64, "i0_64");
    LLVMValueRef i1_64 = LLVMBuildZExt(bld, i1, i64, "i1_64");
    LLVMValueRef gid_64 = LLVMBuildZExt(bld, gid, i64, "gid_64");

    LLVMValueRef gep0 = LLVMBuildGEP2(bld, f32, in0, &i0_64, 1, "p0");
    LLVMValueRef gep1 = LLVMBuildGEP2(bld, f32, in1, &i1_64, 1, "p1");
    LLVMValueRef v0 = LLVMBuildLoad2(bld, f32, gep0, "v0");
    LLVMValueRef v1 = LLVMBuildLoad2(bld, f32, gep1, "v1");

    LLVMValueRef result = NULL;
    switch (type) {
    case UOP_ADD: result = LLVMBuildFAdd(bld, v0, v1, "add"); break;
    case UOP_SUB: result = LLVMBuildFSub(bld, v0, v1, "sub"); break;
    case UOP_MUL: result = LLVMBuildFMul(bld, v0, v1, "mul"); break;
    case UOP_DIV: {
        LLVMValueRef eps = LLVMConstReal(f32, 1e-8);
        LLVMValueRef denom = LLVMBuildFAdd(bld, v1, eps, "denom");
        result = LLVMBuildFDiv(bld, v0, denom, "div");
        break;
    }
    case UOP_MAX: {
        LLVMValueRef cmp = LLVMBuildFCmp(bld, LLVMRealOGT, v0, v1, "gt");
        result = LLVMBuildSelect(bld, cmp, v0, v1, "max");
        break;
    }
    case UOP_CMPLT: {
        LLVMValueRef cmp = LLVMBuildFCmp(bld, LLVMRealOLT, v0, v1, "lt");
        result = LLVMBuildUIToFP(bld, cmp, f32, "cmplt");
        break;
    }
    case UOP_POW: {
        LLVMTypeRef pow_params[] = { f32, f32 };
        LLVMTypeRef pow_ft = LLVMFunctionType(f32, pow_params, 2, 0);
        LLVMValueRef pow_fn = LLVMGetNamedFunction(mod, "llvm.pow.f32");
        if (!pow_fn) pow_fn = LLVMAddFunction(mod, "llvm.pow.f32", pow_ft);
        LLVMValueRef args[] = { v0, v1 };
        result = LLVMBuildCall2(bld, pow_ft, pow_fn, args, 2, "pow");
        break;
    }
    case UOP_IDIV: {
        // Integer division on floats: floor(a / b)
        LLVMValueRef div = LLVMBuildFDiv(bld, v0, v1, "fdiv");
        LLVMTypeRef floor_params[] = { f32 };
        LLVMTypeRef floor_ft = LLVMFunctionType(f32, floor_params, 1, 0);
        LLVMValueRef floor_fn = LLVMGetNamedFunction(mod, "llvm.floor.f32");
        if (!floor_fn) floor_fn = LLVMAddFunction(mod, "llvm.floor.f32", floor_ft);
        LLVMValueRef floor_args[] = { div };
        result = LLVMBuildCall2(bld, floor_ft, floor_fn, floor_args, 1, "idiv");
        break;
    }
    case UOP_MOD: {
        // Modulo: a - floor(a/b) * b
        LLVMValueRef div = LLVMBuildFDiv(bld, v0, v1, "fdiv");
        LLVMTypeRef floor_params[] = { f32 };
        LLVMTypeRef floor_ft = LLVMFunctionType(f32, floor_params, 1, 0);
        LLVMValueRef floor_fn = LLVMGetNamedFunction(mod, "llvm.floor.f32");
        if (!floor_fn) floor_fn = LLVMAddFunction(mod, "llvm.floor.f32", floor_ft);
        LLVMValueRef floor_args[] = { div };
        LLVMValueRef floored = LLVMBuildCall2(bld, floor_ft, floor_fn, floor_args, 1, "fl");
        LLVMValueRef prod = LLVMBuildFMul(bld, floored, v1, "prod");
        result = LLVMBuildFSub(bld, v0, prod, "mod");
        break;
    }
    default:
        result = LLVMBuildFAdd(bld, v0, v1, "fallback");
        break;
    }

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &gid_64, 1, "pout");
    LLVMBuildStore(bld, result, gep_out);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Unary: void kernel(ptr(1) in, ptr(1) out, i32 n, i32 in_n)
static LLVMModuleRef gpu_build_unary_op(LLVMContextRef ctx, UOpType type,
                                         const char* fn_name, CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr1, ptr1, i32, i32 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 4, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef in    = LLVMGetParam(fn, 0);
    LLVMValueRef out   = LLVMGetParam(fn, 1);
    LLVMValueRef n     = LLVMGetParam(fn, 2);
    LLVMValueRef in_n  = LLVMGetParam(fn, 3);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);
    emit_bounds_check(bld, ctx, fn, gid, n);

    LLVMValueRef idx = LLVMBuildURem(bld, gid, in_n, "idx");
    LLVMValueRef idx_64 = LLVMBuildZExt(bld, idx, i64, "idx_64");
    LLVMValueRef gid_64 = LLVMBuildZExt(bld, gid, i64, "gid_64");

    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in, &idx_64, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "val");

    // Intrinsic helper
    #define GPU_INTRINSIC1(name_str, arg, res_name) do { \
        LLVMTypeRef _params[] = { f32 }; \
        LLVMTypeRef _ft = LLVMFunctionType(f32, _params, 1, 0); \
        LLVMValueRef _fn = LLVMGetNamedFunction(mod, name_str); \
        if (!_fn) _fn = LLVMAddFunction(mod, name_str, _ft); \
        LLVMValueRef _args[] = { arg }; \
        result = LLVMBuildCall2(bld, _ft, _fn, _args, 1, res_name); \
    } while(0)

    LLVMValueRef result = NULL;
    switch (type) {
    case UOP_NEG:
        result = LLVMBuildFNeg(bld, val, "neg");
        break;
    case UOP_EXP:
        GPU_INTRINSIC1("llvm.exp.f32", val, "exp");
        break;
    case UOP_LOG: {
        LLVMValueRef eps = LLVMConstReal(f32, 1e-8f);
        LLVMValueRef safe = LLVMBuildFAdd(bld, val, eps, "safe");
        GPU_INTRINSIC1("llvm.log.f32", safe, "log");
        break;
    }
    case UOP_SQRT: {
        GPU_INTRINSIC1("llvm.fabs.f32", val, "absv");
        LLVMValueRef abs_val = result;
        GPU_INTRINSIC1("llvm.sqrt.f32", abs_val, "sqrt");
        break;
    }
    case UOP_ABS:
        GPU_INTRINSIC1("llvm.fabs.f32", val, "abs");
        break;
    case UOP_SIN:
        GPU_INTRINSIC1("llvm.sin.f32", val, "sin");
        break;
    case UOP_COS:
        GPU_INTRINSIC1("llvm.cos.f32", val, "cos");
        break;
    case UOP_TAN: {
        LLVMTypeRef ft1_params[] = { f32 };
        LLVMTypeRef ft1 = LLVMFunctionType(f32, ft1_params, 1, 0);
        LLVMValueRef sin_fn = LLVMGetNamedFunction(mod, "llvm.sin.f32");
        if (!sin_fn) sin_fn = LLVMAddFunction(mod, "llvm.sin.f32", ft1);
        LLVMValueRef cos_fn = LLVMGetNamedFunction(mod, "llvm.cos.f32");
        if (!cos_fn) cos_fn = LLVMAddFunction(mod, "llvm.cos.f32", ft1);
        LLVMValueRef s_args[] = { val };
        LLVMValueRef s = LLVMBuildCall2(bld, ft1, sin_fn, s_args, 1, "s");
        LLVMValueRef c_args[] = { val };
        LLVMValueRef c = LLVMBuildCall2(bld, ft1, cos_fn, c_args, 1, "c");
        result = LLVMBuildFDiv(bld, s, c, "tan");
        break;
    }
    case UOP_RECIP: {
        LLVMValueRef one_f = LLVMConstReal(f32, 1.0);
        result = LLVMBuildFDiv(bld, one_f, val, "recip");
        break;
    }
    case UOP_SIGMOID: {
        LLVMValueRef neg = LLVMBuildFNeg(bld, val, "neg");
        GPU_INTRINSIC1("llvm.exp.f32", neg, "e");
        LLVMValueRef e = result;
        LLVMValueRef one_f = LLVMConstReal(f32, 1.0);
        LLVMValueRef denom = LLVMBuildFAdd(bld, one_f, e, "denom");
        result = LLVMBuildFDiv(bld, one_f, denom, "sigmoid");
        break;
    }
    case UOP_TANH: {
        LLVMValueRef two = LLVMConstReal(f32, 2.0);
        LLVMValueRef two_x = LLVMBuildFMul(bld, two, val, "2x");
        LLVMValueRef neg = LLVMBuildFNeg(bld, two_x, "neg2x");
        GPU_INTRINSIC1("llvm.exp.f32", neg, "e");
        LLVMValueRef e = result;
        LLVMValueRef one_f = LLVMConstReal(f32, 1.0);
        LLVMValueRef denom = LLVMBuildFAdd(bld, one_f, e, "denom");
        LLVMValueRef sig = LLVMBuildFDiv(bld, one_f, denom, "sig");
        LLVMValueRef scaled = LLVMBuildFMul(bld, two, sig, "scaled");
        result = LLVMBuildFSub(bld, scaled, one_f, "tanh");
        break;
    }
    case UOP_ELU: {
        // ELU: x > 0 ? x : alpha*(exp(x)-1), alpha=1.0
        LLVMValueRef alpha = LLVMConstReal(f32, 1.0);
        LLVMValueRef zero = LLVMConstReal(f32, 0.0);
        LLVMValueRef one_f = LLVMConstReal(f32, 1.0);
        LLVMValueRef cmp = LLVMBuildFCmp(bld, LLVMRealOGT, val, zero, "gt0");
        GPU_INTRINSIC1("llvm.exp.f32", val, "expv");
        LLVMValueRef expv = result;
        LLVMValueRef em1 = LLVMBuildFSub(bld, expv, one_f, "em1");
        LLVMValueRef neg_branch = LLVMBuildFMul(bld, alpha, em1, "aem1");
        result = LLVMBuildSelect(bld, cmp, val, neg_branch, "elu");
        break;
    }
    case UOP_SELU: {
        // SELU: scale*(x > 0 ? x : alpha*(exp(x)-1))
        LLVMValueRef scale = LLVMConstReal(f32, 1.0507009873554804934193349852946);
        LLVMValueRef alpha = LLVMConstReal(f32, 1.6732632423543772848170429916717);
        LLVMValueRef zero = LLVMConstReal(f32, 0.0);
        LLVMValueRef one_f = LLVMConstReal(f32, 1.0);
        LLVMValueRef cmp = LLVMBuildFCmp(bld, LLVMRealOGT, val, zero, "gt0");
        GPU_INTRINSIC1("llvm.exp.f32", val, "expv");
        LLVMValueRef expv = result;
        LLVMValueRef em1 = LLVMBuildFSub(bld, expv, one_f, "em1");
        LLVMValueRef neg_branch = LLVMBuildFMul(bld, alpha, em1, "aem1");
        LLVMValueRef inner = LLVMBuildSelect(bld, cmp, val, neg_branch, "inner");
        result = LLVMBuildFMul(bld, scale, inner, "selu");
        break;
    }
    case UOP_SILU: {
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        LLVMValueRef neg_v = LLVMBuildFNeg(bld, val, "neg");
        GPU_INTRINSIC1("llvm.exp.f32", neg_v, "expneg");
        LLVMValueRef expneg = result;
        LLVMValueRef one_f = LLVMConstReal(f32, 1.0);
        LLVMValueRef denom = LLVMBuildFAdd(bld, one_f, expneg, "denom");
        result = LLVMBuildFDiv(bld, val, denom, "silu");
        break;
    }
    case UOP_MISH: {
        // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        // softplus(x) = max(x,0) + ln(1 + exp(-|x|)) (numerically stable)
        GPU_INTRINSIC1("llvm.fabs.f32", val, "absv");
        LLVMValueRef absv = result;
        LLVMValueRef neg_abs = LLVMBuildFNeg(bld, absv, "negabs");
        GPU_INTRINSIC1("llvm.exp.f32", neg_abs, "ea");
        LLVMValueRef ea = result;
        LLVMValueRef one_f = LLVMConstReal(f32, 1.0);
        LLVMValueRef one_ea = LLVMBuildFAdd(bld, one_f, ea, "onea");
        GPU_INTRINSIC1("llvm.log.f32", one_ea, "logea");
        LLVMValueRef log_part = result;
        LLVMValueRef zero = LLVMConstReal(f32, 0.0);
        LLVMValueRef cmp = LLVMBuildFCmp(bld, LLVMRealOGT, val, zero, "gt0");
        LLVMValueRef max_val = LLVMBuildSelect(bld, cmp, val, zero, "maxv");
        LLVMValueRef sp = LLVMBuildFAdd(bld, max_val, log_part, "sp");
        // tanh(sp) via 2*sigmoid(2*sp)-1
        LLVMValueRef two = LLVMConstReal(f32, 2.0);
        LLVMValueRef two_sp = LLVMBuildFMul(bld, two, sp, "2sp");
        LLVMValueRef neg_2sp = LLVMBuildFNeg(bld, two_sp, "neg2sp");
        GPU_INTRINSIC1("llvm.exp.f32", neg_2sp, "e2sp");
        LLVMValueRef e2sp = result;
        LLVMValueRef denom = LLVMBuildFAdd(bld, one_f, e2sp, "denom");
        LLVMValueRef sig = LLVMBuildFDiv(bld, one_f, denom, "sig");
        LLVMValueRef scaled = LLVMBuildFMul(bld, two, sig, "sc");
        LLVMValueRef tanh_sp = LLVMBuildFSub(bld, scaled, one_f, "tsp");
        result = LLVMBuildFMul(bld, val, tanh_sp, "mish");
        break;
    }
    case UOP_HARDSWISH: {
        // HardSwish: x >= 3 ? x : x <= -3 ? 0 : x*(x+3)/6
        LLVMValueRef three = LLVMConstReal(f32, 3.0);
        LLVMValueRef neg_three = LLVMConstReal(f32, -3.0);
        LLVMValueRef six = LLVMConstReal(f32, 6.0);
        LLVMValueRef zero = LLVMConstReal(f32, 0.0);
        LLVMValueRef cmp_ge3 = LLVMBuildFCmp(bld, LLVMRealOGE, val, three, "ge3");
        LLVMValueRef cmp_le_n3 = LLVMBuildFCmp(bld, LLVMRealOLE, val, neg_three, "len3");
        LLVMValueRef xp3 = LLVMBuildFAdd(bld, val, three, "xp3");
        LLVMValueRef xp3x = LLVMBuildFMul(bld, val, xp3, "xp3x");
        LLVMValueRef mid = LLVMBuildFDiv(bld, xp3x, six, "mid");
        LLVMValueRef sel1 = LLVMBuildSelect(bld, cmp_le_n3, zero, mid, "s1");
        result = LLVMBuildSelect(bld, cmp_ge3, val, sel1, "hardswish");
        break;
    }
    default:
        result = val;
        break;
    }

    #undef GPU_INTRINSIC1

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &gid_64, 1, "pout");
    LLVMBuildStore(bld, result, gep_out);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Fill: void kernel(ptr(1) out, i32 n, float val)
static LLVMModuleRef gpu_build_fill_op(LLVMContextRef ctx, const char* fn_name,
                                        float fill_value, CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr1, i32 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 2, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef out = LLVMGetParam(fn, 0);
    LLVMValueRef n   = LLVMGetParam(fn, 1);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);
    emit_bounds_check(bld, ctx, fn, gid, n);

    LLVMValueRef gid_64 = LLVMBuildZExt(bld, gid, i64, "gid_64");
    LLVMValueRef val = LLVMConstReal(f32, (double)fill_value);
    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &gid_64, 1, "pout");
    LLVMBuildStore(bld, val, gep_out);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Where: void kernel(ptr(1) cond, ptr(1) a, ptr(1) b, ptr(1) out,
//                     i32 n, i32 cn, i32 an, i32 bn)
static LLVMModuleRef gpu_build_where_op(LLVMContextRef ctx, const char* fn_name,
                                         CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr1, ptr1, ptr1, ptr1, i32, i32, i32, i32 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 8, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef cond_p = LLVMGetParam(fn, 0);
    LLVMValueRef a      = LLVMGetParam(fn, 1);
    LLVMValueRef b      = LLVMGetParam(fn, 2);
    LLVMValueRef out    = LLVMGetParam(fn, 3);
    LLVMValueRef n      = LLVMGetParam(fn, 4);
    LLVMValueRef cn     = LLVMGetParam(fn, 5);
    LLVMValueRef an     = LLVMGetParam(fn, 6);
    LLVMValueRef bn     = LLVMGetParam(fn, 7);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);
    emit_bounds_check(bld, ctx, fn, gid, n);

    LLVMValueRef ic = LLVMBuildURem(bld, gid, cn, "ic");
    LLVMValueRef ia = LLVMBuildURem(bld, gid, an, "ia");
    LLVMValueRef ib = LLVMBuildURem(bld, gid, bn, "ib");

    LLVMValueRef ic_64 = LLVMBuildZExt(bld, ic, i64, "ic_64");
    LLVMValueRef ia_64 = LLVMBuildZExt(bld, ia, i64, "ia_64");
    LLVMValueRef ib_64 = LLVMBuildZExt(bld, ib, i64, "ib_64");
    LLVMValueRef gid_64 = LLVMBuildZExt(bld, gid, i64, "gid_64");

    LLVMValueRef gep_c = LLVMBuildGEP2(bld, f32, cond_p, &ic_64, 1, "pc");
    LLVMValueRef gep_a = LLVMBuildGEP2(bld, f32, a, &ia_64, 1, "pa");
    LLVMValueRef gep_b = LLVMBuildGEP2(bld, f32, b, &ib_64, 1, "pb");

    LLVMValueRef vc = LLVMBuildLoad2(bld, f32, gep_c, "vc");
    LLVMValueRef va = LLVMBuildLoad2(bld, f32, gep_a, "va");
    LLVMValueRef vb = LLVMBuildLoad2(bld, f32, gep_b, "vb");

    LLVMValueRef zero_f = LLVMConstReal(f32, 0.0);
    LLVMValueRef is_true = LLVMBuildFCmp(bld, LLVMRealONE, vc, zero_f, "is_true");
    LLVMValueRef result = LLVMBuildSelect(bld, is_true, va, vb, "sel");

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &gid_64, 1, "pout");
    LLVMBuildStore(bld, result, gep_out);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Reduction: void kernel(ptr(1) in, ptr(1) out, i32 n)
// Uses naive atomicAdd to out[0]. Correct but not optimal.
static LLVMModuleRef gpu_build_reduction(LLVMContextRef ctx, UOpType type,
                                          const char* fn_name, CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr1, ptr1, i32 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 3, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef in_p = LLVMGetParam(fn, 0);
    LLVMValueRef out_p = LLVMGetParam(fn, 1);
    LLVMValueRef n    = LLVMGetParam(fn, 2);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);
    emit_bounds_check(bld, ctx, fn, gid, n);

    LLVMValueRef gid_64 = LLVMBuildZExt(bld, gid, i64, "gid_64");
    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in_p, &gid_64, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "val");

    // For MEAN, divide each element by n before accumulating
    if (type == UOP_MEAN) {
        LLVMValueRef n_f = LLVMBuildUIToFP(bld, n, f32, "n_f");
        val = LLVMBuildFDiv(bld, val, n_f, "mean_val");
    }

    // atomicrmw fadd on out[0]
    LLVMValueRef zero_64 = LLVMConstInt(i64, 0, 0);
    LLVMValueRef out_gep = LLVMBuildGEP2(bld, f32, out_p, &zero_64, 1, "outp");

    if (type == UOP_SUM || type == UOP_MEAN) {
        LLVMBuildAtomicRMW(bld, LLVMAtomicRMWBinOpFAdd, out_gep, val,
                           LLVMAtomicOrderingMonotonic, 0);
    } else if (type == UOP_MAX_REDUCE) {
        LLVMBuildAtomicRMW(bld, LLVMAtomicRMWBinOpFMax, out_gep, val,
                           LLVMAtomicOrderingMonotonic, 0);
    } else if (type == UOP_MIN_REDUCE) {
        LLVMBuildAtomicRMW(bld, LLVMAtomicRMWBinOpFMin, out_gep, val,
                           LLVMAtomicOrderingMonotonic, 0);
    }

    LLVMBuildRetVoid(bld);
    LLVMDisposeBuilder(bld);
    return mod;
}

// Matmul: void kernel(ptr(1) A, ptr(1) B, ptr(1) C, i32 M, i32 N, i32 K)
// One thread per output element: C[row][col] = sum_k A[row][k] * B[k][col]
static LLVMModuleRef gpu_build_matmul(LLVMContextRef ctx, const char* fn_name,
                                       CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr1, ptr1, ptr1, i32, i32, i32 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 6, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef A = LLVMGetParam(fn, 0);
    LLVMValueRef B = LLVMGetParam(fn, 1);
    LLVMValueRef C = LLVMGetParam(fn, 2);
    LLVMValueRef M = LLVMGetParam(fn, 3);
    LLVMValueRef N = LLVMGetParam(fn, 4);
    LLVMValueRef K = LLVMGetParam(fn, 5);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);
    LLVMValueRef total = LLVMBuildMul(bld, M, N, "total");
    emit_bounds_check(bld, ctx, fn, gid, total);

    // row = gid / N, col = gid % N
    LLVMValueRef row = LLVMBuildUDiv(bld, gid, N, "row");
    LLVMValueRef col = LLVMBuildURem(bld, gid, N, "col");

    // Inner loop over K
    LLVMBasicBlockRef k_hdr = LLVMAppendBasicBlockInContext(ctx, fn, "k.hdr");
    LLVMBasicBlockRef k_body = LLVMAppendBasicBlockInContext(ctx, fn, "k.body");
    LLVMBasicBlockRef k_exit = LLVMAppendBasicBlockInContext(ctx, fn, "k.exit");

    LLVMValueRef zero_i32 = LLVMConstInt(i32, 0, 0);
    LLVMValueRef one_i32 = LLVMConstInt(i32, 1, 0);
    LLVMValueRef zero_f32 = LLVMConstReal(f32, 0.0);

    LLVMBuildBr(bld, k_hdr);

    // k header
    LLVMPositionBuilderAtEnd(bld, k_hdr);
    LLVMValueRef k_i = LLVMBuildPhi(bld, i32, "k");
    LLVMValueRef acc = LLVMBuildPhi(bld, f32, "acc");
    LLVMValueRef k_cond = LLVMBuildICmp(bld, LLVMIntULT, k_i, K, "k.cond");
    LLVMBuildCondBr(bld, k_cond, k_body, k_exit);

    // k body: acc += A[row*K+k] * B[k*N+col]
    LLVMPositionBuilderAtEnd(bld, k_body);
    LLVMValueRef rk = LLVMBuildMul(bld, row, K, "rk");
    LLVMValueRef rk_k = LLVMBuildAdd(bld, rk, k_i, "rk_k");
    LLVMValueRef rk_k_64 = LLVMBuildZExt(bld, rk_k, i64, "rk_k_64");
    LLVMValueRef a_gep = LLVMBuildGEP2(bld, f32, A, &rk_k_64, 1, "A.p");
    LLVMValueRef a_val = LLVMBuildLoad2(bld, f32, a_gep, "A.v");

    LLVMValueRef kn = LLVMBuildMul(bld, k_i, N, "kn");
    LLVMValueRef kn_c = LLVMBuildAdd(bld, kn, col, "kn_c");
    LLVMValueRef kn_c_64 = LLVMBuildZExt(bld, kn_c, i64, "kn_c_64");
    LLVMValueRef b_gep = LLVMBuildGEP2(bld, f32, B, &kn_c_64, 1, "B.p");
    LLVMValueRef b_val = LLVMBuildLoad2(bld, f32, b_gep, "B.v");

    LLVMValueRef prod = LLVMBuildFMul(bld, a_val, b_val, "prod");
    LLVMValueRef new_acc = LLVMBuildFAdd(bld, acc, prod, "new_acc");
    LLVMValueRef k_next = LLVMBuildAdd(bld, k_i, one_i32, "k.next");
    LLVMBuildBr(bld, k_hdr);

    // Wire phi
    // k_hdr's incoming: from body (after bounds check) and from k_body
    LLVMValueRef k_vals[] = { zero_i32, k_next };
    LLVMBasicBlockRef k_bbs[] = { LLVMGetPreviousBasicBlock(k_hdr), k_body };
    LLVMAddIncoming(k_i, k_vals, k_bbs, 2);
    LLVMValueRef acc_vals[] = { zero_f32, new_acc };
    LLVMAddIncoming(acc, acc_vals, k_bbs, 2);

    // k exit: store C[row*N+col] = acc
    LLVMPositionBuilderAtEnd(bld, k_exit);
    LLVMValueRef rc = LLVMBuildMul(bld, row, N, "rc");
    LLVMValueRef rc_c = LLVMBuildAdd(bld, rc, col, "rc_c");
    LLVMValueRef rc_c_64 = LLVMBuildZExt(bld, rc_c, i64, "rc_c_64");
    LLVMValueRef c_gep = LLVMBuildGEP2(bld, f32, C, &rc_c_64, 1, "C.p");
    LLVMBuildStore(bld, acc, c_gep);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Conv2D: void kernel(ptr(1) input, ptr(1) weight, ptr(1) bias, ptr(1) output,
//                      i32 batch, i32 in_c, i32 in_h, i32 in_w,
//                      i32 out_c, i32 out_h, i32 out_w,
//                      i32 kernel_h, i32 kernel_w,
//                      i32 stride_h, i32 stride_w,
//                      i32 pad_h, i32 pad_w)
// One thread per output element. Direct convolution with padding.
static LLVMModuleRef gpu_build_conv2d(LLVMContextRef ctx, const char* fn_name,
                                       CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    // 17 params: 4 pointers + 13 i32s
    LLVMTypeRef params[] = {
        ptr1, ptr1, ptr1, ptr1,     // input, weight, bias, output
        i32, i32, i32, i32,         // batch, in_c, in_h, in_w
        i32, i32, i32,              // out_c, out_h, out_w
        i32, i32,                   // kernel_h, kernel_w
        i32, i32,                   // stride_h, stride_w
        i32, i32                    // pad_h, pad_w
    };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 17, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef p_input  = LLVMGetParam(fn, 0);
    LLVMValueRef p_weight = LLVMGetParam(fn, 1);
    LLVMValueRef p_bias   = LLVMGetParam(fn, 2);
    LLVMValueRef p_output = LLVMGetParam(fn, 3);
    LLVMValueRef p_batch  = LLVMGetParam(fn, 4);
    LLVMValueRef p_in_c   = LLVMGetParam(fn, 5);
    LLVMValueRef p_in_h   = LLVMGetParam(fn, 6);
    LLVMValueRef p_in_w   = LLVMGetParam(fn, 7);
    LLVMValueRef p_out_c  = LLVMGetParam(fn, 8);
    LLVMValueRef p_out_h  = LLVMGetParam(fn, 9);
    LLVMValueRef p_out_w  = LLVMGetParam(fn, 10);
    LLVMValueRef p_kh     = LLVMGetParam(fn, 11);
    LLVMValueRef p_kw     = LLVMGetParam(fn, 12);
    LLVMValueRef p_sh     = LLVMGetParam(fn, 13);
    LLVMValueRef p_sw     = LLVMGetParam(fn, 14);
    LLVMValueRef p_ph     = LLVMGetParam(fn, 15);
    LLVMValueRef p_pw     = LLVMGetParam(fn, 16);

    (void)p_batch; // used indirectly via total element count

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);

    // total = batch * out_c * out_h * out_w (passed as grid size, recompute for bounds)
    LLVMValueRef oc_oh = LLVMBuildMul(bld, p_out_c, p_out_h, "oc_oh");
    LLVMValueRef oc_oh_ow = LLVMBuildMul(bld, oc_oh, p_out_w, "oc_oh_ow");
    LLVMValueRef total = LLVMBuildMul(bld, p_batch, oc_oh_ow, "total");
    emit_bounds_check(bld, ctx, fn, gid, total);

    LLVMValueRef zero_i32 = LLVMConstInt(i32, 0, 0);
    LLVMValueRef one_i32  = LLVMConstInt(i32, 1, 0);

    // Decompose gid -> b, oc, oh, ow
    // oh_ow = out_h * out_w
    LLVMValueRef oh_ow = LLVMBuildMul(bld, p_out_h, p_out_w, "oh_ow");
    // oc_oh_ow already computed above
    // b = gid / (out_c * out_h * out_w)
    LLVMValueRef b_val = LLVMBuildUDiv(bld, gid, oc_oh_ow, "b");
    // rem1 = gid % (out_c * out_h * out_w)
    LLVMValueRef rem1 = LLVMBuildURem(bld, gid, oc_oh_ow, "rem1");
    // oc = rem1 / (out_h * out_w)
    LLVMValueRef oc_val = LLVMBuildUDiv(bld, rem1, oh_ow, "oc");
    // rem2 = rem1 % (out_h * out_w)
    LLVMValueRef rem2 = LLVMBuildURem(bld, rem1, oh_ow, "rem2");
    // oh = rem2 / out_w
    LLVMValueRef oh_val = LLVMBuildUDiv(bld, rem2, p_out_w, "oh");
    // ow = rem2 % out_w
    LLVMValueRef ow_val = LLVMBuildURem(bld, rem2, p_out_w, "ow");

    // acc = bias[oc]
    LLVMValueRef oc_64 = LLVMBuildZExt(bld, oc_val, i64, "oc_64");
    LLVMValueRef bias_gep = LLVMBuildGEP2(bld, f32, p_bias, &oc_64, 1, "bias_p");
    LLVMValueRef bias_val = LLVMBuildLoad2(bld, f32, bias_gep, "bias_v");

    // Precompute: in_h_w = in_h * in_w, kh_kw = kernel_h * kernel_w
    // ic_kh_kw = in_c * kernel_h * kernel_w
    LLVMValueRef in_hw = LLVMBuildMul(bld, p_in_h, p_in_w, "in_hw");
    LLVMValueRef in_c_hw = LLVMBuildMul(bld, p_in_c, in_hw, "in_c_hw");
    LLVMValueRef kh_kw = LLVMBuildMul(bld, p_kh, p_kw, "kh_kw");
    LLVMValueRef ic_kh_kw = LLVMBuildMul(bld, p_in_c, kh_kw, "ic_kh_kw");

    // Outer loop: ic
    LLVMBasicBlockRef ic_hdr  = LLVMAppendBasicBlockInContext(ctx, fn, "ic.hdr");
    LLVMBasicBlockRef ic_body = LLVMAppendBasicBlockInContext(ctx, fn, "ic.body");
    LLVMBasicBlockRef ic_exit = LLVMAppendBasicBlockInContext(ctx, fn, "ic.exit");
    LLVMBasicBlockRef body_bb = LLVMGetInsertBlock(bld); // "body" from bounds check

    LLVMBuildBr(bld, ic_hdr);

    LLVMPositionBuilderAtEnd(bld, ic_hdr);
    LLVMValueRef ic_phi = LLVMBuildPhi(bld, i32, "ic");
    LLVMValueRef acc_ic = LLVMBuildPhi(bld, f32, "acc_ic");
    LLVMValueRef ic_cond = LLVMBuildICmp(bld, LLVMIntULT, ic_phi, p_in_c, "ic.cond");
    LLVMBuildCondBr(bld, ic_cond, ic_body, ic_exit);

    LLVMPositionBuilderAtEnd(bld, ic_body);

    // Middle loop: kh
    LLVMBasicBlockRef kh_hdr  = LLVMAppendBasicBlockInContext(ctx, fn, "kh.hdr");
    LLVMBasicBlockRef kh_body = LLVMAppendBasicBlockInContext(ctx, fn, "kh.body");
    LLVMBasicBlockRef kh_exit = LLVMAppendBasicBlockInContext(ctx, fn, "kh.exit");

    LLVMBuildBr(bld, kh_hdr);

    LLVMPositionBuilderAtEnd(bld, kh_hdr);
    LLVMValueRef kh_phi = LLVMBuildPhi(bld, i32, "kh");
    LLVMValueRef acc_kh = LLVMBuildPhi(bld, f32, "acc_kh");
    LLVMValueRef kh_cond = LLVMBuildICmp(bld, LLVMIntULT, kh_phi, p_kh, "kh.cond");
    LLVMBuildCondBr(bld, kh_cond, kh_body, kh_exit);

    LLVMPositionBuilderAtEnd(bld, kh_body);

    // Inner loop: kw
    LLVMBasicBlockRef kw_hdr  = LLVMAppendBasicBlockInContext(ctx, fn, "kw.hdr");
    LLVMBasicBlockRef kw_body = LLVMAppendBasicBlockInContext(ctx, fn, "kw.body");
    LLVMBasicBlockRef kw_exit = LLVMAppendBasicBlockInContext(ctx, fn, "kw.exit");
    LLVMBasicBlockRef kw_pad  = LLVMAppendBasicBlockInContext(ctx, fn, "kw.pad");

    LLVMBuildBr(bld, kw_hdr);

    LLVMPositionBuilderAtEnd(bld, kw_hdr);
    LLVMValueRef kw_phi = LLVMBuildPhi(bld, i32, "kw");
    LLVMValueRef acc_kw = LLVMBuildPhi(bld, f32, "acc_kw");
    LLVMValueRef kw_cond_v = LLVMBuildICmp(bld, LLVMIntULT, kw_phi, p_kw, "kw.cond");
    LLVMBuildCondBr(bld, kw_cond_v, kw_body, kw_exit);

    // kw body: compute ih, iw, bounds check, accumulate
    LLVMPositionBuilderAtEnd(bld, kw_body);

    // ih = oh * stride_h - pad_h + kh  (signed arithmetic)
    LLVMValueRef oh_sh = LLVMBuildMul(bld, oh_val, p_sh, "oh_sh");
    LLVMValueRef oh_sh_sub = LLVMBuildSub(bld, oh_sh, p_ph, "oh_sh_sub");
    LLVMValueRef ih_val = LLVMBuildAdd(bld, oh_sh_sub, kh_phi, "ih");

    // iw = ow * stride_w - pad_w + kw
    LLVMValueRef ow_sw = LLVMBuildMul(bld, ow_val, p_sw, "ow_sw");
    LLVMValueRef ow_sw_sub = LLVMBuildSub(bld, ow_sw, p_pw, "ow_sw_sub");
    LLVMValueRef iw_val = LLVMBuildAdd(bld, ow_sw_sub, kw_phi, "iw");

    // Bounds check: ih >= 0 && ih < in_h && iw >= 0 && iw < in_w
    // Using signed comparison (SGE, SLT) since ih/iw can be negative
    LLVMValueRef ih_ge0 = LLVMBuildICmp(bld, LLVMIntSGE, ih_val, zero_i32, "ih_ge0");
    LLVMValueRef ih_lt  = LLVMBuildICmp(bld, LLVMIntSLT, ih_val, p_in_h, "ih_lt");
    LLVMValueRef iw_ge0 = LLVMBuildICmp(bld, LLVMIntSGE, iw_val, zero_i32, "iw_ge0");
    LLVMValueRef iw_lt  = LLVMBuildICmp(bld, LLVMIntSLT, iw_val, p_in_w, "iw_lt");
    LLVMValueRef h_ok = LLVMBuildAnd(bld, ih_ge0, ih_lt, "h_ok");
    LLVMValueRef w_ok = LLVMBuildAnd(bld, iw_ge0, iw_lt, "w_ok");
    LLVMValueRef in_bounds = LLVMBuildAnd(bld, h_ok, w_ok, "in_bounds");

    LLVMBasicBlockRef kw_load = LLVMAppendBasicBlockInContext(ctx, fn, "kw.load");
    LLVMBuildCondBr(bld, in_bounds, kw_load, kw_pad);

    // kw_load: load input and weight, accumulate
    LLVMPositionBuilderAtEnd(bld, kw_load);

    // input[b*in_c*in_h*in_w + ic*in_h*in_w + ih*in_w + iw]
    LLVMValueRef b_off = LLVMBuildMul(bld, b_val, in_c_hw, "b_off");
    LLVMValueRef ic_off = LLVMBuildMul(bld, ic_phi, in_hw, "ic_off");
    LLVMValueRef ih_off = LLVMBuildMul(bld, ih_val, p_in_w, "ih_off");
    LLVMValueRef in_idx = LLVMBuildAdd(bld, b_off, ic_off, "in1");
    in_idx = LLVMBuildAdd(bld, in_idx, ih_off, "in2");
    in_idx = LLVMBuildAdd(bld, in_idx, iw_val, "in_idx");
    LLVMValueRef in_idx_64 = LLVMBuildZExt(bld, in_idx, i64, "in_idx_64");
    LLVMValueRef in_gep = LLVMBuildGEP2(bld, f32, p_input, &in_idx_64, 1, "in_p");
    LLVMValueRef in_val = LLVMBuildLoad2(bld, f32, in_gep, "in_v");

    // weight[oc*in_c*kernel_h*kernel_w + ic*kernel_h*kernel_w + kh*kernel_w + kw]
    LLVMValueRef oc_w_off = LLVMBuildMul(bld, oc_val, ic_kh_kw, "oc_w_off");
    LLVMValueRef ic_w_off = LLVMBuildMul(bld, ic_phi, kh_kw, "ic_w_off");
    LLVMValueRef kh_w_off = LLVMBuildMul(bld, kh_phi, p_kw, "kh_w_off");
    LLVMValueRef w_idx = LLVMBuildAdd(bld, oc_w_off, ic_w_off, "w1");
    w_idx = LLVMBuildAdd(bld, w_idx, kh_w_off, "w2");
    w_idx = LLVMBuildAdd(bld, w_idx, kw_phi, "w_idx");
    LLVMValueRef w_idx_64 = LLVMBuildZExt(bld, w_idx, i64, "w_idx_64");
    LLVMValueRef w_gep = LLVMBuildGEP2(bld, f32, p_weight, &w_idx_64, 1, "w_p");
    LLVMValueRef w_val = LLVMBuildLoad2(bld, f32, w_gep, "w_v");

    LLVMValueRef prod = LLVMBuildFMul(bld, in_val, w_val, "prod");
    LLVMValueRef new_acc_load = LLVMBuildFAdd(bld, acc_kw, prod, "new_acc_load");
    LLVMBuildBr(bld, kw_pad);

    // kw_pad: merge point for in-bounds / out-of-bounds
    LLVMPositionBuilderAtEnd(bld, kw_pad);
    LLVMValueRef acc_merge = LLVMBuildPhi(bld, f32, "acc_merge");
    LLVMValueRef merge_vals[] = { acc_kw, new_acc_load };
    LLVMBasicBlockRef merge_bbs[] = { kw_body, kw_load };
    LLVMAddIncoming(acc_merge, merge_vals, merge_bbs, 2);

    LLVMValueRef kw_next = LLVMBuildAdd(bld, kw_phi, one_i32, "kw.next");
    LLVMBuildBr(bld, kw_hdr);

    // Wire kw phi nodes
    LLVMValueRef kw_phi_vals[] = { zero_i32, kw_next };
    LLVMBasicBlockRef kw_phi_bbs[] = { kh_body, kw_pad };
    LLVMAddIncoming(kw_phi, kw_phi_vals, kw_phi_bbs, 2);
    LLVMValueRef acc_kw_vals[] = { acc_kh, acc_merge };
    LLVMAddIncoming(acc_kw, acc_kw_vals, kw_phi_bbs, 2);

    // kw exit: end of inner loop
    LLVMPositionBuilderAtEnd(bld, kw_exit);
    LLVMValueRef kh_next = LLVMBuildAdd(bld, kh_phi, one_i32, "kh.next");
    LLVMBuildBr(bld, kh_hdr);

    // Wire kh phi nodes
    LLVMValueRef kh_phi_vals[] = { zero_i32, kh_next };
    LLVMBasicBlockRef kh_phi_bbs[] = { ic_body, kw_exit };
    LLVMAddIncoming(kh_phi, kh_phi_vals, kh_phi_bbs, 2);
    LLVMValueRef acc_kh_vals[] = { acc_ic, acc_kw };
    LLVMAddIncoming(acc_kh, acc_kh_vals, kh_phi_bbs, 2);

    // kh exit: end of middle loop
    LLVMPositionBuilderAtEnd(bld, kh_exit);
    LLVMValueRef ic_next = LLVMBuildAdd(bld, ic_phi, one_i32, "ic.next");
    LLVMBuildBr(bld, ic_hdr);

    // Wire ic phi nodes
    LLVMValueRef ic_phi_vals[] = { zero_i32, ic_next };
    LLVMBasicBlockRef ic_phi_bbs[] = { body_bb, kh_exit };
    LLVMAddIncoming(ic_phi, ic_phi_vals, ic_phi_bbs, 2);
    LLVMValueRef acc_ic_vals[] = { bias_val, acc_kh };
    LLVMAddIncoming(acc_ic, acc_ic_vals, ic_phi_bbs, 2);

    // ic exit: store result to output[gid]
    LLVMPositionBuilderAtEnd(bld, ic_exit);
    LLVMValueRef gid_64 = LLVMBuildZExt(bld, gid, i64, "gid_64");
    LLVMValueRef out_gep = LLVMBuildGEP2(bld, f32, p_output, &gid_64, 1, "out_p");
    LLVMBuildStore(bld, acc_ic, out_gep);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Gather: void kernel(ptr(1) input, ptr(1) indices, ptr(1) out, i32 N, i32 C)
static LLVMModuleRef gpu_build_gather_op(LLVMContextRef ctx, const char* fn_name,
                                          CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr1, ptr1, ptr1, i32, i32 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 5, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef input   = LLVMGetParam(fn, 0);
    LLVMValueRef indices = LLVMGetParam(fn, 1);
    LLVMValueRef out     = LLVMGetParam(fn, 2);
    LLVMValueRef N       = LLVMGetParam(fn, 3);
    LLVMValueRef C_val   = LLVMGetParam(fn, 4);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);
    emit_bounds_check(bld, ctx, fn, gid, N);

    LLVMValueRef gid_64 = LLVMBuildZExt(bld, gid, i64, "gid_64");

    // idx = (int32_t)indices[gid]
    LLVMValueRef gep_idx = LLVMBuildGEP2(bld, f32, indices, &gid_64, 1, "pidx");
    LLVMValueRef idx_f = LLVMBuildLoad2(bld, f32, gep_idx, "idx_f");
    LLVMValueRef idx = LLVMBuildFPToSI(bld, idx_f, i32, "idx");

    // offset = gid * C + idx
    LLVMValueRef row = LLVMBuildMul(bld, gid, C_val, "row");
    LLVMValueRef offset = LLVMBuildAdd(bld, row, idx, "offset");
    LLVMValueRef offset_64 = LLVMBuildZExt(bld, offset, i64, "offset_64");

    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, input, &offset_64, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "val");

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &gid_64, 1, "pout");
    LLVMBuildStore(bld, val, gep_out);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Permute 2D: void kernel(ptr(1) in, ptr(1) out, i32 rows, i32 cols)
// out[col*rows+row] = in[gid] where row=gid/cols, col=gid%cols
static LLVMModuleRef gpu_build_permute_2d(LLVMContextRef ctx, const char* fn_name,
                                           CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr1, ptr1, i32, i32 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 4, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef in_p  = LLVMGetParam(fn, 0);
    LLVMValueRef out   = LLVMGetParam(fn, 1);
    LLVMValueRef rows  = LLVMGetParam(fn, 2);
    LLVMValueRef cols  = LLVMGetParam(fn, 3);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);
    LLVMValueRef total = LLVMBuildMul(bld, rows, cols, "total");
    emit_bounds_check(bld, ctx, fn, gid, total);

    LLVMValueRef row = LLVMBuildUDiv(bld, gid, cols, "row");
    LLVMValueRef col = LLVMBuildURem(bld, gid, cols, "col");

    // out[col*rows+row] = in[row*cols+col] = in[gid]
    LLVMValueRef gid_64 = LLVMBuildZExt(bld, gid, i64, "gid_64");
    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in_p, &gid_64, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "v");

    LLVMValueRef out_off = LLVMBuildAdd(bld, LLVMBuildMul(bld, col, rows, "cr"), row, "out_off");
    LLVMValueRef out_off_64 = LLVMBuildZExt(bld, out_off, i64, "out_off_64");
    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &out_off_64, 1, "pout");
    LLVMBuildStore(bld, val, gep_out);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Expand: void kernel(ptr(1) in, ptr(1) out, i32 out_n, i32 in_n)
static LLVMModuleRef gpu_build_expand_op(LLVMContextRef ctx, const char* fn_name,
                                          CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr1, ptr1, i32, i32 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 4, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef in_p  = LLVMGetParam(fn, 0);
    LLVMValueRef out   = LLVMGetParam(fn, 1);
    LLVMValueRef out_n = LLVMGetParam(fn, 2);
    LLVMValueRef in_n  = LLVMGetParam(fn, 3);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);
    emit_bounds_check(bld, ctx, fn, gid, out_n);

    LLVMValueRef idx = LLVMBuildURem(bld, gid, in_n, "idx");
    LLVMValueRef idx_64 = LLVMBuildZExt(bld, idx, i64, "idx_64");
    LLVMValueRef gid_64 = LLVMBuildZExt(bld, gid, i64, "gid_64");

    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in_p, &idx_64, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "v");

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &gid_64, 1, "pout");
    LLVMBuildStore(bld, val, gep_out);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Reshape (memcpy): void kernel(ptr(1) in, ptr(1) out, i32 n)
static LLVMModuleRef gpu_build_reshape_op(LLVMContextRef ctx, const char* fn_name,
                                           CMLGPUCodegen* cg) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr1   = LLVMPointerTypeInContext(ctx, 1);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr1, ptr1, i32 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 3, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    configure_gpu_module(mod, fn, cg);

    LLVMValueRef in_p = LLVMGetParam(fn, 0);
    LLVMValueRef out  = LLVMGetParam(fn, 1);
    LLVMValueRef n    = LLVMGetParam(fn, 2);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef gid = emit_global_thread_id(bld, mod, ctx, cg->target);
    emit_bounds_check(bld, ctx, fn, gid, n);

    LLVMValueRef gid_64 = LLVMBuildZExt(bld, gid, i64, "gid_64");
    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in_p, &gid_64, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "v");
    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &gid_64, 1, "pout");
    LLVMBuildStore(bld, val, gep_out);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

static char* emit_gpu_code(CMLGPUCodegen* cg, LLVMModuleRef mod, size_t* out_size) {
    // Verify
    char* err = NULL;
    if (LLVMVerifyModule(mod, LLVMReturnStatusAction, &err) != 0) {
        LOG_ERROR("GPU codegen: Module verification failed: %s", err ? err : "unknown");
        LLVMDisposeMessage(err);
        return NULL;
    }
    LLVMDisposeMessage(err);

    // Create target machine
    LLVMTargetRef target = NULL;
    err = NULL;
    if (LLVMGetTargetFromTriple(cg->target_triple, &target, &err) != 0) {
        LOG_ERROR("GPU codegen: Failed to get target for '%s': %s",
                  cg->target_triple, err ? err : "unknown");
        LLVMDisposeMessage(err);
        return NULL;
    }

    LLVMTargetMachineRef tm = LLVMCreateTargetMachine(
        target, cg->target_triple, cg->target_cpu, "",
        LLVMCodeGenLevelDefault, LLVMRelocDefault, LLVMCodeModelDefault);

    if (!tm) {
        LOG_ERROR("GPU codegen: Failed to create target machine");
        return NULL;
    }

    // Emit code
    LLVMCodeGenFileType file_type;
    if (cg->target == GPU_TARGET_CUDA) {
        file_type = LLVMAssemblyFile;  // PTX is assembly text
    } else {
        file_type = LLVMObjectFile;    // HSACO is ELF object
    }

    LLVMMemoryBufferRef buf = NULL;
    err = NULL;
    if (LLVMTargetMachineEmitToMemoryBuffer(tm, mod, file_type, &err, &buf) != 0) {
        LOG_ERROR("GPU codegen: Emission failed: %s", err ? err : "unknown");
        LLVMDisposeMessage(err);
        LLVMDisposeTargetMachine(tm);
        return NULL;
    }

    size_t size = LLVMGetBufferSize(buf);
    const char* data = LLVMGetBufferStart(buf);

    char* result = malloc(size + 1);
    memcpy(result, data, size);
    result[size] = '\0';  // null-terminate for PTX text

    if (out_size) *out_size = size;

    LLVMDisposeMemoryBuffer(buf);
    LLVMDisposeTargetMachine(tm);

    LOG_DEBUG("GPU codegen: Emitted %zu bytes of %s code",
              size, cg->target == GPU_TARGET_CUDA ? "PTX" : "HSACO");

    return result;
}

static bool is_binary_op(UOpType type) {
    return type == UOP_ADD || type == UOP_SUB || type == UOP_MUL ||
           type == UOP_DIV || type == UOP_MAX || type == UOP_CMPLT ||
           type == UOP_POW || type == UOP_IDIV || type == UOP_MOD;
}

static bool is_unary_op(UOpType type) {
    return type == UOP_NEG || type == UOP_EXP || type == UOP_LOG ||
           type == UOP_SQRT || type == UOP_ABS || type == UOP_SIN ||
           type == UOP_COS || type == UOP_TAN || type == UOP_RECIP ||
           type == UOP_SIGMOID || type == UOP_TANH ||
           type == UOP_ELU || type == UOP_SELU || type == UOP_MISH ||
           type == UOP_SILU || type == UOP_HARDSWISH;
}

static bool is_reduction(UOpType type) {
    return type == UOP_SUM || type == UOP_MEAN || type == UOP_MAX_REDUCE ||
           type == UOP_MIN_REDUCE;
}

// Upload tensor data to GPU, returning device pointer
static void* gpu_upload(CMLGPUCodegen* cg, float* host_data, size_t numel) {
    size_t size = numel * sizeof(float);
    if (cg->target == GPU_TARGET_CUDA) {
        CUdeviceptr dptr = cml_cuda_malloc(cg->cuda, size);
        if (!dptr) return NULL;
        if (cml_cuda_memcpy_h2d(cg->cuda, dptr, host_data, size) != 0) {
            cml_cuda_free(cg->cuda, dptr);
            return NULL;
        }
        return (void*)(uintptr_t)dptr;
    } else {
        hipDeviceptr_t dptr = cml_rocm_malloc(cg->rocm, size);
        if (!dptr) return NULL;
        if (cml_rocm_memcpy_h2d(cg->rocm, dptr, host_data, size) != 0) {
            cml_rocm_free(cg->rocm, dptr);
            return NULL;
        }
        return dptr;
    }
}

// Allocate zero-initialized GPU memory
static void* gpu_alloc_zero(CMLGPUCodegen* cg, size_t numel) {
    size_t size = numel * sizeof(float);
    float* zeros = calloc(numel, sizeof(float));
    if (!zeros) return NULL;

    void* dptr = NULL;
    if (cg->target == GPU_TARGET_CUDA) {
        CUdeviceptr cptr = cml_cuda_malloc(cg->cuda, size);
        if (cptr) {
            cml_cuda_memcpy_h2d(cg->cuda, cptr, zeros, size);
            dptr = (void*)(uintptr_t)cptr;
        }
    } else {
        hipDeviceptr_t hptr = cml_rocm_malloc(cg->rocm, size);
        if (hptr) {
            cml_rocm_memcpy_h2d(cg->rocm, hptr, zeros, size);
            dptr = hptr;
        }
    }
    free(zeros);
    return dptr;
}

// Allocate GPU memory filled with a constant value
static void* gpu_alloc_filled(CMLGPUCodegen* cg, size_t numel, float fill_val) {
    size_t size = numel * sizeof(float);
    float* buf = malloc(size);
    if (!buf) return NULL;
    for (size_t i = 0; i < numel; i++) buf[i] = fill_val;

    void* dptr = NULL;
    if (cg->target == GPU_TARGET_CUDA) {
        CUdeviceptr cptr = cml_cuda_malloc(cg->cuda, size);
        if (cptr) {
            cml_cuda_memcpy_h2d(cg->cuda, cptr, buf, size);
            dptr = (void*)(uintptr_t)cptr;
        }
    } else {
        hipDeviceptr_t hptr = cml_rocm_malloc(cg->rocm, size);
        if (hptr) {
            cml_rocm_memcpy_h2d(cg->rocm, hptr, buf, size);
            dptr = hptr;
        }
    }
    free(buf);
    return dptr;
}

// Download GPU data to host
static int gpu_download(CMLGPUCodegen* cg, void* dptr, float* host_data, size_t numel) {
    size_t size = numel * sizeof(float);
    if (cg->target == GPU_TARGET_CUDA) {
        return cml_cuda_memcpy_d2h(cg->cuda, host_data, (CUdeviceptr)(uintptr_t)dptr, size);
    } else {
        return cml_rocm_memcpy_d2h(cg->rocm, host_data, dptr, size);
    }
}

static void gpu_free(CMLGPUCodegen* cg, void* dptr) {
    if (!dptr) return;
    if (cg->target == GPU_TARGET_CUDA) {
        cml_cuda_free(cg->cuda, (CUdeviceptr)(uintptr_t)dptr);
    } else {
        cml_rocm_free(cg->rocm, dptr);
    }
}

static int gpu_sync(CMLGPUCodegen* cg) {
    if (cg->target == GPU_TARGET_CUDA) {
        return cml_cuda_synchronize(cg->cuda);
    } else {
        return cml_rocm_synchronize(cg->rocm);
    }
}

static int gpu_compile_and_launch(CMLGPUCodegen* cg, LLVMModuleRef mod,
                                   const char* fn_name, int numel,
                                   void** args, int num_args) {
    // Emit PTX/HSACO
    size_t code_size = 0;
    char* code = emit_gpu_code(cg, mod, &code_size);
    if (!code) {
        LLVMDisposeModule(mod);
        return -1;
    }

    // Module was consumed by emit_gpu_code verification only — dispose it now
    LLVMDisposeModule(mod);

    int block_size = cg->default_block_size;
    int grid_size = (numel + block_size - 1) / block_size;

    int result = -1;

    if (cg->target == GPU_TARGET_CUDA) {
        CMLCUDAKernel* kernel = cml_cuda_compile_ptx(cg->cuda, code, fn_name);
        if (kernel) {
            cml_cuda_kernel_set_launch_config(kernel, grid_size, 1, 1,
                                               block_size, 1, 1);
            result = cml_cuda_launch_kernel(cg->cuda, kernel, args, num_args);
            cml_cuda_kernel_free(cg->cuda, kernel);
        }
    } else {
        CMLROCmKernel* kernel = cml_rocm_compile_hsaco(cg->rocm, code, fn_name);
        if (kernel) {
            kernel->grid_dim[0] = grid_size;
            kernel->grid_dim[1] = 1;
            kernel->grid_dim[2] = 1;
            kernel->block_dim[0] = block_size;
            kernel->block_dim[1] = 1;
            kernel->block_dim[2] = 1;
            result = cml_rocm_launch_kernel(cg->rocm, kernel, args, num_args);
            cml_rocm_kernel_free(cg->rocm, kernel);
        }
    }

    free(code);
    return result;
}

static int gpu_execute_node(CMLGPUCodegen* cg, struct IRNode* node) {
    if (!node || !node->output) return -1;

    Tensor* out = node->output;
    UOpType type = node->type;

    // Unsupported ops fall back to CPU
    if (type == UOP_STRIDE || type == UOP_SLICE) {
        return cpu_execute_node(node);
    }

    // Allocate output host memory if needed
    if (!out->data && out->numel > 0) {
        size_t size = out->numel * sizeof(float);
        out->data = cml_buffer_cache_alloc(size);
        if (!out->data) {
            LOG_ERROR("GPU codegen: Failed to allocate output tensor");
            return -1;
        }
        out->owns_data = true;
    }

    // Build unique kernel name
    char fn_name[64];
    snprintf(fn_name, sizeof(fn_name), "gpu_k%d", cg->kernel_count++);

    LLVMContextRef ctx = LLVMContextCreate();
    LLVMModuleRef mod = NULL;

    // Build the GPU kernel module
    if (is_binary_op(type)) {
        mod = gpu_build_binary_op(ctx, type, fn_name, cg);
    } else if (is_unary_op(type)) {
        mod = gpu_build_unary_op(ctx, type, fn_name, cg);
    } else if (is_reduction(type)) {
        mod = gpu_build_reduction(ctx, type, fn_name, cg);
    } else if (type == UOP_MATMUL) {
        /* WMMA Tensor Core path: when available and beneficial, generate
         * a WMMA kernel instead of the scalar matmul. The WMMA kernel is
         * compiled via NVRTC (separate path), so we fall through to the
         * LLVM NVPTX matmul as the default. */
        mod = gpu_build_matmul(ctx, fn_name, cg);
    } else if (type == UOP_CONV2D) {
        mod = gpu_build_conv2d(ctx, fn_name, cg);
    } else if (type == UOP_FILL) {
        FillParams* p = (FillParams*)node->params;
        float fill_val = p ? p->value : 0.0f;
        mod = gpu_build_fill_op(ctx, fn_name, fill_val, cg);
    } else if (type == UOP_WHERE) {
        mod = gpu_build_where_op(ctx, fn_name, cg);
    } else if (type == UOP_GATHER) {
        mod = gpu_build_gather_op(ctx, fn_name, cg);
    } else if (type == UOP_PERMUTE) {
        if (node->num_inputs >= 1 && node->inputs[0] && node->inputs[0]->ndim == 2) {
            mod = gpu_build_permute_2d(ctx, fn_name, cg);
        } else {
            LLVMContextDispose(ctx);
            return cpu_execute_node(node);
        }
    } else if (type == UOP_RESHAPE) {
        mod = gpu_build_reshape_op(ctx, fn_name, cg);
    } else if (type == UOP_EXPAND) {
        mod = gpu_build_expand_op(ctx, fn_name, cg);
    } else {
        LOG_DEBUG("GPU codegen: Unsupported op %d, falling back to CPU", type);
        LLVMContextDispose(ctx);
        return cpu_execute_node(node);
    }

    if (!mod) {
        LLVMContextDispose(ctx);
        return cpu_execute_node(node);
    }

    // Execute on GPU: upload inputs, launch kernel, download output
    int result = -1;

    if (is_binary_op(type)) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data)
            goto fallback;

        void* d_in0 = gpu_upload(cg, (float*)node->inputs[0]->data, node->inputs[0]->numel);
        void* d_in1 = gpu_upload(cg, (float*)node->inputs[1]->data, node->inputs[1]->numel);
        void* d_out = gpu_alloc_zero(cg, out->numel);
        if (!d_in0 || !d_in1 || !d_out) goto binary_cleanup;

        int32_t n = (int32_t)out->numel;
        int32_t n0 = (int32_t)node->inputs[0]->numel;
        int32_t n1 = (int32_t)node->inputs[1]->numel;
        void* args[] = { &d_in0, &d_in1, &d_out, &n, &n0, &n1 };
        if (gpu_compile_and_launch(cg, mod, fn_name, n, args, 6) == 0) {
            mod = NULL; // consumed
            gpu_sync(cg);
            gpu_download(cg, d_out, (float*)out->data, out->numel);
            result = 0;
        }
    binary_cleanup:
        gpu_free(cg, d_in0); gpu_free(cg, d_in1); gpu_free(cg, d_out);
    }
    else if (is_unary_op(type)) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            goto fallback;

        void* d_in = gpu_upload(cg, (float*)node->inputs[0]->data, node->inputs[0]->numel);
        void* d_out = gpu_alloc_zero(cg, out->numel);
        if (!d_in || !d_out) goto unary_cleanup;

        int32_t n = (int32_t)out->numel;
        int32_t in_n = (int32_t)node->inputs[0]->numel;
        void* args[] = { &d_in, &d_out, &n, &in_n };
        if (gpu_compile_and_launch(cg, mod, fn_name, n, args, 4) == 0) {
            mod = NULL;
            gpu_sync(cg);
            gpu_download(cg, d_out, (float*)out->data, out->numel);
            result = 0;
        }
    unary_cleanup:
        gpu_free(cg, d_in); gpu_free(cg, d_out);
    }
    else if (is_reduction(type)) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            goto fallback;

        void* d_in = gpu_upload(cg, (float*)node->inputs[0]->data, node->inputs[0]->numel);
        void* d_out;
        if (type == UOP_MIN_REDUCE)
            d_out = gpu_alloc_filled(cg, out->numel, FLT_MAX);
        else if (type == UOP_MAX_REDUCE)
            d_out = gpu_alloc_filled(cg, out->numel, -FLT_MAX);
        else
            d_out = gpu_alloc_zero(cg, out->numel);  // zero-init for atomicAdd
        if (!d_in || !d_out) goto reduce_cleanup;

        int32_t n = (int32_t)node->inputs[0]->numel;
        void* args[] = { &d_in, &d_out, &n };
        if (gpu_compile_and_launch(cg, mod, fn_name, n, args, 3) == 0) {
            mod = NULL;
            gpu_sync(cg);
            gpu_download(cg, d_out, (float*)out->data, out->numel);
            result = 0;
        }
    reduce_cleanup:
        gpu_free(cg, d_in); gpu_free(cg, d_out);
    }
    else if (type == UOP_MATMUL) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data)
            goto fallback;
        Tensor* a = node->inputs[0];
        Tensor* b = node->inputs[1];
        if (a->ndim < 2 || b->ndim < 2) goto fallback;

        void* d_A = gpu_upload(cg, (float*)a->data, a->numel);
        void* d_B = gpu_upload(cg, (float*)b->data, b->numel);
        void* d_C = gpu_alloc_zero(cg, out->numel);
        if (!d_A || !d_B || !d_C) goto matmul_cleanup;

        int32_t M = (int32_t)a->shape[a->ndim - 2];
        int32_t K = (int32_t)a->shape[a->ndim - 1];
        int32_t N = (int32_t)b->shape[b->ndim - 1];
        void* args[] = { &d_A, &d_B, &d_C, &M, &N, &K };
        if (gpu_compile_and_launch(cg, mod, fn_name, M * N, args, 6) == 0) {
            mod = NULL;
            gpu_sync(cg);
            gpu_download(cg, d_C, (float*)out->data, out->numel);
            result = 0;
        }
    matmul_cleanup:
        gpu_free(cg, d_A); gpu_free(cg, d_B); gpu_free(cg, d_C);
    }
    else if (type == UOP_CONV2D) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data)
            goto fallback;
        Conv2DParams* cp = (Conv2DParams*)node->params;
        if (!cp) goto fallback;

        Tensor* input = node->inputs[0];   // [batch, in_c, in_h, in_w]
        Tensor* weight = node->inputs[1];  // [out_c, in_c, kh, kw]
        Tensor* bias_t = (node->num_inputs >= 3) ? node->inputs[2] : NULL;

        if (input->ndim < 4 || weight->ndim < 4) goto fallback;

        void* d_input = gpu_upload(cg, (float*)input->data, input->numel);
        void* d_weight = gpu_upload(cg, (float*)weight->data, weight->numel);
        void* d_bias = NULL;
        if (bias_t && bias_t->data) {
            d_bias = gpu_upload(cg, (float*)bias_t->data, bias_t->numel);
        } else {
            d_bias = gpu_alloc_zero(cg, weight->shape[0]); // out_c zeros
        }
        void* d_out = gpu_alloc_zero(cg, out->numel);
        if (!d_input || !d_weight || !d_bias || !d_out) goto conv2d_cleanup;

        int32_t batch     = (int32_t)input->shape[0];
        int32_t in_c      = (int32_t)input->shape[1];
        int32_t in_h      = (int32_t)input->shape[2];
        int32_t in_w      = (int32_t)input->shape[3];
        int32_t out_c_val = (int32_t)weight->shape[0];
        int32_t out_h_val = (int32_t)out->shape[2];
        int32_t out_w_val = (int32_t)out->shape[3];
        int32_t kh        = (int32_t)weight->shape[2];
        int32_t kw        = (int32_t)weight->shape[3];
        int32_t sh        = cp->stride ? (int32_t)cp->stride[0] : 1;
        int32_t sw        = cp->stride ? (int32_t)(cp->stride[1] ? cp->stride[1] : cp->stride[0]) : 1;
        int32_t ph        = cp->padding ? (int32_t)cp->padding[0] : 0;
        int32_t pw        = cp->padding ? (int32_t)(cp->padding[1] ? cp->padding[1] : cp->padding[0]) : 0;

        void* args[] = { &d_input, &d_weight, &d_bias, &d_out,
                         &batch, &in_c, &in_h, &in_w,
                         &out_c_val, &out_h_val, &out_w_val,
                         &kh, &kw, &sh, &sw, &ph, &pw };

        int total_elems = batch * out_c_val * out_h_val * out_w_val;
        if (gpu_compile_and_launch(cg, mod, fn_name, total_elems, args, 17) == 0) {
            mod = NULL;
            gpu_sync(cg);
            gpu_download(cg, d_out, (float*)out->data, out->numel);
            result = 0;
        }
    conv2d_cleanup:
        gpu_free(cg, d_input); gpu_free(cg, d_weight);
        gpu_free(cg, d_bias); gpu_free(cg, d_out);
    }
    else if (type == UOP_FILL) {
        void* d_out = gpu_alloc_zero(cg, out->numel);
        if (!d_out) goto fill_cleanup;

        int32_t n = (int32_t)out->numel;
        void* args[] = { &d_out, &n };
        if (gpu_compile_and_launch(cg, mod, fn_name, n, args, 2) == 0) {
            mod = NULL;
            gpu_sync(cg);
            gpu_download(cg, d_out, (float*)out->data, out->numel);
            result = 0;
        }
    fill_cleanup:
        gpu_free(cg, d_out);
    }
    else if (type == UOP_WHERE) {
        if (node->num_inputs < 3 || !node->inputs[0]->data ||
            !node->inputs[1]->data || !node->inputs[2]->data)
            goto fallback;

        void* d_c = gpu_upload(cg, (float*)node->inputs[0]->data, node->inputs[0]->numel);
        void* d_a = gpu_upload(cg, (float*)node->inputs[1]->data, node->inputs[1]->numel);
        void* d_b = gpu_upload(cg, (float*)node->inputs[2]->data, node->inputs[2]->numel);
        void* d_out = gpu_alloc_zero(cg, out->numel);
        if (!d_c || !d_a || !d_b || !d_out) goto where_cleanup;

        int32_t n = (int32_t)out->numel;
        int32_t cn = (int32_t)node->inputs[0]->numel;
        int32_t an = (int32_t)node->inputs[1]->numel;
        int32_t bn = (int32_t)node->inputs[2]->numel;
        void* args[] = { &d_c, &d_a, &d_b, &d_out, &n, &cn, &an, &bn };
        if (gpu_compile_and_launch(cg, mod, fn_name, n, args, 8) == 0) {
            mod = NULL;
            gpu_sync(cg);
            gpu_download(cg, d_out, (float*)out->data, out->numel);
            result = 0;
        }
    where_cleanup:
        gpu_free(cg, d_c); gpu_free(cg, d_a); gpu_free(cg, d_b); gpu_free(cg, d_out);
    }
    else if (type == UOP_GATHER) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data)
            goto fallback;
        Tensor* input = node->inputs[0];
        if (input->ndim < 2) goto fallback;

        void* d_input = gpu_upload(cg, (float*)input->data, input->numel);
        void* d_indices = gpu_upload(cg, (float*)node->inputs[1]->data, node->inputs[1]->numel);
        void* d_out = gpu_alloc_zero(cg, out->numel);
        if (!d_input || !d_indices || !d_out) goto gather_cleanup;

        int32_t N = (int32_t)out->numel;
        int32_t C = (int32_t)input->shape[input->ndim - 1];
        void* args[] = { &d_input, &d_indices, &d_out, &N, &C };
        if (gpu_compile_and_launch(cg, mod, fn_name, N, args, 5) == 0) {
            mod = NULL;
            gpu_sync(cg);
            gpu_download(cg, d_out, (float*)out->data, out->numel);
            result = 0;
        }
    gather_cleanup:
        gpu_free(cg, d_input); gpu_free(cg, d_indices); gpu_free(cg, d_out);
    }
    else if (type == UOP_PERMUTE) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            goto fallback;
        Tensor* inp = node->inputs[0];

        void* d_in = gpu_upload(cg, (float*)inp->data, inp->numel);
        void* d_out = gpu_alloc_zero(cg, out->numel);
        if (!d_in || !d_out) goto permute_cleanup;

        int32_t rows = (int32_t)inp->shape[0];
        int32_t cols = (int32_t)inp->shape[1];
        void* args[] = { &d_in, &d_out, &rows, &cols };
        if (gpu_compile_and_launch(cg, mod, fn_name, rows * cols, args, 4) == 0) {
            mod = NULL;
            gpu_sync(cg);
            gpu_download(cg, d_out, (float*)out->data, out->numel);
            result = 0;
        }
    permute_cleanup:
        gpu_free(cg, d_in); gpu_free(cg, d_out);
    }
    else if (type == UOP_RESHAPE || type == UOP_EXPAND) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            goto fallback;

        void* d_in = gpu_upload(cg, (float*)node->inputs[0]->data, node->inputs[0]->numel);
        void* d_out = gpu_alloc_zero(cg, out->numel);
        if (!d_in || !d_out) goto copy_cleanup;

        if (type == UOP_RESHAPE) {
            int32_t n = (int32_t)out->numel;
            void* args[] = { &d_in, &d_out, &n };
            if (gpu_compile_and_launch(cg, mod, fn_name, n, args, 3) == 0) {
                mod = NULL;
                gpu_sync(cg);
                gpu_download(cg, d_out, (float*)out->data, out->numel);
                result = 0;
            }
        } else { // UOP_EXPAND
            int32_t out_n = (int32_t)out->numel;
            int32_t in_n = (int32_t)node->inputs[0]->numel;
            void* args[] = { &d_in, &d_out, &out_n, &in_n };
            if (gpu_compile_and_launch(cg, mod, fn_name, out_n, args, 4) == 0) {
                mod = NULL;
                gpu_sync(cg);
                gpu_download(cg, d_out, (float*)out->data, out->numel);
                result = 0;
            }
        }
    copy_cleanup:
        gpu_free(cg, d_in); gpu_free(cg, d_out);
    }

    if (mod) {
        // Module wasn't consumed (error path) — dispose it
        LLVMDisposeModule(mod);
    }

    if (result == 0) {
        node->is_executed = true;
        out->is_executed = true;
        return 0;
    }

fallback:
    if (mod) LLVMDisposeModule(mod);
    LOG_DEBUG("GPU codegen: Falling back to CPU for op %d", type);
    return cpu_execute_node(node);
}

int cml_gpu_execute(CMLGPUCodegen* cg, CMLGraph_t ir) {
    if (!cg || !ir || !cg->initialized) return -1;

    LOG_DEBUG("GPU codegen: Executing IR graph");

    struct IRNode* node = ir->head;
    while (node) {
        if (!node->is_executed) {
            if (gpu_execute_node(cg, node) != 0) {
                LOG_WARNING("GPU codegen: Node execution failed, using CPU fallback");
                cpu_execute_node(node);
                node->is_executed = true;
                if (node->output) node->output->is_executed = true;
            }
        }
        node = node->next;
    }

    ir->is_executed = true;
    return 0;
}

int cml_gpu_execute_up_to(CMLGPUCodegen* cg, CMLGraph_t ir, struct IRNode* target) {
    if (!cg || !ir || !target || !cg->initialized) return -1;

    struct IRNode* node = ir->head;
    while (node) {
        if (!node->is_executed) {
            if (gpu_execute_node(cg, node) != 0) {
                LOG_WARNING("GPU codegen: Node execution failed, using CPU fallback");
                cpu_execute_node(node);
                node->is_executed = true;
                if (node->output) node->output->is_executed = true;
            }
        }
        if (node == target) break;
        node = node->next;
    }

    return 0;
}

#endif // CML_HAS_LLVM_BACKEND
