/**
 * @file llvm_backend.c
 * @brief Direct LLVM IR backend — builds LLVM IR from UOps and JIT-executes
 *
 * Uses the LLVM C API directly for IR building, PassBuilder for
 * optimization, and ORC LLJIT for JIT compilation.
 *
 * ABI: All kernels use raw float* pointers (no memref descriptors).
 * Signature: void kernel(float* in0, float* in1, ..., float* out, i64 numel,
 *                        i64 in0_numel, i64 in1_numel, ...)
 */

#ifdef CML_HAS_LLVM_BACKEND

#include "ops/ir/llvm/llvm_backend.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "core/logging.h"
#include "backend/blas.h"

#include <llvm-c/Core.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <llvm-c/LLJIT.h>
#include <llvm-c/Orc.h>
#include <llvm-c/Transforms/PassBuilder.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

struct CMLLLVMBackend {
    LLVMOrcLLJITRef jit;
    LLVMTargetMachineRef tm;
    bool initialized;
    int kernel_count; // for unique naming
};

static bool g_llvm_targets_initialized = false;

CMLLLVMBackend* cml_llvm_backend_init(void) {
    if (!g_llvm_targets_initialized) {
        LLVMInitializeNativeTarget();
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
        g_llvm_targets_initialized = true;
    }

    CMLLLVMBackend* b = calloc(1, sizeof(CMLLLVMBackend));
    if (!b) return NULL;

    // Create target machine for optimization passes
    char* triple = LLVMGetDefaultTargetTriple();
    LLVMTargetRef target;
    char* err = NULL;
    if (LLVMGetTargetFromTriple(triple, &target, &err) != 0) {
        LOG_ERROR("LLVM: Failed to get target: %s", err ? err : "unknown");
        LLVMDisposeMessage(err);
        LLVMDisposeMessage(triple);
        free(b);
        return NULL;
    }

    b->tm = LLVMCreateTargetMachine(
        target, triple, "generic", "",
        LLVMCodeGenLevelAggressive, LLVMRelocDefault, LLVMCodeModelDefault);
    LLVMDisposeMessage(triple);

    if (!b->tm) {
        LOG_ERROR("LLVM: Failed to create target machine");
        free(b);
        return NULL;
    }

    b->initialized = true;
    b->kernel_count = 0;
    LOG_DEBUG("LLVM JIT backend initialized");
    return b;
}

void cml_llvm_backend_destroy(CMLLLVMBackend* backend) {
    if (!backend) return;
    if (backend->jit) {
        LLVMOrcDisposeLLJIT(backend->jit);
    }
    if (backend->tm) {
        LLVMDisposeTargetMachine(backend->tm);
    }
    free(backend);
}

// Emit a simple loop: for (i64 i = 0; i < n; i++) { body }
// Returns the loop induction variable (i) and positions builder at the loop body
typedef struct {
    LLVMValueRef i;           // induction variable (phi)
    LLVMBasicBlockRef body;   // loop body block
    LLVMBasicBlockRef exit;   // exit block
    LLVMBasicBlockRef header; // header block (for phi wiring)
} LoopInfo;

static LoopInfo emit_loop(LLVMBuilderRef bld, LLVMContextRef ctx,
                          LLVMValueRef fn, LLVMValueRef n,
                          const char* name) {
    LoopInfo info;
    LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx);

    char hdr_name[64], body_name[64], exit_name[64];
    snprintf(hdr_name, sizeof(hdr_name), "%s.hdr", name);
    snprintf(body_name, sizeof(body_name), "%s.body", name);
    snprintf(exit_name, sizeof(exit_name), "%s.exit", name);

    info.header = LLVMAppendBasicBlockInContext(ctx, fn, hdr_name);
    info.body   = LLVMAppendBasicBlockInContext(ctx, fn, body_name);
    info.exit   = LLVMAppendBasicBlockInContext(ctx, fn, exit_name);

    // Current block -> header
    LLVMBuildBr(bld, info.header);

    // Header: i = phi, cond branch
    LLVMPositionBuilderAtEnd(bld, info.header);
    info.i = LLVMBuildPhi(bld, i64, "i");
    LLVMValueRef cond = LLVMBuildICmp(bld, LLVMIntULT, info.i, n, "cond");
    LLVMBuildCondBr(bld, cond, info.body, info.exit);

    // Body (caller fills in)
    LLVMPositionBuilderAtEnd(bld, info.body);

    return info;
}

// Close the loop: wire the phi, branch back to header
static void close_loop(LLVMBuilderRef bld, LoopInfo* info,
                       LLVMBasicBlockRef entry_bb) {
    LLVMTypeRef i64 = LLVMInt64TypeInContext(LLVMGetTypeContext(LLVMTypeOf(info->i)));
    LLVMValueRef one = LLVMConstInt(i64, 1, 0);
    LLVMValueRef i_next = LLVMBuildAdd(bld, info->i, one, "i.next");
    LLVMBuildBr(bld, info->header);

    LLVMValueRef zero = LLVMConstInt(i64, 0, 0);
    LLVMValueRef incoming_vals[] = { zero, i_next };
    LLVMBasicBlockRef incoming_bbs[] = { entry_bb, info->body };
    LLVMAddIncoming(info->i, incoming_vals, incoming_bbs, 2);
}

// Build elementwise binary op: out[i] = op(in0[i % n0], in1[i % n1])
static LLVMModuleRef build_binary_op(LLVMContextRef ctx, UOpType type,
                                     const char* fn_name) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    // void fn(ptr in0, ptr in1, ptr out, i64 out_n, i64 in0_n, i64 in1_n)
    LLVMTypeRef params[] = { ptr, ptr, ptr, i64, i64, i64 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 6, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef in0   = LLVMGetParam(fn, 0);
    LLVMValueRef in1   = LLVMGetParam(fn, 1);
    LLVMValueRef out   = LLVMGetParam(fn, 2);
    LLVMValueRef out_n = LLVMGetParam(fn, 3);
    LLVMValueRef in0_n = LLVMGetParam(fn, 4);
    LLVMValueRef in1_n = LLVMGetParam(fn, 5);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "elem");

    // Compute broadcast indices: i0 = (in0_n == 1) ? 0 : i % in0_n
    LLVMValueRef one = LLVMConstInt(i64, 1, 0);
    LLVMValueRef is_scalar0 = LLVMBuildICmp(bld, LLVMIntEQ, in0_n, one, "sc0");
    LLVMValueRef mod0 = LLVMBuildURem(bld, loop.i, in0_n, "mod0");
    LLVMValueRef zero_i64 = LLVMConstInt(i64, 0, 0);
    LLVMValueRef i0 = LLVMBuildSelect(bld, is_scalar0, zero_i64, mod0, "i0");

    LLVMValueRef is_scalar1 = LLVMBuildICmp(bld, LLVMIntEQ, in1_n, one, "sc1");
    LLVMValueRef mod1 = LLVMBuildURem(bld, loop.i, in1_n, "mod1");
    LLVMValueRef i1 = LLVMBuildSelect(bld, is_scalar1, zero_i64, mod1, "i1");

    // Load inputs
    LLVMValueRef gep0 = LLVMBuildGEP2(bld, f32, in0, &i0, 1, "p0");
    LLVMValueRef gep1 = LLVMBuildGEP2(bld, f32, in1, &i1, 1, "p1");
    LLVMValueRef v0 = LLVMBuildLoad2(bld, f32, gep0, "v0");
    LLVMValueRef v1 = LLVMBuildLoad2(bld, f32, gep1, "v1");

    // Apply operation
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
        // max(a, b) = a > b ? a : b
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
        // Use llvm.pow intrinsic
        LLVMTypeRef pow_args[] = { f32 };
        LLVMValueRef pow_fn = LLVMGetIntrinsicDeclaration(
            mod, LLVMLookupIntrinsicID("llvm.pow", 8), pow_args, 1);
        LLVMValueRef args[] = { v0, v1 };
        result = LLVMBuildCall2(bld, LLVMGetReturnType(LLVMGlobalGetValueType(pow_fn)),
                                pow_fn, args, 2, "pow");
        // Actually: use LLVMBuildCall2 with proper fn type
        LLVMTypeRef pow_fn_type = LLVMFunctionType(f32, (LLVMTypeRef[]){f32, f32}, 2, 0);
        (void)pow_fn_type;
        break;
    }
    default:
        result = LLVMBuildFAdd(bld, v0, v1, "fallback");
        break;
    }

    // Store result
    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout");
    LLVMBuildStore(bld, result, gep_out);

    close_loop(bld, &loop, entry);

    // Exit
    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Build elementwise unary op: out[i] = op(in[i % n_in])
static LLVMModuleRef build_unary_op(LLVMContextRef ctx, UOpType type,
                                    const char* fn_name) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    // void fn(ptr in, ptr out, i64 out_n, i64 in_n)
    LLVMTypeRef params[] = { ptr, ptr, i64, i64 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 4, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef in    = LLVMGetParam(fn, 0);
    LLVMValueRef out   = LLVMGetParam(fn, 1);
    LLVMValueRef out_n = LLVMGetParam(fn, 2);
    LLVMValueRef in_n  = LLVMGetParam(fn, 3);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "elem");

    // Broadcast index
    LLVMValueRef one = LLVMConstInt(i64, 1, 0);
    LLVMValueRef is_scalar = LLVMBuildICmp(bld, LLVMIntEQ, in_n, one, "sc");
    LLVMValueRef mod_i = LLVMBuildURem(bld, loop.i, in_n, "mod_i");
    LLVMValueRef zero_i64 = LLVMConstInt(i64, 0, 0);
    LLVMValueRef idx = LLVMBuildSelect(bld, is_scalar, zero_i64, mod_i, "idx");

    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in, &idx, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "val");

    LLVMValueRef result = NULL;

    // Helper to get intrinsic declaration
    #define GET_INTRINSIC_F32(name_str, name_len) \
        LLVMGetIntrinsicDeclaration(mod, LLVMLookupIntrinsicID(name_str, name_len), \
                                    (LLVMTypeRef[]){f32}, 1)
    #define CALL_INTRINSIC1(intrinsic, arg, res_name) do { \
        LLVMTypeRef _ft = LLVMFunctionType(f32, (LLVMTypeRef[]){f32}, 1, 0); \
        result = LLVMBuildCall2(bld, _ft, intrinsic, (LLVMValueRef[]){arg}, 1, res_name); \
    } while(0)

    switch (type) {
    case UOP_NEG:
        result = LLVMBuildFNeg(bld, val, "neg");
        break;
    case UOP_EXP:
        CALL_INTRINSIC1(GET_INTRINSIC_F32("llvm.exp", 8), val, "exp");
        break;
    case UOP_LOG: {
        LLVMValueRef eps = LLVMConstReal(f32, 1e-8f);
        LLVMValueRef safe = LLVMBuildFAdd(bld, val, eps, "safe");
        CALL_INTRINSIC1(GET_INTRINSIC_F32("llvm.log", 8), safe, "log");
        break;
    }
    case UOP_SQRT: {
        LLVMValueRef abs_val;
        CALL_INTRINSIC1(GET_INTRINSIC_F32("llvm.fabs", 9), val, "absv");
        abs_val = result;
        CALL_INTRINSIC1(GET_INTRINSIC_F32("llvm.sqrt", 9), abs_val, "sqrt");
        break;
    }
    case UOP_ABS:
        CALL_INTRINSIC1(GET_INTRINSIC_F32("llvm.fabs", 9), val, "abs");
        break;
    case UOP_SIN:
        CALL_INTRINSIC1(GET_INTRINSIC_F32("llvm.sin", 8), val, "sin");
        break;
    case UOP_COS:
        CALL_INTRINSIC1(GET_INTRINSIC_F32("llvm.cos", 8), val, "cos");
        break;
    case UOP_TAN: {
        // tan(x) = sin(x) / cos(x)
        LLVMValueRef sin_fn = GET_INTRINSIC_F32("llvm.sin", 8);
        LLVMValueRef cos_fn = GET_INTRINSIC_F32("llvm.cos", 8);
        LLVMTypeRef ft1 = LLVMFunctionType(f32, (LLVMTypeRef[]){f32}, 1, 0);
        LLVMValueRef s = LLVMBuildCall2(bld, ft1, sin_fn, (LLVMValueRef[]){val}, 1, "s");
        LLVMValueRef c = LLVMBuildCall2(bld, ft1, cos_fn, (LLVMValueRef[]){val}, 1, "c");
        result = LLVMBuildFDiv(bld, s, c, "tan");
        break;
    }
    case UOP_RECIP: {
        LLVMValueRef one_f = LLVMConstReal(f32, 1.0);
        result = LLVMBuildFDiv(bld, one_f, val, "recip");
        break;
    }
    case UOP_SIGMOID: {
        // sigmoid(x) = 1 / (1 + exp(-x))
        LLVMValueRef neg = LLVMBuildFNeg(bld, val, "neg");
        LLVMValueRef exp_fn = GET_INTRINSIC_F32("llvm.exp", 8);
        LLVMTypeRef ft1 = LLVMFunctionType(f32, (LLVMTypeRef[]){f32}, 1, 0);
        LLVMValueRef e = LLVMBuildCall2(bld, ft1, exp_fn, (LLVMValueRef[]){neg}, 1, "e");
        LLVMValueRef one_f = LLVMConstReal(f32, 1.0);
        LLVMValueRef denom = LLVMBuildFAdd(bld, one_f, e, "denom");
        result = LLVMBuildFDiv(bld, one_f, denom, "sigmoid");
        break;
    }
    case UOP_TANH: {
        // tanh(x) = 2*sigmoid(2x) - 1
        LLVMValueRef two = LLVMConstReal(f32, 2.0);
        LLVMValueRef two_x = LLVMBuildFMul(bld, two, val, "2x");
        LLVMValueRef neg = LLVMBuildFNeg(bld, two_x, "neg2x");
        LLVMValueRef exp_fn = GET_INTRINSIC_F32("llvm.exp", 8);
        LLVMTypeRef ft1 = LLVMFunctionType(f32, (LLVMTypeRef[]){f32}, 1, 0);
        LLVMValueRef e = LLVMBuildCall2(bld, ft1, exp_fn, (LLVMValueRef[]){neg}, 1, "e");
        LLVMValueRef one_f = LLVMConstReal(f32, 1.0);
        LLVMValueRef denom = LLVMBuildFAdd(bld, one_f, e, "denom");
        LLVMValueRef sig = LLVMBuildFDiv(bld, one_f, denom, "sig");
        LLVMValueRef scaled = LLVMBuildFMul(bld, two, sig, "scaled");
        result = LLVMBuildFSub(bld, scaled, one_f, "tanh");
        break;
    }
    default:
        result = val; // passthrough
        break;
    }

    #undef GET_INTRINSIC_F32
    #undef CALL_INTRINSIC1

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout");
    LLVMBuildStore(bld, result, gep_out);

    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Build reduction: out[0] = reduce(in[0..n])
static LLVMModuleRef build_reduction(LLVMContextRef ctx, UOpType type,
                                     const char* fn_name) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    // void fn(ptr in, ptr out, i64 n)
    LLVMTypeRef params[] = { ptr, ptr, i64 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 3, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef in_p  = LLVMGetParam(fn, 0);
    LLVMValueRef out_p = LLVMGetParam(fn, 1);
    LLVMValueRef n     = LLVMGetParam(fn, 2);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBasicBlockRef loop  = LLVMAppendBasicBlockInContext(ctx, fn, "loop");
    LLVMBasicBlockRef done  = LLVMAppendBasicBlockInContext(ctx, fn, "done");

    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);

    LLVMValueRef zero_i64 = LLVMConstInt(i64, 0, 0);
    LLVMValueRef one_i64  = LLVMConstInt(i64, 1, 0);

    float init_val = 0.0f;
    if (type == UOP_MAX_REDUCE) init_val = -HUGE_VALF;
    LLVMValueRef init = LLVMConstReal(f32, init_val);

    // entry -> loop
    LLVMPositionBuilderAtEnd(bld, entry);
    LLVMBuildBr(bld, loop);

    // loop
    LLVMPositionBuilderAtEnd(bld, loop);
    LLVMValueRef i   = LLVMBuildPhi(bld, i64, "i");
    LLVMValueRef acc = LLVMBuildPhi(bld, f32, "acc");

    LLVMValueRef gep = LLVMBuildGEP2(bld, f32, in_p, &i, 1, "p");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep, "v");

    LLVMValueRef new_acc;
    switch (type) {
    case UOP_SUM:
    case UOP_MEAN:
        new_acc = LLVMBuildFAdd(bld, acc, val, "sum");
        break;
    case UOP_MAX_REDUCE: {
        LLVMValueRef cmp = LLVMBuildFCmp(bld, LLVMRealOGT, val, acc, "gt");
        new_acc = LLVMBuildSelect(bld, cmp, val, acc, "max");
        break;
    }
    default:
        new_acc = LLVMBuildFAdd(bld, acc, val, "sum");
        break;
    }

    LLVMValueRef i_next = LLVMBuildAdd(bld, i, one_i64, "i.next");
    LLVMValueRef cond = LLVMBuildICmp(bld, LLVMIntULT, i_next, n, "cond");
    LLVMBuildCondBr(bld, cond, loop, done);

    // Wire phi
    LLVMValueRef i_vals[]   = { zero_i64, i_next };
    LLVMBasicBlockRef i_bbs[] = { entry, loop };
    LLVMAddIncoming(i, i_vals, i_bbs, 2);

    LLVMValueRef acc_vals[] = { init, new_acc };
    LLVMAddIncoming(acc, acc_vals, i_bbs, 2);

    // done: store result
    LLVMPositionBuilderAtEnd(bld, done);
    LLVMValueRef final_val = new_acc; // from loop block

    if (type == UOP_MEAN) {
        LLVMValueRef n_f = LLVMBuildUIToFP(bld, n, f32, "n_f");
        final_val = LLVMBuildFDiv(bld, new_acc, n_f, "mean");
    }

    LLVMValueRef out_gep = LLVMBuildGEP2(bld, f32, out_p, &zero_i64, 1, "outp");
    LLVMBuildStore(bld, final_val, out_gep);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Build fill: out[i] = constant_value
// void fn(ptr out, i64 n, float val)
static LLVMModuleRef build_fill_op(LLVMContextRef ctx, const char* fn_name,
                                    float fill_value) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    // void fn(ptr out, i64 n)
    LLVMTypeRef params[] = { ptr, i64 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 2, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef out   = LLVMGetParam(fn, 0);
    LLVMValueRef out_n = LLVMGetParam(fn, 1);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LLVMValueRef val = LLVMConstReal(f32, (double)fill_value);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "fill");

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout");
    LLVMBuildStore(bld, val, gep_out);

    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Build where: out[i] = cond[i] != 0 ? a[i] : b[i]
// void fn(ptr cond, ptr a, ptr b, ptr out, i64 n, i64 cond_n, i64 a_n, i64 b_n)
static LLVMModuleRef build_where_op(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, ptr, ptr, i64, i64, i64, i64 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 8, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef cond   = LLVMGetParam(fn, 0);
    LLVMValueRef a      = LLVMGetParam(fn, 1);
    LLVMValueRef b      = LLVMGetParam(fn, 2);
    LLVMValueRef out    = LLVMGetParam(fn, 3);
    LLVMValueRef out_n  = LLVMGetParam(fn, 4);
    LLVMValueRef cond_n = LLVMGetParam(fn, 5);
    LLVMValueRef a_n    = LLVMGetParam(fn, 6);
    LLVMValueRef b_n    = LLVMGetParam(fn, 7);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "where");

    LLVMValueRef one = LLVMConstInt(i64, 1, 0);
    LLVMValueRef zero_i64 = LLVMConstInt(i64, 0, 0);
    LLVMValueRef zero_f32 = LLVMConstReal(f32, 0.0);

    // Broadcast indices
    LLVMValueRef sc_c = LLVMBuildICmp(bld, LLVMIntEQ, cond_n, one, "sc_c");
    LLVMValueRef mod_c = LLVMBuildURem(bld, loop.i, cond_n, "mod_c");
    LLVMValueRef ic = LLVMBuildSelect(bld, sc_c, zero_i64, mod_c, "ic");

    LLVMValueRef sc_a = LLVMBuildICmp(bld, LLVMIntEQ, a_n, one, "sc_a");
    LLVMValueRef mod_a = LLVMBuildURem(bld, loop.i, a_n, "mod_a");
    LLVMValueRef ia = LLVMBuildSelect(bld, sc_a, zero_i64, mod_a, "ia");

    LLVMValueRef sc_b = LLVMBuildICmp(bld, LLVMIntEQ, b_n, one, "sc_b");
    LLVMValueRef mod_b = LLVMBuildURem(bld, loop.i, b_n, "mod_b");
    LLVMValueRef ib = LLVMBuildSelect(bld, sc_b, zero_i64, mod_b, "ib");

    LLVMValueRef gep_c = LLVMBuildGEP2(bld, f32, cond, &ic, 1, "pc");
    LLVMValueRef gep_a = LLVMBuildGEP2(bld, f32, a, &ia, 1, "pa");
    LLVMValueRef gep_b = LLVMBuildGEP2(bld, f32, b, &ib, 1, "pb");

    LLVMValueRef vc = LLVMBuildLoad2(bld, f32, gep_c, "vc");
    LLVMValueRef va = LLVMBuildLoad2(bld, f32, gep_a, "va");
    LLVMValueRef vb = LLVMBuildLoad2(bld, f32, gep_b, "vb");

    LLVMValueRef is_true = LLVMBuildFCmp(bld, LLVMRealONE, vc, zero_f32, "is_true");
    LLVMValueRef result = LLVMBuildSelect(bld, is_true, va, vb, "sel");

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout");
    LLVMBuildStore(bld, result, gep_out);

    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Build gather: out[i] = input[i * C + indices[i]] (for 2D gather along last dim)
// void fn(ptr input, ptr indices, ptr out, i64 N, i64 C)
static LLVMModuleRef build_gather_op(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, ptr, i64, i64 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 5, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef input   = LLVMGetParam(fn, 0);
    LLVMValueRef indices = LLVMGetParam(fn, 1);
    LLVMValueRef out     = LLVMGetParam(fn, 2);
    LLVMValueRef N       = LLVMGetParam(fn, 3);
    LLVMValueRef C       = LLVMGetParam(fn, 4);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, N, "gather");

    // idx = (int64_t)indices[i]
    LLVMValueRef gep_idx = LLVMBuildGEP2(bld, f32, indices, &loop.i, 1, "pidx");
    LLVMValueRef idx_f = LLVMBuildLoad2(bld, f32, gep_idx, "idx_f");
    LLVMValueRef idx = LLVMBuildFPToSI(bld, idx_f, i64, "idx");

    // offset = i * C + idx
    LLVMValueRef row = LLVMBuildMul(bld, loop.i, C, "row");
    LLVMValueRef offset = LLVMBuildAdd(bld, row, idx, "offset");

    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, input, &offset, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "val");

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout");
    LLVMBuildStore(bld, val, gep_out);

    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Build 2D permute (transpose): out[j*M+i] = in[i*N+j]
// void fn(ptr in, ptr out, i64 M, i64 N)
static LLVMModuleRef build_permute_2d(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, i64, i64 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 4, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef in_p = LLVMGetParam(fn, 0);
    LLVMValueRef out  = LLVMGetParam(fn, 1);
    LLVMValueRef M    = LLVMGetParam(fn, 2);
    LLVMValueRef N    = LLVMGetParam(fn, 3);

    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);

    LLVMBasicBlockRef entry  = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBasicBlockRef i_hdr  = LLVMAppendBasicBlockInContext(ctx, fn, "i.hdr");
    LLVMBasicBlockRef j_hdr  = LLVMAppendBasicBlockInContext(ctx, fn, "j.hdr");
    LLVMBasicBlockRef j_body = LLVMAppendBasicBlockInContext(ctx, fn, "j.body");
    LLVMBasicBlockRef j_exit = LLVMAppendBasicBlockInContext(ctx, fn, "j.exit");
    LLVMBasicBlockRef i_exit = LLVMAppendBasicBlockInContext(ctx, fn, "i.exit");

    LLVMValueRef zero_i64 = LLVMConstInt(i64, 0, 0);
    LLVMValueRef one_i64  = LLVMConstInt(i64, 1, 0);

    LLVMPositionBuilderAtEnd(bld, entry);
    LLVMBuildBr(bld, i_hdr);

    // i loop
    LLVMPositionBuilderAtEnd(bld, i_hdr);
    LLVMValueRef i_val = LLVMBuildPhi(bld, i64, "i");
    LLVMValueRef i_cond = LLVMBuildICmp(bld, LLVMIntULT, i_val, M, "i.cond");
    LLVMBuildCondBr(bld, i_cond, j_hdr, i_exit);

    // j loop
    LLVMPositionBuilderAtEnd(bld, j_hdr);
    LLVMValueRef j_val = LLVMBuildPhi(bld, i64, "j");
    LLVMValueRef j_cond = LLVMBuildICmp(bld, LLVMIntULT, j_val, N, "j.cond");
    LLVMBuildCondBr(bld, j_cond, j_body, j_exit);

    // body: out[j*M+i] = in[i*N+j]
    LLVMPositionBuilderAtEnd(bld, j_body);
    LLVMValueRef in_off = LLVMBuildAdd(bld, LLVMBuildMul(bld, i_val, N, "iN"), j_val, "in_off");
    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in_p, &in_off, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "v");

    LLVMValueRef out_off = LLVMBuildAdd(bld, LLVMBuildMul(bld, j_val, M, "jM"), i_val, "out_off");
    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &out_off, 1, "pout");
    LLVMBuildStore(bld, val, gep_out);

    LLVMValueRef j_next = LLVMBuildAdd(bld, j_val, one_i64, "j.next");
    LLVMBuildBr(bld, j_hdr);

    // j exit
    LLVMPositionBuilderAtEnd(bld, j_exit);
    LLVMValueRef i_next = LLVMBuildAdd(bld, i_val, one_i64, "i.next");
    LLVMBuildBr(bld, i_hdr);

    // i exit
    LLVMPositionBuilderAtEnd(bld, i_exit);
    LLVMBuildRetVoid(bld);

    // Wire phis
    LLVMValueRef j_vals[] = { zero_i64, j_next };
    LLVMBasicBlockRef j_bbs[] = { i_hdr, j_body };
    LLVMAddIncoming(j_val, j_vals, j_bbs, 2);

    LLVMValueRef i_vals[] = { zero_i64, i_next };
    LLVMBasicBlockRef i_bbs[] = { entry, j_exit };
    LLVMAddIncoming(i_val, i_vals, i_bbs, 2);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Build expand (broadcast via modulo): out[i] = in[i % in_n]
// void fn(ptr in, ptr out, i64 out_n, i64 in_n)
static LLVMModuleRef build_expand_op(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, i64, i64 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 4, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef in_p  = LLVMGetParam(fn, 0);
    LLVMValueRef out   = LLVMGetParam(fn, 1);
    LLVMValueRef out_n = LLVMGetParam(fn, 2);
    LLVMValueRef in_n  = LLVMGetParam(fn, 3);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "expand");

    LLVMValueRef idx = LLVMBuildURem(bld, loop.i, in_n, "idx");
    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in_p, &idx, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "v");

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout");
    LLVMBuildStore(bld, val, gep_out);

    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Build reshape: just a memcpy (contiguous data)
// void fn(ptr in, ptr out, i64 n)
static LLVMModuleRef build_reshape_op(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, i64 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 3, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef in_p = LLVMGetParam(fn, 0);
    LLVMValueRef out  = LLVMGetParam(fn, 1);
    LLVMValueRef n    = LLVMGetParam(fn, 2);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, n, "copy");

    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in_p, &loop.i, 1, "pin");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep_in, "v");
    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout");
    LLVMBuildStore(bld, val, gep_out);

    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

// Build matmul: C[m,n] = sum_k A[m,k]*B[k,n]
// void fn(ptr A, ptr B, ptr C, i64 M, i64 N, i64 K)
static LLVMModuleRef build_matmul_kernel(LLVMContextRef ctx,
                                         const char* fn_name) {
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(fn_name, ctx);

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, ptr, i64, i64, i64 };
    LLVMTypeRef fn_type = LLVMFunctionType(void_t, params, 6, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef A = LLVMGetParam(fn, 0);
    LLVMValueRef B = LLVMGetParam(fn, 1);
    LLVMValueRef C = LLVMGetParam(fn, 2);
    LLVMValueRef M = LLVMGetParam(fn, 3);
    LLVMValueRef N = LLVMGetParam(fn, 4);
    LLVMValueRef K = LLVMGetParam(fn, 5);

    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);

    LLVMBasicBlockRef entry     = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBasicBlockRef m_hdr     = LLVMAppendBasicBlockInContext(ctx, fn, "m.hdr");
    LLVMBasicBlockRef n_hdr     = LLVMAppendBasicBlockInContext(ctx, fn, "n.hdr");
    LLVMBasicBlockRef k_hdr     = LLVMAppendBasicBlockInContext(ctx, fn, "k.hdr");
    LLVMBasicBlockRef k_body    = LLVMAppendBasicBlockInContext(ctx, fn, "k.body");
    LLVMBasicBlockRef k_exit    = LLVMAppendBasicBlockInContext(ctx, fn, "k.exit");
    LLVMBasicBlockRef n_exit    = LLVMAppendBasicBlockInContext(ctx, fn, "n.exit");
    LLVMBasicBlockRef m_exit    = LLVMAppendBasicBlockInContext(ctx, fn, "m.exit");

    LLVMValueRef zero_i64 = LLVMConstInt(i64, 0, 0);
    LLVMValueRef one_i64  = LLVMConstInt(i64, 1, 0);
    LLVMValueRef zero_f32 = LLVMConstReal(f32, 0.0);

    // entry -> m_hdr
    LLVMPositionBuilderAtEnd(bld, entry);
    LLVMBuildBr(bld, m_hdr);

    // m loop
    LLVMPositionBuilderAtEnd(bld, m_hdr);
    LLVMValueRef m_i = LLVMBuildPhi(bld, i64, "m");
    LLVMValueRef m_cond = LLVMBuildICmp(bld, LLVMIntULT, m_i, M, "m.cond");
    LLVMBuildCondBr(bld, m_cond, n_hdr, m_exit);

    // n loop
    LLVMPositionBuilderAtEnd(bld, n_hdr);
    LLVMValueRef n_i = LLVMBuildPhi(bld, i64, "n");
    LLVMValueRef n_cond = LLVMBuildICmp(bld, LLVMIntULT, n_i, N, "n.cond");
    LLVMBuildCondBr(bld, n_cond, k_hdr, n_exit);

    // k loop header
    LLVMPositionBuilderAtEnd(bld, k_hdr);
    LLVMValueRef k_i  = LLVMBuildPhi(bld, i64, "k");
    LLVMValueRef acc   = LLVMBuildPhi(bld, f32, "acc");
    LLVMValueRef k_cond = LLVMBuildICmp(bld, LLVMIntULT, k_i, K, "k.cond");
    LLVMBuildCondBr(bld, k_cond, k_body, k_exit);

    // k body: acc += A[m*K+k] * B[k*N+n]
    LLVMPositionBuilderAtEnd(bld, k_body);
    LLVMValueRef mk   = LLVMBuildMul(bld, m_i, K, "mk");
    LLVMValueRef mk_k = LLVMBuildAdd(bld, mk, k_i, "mk_k");
    LLVMValueRef a_gep = LLVMBuildGEP2(bld, f32, A, &mk_k, 1, "A.p");
    LLVMValueRef a_val = LLVMBuildLoad2(bld, f32, a_gep, "A.v");

    LLVMValueRef kn   = LLVMBuildMul(bld, k_i, N, "kn");
    LLVMValueRef kn_n = LLVMBuildAdd(bld, kn, n_i, "kn_n");
    LLVMValueRef b_gep = LLVMBuildGEP2(bld, f32, B, &kn_n, 1, "B.p");
    LLVMValueRef b_val = LLVMBuildLoad2(bld, f32, b_gep, "B.v");

    LLVMValueRef prod     = LLVMBuildFMul(bld, a_val, b_val, "prod");
    LLVMValueRef new_acc  = LLVMBuildFAdd(bld, acc, prod, "new_acc");
    LLVMValueRef k_next   = LLVMBuildAdd(bld, k_i, one_i64, "k.next");
    LLVMBuildBr(bld, k_hdr);

    // Wire k phi
    LLVMValueRef k_vals[] = { zero_i64, k_next };
    LLVMBasicBlockRef k_bbs[] = { n_hdr, k_body };
    LLVMAddIncoming(k_i, k_vals, k_bbs, 2);
    LLVMValueRef acc_vals[] = { zero_f32, new_acc };
    LLVMAddIncoming(acc, acc_vals, k_bbs, 2);

    // k_exit: store C[m*N+n] = acc
    LLVMPositionBuilderAtEnd(bld, k_exit);
    LLVMValueRef mn   = LLVMBuildMul(bld, m_i, N, "mn");
    LLVMValueRef mn_n = LLVMBuildAdd(bld, mn, n_i, "mn_n");
    LLVMValueRef c_gep = LLVMBuildGEP2(bld, f32, C, &mn_n, 1, "C.p");
    LLVMBuildStore(bld, acc, c_gep);
    LLVMValueRef n_next = LLVMBuildAdd(bld, n_i, one_i64, "n.next");
    LLVMBuildBr(bld, n_hdr);

    // n_exit: m++
    LLVMPositionBuilderAtEnd(bld, n_exit);
    LLVMValueRef m_next = LLVMBuildAdd(bld, m_i, one_i64, "m.next");
    LLVMBuildBr(bld, m_hdr);

    // m_exit: ret
    LLVMPositionBuilderAtEnd(bld, m_exit);
    LLVMBuildRetVoid(bld);

    // Wire n phi
    LLVMValueRef n_vals[] = { zero_i64, n_next };
    LLVMBasicBlockRef n_bbs[] = { m_hdr, k_exit };
    LLVMAddIncoming(n_i, n_vals, n_bbs, 2);

    // Wire m phi
    LLVMValueRef m_vals[] = { zero_i64, m_next };
    LLVMBasicBlockRef m_bbs[] = { entry, n_exit };
    LLVMAddIncoming(m_i, m_vals, m_bbs, 2);

    LLVMDisposeBuilder(bld);
    return mod;
}

typedef void (*kernel_fn_t)(void);

static kernel_fn_t compile_and_lookup(CMLLLVMBackend* backend,
                                      LLVMModuleRef mod,
                                      const char* fn_name) {
    // Verify module
    char* err = NULL;
    if (LLVMVerifyModule(mod, LLVMReturnStatusAction, &err) != 0) {
        LOG_ERROR("LLVM: Module verification failed: %s", err ? err : "unknown");
        LLVMDisposeMessage(err);
        LLVMDisposeModule(mod);
        return NULL;
    }
    LLVMDisposeMessage(err);

    // Optimize with O2
    LLVMPassBuilderOptionsRef opts = LLVMCreatePassBuilderOptions();
    LLVMPassBuilderOptionsSetLoopVectorization(opts, 1);
    LLVMPassBuilderOptionsSetSLPVectorization(opts, 1);
    LLVMPassBuilderOptionsSetLoopUnrolling(opts, 1);

    LLVMErrorRef error = LLVMRunPasses(mod, "default<O2>", backend->tm, opts);
    LLVMDisposePassBuilderOptions(opts);

    if (error) {
        char* msg = LLVMGetErrorMessage(error);
        LOG_WARNING("LLVM: O2 optimization failed: %s, trying without", msg);
        LLVMDisposeErrorMessage(msg);
        // Continue without optimization — the IR is still valid
    }

    // Create a fresh LLJIT for this kernel
    LLVMOrcLLJITRef jit = NULL;
    error = LLVMOrcCreateLLJIT(&jit, NULL);
    if (error) {
        char* msg = LLVMGetErrorMessage(error);
        LOG_ERROR("LLVM: Failed to create LLJIT: %s", msg);
        LLVMDisposeErrorMessage(msg);
        LLVMDisposeModule(mod);
        return NULL;
    }

    // Wrap module for ORC
    LLVMContextRef mod_ctx = LLVMGetModuleContext(mod);
    LLVMOrcThreadSafeContextRef tsc = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(mod_ctx);
    LLVMOrcThreadSafeModuleRef tsm = LLVMOrcCreateNewThreadSafeModule(mod, tsc);
    LLVMOrcDisposeThreadSafeContext(tsc);

    LLVMOrcJITDylibRef jd = LLVMOrcLLJITGetMainJITDylib(jit);
    error = LLVMOrcLLJITAddLLVMIRModule(jit, jd, tsm);
    if (error) {
        char* msg = LLVMGetErrorMessage(error);
        LOG_ERROR("LLVM: Failed to add module: %s", msg);
        LLVMDisposeErrorMessage(msg);
        LLVMOrcDisposeLLJIT(jit);
        return NULL;
    }

    // Lookup
    LLVMOrcExecutorAddress addr = 0;
    error = LLVMOrcLLJITLookup(jit, &addr, fn_name);
    if (error) {
        char* msg = LLVMGetErrorMessage(error);
        LOG_ERROR("LLVM: Failed to lookup '%s': %s", fn_name, msg);
        LLVMDisposeErrorMessage(msg);
        LLVMOrcDisposeLLJIT(jit);
        return NULL;
    }

    // Store JIT for cleanup (we keep the latest one alive)
    // In production, we'd cache these. For now, destroy old, keep new.
    if (backend->jit) {
        LLVMOrcDisposeLLJIT(backend->jit);
    }
    backend->jit = jit;

    return (kernel_fn_t)(uintptr_t)addr;
}

static bool is_binary_op(UOpType type) {
    return type == UOP_ADD || type == UOP_SUB || type == UOP_MUL ||
           type == UOP_DIV || type == UOP_MAX || type == UOP_CMPLT ||
           type == UOP_POW;
}

static bool is_unary_op(UOpType type) {
    return type == UOP_NEG || type == UOP_EXP || type == UOP_LOG ||
           type == UOP_SQRT || type == UOP_ABS || type == UOP_SIN ||
           type == UOP_COS || type == UOP_TAN || type == UOP_RECIP ||
           type == UOP_SIGMOID || type == UOP_TANH;
}

static bool is_reduction(UOpType type) {
    return type == UOP_SUM || type == UOP_MEAN || type == UOP_MAX_REDUCE;
}

static int llvm_execute_node(CMLLLVMBackend* backend, struct IRNode* node) {
    if (!node || !node->output) return -1;

    Tensor* out = node->output;

    // Allocate output if needed
    if (!out->data && out->numel > 0) {
        size_t size = out->numel * sizeof(float);
        out->data = cml_buffer_cache_alloc(size);
        if (!out->data) {
            LOG_ERROR("LLVM: Failed to allocate output tensor");
            return -1;
        }
        out->owns_data = true;
    }

    UOpType type = node->type;

    // Conv2D and stride/slice with complex params still use CPU
    if (type == UOP_CONV2D || type == UOP_STRIDE || type == UOP_SLICE) {
        return cpu_execute_node(node);
    }

    // For MATMUL, prefer BLAS if available
    if (type == UOP_MATMUL) {
        extern CMLBlasContext* get_blas_context(void);
        CMLBlasContext* blas = get_blas_context();
        if (blas && blas->initialized) {
            return cpu_execute_node(node);
        }
    }

    // Build and JIT the kernel
    char fn_name[64];
    snprintf(fn_name, sizeof(fn_name), "cml_k%d", backend->kernel_count++);

    LLVMContextRef ctx = LLVMContextCreate();
    LLVMModuleRef mod = NULL;

    if (is_binary_op(type)) {
        mod = build_binary_op(ctx, type, fn_name);
    } else if (is_unary_op(type)) {
        mod = build_unary_op(ctx, type, fn_name);
    } else if (is_reduction(type)) {
        mod = build_reduction(ctx, type, fn_name);
    } else if (type == UOP_MATMUL) {
        mod = build_matmul_kernel(ctx, fn_name);
    } else if (type == UOP_FILL) {
        FillParams* p = (FillParams*)node->params;
        float fill_val = p ? p->value : 0.0f;
        mod = build_fill_op(ctx, fn_name, fill_val);
    } else if (type == UOP_WHERE) {
        mod = build_where_op(ctx, fn_name);
    } else if (type == UOP_GATHER) {
        mod = build_gather_op(ctx, fn_name);
    } else if (type == UOP_PERMUTE) {
        // Only handle 2D transpose via JIT; higher dims fall back to CPU
        if (node->num_inputs >= 1 && node->inputs[0] && node->inputs[0]->ndim == 2) {
            mod = build_permute_2d(ctx, fn_name);
        } else {
            LLVMContextDispose(ctx);
            return cpu_execute_node(node);
        }
    } else if (type == UOP_RESHAPE) {
        mod = build_reshape_op(ctx, fn_name);
    } else if (type == UOP_EXPAND) {
        mod = build_expand_op(ctx, fn_name);
    } else {
        LOG_DEBUG("LLVM: Unsupported op %d, falling back to CPU", type);
        LLVMContextDispose(ctx);
        return cpu_execute_node(node);
    }

    if (!mod) {
        LLVMContextDispose(ctx);
        return cpu_execute_node(node);
    }

    kernel_fn_t fn = compile_and_lookup(backend, mod, fn_name);
    if (!fn) {
        // Module ownership was handled by compile_and_lookup
        return cpu_execute_node(node);
    }

    // Execute kernel based on type
    if (is_binary_op(type)) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data) {
            return cpu_execute_node(node);
        }
        typedef void (*binary_fn_t)(float*, float*, float*, int64_t, int64_t, int64_t);
        binary_fn_t bfn = (binary_fn_t)(void*)fn;
        bfn((float*)node->inputs[0]->data,
            (float*)node->inputs[1]->data,
            (float*)out->data,
            (int64_t)out->numel,
            (int64_t)node->inputs[0]->numel,
            (int64_t)node->inputs[1]->numel);
    } else if (is_unary_op(type)) {
        if (node->num_inputs < 1 || !node->inputs[0]->data) {
            return cpu_execute_node(node);
        }
        typedef void (*unary_fn_t)(float*, float*, int64_t, int64_t);
        unary_fn_t ufn = (unary_fn_t)(void*)fn;
        ufn((float*)node->inputs[0]->data,
            (float*)out->data,
            (int64_t)out->numel,
            (int64_t)node->inputs[0]->numel);
    } else if (is_reduction(type)) {
        if (node->num_inputs < 1 || !node->inputs[0]->data) {
            return cpu_execute_node(node);
        }
        typedef void (*reduce_fn_t)(float*, float*, int64_t);
        reduce_fn_t rfn = (reduce_fn_t)(void*)fn;
        rfn((float*)node->inputs[0]->data,
            (float*)out->data,
            (int64_t)node->inputs[0]->numel);
    } else if (type == UOP_MATMUL) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data) {
            return cpu_execute_node(node);
        }
        Tensor* a = node->inputs[0];
        Tensor* b = node->inputs[1];
        if (a->ndim < 2 || b->ndim < 2) return cpu_execute_node(node);

        typedef void (*matmul_fn_t)(float*, float*, float*, int64_t, int64_t, int64_t);
        matmul_fn_t mfn = (matmul_fn_t)(void*)fn;
        int64_t M = a->shape[a->ndim - 2];
        int64_t K = a->shape[a->ndim - 1];
        int64_t N = b->shape[b->ndim - 1];
        mfn((float*)a->data, (float*)b->data, (float*)out->data, M, N, K);
    } else if (type == UOP_FILL) {
        typedef void (*fill_fn_t)(float*, int64_t);
        fill_fn_t ffn = (fill_fn_t)(void*)fn;
        ffn((float*)out->data, (int64_t)out->numel);
    } else if (type == UOP_WHERE) {
        if (node->num_inputs < 3 || !node->inputs[0]->data ||
            !node->inputs[1]->data || !node->inputs[2]->data) {
            return cpu_execute_node(node);
        }
        typedef void (*where_fn_t)(float*, float*, float*, float*,
                                   int64_t, int64_t, int64_t, int64_t);
        where_fn_t wfn = (where_fn_t)(void*)fn;
        wfn((float*)node->inputs[0]->data,
            (float*)node->inputs[1]->data,
            (float*)node->inputs[2]->data,
            (float*)out->data,
            (int64_t)out->numel,
            (int64_t)node->inputs[0]->numel,
            (int64_t)node->inputs[1]->numel,
            (int64_t)node->inputs[2]->numel);
    } else if (type == UOP_GATHER) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data) {
            return cpu_execute_node(node);
        }
        Tensor* input = node->inputs[0];
        if (input->ndim < 2) return cpu_execute_node(node);
        typedef void (*gather_fn_t)(float*, float*, float*, int64_t, int64_t);
        gather_fn_t gfn = (gather_fn_t)(void*)fn;
        int64_t N = (int64_t)out->numel;
        int64_t C = (int64_t)input->shape[input->ndim - 1];
        gfn((float*)input->data, (float*)node->inputs[1]->data,
            (float*)out->data, N, C);
    } else if (type == UOP_PERMUTE) {
        if (node->num_inputs < 1 || !node->inputs[0]->data) {
            return cpu_execute_node(node);
        }
        Tensor* inp = node->inputs[0];
        typedef void (*permute_fn_t)(float*, float*, int64_t, int64_t);
        permute_fn_t pfn = (permute_fn_t)(void*)fn;
        int64_t M = (int64_t)inp->shape[0];
        int64_t N = (int64_t)inp->shape[1];
        pfn((float*)inp->data, (float*)out->data, M, N);
    } else if (type == UOP_RESHAPE) {
        if (node->num_inputs < 1 || !node->inputs[0]->data) {
            return cpu_execute_node(node);
        }
        typedef void (*copy_fn_t)(float*, float*, int64_t);
        copy_fn_t cfn = (copy_fn_t)(void*)fn;
        cfn((float*)node->inputs[0]->data, (float*)out->data,
            (int64_t)out->numel);
    } else if (type == UOP_EXPAND) {
        if (node->num_inputs < 1 || !node->inputs[0]->data) {
            return cpu_execute_node(node);
        }
        typedef void (*expand_fn_t)(float*, float*, int64_t, int64_t);
        expand_fn_t efn = (expand_fn_t)(void*)fn;
        efn((float*)node->inputs[0]->data, (float*)out->data,
            (int64_t)out->numel, (int64_t)node->inputs[0]->numel);
    }

    node->is_executed = true;
    out->is_executed = true;
    return 0;
}

int cml_llvm_execute(CMLLLVMBackend* backend, CMLGraph_t ir) {
    if (!backend || !ir) return -1;

    LOG_DEBUG("LLVM JIT: Executing IR graph");

    struct IRNode* node = ir->head;
    while (node) {
        if (!node->is_executed) {
            if (llvm_execute_node(backend, node) != 0) {
                LOG_WARNING("LLVM JIT: Node execution failed, using CPU fallback");
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

int cml_llvm_execute_up_to(CMLLLVMBackend* backend, CMLGraph_t ir,
                           struct IRNode* target_node) {
    if (!backend || !ir || !target_node) return -1;

    struct IRNode* node = ir->head;
    while (node) {
        if (!node->is_executed) {
            if (llvm_execute_node(backend, node) != 0) {
                LOG_WARNING("LLVM JIT: Node execution failed, using CPU fallback");
                cpu_execute_node(node);
                node->is_executed = true;
                if (node->output) node->output->is_executed = true;
            }
        }
        if (node == target_node) break;
        node = node->next;
    }

    return 0;
}

#endif // CML_HAS_LLVM_BACKEND
