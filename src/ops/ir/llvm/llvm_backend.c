
#ifdef CML_HAS_LLVM_BACKEND

#include "ops/ir/llvm/llvm_backend.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "ops/ir/process_replay.h"
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
#include "alloc/cml_allocator.h"

/* Kernel cache: direct-mapped by UOpType (max ~60 ops, fits easily). */
#define OP_CACHE_SIZE 256

typedef void (*kernel_fn_t)(void);

struct CMLLLVMBackend {
    LLVMOrcLLJITRef     jit;          /* persistent; lives until destroy    */
    LLVMTargetMachineRef tm;
    LLVMContextRef       ctx;          /* shared; outlives all modules       */
    bool                 initialized;
    int                  kernel_count; /* for unique symbol names            */
    kernel_fn_t          op_cache[OP_CACHE_SIZE]; /* index = UOpType        */
};

static bool g_llvm_targets_initialized = false;

/* -------------------------------------------------------------------------
 * Backend init / destroy
 * ---------------------------------------------------------------------- */

CMLLLVMBackend* cml_llvm_backend_init(void) {
    if (!g_llvm_targets_initialized) {
        LLVMInitializeNativeTarget();
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
        g_llvm_targets_initialized = true;
    }

    CMLLLVMBackend* b = cml_calloc(1, sizeof(CMLLLVMBackend));
    if (!b) return NULL;

    char* triple = LLVMGetDefaultTargetTriple();
    LLVMTargetRef target;
    char* err = NULL;
    if (LLVMGetTargetFromTriple(triple, &target, &err) != 0) {
        LOG_ERROR("LLVM: Failed to get target: %s", err ? err : "unknown");
        LLVMDisposeMessage(err);
        LLVMDisposeMessage(triple);
        cml_free(b);
        return NULL;
    }

    /* Use native CPU so LLVM emits AVX2/AVX-512/NEON/SVE as available. */
    char* cpu      = LLVMGetHostCPUName();
    char* features = LLVMGetHostCPUFeatures();
    b->tm = LLVMCreateTargetMachine(
        target, triple, cpu, features,
        LLVMCodeGenLevelAggressive, LLVMRelocDefault, LLVMCodeModelDefault);
    LLVMDisposeMessage(cpu);
    LLVMDisposeMessage(features);
    LLVMDisposeMessage(triple);

    if (!b->tm) {
        LOG_ERROR("LLVM: Failed to create target machine");
        cml_free(b);
        return NULL;
    }

    /* Single persistent LLJIT — all kernels share it. */
    LLVMErrorRef jit_err = LLVMOrcCreateLLJIT(&b->jit, NULL);
    if (jit_err) {
        char* msg = LLVMGetErrorMessage(jit_err);
        LOG_ERROR("LLVM: Failed to create LLJIT: %s", msg);
        LLVMDisposeErrorMessage(msg);
        LLVMDisposeTargetMachine(b->tm);
        free(b);
        return NULL;
    }

    /* Shared context for all modules. */
    b->ctx = LLVMContextCreate();
    if (!b->ctx) {
        LLVMOrcDisposeLLJIT(b->jit);
        LLVMDisposeTargetMachine(b->tm);
        free(b);
        return NULL;
    }

    b->initialized = true;
    LOG_DEBUG("LLVM JIT backend initialized (native CPU, persistent JIT)");
    return b;
}

void cml_llvm_backend_destroy(CMLLLVMBackend* backend) {
    if (!backend) return;
    if (backend->jit)  LLVMOrcDisposeLLJIT(backend->jit);
    if (backend->tm)   LLVMDisposeTargetMachine(backend->tm);
    if (backend->ctx)  LLVMContextDispose(backend->ctx);
    free(backend);
}

/* -------------------------------------------------------------------------
 * Attribute helpers
 * ---------------------------------------------------------------------- */

/* Add noalias to the first n_ptrs pointer parameters (0-indexed). */
static void add_noalias(LLVMContextRef ctx, LLVMValueRef fn, unsigned n_ptrs) {
    unsigned kind = LLVMGetEnumAttributeKindForName("noalias", 7);
    if (!kind) return; /* older LLVM that doesn't support it */
    for (unsigned i = 0; i < n_ptrs; i++) {
        LLVMAttributeRef a = LLVMCreateEnumAttribute(ctx, kind, 0);
        LLVMAddAttributeAtIndex(fn, i + 1 /* 1-indexed */, a);
    }
}

/* -------------------------------------------------------------------------
 * Loop helpers
 * ---------------------------------------------------------------------- */

typedef struct {
    LLVMValueRef      i;      /* phi (induction variable)   */
    LLVMBasicBlockRef body;
    LLVMBasicBlockRef exit;
    LLVMBasicBlockRef header;
} LoopInfo;

static LoopInfo emit_loop(LLVMBuilderRef bld, LLVMContextRef ctx,
                          LLVMValueRef fn, LLVMValueRef n, const char* name) {
    LoopInfo info;
    LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx);

    char h[64], body[64], ex[64];
    snprintf(h,    sizeof(h),    "%s.hdr",  name);
    snprintf(body, sizeof(body), "%s.body", name);
    snprintf(ex,   sizeof(ex),   "%s.exit", name);

    info.header = LLVMAppendBasicBlockInContext(ctx, fn, h);
    info.body   = LLVMAppendBasicBlockInContext(ctx, fn, body);
    info.exit   = LLVMAppendBasicBlockInContext(ctx, fn, ex);

    LLVMBuildBr(bld, info.header);

    LLVMPositionBuilderAtEnd(bld, info.header);
    info.i = LLVMBuildPhi(bld, i64, "i");
    LLVMValueRef cond = LLVMBuildICmp(bld, LLVMIntULT, info.i, n, "cond");
    LLVMBuildCondBr(bld, cond, info.body, info.exit);

    LLVMPositionBuilderAtEnd(bld, info.body);
    return info;
}

static void close_loop(LLVMBuilderRef bld, LoopInfo* info,
                       LLVMBasicBlockRef entry_bb) {
    LLVMTypeRef i64  = LLVMInt64TypeInContext(LLVMGetTypeContext(LLVMTypeOf(info->i)));
    LLVMValueRef one = LLVMConstInt(i64, 1, 0);
    LLVMValueRef i_next = LLVMBuildAdd(bld, info->i, one, "i.next");
    LLVMBuildBr(bld, info->header);

    LLVMValueRef zero = LLVMConstInt(i64, 0, 0);
    LLVMValueRef in_vals[] = { zero, i_next };
    LLVMBasicBlockRef in_bbs[] = { entry_bb, info->body };
    LLVMAddIncoming(info->i, in_vals, in_bbs, 2);
}

/* -------------------------------------------------------------------------
 * Intrinsic helpers
 * ---------------------------------------------------------------------- */

#define INTR1(mod, ctx, name, name_len, f32) \
    LLVMGetIntrinsicDeclaration(mod, LLVMLookupIntrinsicID(name, name_len), \
                                (LLVMTypeRef[]){f32}, 1)

static LLVMValueRef call1(LLVMBuilderRef bld, LLVMTypeRef f32,
                          LLVMValueRef intr, LLVMValueRef arg,
                          const char* res) {
    LLVMTypeRef ft = LLVMFunctionType(f32, (LLVMTypeRef[]){f32}, 1, 0);
    return LLVMBuildCall2(bld, ft, intr, (LLVMValueRef[]){arg}, 1, res);
}

/* Declare an external C libm function: float name(float) */
static LLVMValueRef extern_f32(LLVMModuleRef mod, LLVMContextRef ctx,
                                const char* name) {
    LLVMValueRef fn = LLVMGetNamedFunction(mod, name);
    if (fn) return fn;
    LLVMTypeRef f32 = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ft  = LLVMFunctionType(f32, (LLVMTypeRef[]){f32}, 1, 0);
    fn = LLVMAddFunction(mod, name, ft);
    LLVMSetLinkage(fn, LLVMExternalLinkage);
    return fn;
}

/* -------------------------------------------------------------------------
 * Binary elementwise: out[i] = op(in0[i%n0], in1[i%n1])
 * Signature: void(ptr in0, ptr in1, ptr out, i64 out_n, i64 in0_n, i64 in1_n)
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_binary_op(LLVMContextRef ctx, UOpType type,
                                     const char* fn_name) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, ptr, i64, i64, i64 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 6, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 3); /* in0, in1, out are noalias */

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

    LLVMValueRef one      = LLVMConstInt(i64, 1, 0);
    LLVMValueRef zero_i64 = LLVMConstInt(i64, 0, 0);

    LLVMValueRef sc0  = LLVMBuildICmp(bld, LLVMIntEQ, in0_n, one, "sc0");
    LLVMValueRef mod0 = LLVMBuildURem(bld, loop.i, in0_n, "mod0");
    LLVMValueRef i0   = LLVMBuildSelect(bld, sc0, zero_i64, mod0, "i0");

    LLVMValueRef sc1  = LLVMBuildICmp(bld, LLVMIntEQ, in1_n, one, "sc1");
    LLVMValueRef mod1 = LLVMBuildURem(bld, loop.i, in1_n, "mod1");
    LLVMValueRef i1   = LLVMBuildSelect(bld, sc1, zero_i64, mod1, "i1");

    LLVMValueRef gep0 = LLVMBuildGEP2(bld, f32, in0, &i0, 1, "p0");
    LLVMValueRef gep1 = LLVMBuildGEP2(bld, f32, in1, &i1, 1, "p1");
    LLVMValueRef v0   = LLVMBuildLoad2(bld, f32, gep0, "v0");
    LLVMValueRef v1   = LLVMBuildLoad2(bld, f32, gep1, "v1");

    LLVMValueRef result = NULL;
    switch (type) {
    case UOP_ADD:  result = LLVMBuildFAdd(bld, v0, v1, "r"); break;
    case UOP_SUB:  result = LLVMBuildFSub(bld, v0, v1, "r"); break;
    case UOP_MUL:  result = LLVMBuildFMul(bld, v0, v1, "r"); break;
    case UOP_DIV: {
        LLVMValueRef eps   = LLVMConstReal(f32, 1e-8);
        LLVMValueRef denom = LLVMBuildFAdd(bld, v1, eps, "denom");
        result = LLVMBuildFDiv(bld, v0, denom, "r");
        break;
    }
    case UOP_MAX: {
        LLVMValueRef cmp = LLVMBuildFCmp(bld, LLVMRealOGT, v0, v1, "gt");
        result = LLVMBuildSelect(bld, cmp, v0, v1, "r");
        break;
    }
    case UOP_CMPLT: {
        LLVMValueRef cmp = LLVMBuildFCmp(bld, LLVMRealOLT, v0, v1, "lt");
        result = LLVMBuildUIToFP(bld, cmp, f32, "r");
        break;
    }
    case UOP_POW: {
        LLVMValueRef pow_fn = INTR1(mod, ctx, "llvm.pow", 8, f32);
        LLVMTypeRef  ft     = LLVMFunctionType(f32, (LLVMTypeRef[]){f32,f32}, 2, 0);
        result = LLVMBuildCall2(bld, ft, pow_fn, (LLVMValueRef[]){v0,v1}, 2, "r");
        break;
    }
    default:
        result = LLVMBuildFAdd(bld, v0, v1, "r");
        break;
    }

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout");
    LLVMBuildStore(bld, result, gep_out);
    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);
    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * Unary elementwise: out[i] = op(in[i%n_in])
 * Signature: void(ptr in, ptr out, i64 out_n, i64 in_n)
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_unary_op(LLVMContextRef ctx, UOpType type,
                                    const char* fn_name) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, i64, i64 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 4, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 2); /* in, out are noalias */

    LLVMValueRef in_p  = LLVMGetParam(fn, 0);
    LLVMValueRef out   = LLVMGetParam(fn, 1);
    LLVMValueRef out_n = LLVMGetParam(fn, 2);
    LLVMValueRef in_n  = LLVMGetParam(fn, 3);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "elem");

    LLVMValueRef one      = LLVMConstInt(i64, 1, 0);
    LLVMValueRef zero_i64 = LLVMConstInt(i64, 0, 0);
    LLVMValueRef sc       = LLVMBuildICmp(bld, LLVMIntEQ, in_n, one, "sc");
    LLVMValueRef mod_i    = LLVMBuildURem(bld, loop.i, in_n, "mod_i");
    LLVMValueRef idx      = LLVMBuildSelect(bld, sc, zero_i64, mod_i, "idx");

    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in_p, &idx, 1, "pin");
    LLVMValueRef val    = LLVMBuildLoad2(bld, f32, gep_in, "val");

    LLVMValueRef result = NULL;
    LLVMValueRef zero_f = LLVMConstReal(f32, 0.0);
    LLVMValueRef one_f  = LLVMConstReal(f32, 1.0);

    switch (type) {
    case UOP_NEG:
        result = LLVMBuildFNeg(bld, val, "r");
        break;

    case UOP_ABS:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.fabs", 9, f32), val, "r");
        break;

    case UOP_SQRT: {
        LLVMValueRef av = call1(bld, f32, INTR1(mod, ctx, "llvm.fabs", 9, f32), val, "av");
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.sqrt", 9, f32), av, "r");
        break;
    }

    case UOP_RSQRT: {
        LLVMValueRef av  = call1(bld, f32, INTR1(mod, ctx, "llvm.fabs", 9, f32), val, "av");
        LLVMValueRef sq  = call1(bld, f32, INTR1(mod, ctx, "llvm.sqrt", 9, f32), av, "sq");
        LLVMValueRef eps = LLVMConstReal(f32, 1e-8f);
        LLVMValueRef s   = LLVMBuildFAdd(bld, sq, eps, "s");
        result = LLVMBuildFDiv(bld, one_f, s, "r");
        break;
    }

    case UOP_SQUARE:
        result = LLVMBuildFMul(bld, val, val, "r");
        break;

    case UOP_EXP:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.exp", 8, f32), val, "r");
        break;

    case UOP_EXP2:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.exp2", 9, f32), val, "r");
        break;

    case UOP_LOG: {
        LLVMValueRef eps  = LLVMConstReal(f32, 1e-8f);
        LLVMValueRef safe = LLVMBuildFAdd(bld, val, eps, "safe");
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.log", 8, f32), safe, "r");
        break;
    }

    case UOP_LOG2: {
        LLVMValueRef eps  = LLVMConstReal(f32, 1e-8f);
        LLVMValueRef safe = LLVMBuildFAdd(bld, val, eps, "safe");
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.log2", 9, f32), safe, "r");
        break;
    }

    case UOP_SIN:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.sin", 8, f32), val, "r");
        break;

    case UOP_COS:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.cos", 8, f32), val, "r");
        break;

    case UOP_TAN: {
        LLVMValueRef s  = call1(bld, f32, INTR1(mod, ctx, "llvm.sin", 8, f32), val, "s");
        LLVMValueRef c  = call1(bld, f32, INTR1(mod, ctx, "llvm.cos", 8, f32), val, "c");
        LLVMValueRef eps= LLVMConstReal(f32, 1e-8f);
        LLVMValueRef cd = LLVMBuildFAdd(bld, c, eps, "cd");
        result = LLVMBuildFDiv(bld, s, cd, "r");
        break;
    }

    case UOP_ASIN: {
        LLVMValueRef f = extern_f32(mod, ctx, "asinf");
        LLVMTypeRef  t = LLVMFunctionType(f32, (LLVMTypeRef[]){f32}, 1, 0);
        result = LLVMBuildCall2(bld, t, f, (LLVMValueRef[]){val}, 1, "r");
        break;
    }

    case UOP_ACOS: {
        LLVMValueRef f = extern_f32(mod, ctx, "acosf");
        LLVMTypeRef  t = LLVMFunctionType(f32, (LLVMTypeRef[]){f32}, 1, 0);
        result = LLVMBuildCall2(bld, t, f, (LLVMValueRef[]){val}, 1, "r");
        break;
    }

    case UOP_ATAN: {
        LLVMValueRef f = extern_f32(mod, ctx, "atanf");
        LLVMTypeRef  t = LLVMFunctionType(f32, (LLVMTypeRef[]){f32}, 1, 0);
        result = LLVMBuildCall2(bld, t, f, (LLVMValueRef[]){val}, 1, "r");
        break;
    }

    case UOP_ERF: {
        LLVMValueRef f = extern_f32(mod, ctx, "erff");
        LLVMTypeRef  t = LLVMFunctionType(f32, (LLVMTypeRef[]){f32}, 1, 0);
        result = LLVMBuildCall2(bld, t, f, (LLVMValueRef[]){val}, 1, "r");
        break;
    }

    case UOP_FLOOR:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.floor", 10, f32), val, "r");
        break;

    case UOP_CEIL:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.ceil", 9, f32), val, "r");
        break;

    case UOP_ROUND:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.round", 10, f32), val, "r");
        break;

    case UOP_SIGN: {
        /* sign(x) = (x>0) - (x<0) as float */
        LLVMValueRef gt = LLVMBuildFCmp(bld, LLVMRealOGT, val, zero_f, "gt");
        LLVMValueRef lt = LLVMBuildFCmp(bld, LLVMRealOLT, val, zero_f, "lt");
        LLVMValueRef gf = LLVMBuildUIToFP(bld, gt, f32, "gf");
        LLVMValueRef lf = LLVMBuildUIToFP(bld, lt, f32, "lf");
        result = LLVMBuildFSub(bld, gf, lf, "r");
        break;
    }

    case UOP_RECIP: {
        LLVMValueRef eps = LLVMConstReal(f32, 1e-8f);
        LLVMValueRef d   = LLVMBuildFAdd(bld, val, eps, "d");
        result = LLVMBuildFDiv(bld, one_f, d, "r");
        break;
    }

    case UOP_RELU: {
        /* max(x, 0) — LLVM recognises this and emits vmaxps */
        LLVMValueRef cmp = LLVMBuildFCmp(bld, LLVMRealOGT, val, zero_f, "gt");
        result = LLVMBuildSelect(bld, cmp, val, zero_f, "r");
        break;
    }

    case UOP_RELU6: {
        LLVMValueRef six = LLVMConstReal(f32, 6.0f);
        LLVMValueRef g0  = LLVMBuildFCmp(bld, LLVMRealOGT, val, zero_f, "g0");
        LLVMValueRef cl  = LLVMBuildSelect(bld, g0, val, zero_f, "cl");
        LLVMValueRef l6  = LLVMBuildFCmp(bld, LLVMRealOLT, cl, six, "l6");
        result = LLVMBuildSelect(bld, l6, cl, six, "r");
        break;
    }

    case UOP_SIGMOID: {
        /* 1 / (1 + exp(-x)) */
        LLVMValueRef neg  = LLVMBuildFNeg(bld, val, "nx");
        LLVMValueRef e    = call1(bld, f32, INTR1(mod, ctx, "llvm.exp", 8, f32), neg, "e");
        LLVMValueRef denom= LLVMBuildFAdd(bld, one_f, e, "d");
        result = LLVMBuildFDiv(bld, one_f, denom, "r");
        break;
    }

    case UOP_TANH: {
        /* 2*sigmoid(2x) - 1 */
        LLVMValueRef two  = LLVMConstReal(f32, 2.0f);
        LLVMValueRef tx   = LLVMBuildFMul(bld, two, val, "tx");
        LLVMValueRef neg  = LLVMBuildFNeg(bld, tx, "ntx");
        LLVMValueRef e    = call1(bld, f32, INTR1(mod, ctx, "llvm.exp", 8, f32), neg, "e");
        LLVMValueRef denom= LLVMBuildFAdd(bld, one_f, e, "d");
        LLVMValueRef sig  = LLVMBuildFDiv(bld, one_f, denom, "sig");
        LLVMValueRef sc   = LLVMBuildFMul(bld, two, sig, "sc");
        result = LLVMBuildFSub(bld, sc, one_f, "r");
        break;
    }

    case UOP_SILU: {
        /* x * sigmoid(x) */
        LLVMValueRef neg  = LLVMBuildFNeg(bld, val, "nx");
        LLVMValueRef e    = call1(bld, f32, INTR1(mod, ctx, "llvm.exp", 8, f32), neg, "e");
        LLVMValueRef denom= LLVMBuildFAdd(bld, one_f, e, "d");
        LLVMValueRef sig  = LLVMBuildFDiv(bld, one_f, denom, "sig");
        result = LLVMBuildFMul(bld, val, sig, "r");
        break;
    }

    case UOP_QUICK_GELU: {
        /* QuickGELU: x * sigmoid(1.702 * x) */
        LLVMValueRef c    = LLVMConstReal(f32, 1.702f);
        LLVMValueRef cx   = LLVMBuildFMul(bld, c, val, "cx");
        LLVMValueRef neg  = LLVMBuildFNeg(bld, cx, "ncx");
        LLVMValueRef e    = call1(bld, f32, INTR1(mod, ctx, "llvm.exp", 8, f32), neg, "e");
        LLVMValueRef denom= LLVMBuildFAdd(bld, one_f, e, "d");
        LLVMValueRef sig  = LLVMBuildFDiv(bld, one_f, denom, "sig");
        result = LLVMBuildFMul(bld, val, sig, "r");
        break;
    }

    default:
        result = val; /* passthrough */
        break;
    }

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout");
    LLVMBuildStore(bld, result, gep_out);
    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);
    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * Reduction: out[0] = reduce(in[0..n])
 * Signature: void(ptr in, ptr out, i64 n)
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_reduction(LLVMContextRef ctx, UOpType type,
                                     const char* fn_name) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, i64 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 3, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 2);

    LLVMValueRef in_p  = LLVMGetParam(fn, 0);
    LLVMValueRef out_p = LLVMGetParam(fn, 1);
    LLVMValueRef n     = LLVMGetParam(fn, 2);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBasicBlockRef loop  = LLVMAppendBasicBlockInContext(ctx, fn, "loop");
    LLVMBasicBlockRef done  = LLVMAppendBasicBlockInContext(ctx, fn, "done");

    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);

    LLVMValueRef zero_i64 = LLVMConstInt(i64, 0, 0);
    LLVMValueRef one_i64  = LLVMConstInt(i64, 1, 0);

    float init_val = (type == UOP_MAX_REDUCE) ? -3.402823466e+38f : 0.0f;
    LLVMValueRef init = LLVMConstReal(f32, (double)init_val);

    LLVMPositionBuilderAtEnd(bld, entry);
    LLVMBuildBr(bld, loop);

    LLVMPositionBuilderAtEnd(bld, loop);
    LLVMValueRef i   = LLVMBuildPhi(bld, i64, "i");
    LLVMValueRef acc = LLVMBuildPhi(bld, f32, "acc");

    LLVMValueRef gep = LLVMBuildGEP2(bld, f32, in_p, &i, 1, "p");
    LLVMValueRef val = LLVMBuildLoad2(bld, f32, gep, "v");

    LLVMValueRef new_acc;
    switch (type) {
    case UOP_MAX_REDUCE: {
        LLVMValueRef cmp = LLVMBuildFCmp(bld, LLVMRealOGT, val, acc, "gt");
        new_acc = LLVMBuildSelect(bld, cmp, val, acc, "mx");
        break;
    }
    default: /* SUM, MEAN */
        new_acc = LLVMBuildFAdd(bld, acc, val, "s");
        break;
    }

    LLVMValueRef i_next = LLVMBuildAdd(bld, i, one_i64, "i.next");
    LLVMValueRef cond   = LLVMBuildICmp(bld, LLVMIntULT, i_next, n, "cond");
    LLVMBuildCondBr(bld, cond, loop, done);

    LLVMValueRef i_vals[]   = { zero_i64, i_next };
    LLVMBasicBlockRef i_bbs[] = { entry, loop };
    LLVMAddIncoming(i,   i_vals, i_bbs, 2);
    LLVMValueRef acc_vals[] = { init, new_acc };
    LLVMAddIncoming(acc, acc_vals, i_bbs, 2);

    LLVMPositionBuilderAtEnd(bld, done);
    LLVMValueRef final_val = new_acc;
    if (type == UOP_MEAN) {
        LLVMTypeRef f32t = LLVMFloatTypeInContext(ctx);
        LLVMValueRef nf  = LLVMBuildUIToFP(bld, n, f32t, "nf");
        final_val = LLVMBuildFDiv(bld, new_acc, nf, "mean");
    }

    LLVMValueRef out_gep = LLVMBuildGEP2(bld, f32, out_p, &zero_i64, 1, "outp");
    LLVMBuildStore(bld, final_val, out_gep);
    LLVMBuildRetVoid(bld);

    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * Fill: out[i] = val  (val passed at runtime — allows caching)
 * Signature: void(ptr out, i64 n, float val)
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_fill_op(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, i64, f32 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 3, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 1); /* out is noalias */

    LLVMValueRef out   = LLVMGetParam(fn, 0);
    LLVMValueRef out_n = LLVMGetParam(fn, 1);
    LLVMValueRef fval  = LLVMGetParam(fn, 2);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "fill");
    LLVMValueRef gep = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "p");
    LLVMBuildStore(bld, fval, gep);
    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);
    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * Where: out[i] = cond[i] ? a[i] : b[i]
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_where_op(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, ptr, ptr, i64, i64, i64, i64 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 8, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 4);

    LLVMValueRef cond_p = LLVMGetParam(fn, 0);
    LLVMValueRef a_p    = LLVMGetParam(fn, 1);
    LLVMValueRef b_p    = LLVMGetParam(fn, 2);
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
    LLVMValueRef z   = LLVMConstInt(i64, 0, 0);
    LLVMValueRef zf  = LLVMConstReal(f32, 0.0);

#define BCAST(ptr_v, n_v, suffix) \
    LLVMBuildSelect(bld, LLVMBuildICmp(bld, LLVMIntEQ, n_v, one, "sc"#suffix), \
                    z, LLVMBuildURem(bld, loop.i, n_v, "m"#suffix), "i"#suffix)

    LLVMValueRef ic = BCAST(cond_p, cond_n, c);
    LLVMValueRef ia = BCAST(a_p,    a_n,    a);
    LLVMValueRef ib = BCAST(b_p,    b_n,    b);
#undef BCAST

    LLVMValueRef vc = LLVMBuildLoad2(bld, f32,
        LLVMBuildGEP2(bld, f32, cond_p, &ic, 1, "pc"), "vc");
    LLVMValueRef va = LLVMBuildLoad2(bld, f32,
        LLVMBuildGEP2(bld, f32, a_p,    &ia, 1, "pa"), "va");
    LLVMValueRef vb = LLVMBuildLoad2(bld, f32,
        LLVMBuildGEP2(bld, f32, b_p,    &ib, 1, "pb"), "vb");

    LLVMValueRef is_true = LLVMBuildFCmp(bld, LLVMRealONE, vc, zf, "it");
    LLVMValueRef result  = LLVMBuildSelect(bld, is_true, va, vb, "r");

    LLVMBuildStore(bld, result,
        LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout"));
    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);
    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * Gather: out[i] = input[i*C + (int)indices[i]]
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_gather_op(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, ptr, i64, i64 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 5, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 3);

    LLVMValueRef input   = LLVMGetParam(fn, 0);
    LLVMValueRef indices = LLVMGetParam(fn, 1);
    LLVMValueRef out     = LLVMGetParam(fn, 2);
    LLVMValueRef N       = LLVMGetParam(fn, 3);
    LLVMValueRef C       = LLVMGetParam(fn, 4);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, N, "gather");

    LLVMValueRef idx_f  = LLVMBuildLoad2(bld, f32,
        LLVMBuildGEP2(bld, f32, indices, &loop.i, 1, "pidx"), "idxf");
    LLVMValueRef idx    = LLVMBuildFPToSI(bld, idx_f, i64, "idx");
    LLVMValueRef offset = LLVMBuildAdd(bld, LLVMBuildMul(bld, loop.i, C, "row"), idx, "off");
    LLVMValueRef val    = LLVMBuildLoad2(bld, f32,
        LLVMBuildGEP2(bld, f32, input, &offset, 1, "pin"), "v");
    LLVMBuildStore(bld, val,
        LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout"));

    close_loop(bld, &loop, entry);
    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);
    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * 2D permute (transpose): out[j*M+i] = in[i*N+j]
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_permute_2d(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, i64, i64 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 4, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 2);

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

    LLVMValueRef zero = LLVMConstInt(i64, 0, 0);
    LLVMValueRef one  = LLVMConstInt(i64, 1, 0);

    LLVMPositionBuilderAtEnd(bld, entry);
    LLVMBuildBr(bld, i_hdr);

    LLVMPositionBuilderAtEnd(bld, i_hdr);
    LLVMValueRef iv = LLVMBuildPhi(bld, i64, "i");
    LLVMBuildCondBr(bld, LLVMBuildICmp(bld, LLVMIntULT, iv, M, "ic"), j_hdr, i_exit);

    LLVMPositionBuilderAtEnd(bld, j_hdr);
    LLVMValueRef jv = LLVMBuildPhi(bld, i64, "j");
    LLVMBuildCondBr(bld, LLVMBuildICmp(bld, LLVMIntULT, jv, N, "jc"), j_body, j_exit);

    LLVMPositionBuilderAtEnd(bld, j_body);
    LLVMValueRef in_off  = LLVMBuildAdd(bld, LLVMBuildMul(bld, iv, N, "iN"), jv, "io");
    LLVMValueRef out_off = LLVMBuildAdd(bld, LLVMBuildMul(bld, jv, M, "jM"), iv, "oo");
    LLVMValueRef vv = LLVMBuildLoad2(bld, f32,
        LLVMBuildGEP2(bld, f32, in_p, &in_off, 1, "pin"), "v");
    LLVMBuildStore(bld, vv, LLVMBuildGEP2(bld, f32, out, &out_off, 1, "pout"));
    LLVMValueRef jn = LLVMBuildAdd(bld, jv, one, "j.next");
    LLVMBuildBr(bld, j_hdr);

    LLVMPositionBuilderAtEnd(bld, j_exit);
    LLVMValueRef in2 = LLVMBuildAdd(bld, iv, one, "i.next");
    LLVMBuildBr(bld, i_hdr);

    LLVMPositionBuilderAtEnd(bld, i_exit);
    LLVMBuildRetVoid(bld);

    LLVMValueRef jv_in[] = { zero, jn };  LLVMBasicBlockRef jb[] = { i_hdr, j_body };
    LLVMAddIncoming(jv, jv_in, jb, 2);
    LLVMValueRef iv_in[] = { zero, in2 }; LLVMBasicBlockRef ib[] = { entry, j_exit };
    LLVMAddIncoming(iv, iv_in, ib, 2);

    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * Expand (broadcast): out[i] = in[i % in_n]
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_expand_op(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, i64, i64 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 4, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 2);

    LLVMValueRef in_p  = LLVMGetParam(fn, 0);
    LLVMValueRef out   = LLVMGetParam(fn, 1);
    LLVMValueRef out_n = LLVMGetParam(fn, 2);
    LLVMValueRef in_n  = LLVMGetParam(fn, 3);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "expand");
    LLVMValueRef idx = LLVMBuildURem(bld, loop.i, in_n, "idx");
    LLVMValueRef v   = LLVMBuildLoad2(bld, f32,
        LLVMBuildGEP2(bld, f32, in_p, &idx, 1, "pin"), "v");
    LLVMBuildStore(bld, v, LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout"));
    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);
    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * Reshape: memcpy loop
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_reshape_op(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, i64 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 3, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 2);

    LLVMValueRef in_p = LLVMGetParam(fn, 0);
    LLVMValueRef out  = LLVMGetParam(fn, 1);
    LLVMValueRef n    = LLVMGetParam(fn, 2);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, n, "copy");
    LLVMValueRef v = LLVMBuildLoad2(bld, f32,
        LLVMBuildGEP2(bld, f32, in_p, &loop.i, 1, "pin"), "v");
    LLVMBuildStore(bld, v, LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout"));
    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);
    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * Matmul: C[m,n] = Σ_k A[m,k]*B[k,n]
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_matmul_kernel(LLVMContextRef ctx, const char* fn_name) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, ptr, i64, i64, i64 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 6, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 3);

    LLVMValueRef A = LLVMGetParam(fn, 0);
    LLVMValueRef B = LLVMGetParam(fn, 1);
    LLVMValueRef C = LLVMGetParam(fn, 2);
    LLVMValueRef M = LLVMGetParam(fn, 3);
    LLVMValueRef N = LLVMGetParam(fn, 4);
    LLVMValueRef K = LLVMGetParam(fn, 5);

    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);

    LLVMBasicBlockRef entry  = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBasicBlockRef m_hdr  = LLVMAppendBasicBlockInContext(ctx, fn, "m.hdr");
    LLVMBasicBlockRef n_hdr  = LLVMAppendBasicBlockInContext(ctx, fn, "n.hdr");
    LLVMBasicBlockRef k_hdr  = LLVMAppendBasicBlockInContext(ctx, fn, "k.hdr");
    LLVMBasicBlockRef k_body = LLVMAppendBasicBlockInContext(ctx, fn, "k.body");
    LLVMBasicBlockRef k_exit = LLVMAppendBasicBlockInContext(ctx, fn, "k.exit");
    LLVMBasicBlockRef n_exit = LLVMAppendBasicBlockInContext(ctx, fn, "n.exit");
    LLVMBasicBlockRef m_exit = LLVMAppendBasicBlockInContext(ctx, fn, "m.exit");

    LLVMValueRef z64 = LLVMConstInt(i64, 0, 0);
    LLVMValueRef o64 = LLVMConstInt(i64, 1, 0);
    LLVMValueRef zf  = LLVMConstReal(f32, 0.0);

    LLVMPositionBuilderAtEnd(bld, entry);  LLVMBuildBr(bld, m_hdr);

    LLVMPositionBuilderAtEnd(bld, m_hdr);
    LLVMValueRef mi = LLVMBuildPhi(bld, i64, "m");
    LLVMBuildCondBr(bld, LLVMBuildICmp(bld, LLVMIntULT, mi, M, "mc"), n_hdr, m_exit);

    LLVMPositionBuilderAtEnd(bld, n_hdr);
    LLVMValueRef ni = LLVMBuildPhi(bld, i64, "n");
    LLVMBuildCondBr(bld, LLVMBuildICmp(bld, LLVMIntULT, ni, N, "nc"), k_hdr, n_exit);

    LLVMPositionBuilderAtEnd(bld, k_hdr);
    LLVMValueRef ki  = LLVMBuildPhi(bld, i64, "k");
    LLVMValueRef acc = LLVMBuildPhi(bld, f32, "acc");
    LLVMBuildCondBr(bld, LLVMBuildICmp(bld, LLVMIntULT, ki, K, "kc"), k_body, k_exit);

    LLVMPositionBuilderAtEnd(bld, k_body);
    LLVMValueRef mK    = LLVMBuildAdd(bld, LLVMBuildMul(bld, mi, K, "mK"), ki, "mKk");
    LLVMValueRef av    = LLVMBuildLoad2(bld, f32,
        LLVMBuildGEP2(bld, f32, A, &mK, 1, "Ap"), "av");
    LLVMValueRef kN    = LLVMBuildAdd(bld, LLVMBuildMul(bld, ki, N, "kN"), ni, "kNn");
    LLVMValueRef bv    = LLVMBuildLoad2(bld, f32,
        LLVMBuildGEP2(bld, f32, B, &kN, 1, "Bp"), "bv");
    LLVMValueRef nacc  = LLVMBuildFAdd(bld, acc, LLVMBuildFMul(bld, av, bv, "p"), "na");
    LLVMValueRef kn    = LLVMBuildAdd(bld, ki, o64, "k.next");
    LLVMBuildBr(bld, k_hdr);

    LLVMPositionBuilderAtEnd(bld, k_exit);
    LLVMValueRef mN  = LLVMBuildAdd(bld, LLVMBuildMul(bld, mi, N, "mN"), ni, "mNn");
    LLVMBuildStore(bld, acc, LLVMBuildGEP2(bld, f32, C, &mN, 1, "Cp"));
    LLVMValueRef nn  = LLVMBuildAdd(bld, ni, o64, "n.next");
    LLVMBuildBr(bld, n_hdr);

    LLVMPositionBuilderAtEnd(bld, n_exit);
    LLVMValueRef mn2 = LLVMBuildAdd(bld, mi, o64, "m.next");
    LLVMBuildBr(bld, m_hdr);

    LLVMPositionBuilderAtEnd(bld, m_exit);
    LLVMBuildRetVoid(bld);

    LLVMValueRef kv[]  = {z64, kn};  LLVMBasicBlockRef kb[] = {n_hdr, k_body};
    LLVMAddIncoming(ki,  kv, kb, 2);
    LLVMValueRef av2[] = {zf,  nacc}; LLVMAddIncoming(acc, av2, kb, 2);
    LLVMValueRef nv[]  = {z64, nn};  LLVMBasicBlockRef nb[] = {m_hdr, k_exit};
    LLVMAddIncoming(ni,  nv, nb, 2);
    LLVMValueRef mv[]  = {z64, mn2}; LLVMBasicBlockRef mb[] = {entry, n_exit};
    LLVMAddIncoming(mi,  mv, mb, 2);

    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * Compile + add to persistent JIT; return callable function pointer.
 * ctx must be backend->ctx (shared). ORC borrows but doesn't own it here.
 * ---------------------------------------------------------------------- */
static kernel_fn_t compile_and_lookup(CMLLLVMBackend* backend,
                                      LLVMModuleRef mod,
                                      const char* fn_name) {
    char* err = NULL;
    if (LLVMVerifyModule(mod, LLVMReturnStatusAction, &err) != 0) {
        LOG_ERROR("LLVM: Module verification failed: %s", err ? err : "?");
        LLVMDisposeMessage(err);
        LLVMDisposeModule(mod);
        return NULL;
    }
    LLVMDisposeMessage(err);

    LLVMPassBuilderOptionsRef opts = LLVMCreatePassBuilderOptions();
    LLVMPassBuilderOptionsSetLoopVectorization(opts, 1);
    LLVMPassBuilderOptionsSetSLPVectorization(opts, 1);
    LLVMPassBuilderOptionsSetLoopUnrolling(opts, 1);
    LLVMErrorRef opt_err = LLVMRunPasses(mod, "default<O3>", backend->tm, opts);
    LLVMDisposePassBuilderOptions(opts);
    if (opt_err) {
        char* msg = LLVMGetErrorMessage(opt_err);
        LOG_WARNING("LLVM: O3 pass failed (%s), continuing", msg);
        LLVMDisposeErrorMessage(msg);
    }

#ifdef CML_PROCESS_REPLAY_LLVM
    {
        char* ir = LLVMPrintModuleToString(mod);
        if (ir) { cml_process_replay_record(fn_name, ir, strlen(ir)); LLVMDisposeMessage(ir); }
    }
#endif

    /* Wrap module for ORC — TSC wraps (doesn't own) backend->ctx. */
    LLVMOrcThreadSafeContextRef tsc =
        LLVMOrcCreateNewThreadSafeContextFromLLVMContext(backend->ctx);
    LLVMOrcThreadSafeModuleRef tsm = LLVMOrcCreateNewThreadSafeModule(mod, tsc);
    LLVMOrcDisposeThreadSafeContext(tsc); /* safe: TSM keeps module alive */

    LLVMOrcJITDylibRef jd = LLVMOrcLLJITGetMainJITDylib(backend->jit);
    LLVMErrorRef add_err  = LLVMOrcLLJITAddLLVMIRModule(backend->jit, jd, tsm);
    if (add_err) {
        char* msg = LLVMGetErrorMessage(add_err);
        LOG_ERROR("LLVM: AddLLVMIRModule failed: %s", msg);
        LLVMDisposeErrorMessage(msg);
        return NULL;
    }

    LLVMOrcExecutorAddress addr = 0;
    LLVMErrorRef lkp_err = LLVMOrcLLJITLookup(backend->jit, &addr, fn_name);
    if (lkp_err) {
        char* msg = LLVMGetErrorMessage(lkp_err);
        LOG_ERROR("LLVM: Lookup '%s' failed: %s", fn_name, msg);
        LLVMDisposeErrorMessage(msg);
        return NULL;
    }

    return (kernel_fn_t)(uintptr_t)addr;
}

/* -------------------------------------------------------------------------
 * Op classification
 * ---------------------------------------------------------------------- */
static bool is_binary_op(UOpType t) {
    return t == UOP_ADD || t == UOP_SUB || t == UOP_MUL || t == UOP_DIV ||
           t == UOP_MAX || t == UOP_CMPLT || t == UOP_POW;
}

static bool is_unary_op(UOpType t) {
    switch (t) {
    case UOP_NEG: case UOP_EXP: case UOP_EXP2: case UOP_LOG: case UOP_LOG2:
    case UOP_SQRT: case UOP_RSQRT: case UOP_SQUARE:
    case UOP_ABS: case UOP_SIGN:
    case UOP_SIN: case UOP_COS: case UOP_TAN:
    case UOP_ASIN: case UOP_ACOS: case UOP_ATAN:
    case UOP_TANH: case UOP_SIGMOID: case UOP_RECIP:
    case UOP_RELU: case UOP_RELU6: case UOP_SILU: case UOP_QUICK_GELU:
    case UOP_FLOOR: case UOP_CEIL: case UOP_ROUND: case UOP_ERF:
        return true;
    default:
        return false;
    }
}

static bool is_reduction(UOpType t) {
    return t == UOP_SUM || t == UOP_MEAN || t == UOP_MAX_REDUCE;
}

/* -------------------------------------------------------------------------
 * Per-node JIT execution
 * ---------------------------------------------------------------------- */
static int llvm_execute_node(CMLLLVMBackend* backend, struct IRNode* node) {
    if (!node || !node->output) return -1;

    Tensor* out = node->output;
    if (!out->data && out->numel > 0) {
        out->data = cml_buffer_cache_alloc(out->numel * sizeof(float));
        if (!out->data) { LOG_ERROR("LLVM: OOM for output tensor"); return -1; }
        out->owns_data = true;
    }

    UOpType type = node->type;

    /* Ops that still go to the CPU scalar path. */
    if (type == UOP_CONV2D || type == UOP_STRIDE || type == UOP_SLICE)
        return cpu_execute_node(node);

    /* Prefer BLAS for matmul when available. */
    if (type == UOP_MATMUL) {
        extern CMLBlasContext* get_blas_context(void);
        CMLBlasContext* blas = get_blas_context();
        if (blas && blas->initialized)
            return cpu_execute_node(node);
    }

    /* ---- Kernel cache lookup ----------------------------------------- */
    unsigned cache_idx = (unsigned)type % OP_CACHE_SIZE;
    kernel_fn_t fn = backend->op_cache[cache_idx];

    if (!fn) {
        char fn_name[64];
        snprintf(fn_name, sizeof(fn_name), "cml_k%d", backend->kernel_count++);

        LLVMModuleRef mod = NULL;

        if (is_binary_op(type)) {
            mod = build_binary_op(backend->ctx, type, fn_name);
        } else if (is_unary_op(type)) {
            mod = build_unary_op(backend->ctx, type, fn_name);
        } else if (is_reduction(type)) {
            mod = build_reduction(backend->ctx, type, fn_name);
        } else if (type == UOP_MATMUL) {
            mod = build_matmul_kernel(backend->ctx, fn_name);
        } else if (type == UOP_FILL) {
            mod = build_fill_op(backend->ctx, fn_name);
        } else if (type == UOP_WHERE) {
            mod = build_where_op(backend->ctx, fn_name);
        } else if (type == UOP_GATHER) {
            mod = build_gather_op(backend->ctx, fn_name);
        } else if (type == UOP_PERMUTE) {
            if (node->num_inputs >= 1 && node->inputs[0] && node->inputs[0]->ndim == 2)
                mod = build_permute_2d(backend->ctx, fn_name);
            else
                return cpu_execute_node(node);
        } else if (type == UOP_RESHAPE) {
            mod = build_reshape_op(backend->ctx, fn_name);
        } else if (type == UOP_EXPAND) {
            mod = build_expand_op(backend->ctx, fn_name);
        } else {
            LOG_DEBUG("LLVM: Unsupported op %d, CPU fallback", type);
            return cpu_execute_node(node);
        }

        if (!mod) return cpu_execute_node(node);

        fn = compile_and_lookup(backend, mod, fn_name);
        if (!fn) return cpu_execute_node(node);

        backend->op_cache[cache_idx] = fn;
        LOG_DEBUG("LLVM: Compiled and cached kernel for op %d ('%s')", type, fn_name);
    }

    /* ---- Dispatch ----------------------------------------------------- */
    if (is_binary_op(type)) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data)
            return cpu_execute_node(node);
        typedef void (*bfn_t)(float*, float*, float*, int64_t, int64_t, int64_t);
        ((bfn_t)(void*)fn)(
            (float*)node->inputs[0]->data,
            (float*)node->inputs[1]->data,
            (float*)out->data,
            (int64_t)out->numel,
            (int64_t)node->inputs[0]->numel,
            (int64_t)node->inputs[1]->numel);

    } else if (is_unary_op(type)) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            return cpu_execute_node(node);
        typedef void (*ufn_t)(float*, float*, int64_t, int64_t);
        ((ufn_t)(void*)fn)(
            (float*)node->inputs[0]->data,
            (float*)out->data,
            (int64_t)out->numel,
            (int64_t)node->inputs[0]->numel);

    } else if (is_reduction(type)) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            return cpu_execute_node(node);
        typedef void (*rfn_t)(float*, float*, int64_t);
        ((rfn_t)(void*)fn)(
            (float*)node->inputs[0]->data,
            (float*)out->data,
            (int64_t)node->inputs[0]->numel);

    } else if (type == UOP_MATMUL) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data)
            return cpu_execute_node(node);
        Tensor* a = node->inputs[0]; Tensor* b = node->inputs[1];
        if (a->ndim < 2 || b->ndim < 2) return cpu_execute_node(node);
        typedef void (*mfn_t)(float*, float*, float*, int64_t, int64_t, int64_t);
        ((mfn_t)(void*)fn)(
            (float*)a->data, (float*)b->data, (float*)out->data,
            (int64_t)a->shape[a->ndim-2],
            (int64_t)b->shape[b->ndim-1],
            (int64_t)a->shape[a->ndim-1]);

    } else if (type == UOP_FILL) {
        FillParams* p = (FillParams*)node->params;
        float fv = p ? p->value : 0.0f;
        typedef void (*ffn_t)(float*, int64_t, float);
        ((ffn_t)(void*)fn)((float*)out->data, (int64_t)out->numel, fv);

    } else if (type == UOP_WHERE) {
        if (node->num_inputs < 3 || !node->inputs[0]->data ||
            !node->inputs[1]->data || !node->inputs[2]->data)
            return cpu_execute_node(node);
        typedef void (*wfn_t)(float*, float*, float*, float*,
                              int64_t, int64_t, int64_t, int64_t);
        ((wfn_t)(void*)fn)(
            (float*)node->inputs[0]->data,
            (float*)node->inputs[1]->data,
            (float*)node->inputs[2]->data,
            (float*)out->data,
            (int64_t)out->numel,
            (int64_t)node->inputs[0]->numel,
            (int64_t)node->inputs[1]->numel,
            (int64_t)node->inputs[2]->numel);

    } else if (type == UOP_GATHER) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data)
            return cpu_execute_node(node);
        Tensor* inp = node->inputs[0];
        if (inp->ndim < 2) return cpu_execute_node(node);
        typedef void (*gfn_t)(float*, float*, float*, int64_t, int64_t);
        ((gfn_t)(void*)fn)(
            (float*)inp->data, (float*)node->inputs[1]->data,
            (float*)out->data,
            (int64_t)out->numel,
            (int64_t)inp->shape[inp->ndim-1]);

    } else if (type == UOP_PERMUTE) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            return cpu_execute_node(node);
        Tensor* inp = node->inputs[0];
        typedef void (*pfn_t)(float*, float*, int64_t, int64_t);
        ((pfn_t)(void*)fn)(
            (float*)inp->data, (float*)out->data,
            (int64_t)inp->shape[0], (int64_t)inp->shape[1]);

    } else if (type == UOP_RESHAPE || type == UOP_EXPAND) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            return cpu_execute_node(node);
        typedef void (*cfn_t)(float*, float*, int64_t, int64_t);
        ((cfn_t)(void*)fn)(
            (float*)node->inputs[0]->data,
            (float*)out->data,
            (int64_t)out->numel,
            (int64_t)node->inputs[0]->numel);
    }

    node->is_executed  = true;
    out->is_executed   = true;
    return 0;
}

/* -------------------------------------------------------------------------
 * Public graph execution
 * ---------------------------------------------------------------------- */
int cml_llvm_execute(CMLLLVMBackend* backend, CMLGraph_t ir) {
    if (!backend || !ir) return -1;
    struct IRNode* node = ir->head;
    while (node) {
        if (!node->is_executed) {
            if (llvm_execute_node(backend, node) != 0) {
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

#endif /* CML_HAS_LLVM_BACKEND */
