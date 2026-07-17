
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
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include "alloc/cml_allocator.h"

/* Kernel cache: keyed by (op type + concrete shape) so every kernel is
 * shape-specialized — loop bounds and broadcast patterns are baked in as
 * compile-time constants and LLVM emits width-specific SIMD for each shape.
 * Open-addressed with a short linear-probe window; evict on a full window. */
#define OP_CACHE_SIZE 1024
#define OP_CACHE_PROBE 8

typedef void (*kernel_fn_t)(void);

typedef struct {
    uint64_t     key;   /* 0 == empty slot */
    kernel_fn_t  fn;
} KernelCacheSlot;

struct CMLLLVMBackend {
    LLVMOrcLLJITRef     jit;          /* persistent; lives until destroy    */
    LLVMTargetMachineRef tm;
    /* No shared ctx: each kernel gets its own LLVMContextCreate() so the
     * context lifetime is tied to the ORC TSC/TSM, not to the backend.
     * LLVMOrcCreateNewThreadSafeContextFromLLVMContext transfers ownership,
     * so a shared backend->ctx would become dangling after the first kernel. */
    bool                 initialized;
    int                  kernel_count; /* for unique symbol names            */
    KernelCacheSlot      op_cache[OP_CACHE_SIZE]; /* keyed by shape signature */
};

/* FNV-1a over (type, up to four shape dims). key 0 is reserved for "empty". */
static uint64_t shape_key(UOpType type, int64_t a, int64_t b,
                          int64_t c, int64_t d) {
    uint64_t h = 1469598103934665603ULL;
    uint64_t v[5] = { (uint64_t)type, (uint64_t)a, (uint64_t)b,
                      (uint64_t)c, (uint64_t)d };
    for (int i = 0; i < 5; i++) { h ^= v[i]; h *= 1099511628211ULL; }
    return h ? h : 1;
}

/* Find the cached fn for key, or the slot index to insert into.
 * Returns the fn (non-NULL) on hit; on miss returns NULL and sets *slot to the
 * slot to populate (an empty slot within the probe window, else evict home). */
static kernel_fn_t cache_lookup(struct CMLLLVMBackend* b, uint64_t key,
                                unsigned* slot) {
    unsigned home = (unsigned)(key % OP_CACHE_SIZE);
    for (unsigned p = 0; p < OP_CACHE_PROBE; p++) {
        unsigned idx = (home + p) % OP_CACHE_SIZE;
        if (b->op_cache[idx].key == key) { *slot = idx; return b->op_cache[idx].fn; }
        if (b->op_cache[idx].key == 0)   { *slot = idx; return NULL; }
    }
    *slot = home; /* window full: evict the home slot (JIT'd code stays owned by ORC) */
    return NULL;
}

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
        cml_free(b);
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
    cml_free(backend);
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

/* Element LLVM type for a tensor dtype the JIT emits kernels for (f32/f64/int). */
static LLVMTypeRef elem_type(LLVMContextRef ctx, DType dt) {
    switch (dt) {
    case DTYPE_FLOAT64: return LLVMDoubleTypeInContext(ctx);
    case DTYPE_INT64:   return LLVMInt64TypeInContext(ctx);
    case DTYPE_INT32:   return LLVMInt32TypeInContext(ctx);
    default:            return LLVMFloatTypeInContext(ctx);
    }
}
static int dtype_is_int_jit(DType dt) {
    return dt == DTYPE_INT32 || dt == DTYPE_INT64;
}

/* Resolve an input's element index at loop position i for a shape known at
 * codegen time.  Broadcasting collapses to a constant here: in_n == out_n is a
 * straight index (no arithmetic), in_n == 1 is a constant-0 splat, and only the
 * genuinely-broadcast case emits a urem.  This is what lets LLVM vectorize the
 * common contiguous/scalar cases with no runtime broadcast branch. */
static LLVMValueRef bcast_index(LLVMBuilderRef bld, LLVMContextRef ctx,
                                LLVMValueRef i, int64_t in_n, int64_t out_n) {
    LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx);
    if (in_n == out_n || in_n <= 0) return i;
    if (in_n == 1) return LLVMConstInt(i64, 0, 0);
    return LLVMBuildURem(bld, i, LLVMConstInt(i64, (unsigned long long)in_n, 0), "bidx");
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
 * Shape-specialized: out_n/in0_n/in1_n are baked in as compile-time constants
 * so the trip count is fixed and broadcasting is resolved at codegen time.
 * Signature: void(ptr in0, ptr in1, ptr out, i64, i64, i64) — the three size
 * params are retained for ABI stability but unused (the shapes are constants).
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_binary_op(LLVMContextRef ctx, UOpType type,
                                     const char* fn_name, int64_t out_numel,
                                     int64_t in0_numel, int64_t in1_numel, DType dt) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = elem_type(ctx, dt);  /* element type (f32 or f64) */
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
    LLVMValueRef out_n = LLVMConstInt(i64, (unsigned long long)out_numel, 0);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "elem");

    LLVMValueRef i0 = bcast_index(bld, ctx, loop.i, in0_numel, out_numel);
    LLVMValueRef i1 = bcast_index(bld, ctx, loop.i, in1_numel, out_numel);

    LLVMValueRef gep0 = LLVMBuildGEP2(bld, f32, in0, &i0, 1, "p0");
    LLVMValueRef gep1 = LLVMBuildGEP2(bld, f32, in1, &i1, 1, "p1");
    LLVMValueRef v0   = LLVMBuildLoad2(bld, f32, gep0, "v0");
    LLVMValueRef v1   = LLVMBuildLoad2(bld, f32, gep1, "v1");

    LLVMValueRef result = NULL;
    if (dtype_is_int_jit(dt)) {
        /* Integer arithmetic (i32/i64). Only add/sub/mul/div/max are routed to
         * the JIT for integer dtypes (see the dispatch guard). */
        switch (type) {
        case UOP_ADD: result = LLVMBuildAdd(bld, v0, v1, "r"); break;
        case UOP_SUB: result = LLVMBuildSub(bld, v0, v1, "r"); break;
        case UOP_MUL: result = LLVMBuildMul(bld, v0, v1, "r"); break;
        case UOP_DIV: {
            /* guard divide-by-zero: y==0 ? 0 : x/y (matches interpreter) */
            LLVMValueRef zero = LLVMConstInt(f32, 0, 0);
            LLVMValueRef isz  = LLVMBuildICmp(bld, LLVMIntEQ, v1, zero, "isz");
            LLVMValueRef safe = LLVMBuildSelect(bld, isz, LLVMConstInt(f32, 1, 0), v1, "safe");
            LLVMValueRef q    = LLVMBuildSDiv(bld, v0, safe, "q");
            result = LLVMBuildSelect(bld, isz, zero, q, "r");
            break;
        }
        case UOP_MAX: {
            LLVMValueRef cmp = LLVMBuildICmp(bld, LLVMIntSGT, v0, v1, "gt");
            result = LLVMBuildSelect(bld, cmp, v0, v1, "r");
            break;
        }
        default: result = LLVMBuildAdd(bld, v0, v1, "r"); break;
        }
    } else
    switch (type) {
    case UOP_ADD:  result = LLVMBuildFAdd(bld, v0, v1, "r"); break;
    case UOP_SUB:  result = LLVMBuildFSub(bld, v0, v1, "r"); break;
    case UOP_MUL:  result = LLVMBuildFMul(bld, v0, v1, "r"); break;
    case UOP_DIV:
        result = LLVMBuildFDiv(bld, v0, v1, "r");   /* IEEE: /0 = inf */
        break;
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
                                    const char* fn_name, int64_t out_numel,
                                    int64_t in_numel, DType dt) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = elem_type(ctx, dt);  /* element type (f32 or f64) */
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr, ptr, i64, i64 };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 4, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);
    add_noalias(ctx, fn, 2); /* in, out are noalias */

    LLVMValueRef in_p  = LLVMGetParam(fn, 0);
    LLVMValueRef out   = LLVMGetParam(fn, 1);
    LLVMValueRef out_n = LLVMConstInt(i64, (unsigned long long)out_numel, 0);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "elem");

    LLVMValueRef idx    = bcast_index(bld, ctx, loop.i, in_numel, out_numel);
    LLVMValueRef gep_in = LLVMBuildGEP2(bld, f32, in_p, &idx, 1, "pin");
    LLVMValueRef val    = LLVMBuildLoad2(bld, f32, gep_in, "val");

    LLVMValueRef result = NULL;
    if (dtype_is_int_jit(dt)) {
        /* Integer unary (i32/i64): only neg/abs/square are routed here. */
        LLVMValueRef zero = LLVMConstInt(f32, 0, 0);
        switch (type) {
        case UOP_NEG:    result = LLVMBuildSub(bld, zero, val, "r"); break;
        case UOP_SQUARE: result = LLVMBuildMul(bld, val, val, "r"); break;
        case UOP_ABS: {
            LLVMValueRef neg = LLVMBuildSub(bld, zero, val, "neg");
            LLVMValueRef isn = LLVMBuildICmp(bld, LLVMIntSLT, val, zero, "isn");
            result = LLVMBuildSelect(bld, isn, neg, val, "r");
            break;
        }
        default: result = val; break;
        }
        goto store_result;
    }

    LLVMValueRef zero_f = LLVMConstReal(f32, 0.0);
    LLVMValueRef one_f  = LLVMConstReal(f32, 1.0);

    switch (type) {
    case UOP_NEG:
        result = LLVMBuildFNeg(bld, val, "r");
        break;

    case UOP_ABS:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.fabs", 9, f32), val, "r");
        break;

    case UOP_SQRT:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.sqrt", 9, f32), val, "r");
        break;

    case UOP_RSQRT: {
        LLVMValueRef sq  = call1(bld, f32, INTR1(mod, ctx, "llvm.sqrt", 9, f32), val, "sq");
        result = LLVMBuildFDiv(bld, one_f, sq, "r");
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

    case UOP_LOG:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.log", 8, f32), val, "r");
        break;

    case UOP_LOG2:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.log2", 9, f32), val, "r");
        break;

    case UOP_SIN:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.sin", 8, f32), val, "r");
        break;

    case UOP_COS:
        result = call1(bld, f32, INTR1(mod, ctx, "llvm.cos", 8, f32), val, "r");
        break;

    case UOP_TAN: {
        LLVMValueRef s  = call1(bld, f32, INTR1(mod, ctx, "llvm.sin", 8, f32), val, "s");
        LLVMValueRef c  = call1(bld, f32, INTR1(mod, ctx, "llvm.cos", 8, f32), val, "c");
        result = LLVMBuildFDiv(bld, s, c, "r");
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
        result = LLVMBuildFDiv(bld, one_f, val, "r");   /* IEEE: 1/0 = inf */
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

store_result:;
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
                                     const char* fn_name, int64_t n_elems) {
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
    LLVMValueRef n     = LLVMConstInt(i64, (unsigned long long)n_elems, 0);

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
 * Per-axis reduction: out[p] = reduce over `count` elements at stride `inner`
 * starting at base(p) = (p/inner)*count*inner + (p%inner), for p in [0, nout).
 * This is the same (outer,inner,count) formulation the interpreter uses; all
 * three are baked as constants. Same signature as the global reducer (the i64
 * param is ignored) so the dispatch call-through is uniform.
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_reduction_axis(LLVMContextRef ctx, UOpType type,
                                          const char* fn_name, int64_t outer,
                                          int64_t inner, int64_t count) {
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

    LLVMValueRef c_nout   = LLVMConstInt(i64, (unsigned long long)(outer * inner), 0);
    LLVMValueRef c_inner  = LLVMConstInt(i64, (unsigned long long)inner, 0);
    LLVMValueRef c_count  = LLVMConstInt(i64, (unsigned long long)count, 0);
    LLVMValueRef c_cinner = LLVMConstInt(i64, (unsigned long long)(count * inner), 0);
    LLVMValueRef z   = LLVMConstInt(i64, 0, 0);
    LLVMValueRef one = LLVMConstInt(i64, 1, 0);
    float init_val = (type == UOP_MAX_REDUCE) ? -3.402823466e+38f : 0.0f;
    LLVMValueRef init = LLVMConstReal(f32, (double)init_val);

    LLVMBasicBlockRef entry  = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBasicBlockRef p_hdr  = LLVMAppendBasicBlockInContext(ctx, fn, "p.hdr");
    LLVMBasicBlockRef p_body = LLVMAppendBasicBlockInContext(ctx, fn, "p.body");
    LLVMBasicBlockRef r_hdr  = LLVMAppendBasicBlockInContext(ctx, fn, "r.hdr");
    LLVMBasicBlockRef r_body = LLVMAppendBasicBlockInContext(ctx, fn, "r.body");
    LLVMBasicBlockRef r_done = LLVMAppendBasicBlockInContext(ctx, fn, "r.done");
    LLVMBasicBlockRef p_exit = LLVMAppendBasicBlockInContext(ctx, fn, "p.exit");

    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);
    LLVMBuildBr(bld, p_hdr);

    LLVMPositionBuilderAtEnd(bld, p_hdr);
    LLVMValueRef p = LLVMBuildPhi(bld, i64, "p");
    LLVMBuildCondBr(bld, LLVMBuildICmp(bld, LLVMIntULT, p, c_nout, "pc"), p_body, p_exit);

    LLVMPositionBuilderAtEnd(bld, p_body);
    LLVMValueRef base = LLVMBuildAdd(bld,
        LLVMBuildMul(bld, LLVMBuildUDiv(bld, p, c_inner, "pd"), c_cinner, "b0"),
        LLVMBuildURem(bld, p, c_inner, "pm"), "base");
    LLVMBuildBr(bld, r_hdr);

    LLVMPositionBuilderAtEnd(bld, r_hdr);
    LLVMValueRef r   = LLVMBuildPhi(bld, i64, "r");
    LLVMValueRef acc = LLVMBuildPhi(bld, f32, "acc");
    LLVMBuildCondBr(bld, LLVMBuildICmp(bld, LLVMIntULT, r, c_count, "rc"), r_body, r_done);

    LLVMPositionBuilderAtEnd(bld, r_body);
    LLVMValueRef idx = LLVMBuildAdd(bld, base, LLVMBuildMul(bld, r, c_inner, "ri"), "idx");
    LLVMValueRef v   = LLVMBuildLoad2(bld, f32, LLVMBuildGEP2(bld, f32, in_p, &idx, 1, "gp"), "v");
    LLVMValueRef nacc;
    if (type == UOP_MAX_REDUCE) {
        LLVMValueRef c = LLVMBuildFCmp(bld, LLVMRealOGT, v, acc, "gt");
        nacc = LLVMBuildSelect(bld, c, v, acc, "mx");
    } else {
        nacc = LLVMBuildFAdd(bld, acc, v, "s");
    }
    LLVMValueRef rnext = LLVMBuildAdd(bld, r, one, "rn");
    LLVMBuildBr(bld, r_hdr);

    LLVMValueRef r_in[]   = { z, rnext };  LLVMBasicBlockRef rb[] = { p_body, r_body };
    LLVMAddIncoming(r, r_in, rb, 2);
    LLVMValueRef acc_in[] = { init, nacc }; LLVMAddIncoming(acc, acc_in, rb, 2);

    LLVMPositionBuilderAtEnd(bld, r_done);
    LLVMValueRef final_v = acc;
    if (type == UOP_MEAN) {
        LLVMValueRef cf = LLVMBuildUIToFP(bld, c_count, f32, "cf");
        final_v = LLVMBuildFDiv(bld, acc, cf, "mean");
    }
    LLVMBuildStore(bld, final_v, LLVMBuildGEP2(bld, f32, out_p, &p, 1, "op"));
    LLVMValueRef pnext = LLVMBuildAdd(bld, p, one, "pn");
    LLVMBuildBr(bld, p_hdr);

    LLVMValueRef p_in[] = { z, pnext }; LLVMBasicBlockRef pb[] = { entry, r_done };
    LLVMAddIncoming(p, p_in, pb, 2);

    LLVMPositionBuilderAtEnd(bld, p_exit);
    LLVMBuildRetVoid(bld);
    LLVMDisposeBuilder(bld);
    return mod;
}

/* -------------------------------------------------------------------------
 * Fill: out[i] = val  (val passed at runtime — allows caching)
 * Signature: void(ptr out, i64 n, float val)
 * ---------------------------------------------------------------------- */
static LLVMModuleRef build_fill_op(LLVMContextRef ctx, const char* fn_name,
                                   int64_t out_numel) {
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
    LLVMValueRef out_n = LLVMConstInt(i64, (unsigned long long)out_numel, 0);
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
static LLVMModuleRef build_where_op(LLVMContextRef ctx, const char* fn_name,
                                    int64_t out_numel, int64_t cond_numel,
                                    int64_t a_numel, int64_t b_numel) {
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
    LLVMValueRef out_n  = LLVMConstInt(i64, (unsigned long long)out_numel, 0);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "where");

    LLVMValueRef zf  = LLVMConstReal(f32, 0.0);

    LLVMValueRef ic = bcast_index(bld, ctx, loop.i, cond_numel, out_numel);
    LLVMValueRef ia = bcast_index(bld, ctx, loop.i, a_numel,    out_numel);
    LLVMValueRef ib = bcast_index(bld, ctx, loop.i, b_numel,    out_numel);

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
static LLVMModuleRef build_gather_op(LLVMContextRef ctx, const char* fn_name,
                                     int64_t n_rows, int64_t n_cols) {
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
    LLVMValueRef N       = LLVMConstInt(i64, (unsigned long long)n_rows, 0);
    LLVMValueRef C       = LLVMConstInt(i64, (unsigned long long)n_cols, 0);

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
static LLVMModuleRef build_permute_2d(LLVMContextRef ctx, const char* fn_name,
                                      int64_t rows, int64_t cols) {
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
    LLVMValueRef M    = LLVMConstInt(i64, (unsigned long long)rows, 0);
    LLVMValueRef N    = LLVMConstInt(i64, (unsigned long long)cols, 0);

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
static LLVMModuleRef build_expand_op(LLVMContextRef ctx, const char* fn_name,
                                     int64_t out_numel, int64_t in_numel) {
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
    LLVMValueRef out_n = LLVMConstInt(i64, (unsigned long long)out_numel, 0);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "expand");
    LLVMValueRef idx = bcast_index(bld, ctx, loop.i, in_numel, out_numel);
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
static LLVMModuleRef build_reshape_op(LLVMContextRef ctx, const char* fn_name,
                                      int64_t n_elems) {
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
    LLVMValueRef n    = LLVMConstInt(i64, (unsigned long long)n_elems, 0);

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
static LLVMModuleRef build_matmul_kernel(LLVMContextRef ctx, const char* fn_name,
                                         int64_t dim_m, int64_t dim_n,
                                         int64_t dim_k) {
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
    LLVMValueRef M = LLVMConstInt(i64, (unsigned long long)dim_m, 0);
    LLVMValueRef N = LLVMConstInt(i64, (unsigned long long)dim_n, 0);
    LLVMValueRef K = LLVMConstInt(i64, (unsigned long long)dim_k, 0);

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
 * mod was created in a fresh per-kernel context (LLVMContextCreate).
 * We extract that context via LLVMGetModuleContext and transfer its
 * ownership to the TSC; the JIT then owns TSM→TSC→context.
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

    /* Transfer the module's context to ORC.  LLVMGetModuleContext returns the
     * raw context that was passed to LLVMModuleCreateWithNameInContext when the
     * module was built; wrapping it here transfers ownership to the TSC.
     * The JIT then owns TSM → TSC → context, so there is no dangling pointer. */
    LLVMContextRef mod_ctx = LLVMGetModuleContext(mod);
    LLVMOrcThreadSafeContextRef tsc =
        LLVMOrcCreateNewThreadSafeContextFromLLVMContext(mod_ctx);
    LLVMOrcThreadSafeModuleRef tsm = LLVMOrcCreateNewThreadSafeModule(mod, tsc);
    LLVMOrcDisposeThreadSafeContext(tsc); /* drop our ref; TSM keeps context alive */

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
 * Fused elementwise chain: native JIT codegen
 *
 * Emits ONE function that evaluates the whole elementwise chain in a single
 * loop, keeping intermediates in SSA values (registers) — no intermediate
 * buffers. LLVM's O2 vectorizer turns the arithmetic steps into SIMD. This is
 * the true-codegen path for UOP_FUSED_ELEMENTWISE (the blocked C interpreter in
 * execution.c is the fallback when the JIT is unavailable).
 *   ABI: void kern(const float** in, float* out)
 *   out_numel and each input's numel are baked in as compile-time constants, so
 *   broadcasting resolves at codegen time exactly like the other kernels.
 * ---------------------------------------------------------------------- */
#define FE_UNUSED_REF (-1000000)   /* matches FUSED_UNUSED_REF in execution.c */

/* Emit the scalar result of one primitive elementwise op. Mirrors fused_eval_block
 * (execution.c) and the per-op emission in build_binary_op/build_unary_op. */
static LLVMValueRef fe_emit_op(LLVMBuilderRef bld, LLVMModuleRef mod, LLVMContextRef ctx,
                               LLVMTypeRef f32, UOpType op,
                               LLVMValueRef a, LLVMValueRef b, LLVMValueRef c, float konst) {
    LLVMValueRef zero = LLVMConstReal(f32, 0.0);
    switch (op) {
    case UOP_ADD:     return LLVMBuildFAdd(bld, a, b, "r");
    case UOP_SUB:     return LLVMBuildFSub(bld, a, b, "r");
    case UOP_MUL:     return LLVMBuildFMul(bld, a, b, "r");
    case UOP_DIV:     return LLVMBuildFDiv(bld, a, b, "r");
    case UOP_MAX:     { LLVMValueRef c1=LLVMBuildFCmp(bld,LLVMRealOGT,a,b,"gt"); return LLVMBuildSelect(bld,c1,a,b,"r"); }
    case UOP_MINIMUM: { LLVMValueRef c1=LLVMBuildFCmp(bld,LLVMRealOLT,a,b,"lt"); return LLVMBuildSelect(bld,c1,a,b,"r"); }
    case UOP_POW:     { LLVMValueRef f=INTR1(mod,ctx,"llvm.pow",8,f32);
                        LLVMTypeRef ft=LLVMFunctionType(f32,(LLVMTypeRef[]){f32,f32},2,0);
                        return LLVMBuildCall2(bld,ft,f,(LLVMValueRef[]){a,b},2,"r"); }
    case UOP_NEG:     return LLVMBuildFNeg(bld, a, "r");
    case UOP_RECIP:   return LLVMBuildFDiv(bld, LLVMConstReal(f32,1.0), a, "r");
    case UOP_EXP:     return call1(bld,f32,INTR1(mod,ctx,"llvm.exp",8,f32),a,"r");
    case UOP_LOG:     return call1(bld,f32,INTR1(mod,ctx,"llvm.log",8,f32),a,"r");
    case UOP_SQRT:    return call1(bld,f32,INTR1(mod,ctx,"llvm.sqrt",9,f32),a,"r");
    case UOP_SIN:     return call1(bld,f32,INTR1(mod,ctx,"llvm.sin",8,f32),a,"r");
    case UOP_COS:     return call1(bld,f32,INTR1(mod,ctx,"llvm.cos",8,f32),a,"r");
    case UOP_ABS:     return call1(bld,f32,INTR1(mod,ctx,"llvm.fabs",9,f32),a,"r");
    case UOP_CMPLT:   return LLVMBuildUIToFP(bld, LLVMBuildFCmp(bld,LLVMRealOLT,a,b,"c"), f32, "r");
    case UOP_CMPLE:   return LLVMBuildUIToFP(bld, LLVMBuildFCmp(bld,LLVMRealOLE,a,b,"c"), f32, "r");
    case UOP_CMPGT:   return LLVMBuildUIToFP(bld, LLVMBuildFCmp(bld,LLVMRealOGT,a,b,"c"), f32, "r");
    case UOP_CMPGE:   return LLVMBuildUIToFP(bld, LLVMBuildFCmp(bld,LLVMRealOGE,a,b,"c"), f32, "r");
    case UOP_CMPEQ:   return LLVMBuildUIToFP(bld, LLVMBuildFCmp(bld,LLVMRealOEQ,a,b,"c"), f32, "r");
    case UOP_CMPNE:   return LLVMBuildUIToFP(bld, LLVMBuildFCmp(bld,LLVMRealUNE,a,b,"c"), f32, "r");
    case UOP_WHERE:   { LLVMValueRef nz=LLVMBuildFCmp(bld,LLVMRealONE,a,zero,"nz"); return LLVMBuildSelect(bld,nz,b,c,"r"); }
    case UOP_FILL:    return LLVMConstReal(f32, (double)konst);
    default:          return zero;
    }
}

static LLVMModuleRef build_fused_elementwise(LLVMContextRef ctx, const char* fn_name,
        const FusedElementwiseParams* fp, int64_t out_numel,
        const int64_t* in_numel, int num_inputs) {
    LLVMModuleRef mod  = LLVMModuleCreateWithNameInContext(fn_name, ctx);
    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef ptr    = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    LLVMTypeRef params[] = { ptr /* const float** in */, ptr /* float* out */ };
    LLVMTypeRef fn_type  = LLVMFunctionType(void_t, params, 2, 0);
    LLVMValueRef fn      = LLVMAddFunction(mod, fn_name, fn_type);

    LLVMValueRef in_arr = LLVMGetParam(fn, 0);
    LLVMValueRef out    = LLVMGetParam(fn, 1);
    LLVMValueRef out_n  = LLVMConstInt(i64, (unsigned long long)out_numel, 0);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBuilderRef bld = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(bld, entry);

    /* Load each input base pointer once from the float** array (loop-invariant). */
    LLVMValueRef inptr[32];
    for (int k = 0; k < num_inputs && k < 32; k++) {
        LLVMValueRef idxk = LLVMConstInt(i64, (unsigned long long)k, 0);
        LLVMValueRef gepk = LLVMBuildGEP2(bld, ptr, in_arr, &idxk, 1, "pp");
        inptr[k] = LLVMBuildLoad2(bld, ptr, gepk, "inp");
    }

    LoopInfo loop = emit_loop(bld, ctx, fn, out_n, "fe");

    int ns = fp->num_steps;
    LLVMValueRef reg[256];
    for (int s = 0; s < ns; s++) {
        int refs[3] = { fp->a[s], fp->b[s], fp->c[s] };
        LLVMValueRef ops[3];
        for (int t = 0; t < 3; t++) {
            int ref = refs[t];
            if (ref == FE_UNUSED_REF)      { ops[t] = LLVMConstReal(f32, 0.0); continue; }
            if (ref < 0)                   { ops[t] = reg[-ref - 1]; continue; }
            if (ref >= num_inputs || ref >= 32) { ops[t] = LLVMConstReal(f32, 0.0); continue; }
            LLVMValueRef idx = bcast_index(bld, ctx, loop.i, in_numel[ref], out_numel);
            LLVMValueRef gep = LLVMBuildGEP2(bld, f32, inptr[ref], &idx, 1, "pin");
            ops[t] = LLVMBuildLoad2(bld, f32, gep, "v");
        }
        reg[s] = fe_emit_op(bld, mod, ctx, f32, fp->op[s], ops[0], ops[1], ops[2], fp->konst[s]);
    }

    LLVMValueRef gep_out = LLVMBuildGEP2(bld, f32, out, &loop.i, 1, "pout");
    LLVMBuildStore(bld, reg[ns - 1], gep_out);
    close_loop(bld, &loop, entry);

    LLVMPositionBuilderAtEnd(bld, loop.exit);
    LLVMBuildRetVoid(bld);
    LLVMDisposeBuilder(bld);
    return mod;
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

    /* UOP_EXPAND (broadcast) produces a view whose data ALIASES the smaller
     * source buffer (out->numel > input numel, owns_data==false).  Because
     * out->data is already non-NULL, the allocation below is skipped and a JIT
     * expand kernel would write out->numel elements into the small aliased
     * buffer — a heap overflow that silently corrupts adjacent memory (JIT code
     * is not sanitizer-instrumented).  The interpreter's UOP_EXPAND correctly
     * allocates a fresh full-size buffer and broadcasts, so defer to it. */
    if (node->type == UOP_EXPAND)
        return cpu_execute_node(node);

    if (!out->data && out->numel > 0) {
        /* Size by the actual dtype — f64/int kernels write 8 bytes/elem, not 4;
         * sizeof(float) under-allocated and the kernel overflowed its output. */
        out->data = cml_buffer_cache_alloc(out->numel * cml_dtype_size(out->dtype));
        if (!out->data) { LOG_ERROR("LLVM: OOM for output tensor"); return -1; }
        out->owns_data         = true;
        out->from_buffer_cache = true; /* mirror interpreter: route free to cml_buffer_cache_free */
    }

    UOpType type = node->type;

    /* The JIT emits typed kernels for float32 (all ops), float64 (elementwise
     * binary/unary), and int32/int64 (a restricted elementwise set: add/sub/mul/
     * div/max, neg/abs/square). Any other dtype/op, or mixed-dtype operands, go to
     * the interpreter's dtype-generic path. */
    DType edt = out->dtype;
    int f64_ok = (edt == DTYPE_FLOAT64) && (is_binary_op(type) || is_unary_op(type));
    int int_ok = (edt == DTYPE_INT32 || edt == DTYPE_INT64) &&
                 (type == UOP_ADD || type == UOP_SUB || type == UOP_MUL ||
                  type == UOP_DIV || type == UOP_MAX ||
                  type == UOP_NEG || type == UOP_ABS || type == UOP_SQUARE);
    if (edt != DTYPE_FLOAT32 && !f64_ok && !int_ok)
        return cpu_execute_node(node);
    for (int _i = 0; _i < node->num_inputs && node->inputs; _i++) {
        if (node->inputs[_i] && node->inputs[_i]->dtype != edt)
            return cpu_execute_node(node);
    }

    /* Ops that always go to the CPU interpreter (unsupported by the JIT). */
    if (type == UOP_CONV2D || type == UOP_STRIDE || type == UOP_SLICE)
        return cpu_execute_node(node);

    /* Prefer BLAS for matmul when available; also defer quantized-weight matmul
     * to the interpreter, which has the int8/GGUF dispatch (the JIT kernel would
     * misread the int8 weight buffer as float32). */
    if (type == UOP_MATMUL) {
        if (node->num_inputs >= 2 && node->inputs[1] &&
            node->inputs[1]->quant_type != CML_QUANT_NONE)
            return cpu_execute_node(node);
        /* Matmul with a fused epilogue (bias+activation): the JIT gemm kernel
         * doesn't apply the epilogue, so defer to the interpreter, which runs
         * the gemm (BLAS or naive) and then applies the epilogue in-place. */
        if (node->params)
            return cpu_execute_node(node);
        extern CMLBlasContext* get_blas_context(void);
        CMLBlasContext* blas = get_blas_context();
        if (blas && blas->initialized)
            return cpu_execute_node(node);
    }

    /* ---- Fused elementwise chain: native JIT codegen ----------------- *
     * Bespoke ABI (const float** in, float* out) and a content-hashed cache key
     * (the chain structure, not just a shape, distinguishes kernels), so it is
     * handled here rather than in the shape-keyed path below. */
    if (type == UOP_FUSED_ELEMENTWISE) {
        FusedElementwiseParams* fp = (FusedElementwiseParams*)node->params;
        if (!fp || fp->num_steps <= 0 || fp->num_steps > 256 ||
            node->num_inputs < 0 || node->num_inputs > 32)
            return cpu_execute_node(node);
        int64_t in_numel[32];
        for (int k = 0; k < node->num_inputs; k++) {
            Tensor* it = node->inputs[k];
            if (!it || !it->data || it->dtype != DTYPE_FLOAT32)
                return cpu_execute_node(node);
            in_numel[k] = (int64_t)it->numel;
        }
        int64_t out_numel = (int64_t)out->numel;

        /* FNV-1a content hash: chain ops/refs/consts + output & input shapes. */
        uint64_t key = 1469598103934665603ULL;
        #define FEH(x) do { key ^= (uint64_t)(x); key *= 1099511628211ULL; } while (0)
        FEH(UOP_FUSED_ELEMENTWISE); FEH(out_numel);
        FEH((unsigned)node->num_inputs); FEH((unsigned)fp->num_steps);
        for (int s = 0; s < fp->num_steps; s++) {
            FEH((unsigned)fp->op[s]); FEH((uint32_t)fp->a[s]);
            FEH((uint32_t)fp->b[s]);  FEH((uint32_t)fp->c[s]);
            uint32_t kb; memcpy(&kb, &fp->konst[s], sizeof(kb)); FEH(kb);
        }
        for (int k = 0; k < node->num_inputs; k++) FEH(in_numel[k]);
        #undef FEH
        if (key == 0) key = 0x9E3779B97F4A7C15ULL;  /* 0 marks an empty slot */

        unsigned fslot = 0;
        kernel_fn_t ffn = cache_lookup(backend, key, &fslot);
        if (!ffn) {
            char fn_name[64];
            snprintf(fn_name, sizeof(fn_name), "cml_fe%d", backend->kernel_count++);
            LLVMContextRef kern_ctx = LLVMContextCreate();
            if (!kern_ctx) return cpu_execute_node(node);
            LLVMModuleRef mod = build_fused_elementwise(kern_ctx, fn_name, fp,
                                                        out_numel, in_numel, node->num_inputs);
            if (!mod) { LLVMContextDispose(kern_ctx); return cpu_execute_node(node); }
            ffn = compile_and_lookup(backend, mod, fn_name);
            if (!ffn) return cpu_execute_node(node);
            backend->op_cache[fslot].key = key;
            backend->op_cache[fslot].fn  = ffn;
            LOG_DEBUG("LLVM: compiled fused elementwise kernel steps=%d inputs=%d n=%lld ('%s')",
                      fp->num_steps, node->num_inputs, (long long)out_numel, fn_name);
        }

        const float* inptrs[32];
        for (int k = 0; k < node->num_inputs; k++)
            inptrs[k] = (const float*)node->inputs[k]->data;
        typedef void (*fefn_t)(const float**, float*);
        ((fefn_t)(void*)ffn)(inptrs, (float*)out->data);
        node->is_executed = true;
        out->is_executed  = true;
        return 0;
    }

    /* ---- Gather the concrete shape signature ------------------------- *
     * These sizes are baked into the kernel as compile-time constants, so a
     * distinct shape gets its own specialized kernel.  The input-availability
     * guards are hoisted here (the dispatch section below re-derives the same
     * values); on failure we take the scalar CPU path exactly as before. */
    int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;

    if (is_binary_op(type)) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data)
            return cpu_execute_node(node);
        s0 = (int64_t)out->numel;
        s1 = (int64_t)node->inputs[0]->numel;
        s2 = (int64_t)node->inputs[1]->numel;
    } else if (is_unary_op(type)) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            return cpu_execute_node(node);
        s0 = (int64_t)out->numel;
        s1 = (int64_t)node->inputs[0]->numel;
    } else if (is_reduction(type)) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            return cpu_execute_node(node);
        /* Compute the (outer, inner, count) reduction layout (global reduce is
         * outer=inner=1, count=numel). s0=numel (global), s1/s2/s3=outer/inner/count. */
        Tensor* inp   = node->inputs[0];
        int64_t outer = 1, inner = 1, count = (int64_t)inp->numel;
        ReduceParams* rp = (ReduceParams*)node->params;
        if (rp && rp->num_dims == 1 && inp->ndim >= 1) {
            int rdim = rp->dims[0];
            if (rdim < 0) rdim += inp->ndim;
            if (rdim >= 0 && rdim < inp->ndim) {
                count = (int64_t)inp->shape[rdim];
                inner = 1;
                for (int d = rdim + 1; d < inp->ndim; d++) inner *= (int64_t)inp->shape[d];
                outer = 1;
                for (int d = 0; d < rdim; d++) outer *= (int64_t)inp->shape[d];
            }
        }
        /* If the derived layout doesn't match the output size, fall back. */
        if ((int64_t)out->numel != outer * inner)
            return cpu_execute_node(node);
        s0 = (int64_t)inp->numel;
        s1 = outer; s2 = inner; s3 = count;
    } else if (type == UOP_MATMUL) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data)
            return cpu_execute_node(node);
        Tensor* a = node->inputs[0]; Tensor* b = node->inputs[1];
        if (a->ndim < 2 || b->ndim < 2) return cpu_execute_node(node);
        s0 = (int64_t)a->shape[a->ndim-2];
        s1 = (int64_t)b->shape[b->ndim-1];
        s2 = (int64_t)a->shape[a->ndim-1];
    } else if (type == UOP_FILL) {
        s0 = (int64_t)out->numel;
    } else if (type == UOP_WHERE) {
        if (node->num_inputs < 3 || !node->inputs[0]->data ||
            !node->inputs[1]->data || !node->inputs[2]->data)
            return cpu_execute_node(node);
        s0 = (int64_t)out->numel;
        s1 = (int64_t)node->inputs[0]->numel;
        s2 = (int64_t)node->inputs[1]->numel;
        s3 = (int64_t)node->inputs[2]->numel;
    } else if (type == UOP_GATHER) {
        if (node->num_inputs < 2 || !node->inputs[0]->data || !node->inputs[1]->data)
            return cpu_execute_node(node);
        Tensor* inp  = node->inputs[0];
        Tensor* idx  = node->inputs[1];
        /* The JIT kernel implements exactly out[i] = input[i*C + idx[i]] for a 2D
         * input [R,C], 1D indices [R], gathering the LAST dim, output [R]. Any
         * other config (generic N-d gather, non-last dim) uses the interpreter,
         * which also bounds-checks indices. */
        GatherParams* gp = (GatherParams*)node->params;
        int dim = gp ? gp->dim : -1;
        if (dim < 0) dim += inp->ndim;
        if (inp->ndim != 2 || idx->ndim != 1 || dim != inp->ndim - 1 ||
            out->numel != (size_t)inp->shape[0] || idx->numel != (size_t)inp->shape[0])
            return cpu_execute_node(node);
        s0 = (int64_t)out->numel;
        s1 = (int64_t)inp->shape[inp->ndim-1];
    } else if (type == UOP_PERMUTE) {
        if (node->num_inputs < 1 || !node->inputs[0]->data ||
            node->inputs[0]->ndim != 2)
            return cpu_execute_node(node);
        s0 = (int64_t)node->inputs[0]->shape[0];
        s1 = (int64_t)node->inputs[0]->shape[1];
    } else if (type == UOP_RESHAPE || type == UOP_EXPAND) {
        if (node->num_inputs < 1 || !node->inputs[0]->data)
            return cpu_execute_node(node);
        s0 = (int64_t)out->numel;
        s1 = (int64_t)node->inputs[0]->numel;
    } else {
        LOG_DEBUG("LLVM: Unsupported op %d, CPU fallback", type);
        return cpu_execute_node(node);
    }

    /* ---- Shape-keyed kernel cache lookup ----------------------------- */
    /* Fold the element dtype into the key so f32 and f64 kernels are distinct. */
    uint64_t key = shape_key(type, s0, s1, s2, s3) ^ ((uint64_t)edt * 0x9E3779B97F4A7C15ULL);
    unsigned slot = 0;
    kernel_fn_t fn = cache_lookup(backend, key, &slot);

    if (!fn) {
        char fn_name[64];
        snprintf(fn_name, sizeof(fn_name), "cml_k%d", backend->kernel_count++);

        /* Fresh context per kernel: LLVMOrcCreateNewThreadSafeContextFromLLVMContext
         * transfers ownership, so a shared backend context becomes dangling after the
         * first AddLLVMIRModule.  Give each kernel its own context; compile_and_lookup
         * transfers ownership to ORC (TSC → TSM → JIT). */
        LLVMContextRef kern_ctx = LLVMContextCreate();
        if (!kern_ctx) return cpu_execute_node(node);

        LLVMModuleRef mod = NULL;

        if (is_binary_op(type)) {
            mod = build_binary_op(kern_ctx, type, fn_name, s0, s1, s2, edt);
        } else if (is_unary_op(type)) {
            mod = build_unary_op(kern_ctx, type, fn_name, s0, s1, edt);
        } else if (is_reduction(type)) {
            if (out->numel == 1)
                mod = build_reduction(kern_ctx, type, fn_name, s0);           /* global */
            else
                mod = build_reduction_axis(kern_ctx, type, fn_name, s1, s2, s3); /* per-axis */
        } else if (type == UOP_MATMUL) {
            mod = build_matmul_kernel(kern_ctx, fn_name, s0, s1, s2);
        } else if (type == UOP_FILL) {
            mod = build_fill_op(kern_ctx, fn_name, s0);
        } else if (type == UOP_WHERE) {
            mod = build_where_op(kern_ctx, fn_name, s0, s1, s2, s3);
        } else if (type == UOP_GATHER) {
            mod = build_gather_op(kern_ctx, fn_name, s0, s1);
        } else if (type == UOP_PERMUTE) {
            mod = build_permute_2d(kern_ctx, fn_name, s0, s1);
        } else if (type == UOP_RESHAPE) {
            mod = build_reshape_op(kern_ctx, fn_name, s0);
        } else if (type == UOP_EXPAND) {
            mod = build_expand_op(kern_ctx, fn_name, s0, s1);
        } else {
            LLVMContextDispose(kern_ctx);
            return cpu_execute_node(node);
        }

        if (!mod) { LLVMContextDispose(kern_ctx); return cpu_execute_node(node); }

        fn = compile_and_lookup(backend, mod, fn_name);
        /* kern_ctx ownership transferred to JIT via compile_and_lookup; do not free. */
        if (!fn) return cpu_execute_node(node);

        backend->op_cache[slot].key = key;
        backend->op_cache[slot].fn  = fn;
        LOG_DEBUG("LLVM: Compiled shape-specialized kernel op=%d shape=[%lld,%lld,%lld,%lld] ('%s')",
                  type, (long long)s0, (long long)s1, (long long)s2, (long long)s3, fn_name);
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
/* Execute a single node via the JIT (falls back to the interpreter internally
 * for unsupported ops / shapes). Public entry for the default execution path. */
int cml_llvm_execute_node(CMLLLVMBackend* backend, struct IRNode* node) {
    if (!backend || !node) return -1;
    return llvm_execute_node(backend, node);
}

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
