#include "ops/ir/gpu/spirv_codegen.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alloc/cml_allocator.h"


#define SPIRV_MAGIC       0x07230203
#define SPIRV_VERSION     0x00010300  /* SPIR-V 1.3 */
#define SPIRV_GENERATOR   0x00000000

/* Opcodes (opcode | (word_count << 16)) */
#define SpvOpCapability           17
#define SpvOpExtInstImport        11
#define SpvOpMemoryModel          14
#define SpvOpEntryPoint           15
#define SpvOpExecutionMode        16
#define SpvOpDecorate             71
#define SpvOpMemberDecorate       72
#define SpvOpTypeVoid             19
#define SpvOpTypeBool             20
#define SpvOpTypeInt              21
#define SpvOpTypeFloat            22
#define SpvOpTypeVector           23
#define SpvOpTypeArray            28
#define SpvOpTypeRuntimeArray     29
#define SpvOpTypeStruct           30
#define SpvOpTypePointer          32
#define SpvOpTypeFunction         33
#define SpvOpConstant             43
#define SpvOpConstantComposite    44
#define SpvOpFunction             54
#define SpvOpFunctionEnd          56
#define SpvOpLabel                248
#define SpvOpReturn               253
#define SpvOpVariable             59
#define SpvOpLoad                 61
#define SpvOpStore                62
#define SpvOpAccessChain          65
#define SpvOpCompositeExtract     81
#define SpvOpIAdd                 128
#define SpvOpIMul                 132
#define SpvOpFAdd                 129
#define SpvOpFSub                 131
#define SpvOpFMul                 133
#define SpvOpFDiv                 136
#define SpvOpFNegate              127
#define SpvOpULessThan            176
#define SpvOpSLessThan            177
#define SpvOpFOrdLessThan         188
#define SpvOpSelectionMerge       247
#define SpvOpBranchConditional    250
#define SpvOpBranch               249
#define SpvOpPhi                  245
#define SpvOpExtInst              12
#define SpvOpUDiv                 134
#define SpvOpConvertUToF          111
#define SpvOpConvertFToU          109
#define SpvOpBitcast              124
#define SpvOpShiftRightLogical    170
#define SpvOpControlBarrier       224
#define SpvOpLoopMerge            246
#define SpvOpUGreaterThanEqual    178

/* Built-in constants */
#define SpvBuiltInLocalInvocationId  27

/* Scope constants */
#define SpvScopeWorkgroup         2

/* Memory semantics */
#define SpvMemorySemanticsWorkgroupMemoryMask 0x100
#define SpvMemorySemanticsAcquireReleaseMask  0x08

/* Decoration constants */
#define SpvDecorationBinding       33
#define SpvDecorationDescriptorSet 34
#define SpvDecorationBuiltIn       11
#define SpvDecorationOffset        35
#define SpvDecorationArrayStride   6
#define SpvDecorationBlock         2
#define SpvDecorationBufferBlock   3

/* Built-in constants */
#define SpvBuiltInGlobalInvocationId 28

/* Storage classes */
#define SpvStorageClassInput        1
#define SpvStorageClassUniform      2
#define SpvStorageClassWorkgroup    4
#define SpvStorageClassStorageBuffer 12
#define SpvStorageClassFunction     7

/* Addressing/memory model */
#define SpvAddressingModelLogical   0
#define SpvMemoryModelGLSL450       1

/* Execution model/mode */
#define SpvExecutionModelGLCompute  5
#define SpvExecutionModeLocalSize   17

/* Capability */
#define SpvCapabilityShader         1

/* GLSL.std.450 extended instruction opcodes */
#define GLSLstd450Exp          27
#define GLSLstd450Log          28
#define GLSLstd450Sqrt         31
#define GLSLstd450FAbs         4
#define GLSLstd450Sin          13
#define GLSLstd450Cos          14
#define GLSLstd450Tan          15
#define GLSLstd450Exp2         29
#define GLSLstd450Log2         30
#define GLSLstd450InverseSqrt  32
#define GLSLstd450FMax         40
#define GLSLstd450FMin         37
#define GLSLstd450Floor        8
#define GLSLstd450Ceil         9
#define GLSLstd450Round        1
#define GLSLstd450Tanh         21
#define GLSLstd450Pow          26
#define GLSLstd450Asin         16
#define GLSLstd450Acos         17
#define GLSLstd450Atan         18
#define GLSLstd450Sinh         19
#define GLSLstd450Cosh         20
#define GLSLstd450FSign        6


SPIRVBuilder* spirv_builder_create(void) {
    SPIRVBuilder* b = (SPIRVBuilder*)cml_calloc(1, sizeof(SPIRVBuilder));
    if (!b) return NULL;
    b->cap = 4096;
    b->words = (uint32_t*)cml_malloc(b->cap * sizeof(uint32_t));
    if (!b->words) { cml_free(b); return NULL; }
    b->len = 0;
    b->next_id = 1;
    return b;
}

void spirv_builder_destroy(SPIRVBuilder* b) {
    if (!b) return;
    cml_free(b->words);
    cml_free(b);
}

void spirv_builder_emit(SPIRVBuilder* b, uint32_t word) {
    if (b->len >= b->cap) {
        size_t ncap = b->cap * 2;
        uint32_t* nw = (uint32_t*)cml_realloc(b->words, ncap * sizeof(uint32_t));
        if (!nw) { b->overflow = true; return; }  /* drop instead of NULL-deref */
        b->words = nw;
        b->cap = ncap;
    }
    b->words[b->len++] = word;
}

uint32_t spirv_builder_alloc_id(SPIRVBuilder* b) {
    return b->next_id++;
}

static void emit_op(SPIRVBuilder* b, uint32_t opcode, uint32_t word_count) {
    spirv_builder_emit(b, (word_count << 16) | opcode);
}

static void __attribute__((unused)) emit_header(SPIRVBuilder* b, uint32_t bound) {
    spirv_builder_emit(b, SPIRV_MAGIC);
    spirv_builder_emit(b, SPIRV_VERSION);
    spirv_builder_emit(b, SPIRV_GENERATOR);
    spirv_builder_emit(b, bound);
    spirv_builder_emit(b, 0); /* reserved */
}

static void emit_capability(SPIRVBuilder* b) {
    emit_op(b, SpvOpCapability, 2);
    spirv_builder_emit(b, SpvCapabilityShader);
}

static uint32_t emit_ext_import(SPIRVBuilder* b) {
    uint32_t id = spirv_builder_alloc_id(b);
    /* "GLSL.std.450" = 4 words of string + null padding */
    emit_op(b, SpvOpExtInstImport, 6);
    spirv_builder_emit(b, id);
    spirv_builder_emit(b, 0x534C4C47); /* "GLSL" */
    spirv_builder_emit(b, 0x6474732E); /* ".std" */
    spirv_builder_emit(b, 0x3035342E); /* ".450" */
    spirv_builder_emit(b, 0x00000000); /* null terminator */
    return id;
}

static void emit_memory_model(SPIRVBuilder* b) {
    emit_op(b, SpvOpMemoryModel, 3);
    spirv_builder_emit(b, SpvAddressingModelLogical);
    spirv_builder_emit(b, SpvMemoryModelGLSL450);
}

static void emit_entry_point(SPIRVBuilder* b, uint32_t func_id, uint32_t global_inv_id) {
    /* "main" = 1 word + null byte padding */
    emit_op(b, SpvOpEntryPoint, 6);
    spirv_builder_emit(b, SpvExecutionModelGLCompute);
    spirv_builder_emit(b, func_id);
    spirv_builder_emit(b, 0x6E69616D); /* "main" */
    spirv_builder_emit(b, 0x00000000); /* null terminator */
    spirv_builder_emit(b, global_inv_id);
}

static void emit_execution_mode(SPIRVBuilder* b, uint32_t func_id, int lx, int ly, int lz) {
    /* OpExecutionMode (opcode 16) <func> LocalSize(17) lx ly lz.
     * The opcode must be SpvOpExecutionMode; SpvExecutionModeLocalSize is the
     * mode *operand* emitted below — previously it was wrongly used as the
     * opcode too, which made every generated module fail SPIR-V validation. */
    emit_op(b, SpvOpExecutionMode, 6);
    spirv_builder_emit(b, func_id);
    spirv_builder_emit(b, SpvExecutionModeLocalSize);
    spirv_builder_emit(b, (uint32_t)lx);
    spirv_builder_emit(b, (uint32_t)ly);
    spirv_builder_emit(b, (uint32_t)lz);
}

static void emit_decorate(SPIRVBuilder* b, uint32_t target, uint32_t decoration, uint32_t value) {
    emit_op(b, SpvOpDecorate, 4);
    spirv_builder_emit(b, target);
    spirv_builder_emit(b, decoration);
    spirv_builder_emit(b, value);
}

static void emit_decorate_no_value(SPIRVBuilder* b, uint32_t target, uint32_t decoration) {
    emit_op(b, SpvOpDecorate, 3);
    spirv_builder_emit(b, target);
    spirv_builder_emit(b, decoration);
}

static void emit_member_decorate(SPIRVBuilder* b, uint32_t struct_id, uint32_t member,
                                  uint32_t decoration, uint32_t value) {
    emit_op(b, SpvOpMemberDecorate, 5);
    spirv_builder_emit(b, struct_id);
    spirv_builder_emit(b, member);
    spirv_builder_emit(b, decoration);
    spirv_builder_emit(b, value);
}

/* One-line spellings of the SPIR-V instructions this file emits by the dozen.
 * Each writes the same word stream the hand-rolled emit_op/emit pairs did. */
static void emit_pointer_type(SPIRVBuilder* b, uint32_t result, uint32_t storage_class,
                              uint32_t pointee) {
    emit_op(b, SpvOpTypePointer, 4);
    spirv_builder_emit(b, result);
    spirv_builder_emit(b, storage_class);
    spirv_builder_emit(b, pointee);
}

static void emit_variable(SPIRVBuilder* b, uint32_t ptr_type, uint32_t result,
                          uint32_t storage_class) {
    emit_op(b, SpvOpVariable, 4);
    spirv_builder_emit(b, ptr_type);
    spirv_builder_emit(b, result);
    spirv_builder_emit(b, storage_class);
}

static void emit_struct(SPIRVBuilder* b, uint32_t result, uint32_t member_type) {
    emit_op(b, SpvOpTypeStruct, 3);
    spirv_builder_emit(b, result);
    spirv_builder_emit(b, member_type);
}

static void emit_load(SPIRVBuilder* b, uint32_t type, uint32_t result, uint32_t pointer) {
    emit_op(b, SpvOpLoad, 4);
    spirv_builder_emit(b, type);
    spirv_builder_emit(b, result);
    spirv_builder_emit(b, pointer);
}

static void emit_store(SPIRVBuilder* b, uint32_t pointer, uint32_t value) {
    emit_op(b, SpvOpStore, 3);
    spirv_builder_emit(b, pointer);
    spirv_builder_emit(b, value);
}

static void emit_composite_extract(SPIRVBuilder* b, uint32_t type, uint32_t result,
                                   uint32_t composite, uint32_t index) {
    emit_op(b, SpvOpCompositeExtract, 5);
    spirv_builder_emit(b, type);
    spirv_builder_emit(b, result);
    spirv_builder_emit(b, composite);
    spirv_builder_emit(b, index);
}

static void emit_runtime_array(SPIRVBuilder* b, uint32_t result, uint32_t element_type) {
    emit_op(b, SpvOpTypeRuntimeArray, 3);
    spirv_builder_emit(b, result);
    spirv_builder_emit(b, element_type);
}

/* The void/bool/uint/float/uvec3/void() set every generated kernel declares. */
static void emit_scalar_types(SPIRVBuilder* b, uint32_t id_void, uint32_t id_bool,
                              uint32_t id_uint, uint32_t id_float, uint32_t id_uint3,
                              uint32_t id_void_fn) {
    emit_op(b, SpvOpTypeVoid, 2);
    spirv_builder_emit(b, id_void);
    emit_op(b, SpvOpTypeBool, 2);
    spirv_builder_emit(b, id_bool);
    emit_op(b, SpvOpTypeInt, 4);
    spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, 32);
    spirv_builder_emit(b, 0); /* unsigned */
    emit_op(b, SpvOpTypeFloat, 3);
    spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, 32);
    emit_op(b, SpvOpTypeVector, 4);
    spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, 3);
    emit_op(b, SpvOpTypeFunction, 3);
    spirv_builder_emit(b, id_void_fn);
    spirv_builder_emit(b, id_void);
}

static void emit_uint_constant_id(SPIRVBuilder* b, uint32_t uint_type, uint32_t result,
                                  uint32_t value) {
    emit_op(b, SpvOpConstant, 4);
    spirv_builder_emit(b, uint_type);
    spirv_builder_emit(b, result);
    spirv_builder_emit(b, value);
}

/* Bind `var` to descriptor set 0 at `binding`. */
static void emit_binding(SPIRVBuilder* b, uint32_t var, uint32_t binding) {
    emit_decorate(b, var, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, var, SpvDecorationBinding, binding);
}

/* OpFunction ... OpLabel: the entry of every generated kernel. */
static void emit_function_entry(SPIRVBuilder* b, uint32_t id_void, uint32_t id_func,
                                uint32_t id_void_fn, uint32_t id_label) {
    emit_op(b, SpvOpFunction, 5);
    spirv_builder_emit(b, id_void);
    spirv_builder_emit(b, id_func);
    spirv_builder_emit(b, 0); /* FunctionControl None */
    spirv_builder_emit(b, id_void_fn);
    emit_op(b, SpvOpLabel, 2);
    spirv_builder_emit(b, id_label);
}

static void emit_access_chain1(SPIRVBuilder* b, uint32_t ptr_type, uint32_t result,
                               uint32_t base, uint32_t index) {
    emit_op(b, SpvOpAccessChain, 5);
    spirv_builder_emit(b, ptr_type);
    spirv_builder_emit(b, result);
    spirv_builder_emit(b, base);
    spirv_builder_emit(b, index);
}

static void emit_access_chain2(SPIRVBuilder* b, uint32_t ptr_type, uint32_t result,
                               uint32_t base, uint32_t index0, uint32_t index1) {
    emit_op(b, SpvOpAccessChain, 6);
    spirv_builder_emit(b, ptr_type);
    spirv_builder_emit(b, result);
    spirv_builder_emit(b, base);
    spirv_builder_emit(b, index0);
    spirv_builder_emit(b, index1);
}

/* Read gl_GlobalInvocationID.x, the flat element index of every kernel. */
static uint32_t emit_global_invocation_x(SPIRVBuilder* b, uint32_t id_uint, uint32_t id_uint3,
                                         uint32_t id_var_gid) {
    uint32_t vec = spirv_builder_alloc_id(b);
    emit_load(b, id_uint3, vec, id_var_gid);
    uint32_t x = spirv_builder_alloc_id(b);
    emit_composite_extract(b, id_uint, x, vec, 0);
    return x;
}

/* Load `buffer[index]` from a struct-of-runtime-array storage buffer. */
static uint32_t emit_load_element(SPIRVBuilder* b, uint32_t elem_type, uint32_t elem_ptr_type,
                                  uint32_t buffer_var, uint32_t member, uint32_t index) {
    uint32_t ptr = spirv_builder_alloc_id(b);
    emit_access_chain2(b, elem_ptr_type, ptr, buffer_var, member, index);
    uint32_t value = spirv_builder_alloc_id(b);
    emit_load(b, elem_type, value, ptr);
    return value;
}

/* Load a single struct member -- how the kernels read their `n` parameter. */
static uint32_t emit_load_scalar(SPIRVBuilder* b, uint32_t type, uint32_t ptr_type,
                                 uint32_t var, uint32_t member) {
    uint32_t ptr = spirv_builder_alloc_id(b);
    emit_access_chain1(b, ptr_type, ptr, var, member);
    uint32_t value = spirv_builder_alloc_id(b);
    emit_load(b, type, value, ptr);
    return value;
}

static void emit_label(SPIRVBuilder* b, uint32_t id_label) {
    emit_op(b, SpvOpLabel, 2);
    spirv_builder_emit(b, id_label);
}

/* Guard the kernel body with `if (index < n)`, using a structured selection
 * that merges at `id_label_end`. */
static void emit_bounds_check(SPIRVBuilder* b, uint32_t id_bool, uint32_t id_index, uint32_t id_n,
                              uint32_t id_label_body, uint32_t id_label_end) {
    uint32_t id_cmp = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpULessThan, 5);
    spirv_builder_emit(b, id_bool);
    spirv_builder_emit(b, id_cmp);
    spirv_builder_emit(b, id_index);
    spirv_builder_emit(b, id_n);

    emit_op(b, SpvOpSelectionMerge, 3);
    spirv_builder_emit(b, id_label_end);
    spirv_builder_emit(b, 0); /* SelectionControl None */
    emit_op(b, SpvOpBranchConditional, 4);
    spirv_builder_emit(b, id_cmp);
    spirv_builder_emit(b, id_label_body);
    spirv_builder_emit(b, id_label_end);
}

static uint32_t emit_float_constant(SPIRVBuilder* b, uint32_t float_type, float value) {
    uint32_t id = spirv_builder_alloc_id(b);
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    emit_op(b, SpvOpConstant, 4);
    spirv_builder_emit(b, float_type);
    spirv_builder_emit(b, id);
    spirv_builder_emit(b, bits);
    return id;
}

static uint32_t __attribute__((unused)) emit_uint_constant(SPIRVBuilder* b, uint32_t uint_type, uint32_t value) {
    uint32_t id = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpConstant, 4);
    spirv_builder_emit(b, uint_type);
    spirv_builder_emit(b, id);
    spirv_builder_emit(b, value);
    return id;
}

uint32_t* spirv_builder_finalize(SPIRVBuilder* b, size_t* out_size) {
    if (b->overflow || b->len == 0) return NULL;  /* truncated → don't ship invalid SPIR-V */
    uint32_t* result = (uint32_t*)cml_malloc(b->len * sizeof(uint32_t));
    if (!result) return NULL;
    memcpy(result, b->words, b->len * sizeof(uint32_t));
    *out_size = b->len * sizeof(uint32_t);
    return result;
}


CMLSPIRVCodegen* cml_spirv_codegen_create(void) {
    CMLSPIRVCodegen* cg = (CMLSPIRVCodegen*)cml_calloc(1, sizeof(CMLSPIRVCodegen));
    if (!cg) return NULL;
    cg->local_size_x = 256;
    cg->local_size_y = 1;
    cg->local_size_z = 1;
    cg->initialized = true;
    return cg;
}

void cml_spirv_codegen_destroy(CMLSPIRVCodegen* cg) {
    cml_free(cg);
}

/* Core IDs every generated kernel allocates before it emits any type. */
typedef struct {
    uint32_t glsl_ext;
    uint32_t id_void, id_bool, id_uint, id_float, id_uint3, id_void_fn, id_rtarray;
} SpirvCoreIds;

/* Start a compute-shader module: reserve the five header words (back-patched by
 * spirv_finish_kernel), emit the capability/import/memory-model triple, and
 * pre-allocate the core type IDs in declaration order. */
static SPIRVBuilder* spirv_begin_kernel(SpirvCoreIds* ids) {
    SPIRVBuilder* b = spirv_builder_create();
    if (!b)
        return NULL;

    for (int i = 0; i < 5; i++)
        spirv_builder_emit(b, 0);

    emit_capability(b);
    ids->glsl_ext = emit_ext_import(b);
    emit_memory_model(b);

    ids->id_void    = spirv_builder_alloc_id(b);
    ids->id_bool    = spirv_builder_alloc_id(b);
    ids->id_uint    = spirv_builder_alloc_id(b);
    ids->id_float   = spirv_builder_alloc_id(b);
    ids->id_uint3   = spirv_builder_alloc_id(b);
    ids->id_void_fn = spirv_builder_alloc_id(b);
    ids->id_rtarray = spirv_builder_alloc_id(b);
    return b;
}

/* Close the entry function, back-patch the module header (the five words
 * reserved up front) and hand the finished word stream to the caller. */
static uint32_t* spirv_finish_kernel(SPIRVBuilder* b, CMLSPIRVCodegen* cg, size_t* out_size) {
    emit_op(b, SpvOpReturn, 1);
    emit_op(b, SpvOpFunctionEnd, 1);

    b->words[0] = SPIRV_MAGIC;
    b->words[1] = SPIRV_VERSION;
    b->words[2] = SPIRV_GENERATOR;
    b->words[3] = b->next_id;
    b->words[4] = 0;

    cg->kernel_count++;
    uint32_t* result = spirv_builder_finalize(b, out_size);
    spirv_builder_destroy(b);
    return result;
}

/*
 * Generate a unary compute shader:
 *   layout(set=0, binding=0) buffer InBuf  { float data[]; } inBuf;
 *   layout(set=0, binding=1) buffer OutBuf { float data[]; } outBuf;
 *   layout(set=0, binding=2) buffer Params { uint n; }       params;
 *
 *   void main() {
 *       uint idx = gl_GlobalInvocationID.x;
 *       if (idx >= n) return;
 *       outBuf.data[idx] = op(inBuf.data[idx]);
 *   }
 */
uint32_t* cml_spirv_gen_unary(CMLSPIRVCodegen* cg, UOpType op, const char* name,
                               size_t* out_size) {
    (void)name;
    if (!cg || !out_size) return NULL;

    SpirvCoreIds ids;
    size_t header_offset = 0;
    SPIRVBuilder* b = spirv_begin_kernel(&ids);
    if (!b) return NULL;

    uint32_t id_void = ids.id_void, id_bool = ids.id_bool, id_uint = ids.id_uint,
             id_float = ids.id_float, id_uint3 = ids.id_uint3,
             id_void_fn = ids.id_void_fn, id_rtarray_f = ids.id_rtarray;
    uint32_t glsl_ext = ids.glsl_ext;
    (void)id_bool;
    uint32_t id_struct_in = spirv_builder_alloc_id(b); /* struct { float[] } */
    uint32_t id_struct_out= spirv_builder_alloc_id(b);
    uint32_t id_struct_p  = spirv_builder_alloc_id(b); /* struct { uint } */

    /* Pointer types */
    uint32_t id_ptr_sb_in = spirv_builder_alloc_id(b); /* StorageBuffer ptr to struct_in */
    uint32_t id_ptr_sb_out= spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_p  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_f  = spirv_builder_alloc_id(b); /* StorageBuffer ptr to float */
    uint32_t id_ptr_sb_u  = spirv_builder_alloc_id(b); /* StorageBuffer ptr to uint */
    uint32_t id_ptr_in_u3 = spirv_builder_alloc_id(b); /* Input ptr to uvec3 */

    /* Variables */
    uint32_t id_var_in    = spirv_builder_alloc_id(b);
    uint32_t id_var_out   = spirv_builder_alloc_id(b);
    uint32_t id_var_p     = spirv_builder_alloc_id(b);
    uint32_t id_var_gid   = spirv_builder_alloc_id(b); /* GlobalInvocationId */

    /* Constants */
    uint32_t id_const_0   = spirv_builder_alloc_id(b); /* uint 0 */

    /* Function */
    uint32_t id_main      = spirv_builder_alloc_id(b);

    /* OpEntryPoint GLCompute %main "main" %gid */
    emit_entry_point(b, id_main, id_var_gid);

    /* OpExecutionMode %main LocalSize 256 1 1 */
    emit_execution_mode(b, id_main, cg->local_size_x, cg->local_size_y, cg->local_size_z);

    /* Decorations */
    emit_decorate(b, id_var_gid, SpvDecorationBuiltIn, SpvBuiltInGlobalInvocationId);

    /* Buffer block decorations */
    emit_decorate_no_value(b, id_struct_in, SpvDecorationBufferBlock);
    emit_decorate_no_value(b, id_struct_out, SpvDecorationBufferBlock);
    emit_decorate_no_value(b, id_struct_p, SpvDecorationBlock);

    emit_member_decorate(b, id_struct_in, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_struct_out, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_struct_p, 0, SpvDecorationOffset, 0);

    emit_decorate(b, id_rtarray_f, SpvDecorationArrayStride, 4);

    emit_binding(b, id_var_in, 0);
    emit_binding(b, id_var_out, 1);
    emit_binding(b, id_var_p, 2);
    /* Type declarations */
    emit_scalar_types(b, id_void, id_bool, id_uint, id_float, id_uint3, id_void_fn);
    emit_runtime_array(b, id_rtarray_f, id_float);
    emit_struct(b, id_struct_in, id_rtarray_f);
    emit_struct(b, id_struct_out, id_rtarray_f);
    emit_struct(b, id_struct_p, id_uint);

    emit_pointer_type(b, id_ptr_sb_in, SpvStorageClassStorageBuffer, id_struct_in);
    emit_pointer_type(b, id_ptr_sb_out, SpvStorageClassStorageBuffer, id_struct_out);
    emit_pointer_type(b, id_ptr_sb_p, SpvStorageClassUniform, id_struct_p);
    emit_pointer_type(b, id_ptr_sb_f, SpvStorageClassStorageBuffer, id_float);
    emit_pointer_type(b, id_ptr_sb_u, SpvStorageClassUniform, id_uint);
    emit_pointer_type(b, id_ptr_in_u3, SpvStorageClassInput, id_uint3);

    /* Constants */
    emit_uint_constant_id(b, id_uint, id_const_0, 0);
    /* Variables */
    emit_variable(b, id_ptr_sb_in, id_var_in, SpvStorageClassStorageBuffer);
    emit_variable(b, id_ptr_sb_out, id_var_out, SpvStorageClassStorageBuffer);
    emit_variable(b, id_ptr_sb_p, id_var_p, SpvStorageClassUniform);
    emit_variable(b, id_ptr_in_u3, id_var_gid, SpvStorageClassInput);

    /* Function body */
    uint32_t id_label_entry = spirv_builder_alloc_id(b);
    uint32_t id_label_body  = spirv_builder_alloc_id(b);
    uint32_t id_label_end   = spirv_builder_alloc_id(b);

    emit_function_entry(b, id_void, id_main, id_void_fn, id_label_entry);
    /* Load GlobalInvocationID.x */
    uint32_t id_gid_x = emit_global_invocation_x(b, id_uint, id_uint3, id_var_gid);
    /* Load n from params */
    uint32_t id_n = emit_load_scalar(b, id_uint, id_ptr_sb_u, id_var_p, id_const_0);
    /* Bounds check: if (idx >= n) return */
    emit_bounds_check(b, id_bool, id_gid_x, id_n, id_label_body, id_label_end);
    /* Body: load input[idx] */
    emit_label(b, id_label_body);
    uint32_t id_val = emit_load_element(b, id_float, id_ptr_sb_f, id_var_in, id_const_0, id_gid_x);
    /* Apply operation */
    uint32_t id_result = spirv_builder_alloc_id(b);
    switch (op) {
    case UOP_NEG:
        emit_op(b, SpvOpFNegate, 4); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, id_val);
        break;
    case UOP_EXP:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Exp); spirv_builder_emit(b, id_val);
        break;
    case UOP_LOG:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Log); spirv_builder_emit(b, id_val);
        break;
    case UOP_SQRT:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Sqrt); spirv_builder_emit(b, id_val);
        break;
    case UOP_ABS:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450FAbs); spirv_builder_emit(b, id_val);
        break;
    case UOP_SIN:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Sin); spirv_builder_emit(b, id_val);
        break;
    case UOP_COS:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Cos); spirv_builder_emit(b, id_val);
        break;
    case UOP_TAN:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Tan); spirv_builder_emit(b, id_val);
        break;
    case UOP_TANH:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Tanh); spirv_builder_emit(b, id_val);
        break;
    case UOP_RECIP: {
        uint32_t id_one = emit_float_constant(b, id_float, 1.0f);
        emit_op(b, SpvOpFDiv, 5); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, id_one);
        spirv_builder_emit(b, id_val);
        break;
    }
    case UOP_EXP2:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Exp2); spirv_builder_emit(b, id_val);
        break;
    case UOP_LOG2:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Log2); spirv_builder_emit(b, id_val);
        break;
    case UOP_RSQRT:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450InverseSqrt); spirv_builder_emit(b, id_val);
        break;
    case UOP_FLOOR:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Floor); spirv_builder_emit(b, id_val);
        break;
    case UOP_CEIL:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Ceil); spirv_builder_emit(b, id_val);
        break;
    case UOP_SIGN:
        emit_op(b, SpvOpExtInst, 6); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_result); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450FSign); spirv_builder_emit(b, id_val);
        break;
    default:
        /* Unsupported unary op: pass through keeps the module structurally valid,
         * but mark it invalid so finalize returns NULL (caller falls back to CPU)
         * rather than silently computing the identity. */
        id_result = id_val;
        b->overflow = true;
        break;
    }

    /* Store result to output[idx] */
    uint32_t id_out_ptr = spirv_builder_alloc_id(b);
    emit_access_chain2(b, id_ptr_sb_f, id_out_ptr, id_var_out, id_const_0, id_gid_x);
    emit_store(b, id_out_ptr, id_result);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_label_end);

    /* End label */
    emit_label(b, id_label_end);
    emit_op(b, SpvOpReturn, 1);
    emit_op(b, SpvOpFunctionEnd, 1);

    /* Patch header */
    b->words[header_offset + 0] = SPIRV_MAGIC;
    b->words[header_offset + 1] = SPIRV_VERSION;
    b->words[header_offset + 2] = SPIRV_GENERATOR;
    b->words[header_offset + 3] = b->next_id;
/* bound */
    b->words[header_offset + 4] = 0;

    cg->kernel_count++;
    uint32_t* result = spirv_builder_finalize(b, out_size);
    spirv_builder_destroy(b);
    return result;
}

uint32_t* cml_spirv_gen_binary(CMLSPIRVCodegen* cg, UOpType op, const char* name,
                                size_t* out_size) {
    (void)name;
    if (!cg || !out_size) return NULL;

    /*
     * Binary shader layout:
     *   binding 0: input A (float[])
     *   binding 1: output  (float[])
     *   binding 2: input B (float[])
     *
     * For simplicity, reuse the same builder pattern as unary but with
     * two loads and the binary op between them.
     */

    SpirvCoreIds ids;
    SPIRVBuilder* b = spirv_begin_kernel(&ids);
    if (!b) return NULL;

    uint32_t id_void = ids.id_void, id_bool = ids.id_bool, id_uint = ids.id_uint,
             id_float = ids.id_float, id_uint3 = ids.id_uint3,
             id_void_fn = ids.id_void_fn, id_rtarray = ids.id_rtarray;
    uint32_t glsl_ext = ids.glsl_ext;
    (void)id_bool;
    uint32_t id_s_a     = spirv_builder_alloc_id(b);
    uint32_t id_s_out   = spirv_builder_alloc_id(b);
    uint32_t id_s_b     = spirv_builder_alloc_id(b);

    uint32_t id_p_sb_a  = spirv_builder_alloc_id(b);
    uint32_t id_p_sb_o  = spirv_builder_alloc_id(b);
    uint32_t id_p_sb_b  = spirv_builder_alloc_id(b);
    uint32_t id_p_sb_f  = spirv_builder_alloc_id(b);
    uint32_t id_p_in_u3 = spirv_builder_alloc_id(b);

    uint32_t id_va      = spirv_builder_alloc_id(b);
    uint32_t id_vo      = spirv_builder_alloc_id(b);
    uint32_t id_vb      = spirv_builder_alloc_id(b);
    uint32_t id_vgid    = spirv_builder_alloc_id(b);
    uint32_t id_c0      = spirv_builder_alloc_id(b);
    uint32_t id_main    = spirv_builder_alloc_id(b);

    emit_entry_point(b, id_main, id_vgid);
    emit_execution_mode(b, id_main, cg->local_size_x, cg->local_size_y, cg->local_size_z);

    /* Decorations */
    emit_decorate(b, id_vgid, SpvDecorationBuiltIn, SpvBuiltInGlobalInvocationId);
    emit_decorate_no_value(b, id_s_a, SpvDecorationBufferBlock);
    emit_decorate_no_value(b, id_s_out, SpvDecorationBufferBlock);
    emit_decorate_no_value(b, id_s_b, SpvDecorationBufferBlock);
    emit_member_decorate(b, id_s_a, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_s_out, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_s_b, 0, SpvDecorationOffset, 0);
    emit_decorate(b, id_rtarray, SpvDecorationArrayStride, 4);
    emit_binding(b, id_va, 0);
    emit_binding(b, id_vo, 1);
    emit_binding(b, id_vb, 2);
    /* Types */
    emit_scalar_types(b, id_void, id_bool, id_uint, id_float, id_uint3, id_void_fn);
    emit_runtime_array(b, id_rtarray, id_float);
    emit_struct(b, id_s_a, id_rtarray);
    emit_struct(b, id_s_out, id_rtarray);
    emit_struct(b, id_s_b, id_rtarray);

    emit_pointer_type(b, id_p_sb_a, SpvStorageClassStorageBuffer, id_s_a);
    emit_pointer_type(b, id_p_sb_o, SpvStorageClassStorageBuffer, id_s_out);
    emit_pointer_type(b, id_p_sb_b, SpvStorageClassStorageBuffer, id_s_b);
    emit_pointer_type(b, id_p_sb_f, SpvStorageClassStorageBuffer, id_float);
    emit_pointer_type(b, id_p_in_u3, SpvStorageClassInput, id_uint3);

    /* Constants */
    emit_uint_constant_id(b, id_uint, id_c0, 0);
    /* Variables */
    emit_variable(b, id_p_sb_a, id_va, SpvStorageClassStorageBuffer);
    emit_variable(b, id_p_sb_o, id_vo, SpvStorageClassStorageBuffer);
    emit_variable(b, id_p_sb_b, id_vb, SpvStorageClassStorageBuffer);
    emit_variable(b, id_p_in_u3, id_vgid, SpvStorageClassInput);

    /* Function */
    uint32_t id_l_entry = spirv_builder_alloc_id(b);

    emit_function_entry(b, id_void, id_main, id_void_fn, id_l_entry);
    /* Load global invocation ID.x */
    uint32_t id_gx = emit_global_invocation_x(b, id_uint, id_uint3, id_vgid);
    /* Load A[idx] */
    uint32_t id_a = emit_load_element(b, id_float, id_p_sb_f, id_va, id_c0, id_gx);
    /* Load B[idx] */
    uint32_t id_bv = emit_load_element(b, id_float, id_p_sb_f, id_vb, id_c0, id_gx);
    /* Apply binary op */
    uint32_t id_res = spirv_builder_alloc_id(b);
    switch (op) {
    case UOP_ADD:
        emit_op(b, SpvOpFAdd, 5); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_res); spirv_builder_emit(b, id_a);
        spirv_builder_emit(b, id_bv);
        break;
    case UOP_SUB:
        emit_op(b, SpvOpFSub, 5); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_res); spirv_builder_emit(b, id_a);
        spirv_builder_emit(b, id_bv);
        break;
    case UOP_MUL:
        emit_op(b, SpvOpFMul, 5); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_res); spirv_builder_emit(b, id_a);
        spirv_builder_emit(b, id_bv);
        break;
    case UOP_DIV:
        emit_op(b, SpvOpFDiv, 5); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_res); spirv_builder_emit(b, id_a);
        spirv_builder_emit(b, id_bv);
        break;
    case UOP_MAX:
        emit_op(b, SpvOpExtInst, 7); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_res); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450FMax); spirv_builder_emit(b, id_a);
        spirv_builder_emit(b, id_bv);
        break;
    case UOP_MINIMUM:
        emit_op(b, SpvOpExtInst, 7); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_res); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450FMin); spirv_builder_emit(b, id_a);
        spirv_builder_emit(b, id_bv);
        break;
    case UOP_POW:
        emit_op(b, SpvOpExtInst, 7); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_res); spirv_builder_emit(b, glsl_ext);
        spirv_builder_emit(b, GLSLstd450Pow); spirv_builder_emit(b, id_a);
        spirv_builder_emit(b, id_bv);
        break;
    default:
        /* Unsupported op (e.g. CMPLT/MOD/IDIV): emit a structurally-valid copy
         * to keep id_res defined, but mark the module invalid so finalize
         * returns NULL and the caller falls back to CPU. Previously this
         * silently emitted FAdd, turning those ops into addition. */
        emit_op(b, SpvOpFAdd, 5); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_res); spirv_builder_emit(b, id_a);
        spirv_builder_emit(b, id_bv);
        b->overflow = true;
        break;
    }

    /* Store result */
    uint32_t id_op = spirv_builder_alloc_id(b);
    emit_access_chain2(b, id_p_sb_f, id_op, id_vo, id_c0, id_gx);
    emit_store(b, id_op, id_res);

    return spirv_finish_kernel(b, cg, out_size);
}

/*
 * Generate a reduction compute shader (SUM or MEAN):
 *   layout(set=0, binding=0) buffer InBuf  { float data[]; } inBuf;
 *   layout(set=0, binding=1) buffer OutBuf { float data[]; } outBuf;
 *   layout(set=0, binding=2) uniform Params { uint n; }      params;
 *   shared float sdata[256];
 *
 *   void main() {
 *       uint tid = gl_LocalInvocationID.x;
 *       uint gid = gl_GlobalInvocationID.x;
 *       sdata[tid] = (gid < n) ? inBuf.data[gid] : 0.0;
 *       barrier();
 *       for (uint s = 128; s > 0; s >>= 1) {
 *           if (tid < s) sdata[tid] += sdata[tid + s];
 *           barrier();
 *       }
 *       if (tid == 0) {
 *           float result = sdata[0];
 *           if (MEAN) result /= float(n);
 *           outBuf.data[0] = result;
 *       }
 *   }
 *
 * Implementation uses unrolled tree reduction (8 steps for 256 threads).
 */
uint32_t* cml_spirv_gen_reduction(CMLSPIRVCodegen* cg, UOpType op, const char* name,
                                   size_t* out_size) {
    (void)name;
    if (!cg || !out_size) return NULL;

    SpirvCoreIds ids;
    SPIRVBuilder* b = spirv_begin_kernel(&ids);
    if (!b) return NULL;

    uint32_t id_void = ids.id_void, id_bool = ids.id_bool, id_uint = ids.id_uint,
             id_float = ids.id_float, id_uint3 = ids.id_uint3,
             id_void_fn = ids.id_void_fn, id_rtarray_f = ids.id_rtarray;
    (void)id_bool;
    uint32_t id_struct_in = spirv_builder_alloc_id(b);
    uint32_t id_struct_out= spirv_builder_alloc_id(b);
    uint32_t id_struct_p  = spirv_builder_alloc_id(b);

    uint32_t id_ptr_sb_in = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_out= spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_p  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_f  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_u  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_in_u3 = spirv_builder_alloc_id(b);
    uint32_t id_ptr_wg_f  = spirv_builder_alloc_id(b); /* Workgroup ptr to float */

    /* Shared memory array type: float[256] */
    uint32_t id_arr_256   = spirv_builder_alloc_id(b);
    uint32_t id_ptr_wg_arr= spirv_builder_alloc_id(b); /* Workgroup ptr to float[256] */

    /* Variables */
    uint32_t id_var_in    = spirv_builder_alloc_id(b);
    uint32_t id_var_out   = spirv_builder_alloc_id(b);
    uint32_t id_var_p     = spirv_builder_alloc_id(b);
    uint32_t id_var_gid   = spirv_builder_alloc_id(b);
    uint32_t id_var_lid   = spirv_builder_alloc_id(b); /* LocalInvocationID */
    uint32_t id_var_sdata = spirv_builder_alloc_id(b); /* shared memory variable */

    /* Constants */
    uint32_t id_const_0u  = spirv_builder_alloc_id(b);
    uint32_t id_const_256u= spirv_builder_alloc_id(b);

    uint32_t id_main      = spirv_builder_alloc_id(b);

    /* Entry point: must list all Input/Output interface variables */
    /* "main" with two interface variables: gid and lid */
    emit_op(b, SpvOpEntryPoint, 7);
    spirv_builder_emit(b, SpvExecutionModelGLCompute);
    spirv_builder_emit(b, id_main);
    spirv_builder_emit(b, 0x6E69616D); /* "main" */
    spirv_builder_emit(b, 0x00000000);
    spirv_builder_emit(b, id_var_gid);
    spirv_builder_emit(b, id_var_lid);

    emit_execution_mode(b, id_main, 256, 1, 1);

    /* Decorations */
    emit_decorate(b, id_var_gid, SpvDecorationBuiltIn, SpvBuiltInGlobalInvocationId);
    emit_decorate(b, id_var_lid, SpvDecorationBuiltIn, SpvBuiltInLocalInvocationId);

    emit_decorate_no_value(b, id_struct_in, SpvDecorationBufferBlock);
    emit_decorate_no_value(b, id_struct_out, SpvDecorationBufferBlock);
    emit_decorate_no_value(b, id_struct_p, SpvDecorationBlock);
    emit_member_decorate(b, id_struct_in, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_struct_out, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_struct_p, 0, SpvDecorationOffset, 0);
    emit_decorate(b, id_rtarray_f, SpvDecorationArrayStride, 4);

    emit_binding(b, id_var_in, 0);
    emit_binding(b, id_var_out, 1);
    emit_binding(b, id_var_p, 2);
    /* Type declarations */
    emit_scalar_types(b, id_void, id_bool, id_uint, id_float, id_uint3, id_void_fn);
    emit_runtime_array(b, id_rtarray_f, id_float);
    emit_struct(b, id_struct_in, id_rtarray_f);
    emit_struct(b, id_struct_out, id_rtarray_f);
    emit_struct(b, id_struct_p, id_uint);

    /* Pointer types */
    emit_pointer_type(b, id_ptr_sb_in, SpvStorageClassStorageBuffer, id_struct_in);
    emit_pointer_type(b, id_ptr_sb_out, SpvStorageClassStorageBuffer, id_struct_out);
    emit_pointer_type(b, id_ptr_sb_p, SpvStorageClassUniform, id_struct_p);
    emit_pointer_type(b, id_ptr_sb_f, SpvStorageClassStorageBuffer, id_float);
    emit_pointer_type(b, id_ptr_sb_u, SpvStorageClassUniform, id_uint);
    emit_pointer_type(b, id_ptr_in_u3, SpvStorageClassInput, id_uint3);
    emit_pointer_type(b, id_ptr_wg_f, SpvStorageClassWorkgroup, id_float);

    /* float[256] array type */
    emit_uint_constant_id(b, id_uint, id_const_256u, 256);
    emit_op(b, SpvOpTypeArray, 4); spirv_builder_emit(b, id_arr_256);
    spirv_builder_emit(b, id_float); spirv_builder_emit(b, id_const_256u);
    emit_pointer_type(b, id_ptr_wg_arr, SpvStorageClassWorkgroup, id_arr_256);

    /* Constants */
    emit_uint_constant_id(b, id_uint, id_const_0u, 0);
    uint32_t id_const_0f = emit_float_constant(b, id_float, 0.0f);

    /* Stride constants for unrolled reduction: 128, 64, 32, 16, 8, 4, 2, 1 */
    uint32_t id_stride[8];
    uint32_t stride_vals[8] = {128, 64, 32, 16, 8, 4, 2, 1};
    for (int i = 0; i < 8; i++) {
        id_stride[i] = emit_uint_constant(b, id_uint, stride_vals[i]);
    }

    /* Scope and memory semantics constants for OpControlBarrier */
    uint32_t id_scope_wg = emit_uint_constant(b, id_uint, SpvScopeWorkgroup);
    uint32_t id_mem_sem   = emit_uint_constant(b, id_uint,
        SpvMemorySemanticsWorkgroupMemoryMask | SpvMemorySemanticsAcquireReleaseMask);

    /* Variables */
    emit_variable(b, id_ptr_sb_in, id_var_in, SpvStorageClassStorageBuffer);
    emit_variable(b, id_ptr_sb_out, id_var_out, SpvStorageClassStorageBuffer);
    emit_variable(b, id_ptr_sb_p, id_var_p, SpvStorageClassUniform);
    emit_variable(b, id_ptr_in_u3, id_var_gid, SpvStorageClassInput);
    emit_variable(b, id_ptr_in_u3, id_var_lid, SpvStorageClassInput);
    emit_variable(b, id_ptr_wg_arr, id_var_sdata, SpvStorageClassWorkgroup);

    uint32_t id_label_entry = spirv_builder_alloc_id(b);

    emit_function_entry(b, id_void, id_main, id_void_fn, id_label_entry);
    /* Load LocalInvocationID.x (tid) */
    uint32_t id_tid = emit_global_invocation_x(b, id_uint, id_uint3, id_var_lid);
    /* Load GlobalInvocationID.x (gid) */
    uint32_t id_gid_x = emit_global_invocation_x(b, id_uint, id_uint3, id_var_gid);
    /* Load n from params */
    uint32_t id_n = emit_load_scalar(b, id_uint, id_ptr_sb_u, id_var_p, id_const_0u);
    /* Bounds check: load input[gid] if gid < n, else 0.0 */
    uint32_t id_cmp_bounds = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpULessThan, 5); spirv_builder_emit(b, id_bool);
    spirv_builder_emit(b, id_cmp_bounds); spirv_builder_emit(b, id_gid_x);
    spirv_builder_emit(b, id_n);

    uint32_t id_label_load = spirv_builder_alloc_id(b);
    uint32_t id_label_skip = spirv_builder_alloc_id(b);
    uint32_t id_label_merge= spirv_builder_alloc_id(b);

    emit_op(b, SpvOpSelectionMerge, 3);
    spirv_builder_emit(b, id_label_merge); spirv_builder_emit(b, 0);
    emit_op(b, SpvOpBranchConditional, 4);
    spirv_builder_emit(b, id_cmp_bounds); spirv_builder_emit(b, id_label_load);
    spirv_builder_emit(b, id_label_skip);

    /* Load path: load inBuf.data[gid] */
    emit_label(b, id_label_load);
    uint32_t id_loaded_val = emit_load_element(b, id_float, id_ptr_sb_f, id_var_in, id_const_0u, id_gid_x);
    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_label_merge);

    /* Skip path: use 0.0 */
    emit_label(b, id_label_skip);
    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_label_merge);

    /* Merge: phi to pick loaded value or 0.0 */
    emit_label(b, id_label_merge);
    uint32_t id_init_val = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpPhi, 7); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_init_val);
    spirv_builder_emit(b, id_loaded_val); spirv_builder_emit(b, id_label_load);
    spirv_builder_emit(b, id_const_0f); spirv_builder_emit(b, id_label_skip);

    /* Store to shared memory: sdata[tid] = init_val */
    uint32_t id_sdata_ptr = spirv_builder_alloc_id(b);
    emit_access_chain1(b, id_ptr_wg_f, id_sdata_ptr, id_var_sdata, id_tid);
    emit_store(b, id_sdata_ptr, id_init_val);

    /* Barrier */
    emit_op(b, SpvOpControlBarrier, 4);
    spirv_builder_emit(b, id_scope_wg);
    spirv_builder_emit(b, id_scope_wg);
    spirv_builder_emit(b, id_mem_sem);

    /* Unrolled tree reduction: 8 steps (128, 64, 32, 16, 8, 4, 2, 1) */
    for (int step = 0; step < 8; step++) {
        uint32_t id_lbl_reduce = spirv_builder_alloc_id(b);
        uint32_t id_lbl_skip2  = spirv_builder_alloc_id(b);

        /* if (tid < stride) */
        uint32_t id_cmp_s = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpULessThan, 5); spirv_builder_emit(b, id_bool);
        spirv_builder_emit(b, id_cmp_s); spirv_builder_emit(b, id_tid);
        spirv_builder_emit(b, id_stride[step]);

        emit_op(b, SpvOpSelectionMerge, 3);
        spirv_builder_emit(b, id_lbl_skip2); spirv_builder_emit(b, 0);
        emit_op(b, SpvOpBranchConditional, 4);
        spirv_builder_emit(b, id_cmp_s); spirv_builder_emit(b, id_lbl_reduce);
        spirv_builder_emit(b, id_lbl_skip2);

        emit_label(b, id_lbl_reduce);
        /* Load sdata[tid] */
        /* Keep the pointer: the accumulated sum is stored back through it. */
        uint32_t id_sd_ptr1 = spirv_builder_alloc_id(b);
        emit_access_chain1(b, id_ptr_wg_f, id_sd_ptr1, id_var_sdata, id_tid);
        uint32_t id_sd_val1 = spirv_builder_alloc_id(b);
        emit_load(b, id_float, id_sd_val1, id_sd_ptr1);
        /* tid + stride */
        uint32_t id_tid_plus = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpIAdd, 5); spirv_builder_emit(b, id_uint);
        spirv_builder_emit(b, id_tid_plus); spirv_builder_emit(b, id_tid);
        spirv_builder_emit(b, id_stride[step]);

        /* Load sdata[tid + stride] */
        uint32_t id_sd_val2 = emit_load_scalar(b, id_float, id_ptr_wg_f, id_var_sdata, id_tid_plus);
        /* sdata[tid] += sdata[tid + stride] */
        uint32_t id_sum = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpFAdd, 5); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_sum); spirv_builder_emit(b, id_sd_val1);
        spirv_builder_emit(b, id_sd_val2);
        emit_store(b, id_sd_ptr1, id_sum);

        emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_lbl_skip2);
        emit_label(b, id_lbl_skip2);
        /* Barrier after each step */
        emit_op(b, SpvOpControlBarrier, 4);
        spirv_builder_emit(b, id_scope_wg);
        spirv_builder_emit(b, id_scope_wg);
        spirv_builder_emit(b, id_mem_sem);
    }

    /* Thread 0 writes result */
    uint32_t id_lbl_write = spirv_builder_alloc_id(b);
    uint32_t id_lbl_end   = spirv_builder_alloc_id(b);

    uint32_t id_cmp_zero = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpULessThan, 5); spirv_builder_emit(b, id_bool);
    spirv_builder_emit(b, id_cmp_zero); spirv_builder_emit(b, id_tid);
    spirv_builder_emit(b, id_stride[7]); /* id for constant 1 */

    emit_op(b, SpvOpSelectionMerge, 3);
    spirv_builder_emit(b, id_lbl_end); spirv_builder_emit(b, 0);
    emit_op(b, SpvOpBranchConditional, 4);
    spirv_builder_emit(b, id_cmp_zero); spirv_builder_emit(b, id_lbl_write);
    spirv_builder_emit(b, id_lbl_end);

    emit_label(b, id_lbl_write);
    /* Load sdata[0] */
    uint32_t id_res_val = emit_load_scalar(b, id_float, id_ptr_wg_f, id_var_sdata, id_const_0u);
    /* For MEAN: divide by n */
    uint32_t id_final_val = id_res_val;
    if (op == UOP_MEAN) {
        uint32_t id_n_float = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpConvertUToF, 4); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_n_float); spirv_builder_emit(b, id_n);

        id_final_val = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpFDiv, 5); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_final_val); spirv_builder_emit(b, id_res_val);
        spirv_builder_emit(b, id_n_float);
    }

    /* Store to outBuf.data[0] */
    uint32_t id_out_ptr = spirv_builder_alloc_id(b);
    emit_access_chain2(b, id_ptr_sb_f, id_out_ptr, id_var_out, id_const_0u, id_const_0u);
    emit_store(b, id_out_ptr, id_final_val);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_lbl_end);

    /* End */
    emit_label(b, id_lbl_end);
    return spirv_finish_kernel(b, cg, out_size);
}

/*
 * Generate a matrix multiplication compute shader:
 *   layout(set=0, binding=0) buffer A { float data[]; } aBuf;
 *   layout(set=0, binding=1) buffer B { float data[]; } bBuf;
 *   layout(set=0, binding=2) buffer C { float data[]; } cBuf;
 *   layout(set=0, binding=3) uniform Params { uint M; uint N; uint K; } params;
 *
 *   layout(local_size_x=16, local_size_y=16) in;
 *
 *   void main() {
 *       uint col = gl_GlobalInvocationID.x;
 *       uint row = gl_GlobalInvocationID.y;
 *       if (row >= M || col >= N) return;
 *       float sum = 0.0;
 *       for (uint k = 0; k < K; k++)
 *           sum += A[row * K + k] * B[k * N + col];
 *       C[row * N + col] = sum;
 *   }
 */
uint32_t* cml_spirv_gen_matmul(CMLSPIRVCodegen* cg, const char* name, size_t* out_size) {
    (void)name;
    if (!cg || !out_size) return NULL;

    SpirvCoreIds ids;
    SPIRVBuilder* b = spirv_begin_kernel(&ids);
    if (!b) return NULL;

    uint32_t id_void = ids.id_void, id_bool = ids.id_bool, id_uint = ids.id_uint,
             id_float = ids.id_float, id_uint3 = ids.id_uint3,
             id_void_fn = ids.id_void_fn, id_rtarray_f = ids.id_rtarray;
    (void)id_bool;
    uint32_t id_struct_a  = spirv_builder_alloc_id(b);
    uint32_t id_struct_b  = spirv_builder_alloc_id(b);
    uint32_t id_struct_c  = spirv_builder_alloc_id(b);
    uint32_t id_struct_p  = spirv_builder_alloc_id(b); /* { M, N, K } */

    uint32_t id_ptr_sb_a  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_b  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_c  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_p  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_f  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_u  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_in_u3 = spirv_builder_alloc_id(b);
    uint32_t id_ptr_fn_f  = spirv_builder_alloc_id(b); /* Function ptr to float (for loop accumulator) */

    /* Variables */
    uint32_t id_var_a     = spirv_builder_alloc_id(b);
    uint32_t id_var_b     = spirv_builder_alloc_id(b);
    uint32_t id_var_c     = spirv_builder_alloc_id(b);
    uint32_t id_var_p     = spirv_builder_alloc_id(b);
    uint32_t id_var_gid   = spirv_builder_alloc_id(b);

    /* Constants */
    uint32_t id_const_0u  = spirv_builder_alloc_id(b);
    uint32_t id_const_1u  = spirv_builder_alloc_id(b);
    uint32_t id_const_2u  = spirv_builder_alloc_id(b);

    uint32_t id_main      = spirv_builder_alloc_id(b);

    /* Entry point */
    emit_entry_point(b, id_main, id_var_gid);

    /* 2D dispatch: local_size_x=16, local_size_y=16, local_size_z=1 */
    emit_execution_mode(b, id_main, 16, 16, 1);

    /* Decorations */
    emit_decorate(b, id_var_gid, SpvDecorationBuiltIn, SpvBuiltInGlobalInvocationId);

    emit_decorate_no_value(b, id_struct_a, SpvDecorationBufferBlock);
    emit_decorate_no_value(b, id_struct_b, SpvDecorationBufferBlock);
    emit_decorate_no_value(b, id_struct_c, SpvDecorationBufferBlock);
    emit_decorate_no_value(b, id_struct_p, SpvDecorationBlock);

    emit_member_decorate(b, id_struct_a, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_struct_b, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_struct_c, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_struct_p, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_struct_p, 1, SpvDecorationOffset, 4);
    emit_member_decorate(b, id_struct_p, 2, SpvDecorationOffset, 8);

    emit_decorate(b, id_rtarray_f, SpvDecorationArrayStride, 4);

    emit_binding(b, id_var_a, 0);
    emit_binding(b, id_var_b, 1);
    emit_binding(b, id_var_c, 2);
    emit_binding(b, id_var_p, 3);
    /* Type declarations */
    emit_scalar_types(b, id_void, id_bool, id_uint, id_float, id_uint3, id_void_fn);
    emit_runtime_array(b, id_rtarray_f, id_float);
    emit_struct(b, id_struct_a, id_rtarray_f);
    emit_struct(b, id_struct_b, id_rtarray_f);
    emit_struct(b, id_struct_c, id_rtarray_f);
    /* Params struct: { uint M, uint N, uint K } */
    emit_op(b, SpvOpTypeStruct, 5); spirv_builder_emit(b, id_struct_p);
    spirv_builder_emit(b, id_uint); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_uint);

    /* Pointer types */
    emit_pointer_type(b, id_ptr_sb_a, SpvStorageClassStorageBuffer, id_struct_a);
    emit_pointer_type(b, id_ptr_sb_b, SpvStorageClassStorageBuffer, id_struct_b);
    emit_pointer_type(b, id_ptr_sb_c, SpvStorageClassStorageBuffer, id_struct_c);
    emit_pointer_type(b, id_ptr_sb_p, SpvStorageClassUniform, id_struct_p);
    emit_pointer_type(b, id_ptr_sb_f, SpvStorageClassStorageBuffer, id_float);
    emit_pointer_type(b, id_ptr_sb_u, SpvStorageClassUniform, id_uint);
    emit_pointer_type(b, id_ptr_in_u3, SpvStorageClassInput, id_uint3);
    emit_pointer_type(b, id_ptr_fn_f, SpvStorageClassFunction, id_float);

    /* Constants */
    emit_uint_constant_id(b, id_uint, id_const_0u, 0);
    emit_uint_constant_id(b, id_uint, id_const_1u, 1);
    emit_uint_constant_id(b, id_uint, id_const_2u, 2);
    uint32_t id_const_0f = emit_float_constant(b, id_float, 0.0f);

    /* Variables */
    emit_variable(b, id_ptr_sb_a, id_var_a, SpvStorageClassStorageBuffer);
    emit_variable(b, id_ptr_sb_b, id_var_b, SpvStorageClassStorageBuffer);
    emit_variable(b, id_ptr_sb_c, id_var_c, SpvStorageClassStorageBuffer);
    emit_variable(b, id_ptr_sb_p, id_var_p, SpvStorageClassUniform);
    emit_variable(b, id_ptr_in_u3, id_var_gid, SpvStorageClassInput);

    uint32_t id_label_entry = spirv_builder_alloc_id(b);
    uint32_t id_label_body  = spirv_builder_alloc_id(b);
    uint32_t id_label_end   = spirv_builder_alloc_id(b);

    emit_function_entry(b, id_void, id_main, id_void_fn, id_label_entry);
    /* Allocate function-local variable for accumulator */
    uint32_t id_var_sum = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpVariable, 5); spirv_builder_emit(b, id_ptr_fn_f);
    spirv_builder_emit(b, id_var_sum); spirv_builder_emit(b, SpvStorageClassFunction);
    spirv_builder_emit(b, id_const_0f);

    /* Load GlobalInvocationID */
    /* Matmul needs both lanes of the invocation ID, not just .x. */
    uint32_t id_gid_vec = spirv_builder_alloc_id(b);
    emit_load(b, id_uint3, id_gid_vec, id_var_gid);
    uint32_t id_col = spirv_builder_alloc_id(b);
    emit_composite_extract(b, id_uint, id_col, id_gid_vec, 0);
    uint32_t id_row = spirv_builder_alloc_id(b);
    emit_composite_extract(b, id_uint, id_row, id_gid_vec, 1);

    /* Load M, N, K from params */
    uint32_t id_M = emit_load_scalar(b, id_uint, id_ptr_sb_u, id_var_p, id_const_0u);
    uint32_t id_N = emit_load_scalar(b, id_uint, id_ptr_sb_u, id_var_p, id_const_1u);
    uint32_t id_K = emit_load_scalar(b, id_uint, id_ptr_sb_u, id_var_p, id_const_2u);
    /* Bounds check: if (row >= M || col >= N) return */
    uint32_t id_cmp_row = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpULessThan, 5); spirv_builder_emit(b, id_bool);
    spirv_builder_emit(b, id_cmp_row); spirv_builder_emit(b, id_row);
    spirv_builder_emit(b, id_M);

    uint32_t id_cmp_col = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpULessThan, 5); spirv_builder_emit(b, id_bool);
    spirv_builder_emit(b, id_cmp_col); spirv_builder_emit(b, id_col);
    spirv_builder_emit(b, id_N);

    /* OpLogicalAnd: need to define it */
    /* Use SpvOpLogicalAnd = 167 */
    uint32_t id_cmp_both = spirv_builder_alloc_id(b);
    emit_op(b, 167 /* SpvOpLogicalAnd */, 5); spirv_builder_emit(b, id_bool);
    spirv_builder_emit(b, id_cmp_both); spirv_builder_emit(b, id_cmp_row);
    spirv_builder_emit(b, id_cmp_col);

    emit_op(b, SpvOpSelectionMerge, 3);
    spirv_builder_emit(b, id_label_end); spirv_builder_emit(b, 0);
    emit_op(b, SpvOpBranchConditional, 4);
    spirv_builder_emit(b, id_cmp_both); spirv_builder_emit(b, id_label_body);
    spirv_builder_emit(b, id_label_end);

    /* Body */
    emit_label(b, id_label_body);
    /* Loop: for (k = 0; k < K; k++) */
    uint32_t id_lbl_loop_hdr  = spirv_builder_alloc_id(b);
    uint32_t id_lbl_loop_body = spirv_builder_alloc_id(b);
    uint32_t id_lbl_loop_cont = spirv_builder_alloc_id(b);
    uint32_t id_lbl_loop_end  = spirv_builder_alloc_id(b);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_lbl_loop_hdr);

    /* Loop header */
    emit_label(b, id_lbl_loop_hdr);
    /* Phi for k */
    uint32_t id_k = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpPhi, 7); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_k);
    spirv_builder_emit(b, id_const_0u); spirv_builder_emit(b, id_label_body);
    /* k_next comes from continue block - we'll patch the ID later */
    uint32_t phi_k_next_slot = b->len; /* remember position for k_next id */
    spirv_builder_emit(b, 0); /* placeholder for k_next */
    spirv_builder_emit(b, id_lbl_loop_cont);

    /* Phi for sum accumulator */
    uint32_t id_sum_phi = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpPhi, 7); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_sum_phi);
    spirv_builder_emit(b, id_const_0f); spirv_builder_emit(b, id_label_body);
    uint32_t phi_sum_next_slot = b->len;
    spirv_builder_emit(b, 0); /* placeholder for sum_next */
    spirv_builder_emit(b, id_lbl_loop_cont);

    /* Loop merge */
    emit_op(b, SpvOpLoopMerge, 4);
    spirv_builder_emit(b, id_lbl_loop_end);
    spirv_builder_emit(b, id_lbl_loop_cont);
    spirv_builder_emit(b, 0); /* None */

    /* Condition: k < K */
    uint32_t id_cmp_k = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpULessThan, 5); spirv_builder_emit(b, id_bool);
    spirv_builder_emit(b, id_cmp_k); spirv_builder_emit(b, id_k);
    spirv_builder_emit(b, id_K);

    emit_op(b, SpvOpBranchConditional, 4);
    spirv_builder_emit(b, id_cmp_k); spirv_builder_emit(b, id_lbl_loop_body);
    spirv_builder_emit(b, id_lbl_loop_end);

    /* Loop body */
    emit_label(b, id_lbl_loop_body);
    /* Compute index for A: row * K + k */
    uint32_t id_row_k = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpIMul, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_row_k); spirv_builder_emit(b, id_row);
    spirv_builder_emit(b, id_K);
    uint32_t id_a_idx = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpIAdd, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_a_idx); spirv_builder_emit(b, id_row_k);
    spirv_builder_emit(b, id_k);

    /* Load A[row * K + k] */
    uint32_t id_a_val = emit_load_element(b, id_float, id_ptr_sb_f, id_var_a, id_const_0u, id_a_idx);
    /* Compute index for B: k * N + col */
    uint32_t id_k_n = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpIMul, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_k_n); spirv_builder_emit(b, id_k);
    spirv_builder_emit(b, id_N);
    uint32_t id_b_idx = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpIAdd, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_b_idx); spirv_builder_emit(b, id_k_n);
    spirv_builder_emit(b, id_col);

    /* Load B[k * N + col] */
    uint32_t id_b_val = emit_load_element(b, id_float, id_ptr_sb_f, id_var_b, id_const_0u, id_b_idx);
    /* sum += A[...] * B[...] */
    uint32_t id_prod = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpFMul, 5); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_prod); spirv_builder_emit(b, id_a_val);
    spirv_builder_emit(b, id_b_val);
    uint32_t id_sum_next = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpFAdd, 5); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_sum_next); spirv_builder_emit(b, id_sum_phi);
    spirv_builder_emit(b, id_prod);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_lbl_loop_cont);

    /* Continue block: k++ */
    emit_label(b, id_lbl_loop_cont);
    uint32_t id_k_next = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpIAdd, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_k_next); spirv_builder_emit(b, id_k);
    spirv_builder_emit(b, id_const_1u);

    /* Patch phi placeholders */
    b->words[phi_k_next_slot] = id_k_next;
    b->words[phi_sum_next_slot] = id_sum_next;

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_lbl_loop_hdr);

    /* Loop end */
    emit_label(b, id_lbl_loop_end);
    /* Compute output index: row * N + col */
    uint32_t id_row_n = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpIMul, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_row_n); spirv_builder_emit(b, id_row);
    spirv_builder_emit(b, id_N);
    uint32_t id_c_idx = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpIAdd, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_c_idx); spirv_builder_emit(b, id_row_n);
    spirv_builder_emit(b, id_col);

    /* Store C[row * N + col] = sum */
    uint32_t id_c_ptr = spirv_builder_alloc_id(b);
    emit_access_chain2(b, id_ptr_sb_f, id_c_ptr, id_var_c, id_const_0u, id_c_idx);
    emit_store(b, id_c_ptr, id_sum_phi);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_label_end);

    /* End label */
    emit_label(b, id_label_end);
    return spirv_finish_kernel(b, cg, out_size);
}

/*
 * Generate a fill compute shader:
 *   layout(set=0, binding=0) buffer OutBuf { float data[]; } outBuf;
 *   layout(set=0, binding=1) uniform Params { uint n; }     params;
 *
 *   void main() {
 *       uint idx = gl_GlobalInvocationID.x;
 *       if (idx >= n) return;
 *       outBuf.data[idx] = value;
 *   }
 */
uint32_t* cml_spirv_gen_fill(CMLSPIRVCodegen* cg, float value, const char* name,
                              size_t* out_size) {
    (void)name;
    if (!cg || !out_size) return NULL;

    SpirvCoreIds ids;
    SPIRVBuilder* b = spirv_begin_kernel(&ids);
    if (!b) return NULL;

    uint32_t id_void = ids.id_void, id_bool = ids.id_bool, id_uint = ids.id_uint,
             id_float = ids.id_float, id_uint3 = ids.id_uint3,
             id_void_fn = ids.id_void_fn, id_rtarray_f = ids.id_rtarray;
    (void)id_bool;
    uint32_t id_struct_out= spirv_builder_alloc_id(b);
    uint32_t id_struct_p  = spirv_builder_alloc_id(b);

    uint32_t id_ptr_sb_out= spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_p  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_f  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_sb_u  = spirv_builder_alloc_id(b);
    uint32_t id_ptr_in_u3 = spirv_builder_alloc_id(b);

    uint32_t id_var_out   = spirv_builder_alloc_id(b);
    uint32_t id_var_p     = spirv_builder_alloc_id(b);
    uint32_t id_var_gid   = spirv_builder_alloc_id(b);

    uint32_t id_const_0   = spirv_builder_alloc_id(b);
    uint32_t id_main      = spirv_builder_alloc_id(b);

    /* Entry point and execution mode */
    emit_entry_point(b, id_main, id_var_gid);
    emit_execution_mode(b, id_main, cg->local_size_x, cg->local_size_y, cg->local_size_z);

    /* Decorations */
    emit_decorate(b, id_var_gid, SpvDecorationBuiltIn, SpvBuiltInGlobalInvocationId);
    emit_decorate_no_value(b, id_struct_out, SpvDecorationBufferBlock);
    emit_decorate_no_value(b, id_struct_p, SpvDecorationBlock);
    emit_member_decorate(b, id_struct_out, 0, SpvDecorationOffset, 0);
    emit_member_decorate(b, id_struct_p, 0, SpvDecorationOffset, 0);
    emit_decorate(b, id_rtarray_f, SpvDecorationArrayStride, 4);
    emit_binding(b, id_var_out, 0);
    emit_binding(b, id_var_p, 1);
    /* Type declarations */
    emit_scalar_types(b, id_void, id_bool, id_uint, id_float, id_uint3, id_void_fn);
    emit_runtime_array(b, id_rtarray_f, id_float);
    emit_struct(b, id_struct_out, id_rtarray_f);
    emit_struct(b, id_struct_p, id_uint);

    emit_pointer_type(b, id_ptr_sb_out, SpvStorageClassStorageBuffer, id_struct_out);
    emit_pointer_type(b, id_ptr_sb_p, SpvStorageClassUniform, id_struct_p);
    emit_pointer_type(b, id_ptr_sb_f, SpvStorageClassStorageBuffer, id_float);
    emit_pointer_type(b, id_ptr_sb_u, SpvStorageClassUniform, id_uint);
    emit_pointer_type(b, id_ptr_in_u3, SpvStorageClassInput, id_uint3);

    /* Constants */
    emit_uint_constant_id(b, id_uint, id_const_0, 0);
    uint32_t id_fill_val = emit_float_constant(b, id_float, value);

    /* Variables */
    emit_variable(b, id_ptr_sb_out, id_var_out, SpvStorageClassStorageBuffer);
    emit_variable(b, id_ptr_sb_p, id_var_p, SpvStorageClassUniform);
    emit_variable(b, id_ptr_in_u3, id_var_gid, SpvStorageClassInput);

    /* Function body */
    uint32_t id_label_entry = spirv_builder_alloc_id(b);
    uint32_t id_label_body  = spirv_builder_alloc_id(b);
    uint32_t id_label_end   = spirv_builder_alloc_id(b);

    emit_function_entry(b, id_void, id_main, id_void_fn, id_label_entry);
    /* Load GlobalInvocationID.x */
    uint32_t id_gid_x = emit_global_invocation_x(b, id_uint, id_uint3, id_var_gid);
    /* Load n from params */
    uint32_t id_n = emit_load_scalar(b, id_uint, id_ptr_sb_u, id_var_p, id_const_0);
    /* Bounds check: if (idx >= n) return */
    emit_bounds_check(b, id_bool, id_gid_x, id_n, id_label_body, id_label_end);
    /* Body: store fill value to output[idx] */
    emit_label(b, id_label_body);
    uint32_t id_out_ptr = spirv_builder_alloc_id(b);
    emit_access_chain2(b, id_ptr_sb_f, id_out_ptr, id_var_out, id_const_0, id_gid_x);
    emit_store(b, id_out_ptr, id_fill_val);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_label_end);

    /* End label */
    emit_label(b, id_label_end);
    return spirv_finish_kernel(b, cg, out_size);
}
