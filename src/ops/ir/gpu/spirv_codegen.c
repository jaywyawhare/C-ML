/**
 * @file spirv_codegen.c
 * @brief SPIR-V binary code generation
 *
 * Emits binary SPIR-V compute shaders: OpCapability Shader, storage buffers,
 * GlobalInvocationID for thread index, GLSL.std.450 for math ops.
 */

#include "ops/ir/gpu/spirv_codegen.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── SPIR-V magic and opcode constants ── */

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

/* ── SPIR-V builder ── */

SPIRVBuilder* spirv_builder_create(void) {
    SPIRVBuilder* b = (SPIRVBuilder*)calloc(1, sizeof(SPIRVBuilder));
    if (!b) return NULL;
    b->cap = 4096;
    b->words = (uint32_t*)malloc(b->cap * sizeof(uint32_t));
    if (!b->words) { free(b); return NULL; }
    b->len = 0;
    b->next_id = 1;
    return b;
}

void spirv_builder_destroy(SPIRVBuilder* b) {
    if (!b) return;
    free(b->words);
    free(b);
}

void spirv_builder_emit(SPIRVBuilder* b, uint32_t word) {
    if (b->len >= b->cap) {
        b->cap *= 2;
        b->words = (uint32_t*)realloc(b->words, b->cap * sizeof(uint32_t));
    }
    b->words[b->len++] = word;
}

uint32_t spirv_builder_alloc_id(SPIRVBuilder* b) {
    return b->next_id++;
}

/* Emit an instruction: opcode with word count header */
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
    emit_op(b, SpvExecutionModeLocalSize, 6);
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

/* Helper: emit a float constant, returns its ID */
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

/* Helper: emit a uint32 constant */
static uint32_t __attribute__((unused)) emit_uint_constant(SPIRVBuilder* b, uint32_t uint_type, uint32_t value) {
    uint32_t id = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpConstant, 4);
    spirv_builder_emit(b, uint_type);
    spirv_builder_emit(b, id);
    spirv_builder_emit(b, value);
    return id;
}

uint32_t* spirv_builder_finalize(SPIRVBuilder* b, size_t* out_size) {
    uint32_t* result = (uint32_t*)malloc(b->len * sizeof(uint32_t));
    if (!result) return NULL;
    memcpy(result, b->words, b->len * sizeof(uint32_t));
    *out_size = b->len * sizeof(uint32_t);
    return result;
}

/* ── Codegen lifecycle ── */

CMLSPIRVCodegen* cml_spirv_codegen_create(void) {
    CMLSPIRVCodegen* cg = (CMLSPIRVCodegen*)calloc(1, sizeof(CMLSPIRVCodegen));
    if (!cg) return NULL;
    cg->local_size_x = 256;
    cg->local_size_y = 1;
    cg->local_size_z = 1;
    cg->initialized = true;
    return cg;
}

void cml_spirv_codegen_destroy(CMLSPIRVCodegen* cg) {
    free(cg);
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

    SPIRVBuilder* b = spirv_builder_create();
    if (!b) return NULL;

    /* Reserve space for header (5 words) — filled in at the end */
    size_t header_offset = b->len;
    for (int i = 0; i < 5; i++) spirv_builder_emit(b, 0);

    /* OpCapability Shader */
    emit_capability(b);

    /* OpExtInstImport "GLSL.std.450" */
    uint32_t glsl_ext = emit_ext_import(b);

    /* OpMemoryModel Logical GLSL450 */
    emit_memory_model(b);

    /* Pre-allocate IDs for types and variables */
    uint32_t id_void      = spirv_builder_alloc_id(b);
    uint32_t id_bool      = spirv_builder_alloc_id(b);
    uint32_t id_uint      = spirv_builder_alloc_id(b);
    uint32_t id_float     = spirv_builder_alloc_id(b);
    uint32_t id_uint3     = spirv_builder_alloc_id(b);
    uint32_t id_void_fn   = spirv_builder_alloc_id(b);

    /* Runtime array and struct types */
    uint32_t id_rtarray_f = spirv_builder_alloc_id(b); /* float[] */
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

    emit_decorate(b, id_var_in, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_var_in, SpvDecorationBinding, 0);
    emit_decorate(b, id_var_out, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_var_out, SpvDecorationBinding, 1);
    emit_decorate(b, id_var_p, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_var_p, SpvDecorationBinding, 2);

    /* Type declarations */
    emit_op(b, SpvOpTypeVoid, 2); spirv_builder_emit(b, id_void);
    emit_op(b, SpvOpTypeBool, 2); spirv_builder_emit(b, id_bool);
    emit_op(b, SpvOpTypeInt, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, 32); spirv_builder_emit(b, 0); /* 32-bit unsigned */
    emit_op(b, SpvOpTypeFloat, 3); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, 32);
    emit_op(b, SpvOpTypeVector, 4); spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_uint); spirv_builder_emit(b, 3);
    emit_op(b, SpvOpTypeFunction, 3); spirv_builder_emit(b, id_void_fn);
    spirv_builder_emit(b, id_void);

    emit_op(b, SpvOpTypeRuntimeArray, 3); spirv_builder_emit(b, id_rtarray_f);
    spirv_builder_emit(b, id_float);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_struct_in);
    spirv_builder_emit(b, id_rtarray_f);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_struct_out);
    spirv_builder_emit(b, id_rtarray_f);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_struct_p);
    spirv_builder_emit(b, id_uint);

    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_in);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_struct_in);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_out);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_struct_out);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_p);
    spirv_builder_emit(b, SpvStorageClassUniform); spirv_builder_emit(b, id_struct_p);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_f);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_float);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_u);
    spirv_builder_emit(b, SpvStorageClassUniform); spirv_builder_emit(b, id_uint);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_in_u3);
    spirv_builder_emit(b, SpvStorageClassInput); spirv_builder_emit(b, id_uint3);

    /* Constants */
    emit_op(b, SpvOpConstant, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_const_0); spirv_builder_emit(b, 0);

    /* Variables */
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_sb_in);
    spirv_builder_emit(b, id_var_in); spirv_builder_emit(b, SpvStorageClassStorageBuffer);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_sb_out);
    spirv_builder_emit(b, id_var_out); spirv_builder_emit(b, SpvStorageClassStorageBuffer);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_sb_p);
    spirv_builder_emit(b, id_var_p); spirv_builder_emit(b, SpvStorageClassUniform);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_in_u3);
    spirv_builder_emit(b, id_var_gid); spirv_builder_emit(b, SpvStorageClassInput);

    /* Function body */
    uint32_t id_label_entry = spirv_builder_alloc_id(b);
    uint32_t id_label_body  = spirv_builder_alloc_id(b);
    uint32_t id_label_end   = spirv_builder_alloc_id(b);

    emit_op(b, SpvOpFunction, 5); spirv_builder_emit(b, id_void);
    spirv_builder_emit(b, id_main); spirv_builder_emit(b, 0); /* None */
    spirv_builder_emit(b, id_void_fn);

    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_entry);

    /* Load GlobalInvocationID.x */
    uint32_t id_gid_vec = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_gid_vec); spirv_builder_emit(b, id_var_gid);

    uint32_t id_gid_x = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpCompositeExtract, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_gid_x); spirv_builder_emit(b, id_gid_vec);
    spirv_builder_emit(b, 0); /* component 0 */

    /* Load n from params */
    uint32_t id_n_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 5); spirv_builder_emit(b, id_ptr_sb_u);
    spirv_builder_emit(b, id_n_ptr); spirv_builder_emit(b, id_var_p);
    spirv_builder_emit(b, id_const_0);

    uint32_t id_n = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_n); spirv_builder_emit(b, id_n_ptr);

    /* Bounds check: if (idx >= n) return */
    uint32_t id_cmp = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpULessThan, 5); spirv_builder_emit(b, id_bool);
    spirv_builder_emit(b, id_cmp); spirv_builder_emit(b, id_gid_x);
    spirv_builder_emit(b, id_n);

    emit_op(b, SpvOpSelectionMerge, 3);
    spirv_builder_emit(b, id_label_end); spirv_builder_emit(b, 0);
    emit_op(b, SpvOpBranchConditional, 4);
    spirv_builder_emit(b, id_cmp); spirv_builder_emit(b, id_label_body);
    spirv_builder_emit(b, id_label_end);

    /* Body: load input[idx] */
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_body);

    uint32_t id_in_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_ptr_sb_f);
    spirv_builder_emit(b, id_in_ptr); spirv_builder_emit(b, id_var_in);
    spirv_builder_emit(b, id_const_0); spirv_builder_emit(b, id_gid_x);

    uint32_t id_val = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_val); spirv_builder_emit(b, id_in_ptr);

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
        /* Unsupported op: just pass through */
        id_result = id_val;
        break;
    }

    /* Store result to output[idx] */
    uint32_t id_out_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_ptr_sb_f);
    spirv_builder_emit(b, id_out_ptr); spirv_builder_emit(b, id_var_out);
    spirv_builder_emit(b, id_const_0); spirv_builder_emit(b, id_gid_x);

    emit_op(b, SpvOpStore, 3);
    spirv_builder_emit(b, id_out_ptr); spirv_builder_emit(b, id_result);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_label_end);

    /* End label */
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_end);
    emit_op(b, SpvOpReturn, 1);
    emit_op(b, SpvOpFunctionEnd, 1);

    /* Patch header */
    b->words[header_offset + 0] = SPIRV_MAGIC;
    b->words[header_offset + 1] = SPIRV_VERSION;
    b->words[header_offset + 2] = SPIRV_GENERATOR;
    b->words[header_offset + 3] = b->next_id; /* bound */
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

    SPIRVBuilder* b = spirv_builder_create();
    if (!b) return NULL;

    /* Header placeholder */
    for (int i = 0; i < 5; i++) spirv_builder_emit(b, 0);

    emit_capability(b);
    uint32_t glsl_ext = emit_ext_import(b);
    emit_memory_model(b);

    /* Allocate type IDs */
    uint32_t id_void    = spirv_builder_alloc_id(b);
    uint32_t id_bool    = spirv_builder_alloc_id(b);
    uint32_t id_uint    = spirv_builder_alloc_id(b);
    uint32_t id_float   = spirv_builder_alloc_id(b);
    uint32_t id_uint3   = spirv_builder_alloc_id(b);
    uint32_t id_void_fn = spirv_builder_alloc_id(b);

    uint32_t id_rtarray = spirv_builder_alloc_id(b);
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
    emit_decorate(b, id_va, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_va, SpvDecorationBinding, 0);
    emit_decorate(b, id_vo, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_vo, SpvDecorationBinding, 1);
    emit_decorate(b, id_vb, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_vb, SpvDecorationBinding, 2);

    /* Types */
    emit_op(b, SpvOpTypeVoid, 2); spirv_builder_emit(b, id_void);
    emit_op(b, SpvOpTypeBool, 2); spirv_builder_emit(b, id_bool);
    (void)id_bool;
    emit_op(b, SpvOpTypeInt, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, 32); spirv_builder_emit(b, 0);
    emit_op(b, SpvOpTypeFloat, 3); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, 32);
    emit_op(b, SpvOpTypeVector, 4); spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_uint); spirv_builder_emit(b, 3);
    emit_op(b, SpvOpTypeFunction, 3); spirv_builder_emit(b, id_void_fn);
    spirv_builder_emit(b, id_void);

    emit_op(b, SpvOpTypeRuntimeArray, 3); spirv_builder_emit(b, id_rtarray);
    spirv_builder_emit(b, id_float);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_s_a);
    spirv_builder_emit(b, id_rtarray);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_s_out);
    spirv_builder_emit(b, id_rtarray);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_s_b);
    spirv_builder_emit(b, id_rtarray);

    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_p_sb_a);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_s_a);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_p_sb_o);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_s_out);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_p_sb_b);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_s_b);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_p_sb_f);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_float);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_p_in_u3);
    spirv_builder_emit(b, SpvStorageClassInput); spirv_builder_emit(b, id_uint3);

    /* Constants */
    emit_op(b, SpvOpConstant, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_c0); spirv_builder_emit(b, 0);

    /* Variables */
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_p_sb_a);
    spirv_builder_emit(b, id_va); spirv_builder_emit(b, SpvStorageClassStorageBuffer);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_p_sb_o);
    spirv_builder_emit(b, id_vo); spirv_builder_emit(b, SpvStorageClassStorageBuffer);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_p_sb_b);
    spirv_builder_emit(b, id_vb); spirv_builder_emit(b, SpvStorageClassStorageBuffer);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_p_in_u3);
    spirv_builder_emit(b, id_vgid); spirv_builder_emit(b, SpvStorageClassInput);

    /* Function */
    uint32_t id_l_entry = spirv_builder_alloc_id(b);

    emit_op(b, SpvOpFunction, 5); spirv_builder_emit(b, id_void);
    spirv_builder_emit(b, id_main); spirv_builder_emit(b, 0);
    spirv_builder_emit(b, id_void_fn);
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_l_entry);

    /* Load global invocation ID.x */
    uint32_t id_gv = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_gv); spirv_builder_emit(b, id_vgid);
    uint32_t id_gx = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpCompositeExtract, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_gx); spirv_builder_emit(b, id_gv);
    spirv_builder_emit(b, 0);

    /* Load A[idx] */
    uint32_t id_ap = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_p_sb_f);
    spirv_builder_emit(b, id_ap); spirv_builder_emit(b, id_va);
    spirv_builder_emit(b, id_c0); spirv_builder_emit(b, id_gx);
    uint32_t id_a = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_a); spirv_builder_emit(b, id_ap);

    /* Load B[idx] */
    uint32_t id_bp = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_p_sb_f);
    spirv_builder_emit(b, id_bp); spirv_builder_emit(b, id_vb);
    spirv_builder_emit(b, id_c0); spirv_builder_emit(b, id_gx);
    uint32_t id_bv = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_bv); spirv_builder_emit(b, id_bp);

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
        /* Fallback: add */
        emit_op(b, SpvOpFAdd, 5); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_res); spirv_builder_emit(b, id_a);
        spirv_builder_emit(b, id_bv);
        break;
    }

    /* Store result */
    uint32_t id_op = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_p_sb_f);
    spirv_builder_emit(b, id_op); spirv_builder_emit(b, id_vo);
    spirv_builder_emit(b, id_c0); spirv_builder_emit(b, id_gx);
    emit_op(b, SpvOpStore, 3);
    spirv_builder_emit(b, id_op); spirv_builder_emit(b, id_res);

    emit_op(b, SpvOpReturn, 1);
    emit_op(b, SpvOpFunctionEnd, 1);

    /* Patch header */
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

uint32_t* cml_spirv_gen_reduction(CMLSPIRVCodegen* cg, UOpType op, const char* name,
                                   size_t* out_size) {
    (void)name; (void)op;
    if (!cg || !out_size) return NULL;

    /* Stub: reduction kernels require shared memory and barriers.
     * For now, return NULL to signal that reductions should fall back to CPU. */
    LOG_DEBUG("SPIR-V reduction codegen not yet implemented for op %d", (int)op);
    *out_size = 0;
    return NULL;
}

uint32_t* cml_spirv_gen_matmul(CMLSPIRVCodegen* cg, const char* name, size_t* out_size) {
    (void)name;
    if (!cg || !out_size) return NULL;

    /* Stub: matmul requires 2D dispatch and tiling.
     * For now, return NULL to fall back to CPU. */
    LOG_DEBUG("SPIR-V matmul codegen not yet implemented");
    *out_size = 0;
    return NULL;
}

uint32_t* cml_spirv_gen_fill(CMLSPIRVCodegen* cg, float value, const char* name,
                              size_t* out_size) {
    (void)name; (void)value;
    if (!cg || !out_size) return NULL;

    LOG_DEBUG("SPIR-V fill codegen not yet implemented");
    *out_size = 0;
    return NULL;
}
