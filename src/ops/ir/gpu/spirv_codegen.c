#include "ops/ir/gpu/spirv_codegen.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>


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
    uint32_t* result = (uint32_t*)malloc(b->len * sizeof(uint32_t));
    if (!result) return NULL;
    memcpy(result, b->words, b->len * sizeof(uint32_t));
    *out_size = b->len * sizeof(uint32_t);
    return result;
}


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

    SPIRVBuilder* b = spirv_builder_create();
    if (!b) return NULL;

    /* Header placeholder */
    for (int i = 0; i < 5; i++) spirv_builder_emit(b, 0);

    emit_capability(b);
    emit_ext_import(b);
    emit_memory_model(b);

    /* Pre-allocate type IDs */
    uint32_t id_void      = spirv_builder_alloc_id(b);
    uint32_t id_bool      = spirv_builder_alloc_id(b);
    uint32_t id_uint      = spirv_builder_alloc_id(b);
    uint32_t id_float     = spirv_builder_alloc_id(b);
    uint32_t id_uint3     = spirv_builder_alloc_id(b);
    uint32_t id_void_fn   = spirv_builder_alloc_id(b);

    uint32_t id_rtarray_f = spirv_builder_alloc_id(b);
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
    spirv_builder_emit(b, 32); spirv_builder_emit(b, 0);
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

    /* Pointer types */
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
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_wg_f);
    spirv_builder_emit(b, SpvStorageClassWorkgroup); spirv_builder_emit(b, id_float);

    /* float[256] array type */
    emit_op(b, SpvOpConstant, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_const_256u); spirv_builder_emit(b, 256);
    emit_op(b, SpvOpTypeArray, 4); spirv_builder_emit(b, id_arr_256);
    spirv_builder_emit(b, id_float); spirv_builder_emit(b, id_const_256u);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_wg_arr);
    spirv_builder_emit(b, SpvStorageClassWorkgroup); spirv_builder_emit(b, id_arr_256);

    /* Constants */
    emit_op(b, SpvOpConstant, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_const_0u); spirv_builder_emit(b, 0);
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
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_sb_in);
    spirv_builder_emit(b, id_var_in); spirv_builder_emit(b, SpvStorageClassStorageBuffer);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_sb_out);
    spirv_builder_emit(b, id_var_out); spirv_builder_emit(b, SpvStorageClassStorageBuffer);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_sb_p);
    spirv_builder_emit(b, id_var_p); spirv_builder_emit(b, SpvStorageClassUniform);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_in_u3);
    spirv_builder_emit(b, id_var_gid); spirv_builder_emit(b, SpvStorageClassInput);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_in_u3);
    spirv_builder_emit(b, id_var_lid); spirv_builder_emit(b, SpvStorageClassInput);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_wg_arr);
    spirv_builder_emit(b, id_var_sdata); spirv_builder_emit(b, SpvStorageClassWorkgroup);

    uint32_t id_label_entry = spirv_builder_alloc_id(b);

    emit_op(b, SpvOpFunction, 5); spirv_builder_emit(b, id_void);
    spirv_builder_emit(b, id_main); spirv_builder_emit(b, 0);
    spirv_builder_emit(b, id_void_fn);
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_entry);

    /* Load LocalInvocationID.x (tid) */
    uint32_t id_lid_vec = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_lid_vec); spirv_builder_emit(b, id_var_lid);
    uint32_t id_tid = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpCompositeExtract, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_tid); spirv_builder_emit(b, id_lid_vec);
    spirv_builder_emit(b, 0);

    /* Load GlobalInvocationID.x (gid) */
    uint32_t id_gid_vec = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_gid_vec); spirv_builder_emit(b, id_var_gid);
    uint32_t id_gid_x = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpCompositeExtract, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_gid_x); spirv_builder_emit(b, id_gid_vec);
    spirv_builder_emit(b, 0);

    /* Load n from params */
    uint32_t id_n_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 5); spirv_builder_emit(b, id_ptr_sb_u);
    spirv_builder_emit(b, id_n_ptr); spirv_builder_emit(b, id_var_p);
    spirv_builder_emit(b, id_const_0u);
    uint32_t id_n = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_n); spirv_builder_emit(b, id_n_ptr);

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
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_load);
    uint32_t id_in_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_ptr_sb_f);
    spirv_builder_emit(b, id_in_ptr); spirv_builder_emit(b, id_var_in);
    spirv_builder_emit(b, id_const_0u); spirv_builder_emit(b, id_gid_x);
    uint32_t id_loaded_val = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_loaded_val); spirv_builder_emit(b, id_in_ptr);
    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_label_merge);

    /* Skip path: use 0.0 */
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_skip);
    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_label_merge);

    /* Merge: phi to pick loaded value or 0.0 */
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_merge);
    uint32_t id_init_val = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpPhi, 7); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_init_val);
    spirv_builder_emit(b, id_loaded_val); spirv_builder_emit(b, id_label_load);
    spirv_builder_emit(b, id_const_0f); spirv_builder_emit(b, id_label_skip);

    /* Store to shared memory: sdata[tid] = init_val */
    uint32_t id_sdata_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 5); spirv_builder_emit(b, id_ptr_wg_f);
    spirv_builder_emit(b, id_sdata_ptr); spirv_builder_emit(b, id_var_sdata);
    spirv_builder_emit(b, id_tid);
    emit_op(b, SpvOpStore, 3);
    spirv_builder_emit(b, id_sdata_ptr); spirv_builder_emit(b, id_init_val);

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

        emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_lbl_reduce);

        /* Load sdata[tid] */
        uint32_t id_sd_ptr1 = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpAccessChain, 5); spirv_builder_emit(b, id_ptr_wg_f);
        spirv_builder_emit(b, id_sd_ptr1); spirv_builder_emit(b, id_var_sdata);
        spirv_builder_emit(b, id_tid);
        uint32_t id_sd_val1 = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_sd_val1); spirv_builder_emit(b, id_sd_ptr1);

        /* tid + stride */
        uint32_t id_tid_plus = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpIAdd, 5); spirv_builder_emit(b, id_uint);
        spirv_builder_emit(b, id_tid_plus); spirv_builder_emit(b, id_tid);
        spirv_builder_emit(b, id_stride[step]);

        /* Load sdata[tid + stride] */
        uint32_t id_sd_ptr2 = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpAccessChain, 5); spirv_builder_emit(b, id_ptr_wg_f);
        spirv_builder_emit(b, id_sd_ptr2); spirv_builder_emit(b, id_var_sdata);
        spirv_builder_emit(b, id_tid_plus);
        uint32_t id_sd_val2 = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_sd_val2); spirv_builder_emit(b, id_sd_ptr2);

        /* sdata[tid] += sdata[tid + stride] */
        uint32_t id_sum = spirv_builder_alloc_id(b);
        emit_op(b, SpvOpFAdd, 5); spirv_builder_emit(b, id_float);
        spirv_builder_emit(b, id_sum); spirv_builder_emit(b, id_sd_val1);
        spirv_builder_emit(b, id_sd_val2);
        emit_op(b, SpvOpStore, 3);
        spirv_builder_emit(b, id_sd_ptr1); spirv_builder_emit(b, id_sum);

        emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_lbl_skip2);
        emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_lbl_skip2);

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

    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_lbl_write);

    /* Load sdata[0] */
    uint32_t id_res_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 5); spirv_builder_emit(b, id_ptr_wg_f);
    spirv_builder_emit(b, id_res_ptr); spirv_builder_emit(b, id_var_sdata);
    spirv_builder_emit(b, id_const_0u);
    uint32_t id_res_val = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_res_val); spirv_builder_emit(b, id_res_ptr);

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
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_ptr_sb_f);
    spirv_builder_emit(b, id_out_ptr); spirv_builder_emit(b, id_var_out);
    spirv_builder_emit(b, id_const_0u); spirv_builder_emit(b, id_const_0u);
    emit_op(b, SpvOpStore, 3);
    spirv_builder_emit(b, id_out_ptr); spirv_builder_emit(b, id_final_val);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_lbl_end);

    /* End */
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_lbl_end);
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

    SPIRVBuilder* b = spirv_builder_create();
    if (!b) return NULL;

    /* Header placeholder */
    for (int i = 0; i < 5; i++) spirv_builder_emit(b, 0);

    emit_capability(b);
    emit_ext_import(b);
    emit_memory_model(b);

    /* Pre-allocate type IDs */
    uint32_t id_void      = spirv_builder_alloc_id(b);
    uint32_t id_bool      = spirv_builder_alloc_id(b);
    uint32_t id_uint      = spirv_builder_alloc_id(b);
    uint32_t id_float     = spirv_builder_alloc_id(b);
    uint32_t id_uint3     = spirv_builder_alloc_id(b);
    uint32_t id_void_fn   = spirv_builder_alloc_id(b);

    uint32_t id_rtarray_f = spirv_builder_alloc_id(b);
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

    emit_decorate(b, id_var_a, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_var_a, SpvDecorationBinding, 0);
    emit_decorate(b, id_var_b, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_var_b, SpvDecorationBinding, 1);
    emit_decorate(b, id_var_c, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_var_c, SpvDecorationBinding, 2);
    emit_decorate(b, id_var_p, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_var_p, SpvDecorationBinding, 3);

    /* Type declarations */
    emit_op(b, SpvOpTypeVoid, 2); spirv_builder_emit(b, id_void);
    emit_op(b, SpvOpTypeBool, 2); spirv_builder_emit(b, id_bool);
    emit_op(b, SpvOpTypeInt, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, 32); spirv_builder_emit(b, 0);
    emit_op(b, SpvOpTypeFloat, 3); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, 32);
    emit_op(b, SpvOpTypeVector, 4); spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_uint); spirv_builder_emit(b, 3);
    emit_op(b, SpvOpTypeFunction, 3); spirv_builder_emit(b, id_void_fn);
    spirv_builder_emit(b, id_void);

    emit_op(b, SpvOpTypeRuntimeArray, 3); spirv_builder_emit(b, id_rtarray_f);
    spirv_builder_emit(b, id_float);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_struct_a);
    spirv_builder_emit(b, id_rtarray_f);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_struct_b);
    spirv_builder_emit(b, id_rtarray_f);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_struct_c);
    spirv_builder_emit(b, id_rtarray_f);
    /* Params struct: { uint M, uint N, uint K } */
    emit_op(b, SpvOpTypeStruct, 5); spirv_builder_emit(b, id_struct_p);
    spirv_builder_emit(b, id_uint); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_uint);

    /* Pointer types */
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_a);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_struct_a);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_b);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_struct_b);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_c);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_struct_c);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_p);
    spirv_builder_emit(b, SpvStorageClassUniform); spirv_builder_emit(b, id_struct_p);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_f);
    spirv_builder_emit(b, SpvStorageClassStorageBuffer); spirv_builder_emit(b, id_float);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_sb_u);
    spirv_builder_emit(b, SpvStorageClassUniform); spirv_builder_emit(b, id_uint);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_in_u3);
    spirv_builder_emit(b, SpvStorageClassInput); spirv_builder_emit(b, id_uint3);
    emit_op(b, SpvOpTypePointer, 4); spirv_builder_emit(b, id_ptr_fn_f);
    spirv_builder_emit(b, SpvStorageClassFunction); spirv_builder_emit(b, id_float);

    /* Constants */
    emit_op(b, SpvOpConstant, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_const_0u); spirv_builder_emit(b, 0);
    emit_op(b, SpvOpConstant, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_const_1u); spirv_builder_emit(b, 1);
    emit_op(b, SpvOpConstant, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_const_2u); spirv_builder_emit(b, 2);
    uint32_t id_const_0f = emit_float_constant(b, id_float, 0.0f);

    /* Variables */
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_sb_a);
    spirv_builder_emit(b, id_var_a); spirv_builder_emit(b, SpvStorageClassStorageBuffer);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_sb_b);
    spirv_builder_emit(b, id_var_b); spirv_builder_emit(b, SpvStorageClassStorageBuffer);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_sb_c);
    spirv_builder_emit(b, id_var_c); spirv_builder_emit(b, SpvStorageClassStorageBuffer);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_sb_p);
    spirv_builder_emit(b, id_var_p); spirv_builder_emit(b, SpvStorageClassUniform);
    emit_op(b, SpvOpVariable, 4); spirv_builder_emit(b, id_ptr_in_u3);
    spirv_builder_emit(b, id_var_gid); spirv_builder_emit(b, SpvStorageClassInput);

    uint32_t id_label_entry = spirv_builder_alloc_id(b);
    uint32_t id_label_body  = spirv_builder_alloc_id(b);
    uint32_t id_label_end   = spirv_builder_alloc_id(b);

    emit_op(b, SpvOpFunction, 5); spirv_builder_emit(b, id_void);
    spirv_builder_emit(b, id_main); spirv_builder_emit(b, 0);
    spirv_builder_emit(b, id_void_fn);
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_entry);

    /* Allocate function-local variable for accumulator */
    uint32_t id_var_sum = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpVariable, 5); spirv_builder_emit(b, id_ptr_fn_f);
    spirv_builder_emit(b, id_var_sum); spirv_builder_emit(b, SpvStorageClassFunction);
    spirv_builder_emit(b, id_const_0f);

    /* Load GlobalInvocationID */
    uint32_t id_gid_vec = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_gid_vec); spirv_builder_emit(b, id_var_gid);

    uint32_t id_col = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpCompositeExtract, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_col); spirv_builder_emit(b, id_gid_vec);
    spirv_builder_emit(b, 0);

    uint32_t id_row = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpCompositeExtract, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_row); spirv_builder_emit(b, id_gid_vec);
    spirv_builder_emit(b, 1);

    /* Load M, N, K from params */
    uint32_t id_m_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 5); spirv_builder_emit(b, id_ptr_sb_u);
    spirv_builder_emit(b, id_m_ptr); spirv_builder_emit(b, id_var_p);
    spirv_builder_emit(b, id_const_0u);
    uint32_t id_M = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_M); spirv_builder_emit(b, id_m_ptr);

    uint32_t id_n_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 5); spirv_builder_emit(b, id_ptr_sb_u);
    spirv_builder_emit(b, id_n_ptr); spirv_builder_emit(b, id_var_p);
    spirv_builder_emit(b, id_const_1u);
    uint32_t id_N = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_N); spirv_builder_emit(b, id_n_ptr);

    uint32_t id_k_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 5); spirv_builder_emit(b, id_ptr_sb_u);
    spirv_builder_emit(b, id_k_ptr); spirv_builder_emit(b, id_var_p);
    spirv_builder_emit(b, id_const_2u);
    uint32_t id_K = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_K); spirv_builder_emit(b, id_k_ptr);

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
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_body);

    /* Loop: for (k = 0; k < K; k++) */
    uint32_t id_lbl_loop_hdr  = spirv_builder_alloc_id(b);
    uint32_t id_lbl_loop_body = spirv_builder_alloc_id(b);
    uint32_t id_lbl_loop_cont = spirv_builder_alloc_id(b);
    uint32_t id_lbl_loop_end  = spirv_builder_alloc_id(b);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_lbl_loop_hdr);

    /* Loop header */
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_lbl_loop_hdr);

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
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_lbl_loop_body);

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
    uint32_t id_a_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_ptr_sb_f);
    spirv_builder_emit(b, id_a_ptr); spirv_builder_emit(b, id_var_a);
    spirv_builder_emit(b, id_const_0u); spirv_builder_emit(b, id_a_idx);
    uint32_t id_a_val = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_a_val); spirv_builder_emit(b, id_a_ptr);

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
    uint32_t id_b_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_ptr_sb_f);
    spirv_builder_emit(b, id_b_ptr); spirv_builder_emit(b, id_var_b);
    spirv_builder_emit(b, id_const_0u); spirv_builder_emit(b, id_b_idx);
    uint32_t id_b_val = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, id_b_val); spirv_builder_emit(b, id_b_ptr);

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
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_lbl_loop_cont);
    uint32_t id_k_next = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpIAdd, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_k_next); spirv_builder_emit(b, id_k);
    spirv_builder_emit(b, id_const_1u);

    /* Patch phi placeholders */
    b->words[phi_k_next_slot] = id_k_next;
    b->words[phi_sum_next_slot] = id_sum_next;

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_lbl_loop_hdr);

    /* Loop end */
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_lbl_loop_end);

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
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_ptr_sb_f);
    spirv_builder_emit(b, id_c_ptr); spirv_builder_emit(b, id_var_c);
    spirv_builder_emit(b, id_const_0u); spirv_builder_emit(b, id_c_idx);
    emit_op(b, SpvOpStore, 3);
    spirv_builder_emit(b, id_c_ptr); spirv_builder_emit(b, id_sum_phi);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_label_end);

    /* End label */
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_end);
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

    SPIRVBuilder* b = spirv_builder_create();
    if (!b) return NULL;

    /* Reserve space for header (5 words) */
    for (int i = 0; i < 5; i++) spirv_builder_emit(b, 0);

    emit_capability(b);
    emit_ext_import(b);
    emit_memory_model(b);

    /* Pre-allocate IDs */
    uint32_t id_void      = spirv_builder_alloc_id(b);
    uint32_t id_bool      = spirv_builder_alloc_id(b);
    uint32_t id_uint      = spirv_builder_alloc_id(b);
    uint32_t id_float     = spirv_builder_alloc_id(b);
    uint32_t id_uint3     = spirv_builder_alloc_id(b);
    uint32_t id_void_fn   = spirv_builder_alloc_id(b);

    uint32_t id_rtarray_f = spirv_builder_alloc_id(b);
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
    emit_decorate(b, id_var_out, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_var_out, SpvDecorationBinding, 0);
    emit_decorate(b, id_var_p, SpvDecorationDescriptorSet, 0);
    emit_decorate(b, id_var_p, SpvDecorationBinding, 1);

    /* Type declarations */
    emit_op(b, SpvOpTypeVoid, 2); spirv_builder_emit(b, id_void);
    emit_op(b, SpvOpTypeBool, 2); spirv_builder_emit(b, id_bool);
    emit_op(b, SpvOpTypeInt, 4); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, 32); spirv_builder_emit(b, 0);
    emit_op(b, SpvOpTypeFloat, 3); spirv_builder_emit(b, id_float);
    spirv_builder_emit(b, 32);
    emit_op(b, SpvOpTypeVector, 4); spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_uint); spirv_builder_emit(b, 3);
    emit_op(b, SpvOpTypeFunction, 3); spirv_builder_emit(b, id_void_fn);
    spirv_builder_emit(b, id_void);

    emit_op(b, SpvOpTypeRuntimeArray, 3); spirv_builder_emit(b, id_rtarray_f);
    spirv_builder_emit(b, id_float);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_struct_out);
    spirv_builder_emit(b, id_rtarray_f);
    emit_op(b, SpvOpTypeStruct, 3); spirv_builder_emit(b, id_struct_p);
    spirv_builder_emit(b, id_uint);

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

    uint32_t id_fill_val = emit_float_constant(b, id_float, value);

    /* Variables */
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
    spirv_builder_emit(b, id_main); spirv_builder_emit(b, 0);
    spirv_builder_emit(b, id_void_fn);
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_entry);

    /* Load GlobalInvocationID.x */
    uint32_t id_gid_vec = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpLoad, 4); spirv_builder_emit(b, id_uint3);
    spirv_builder_emit(b, id_gid_vec); spirv_builder_emit(b, id_var_gid);

    uint32_t id_gid_x = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpCompositeExtract, 5); spirv_builder_emit(b, id_uint);
    spirv_builder_emit(b, id_gid_x); spirv_builder_emit(b, id_gid_vec);
    spirv_builder_emit(b, 0);

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

    /* Body: store fill value to output[idx] */
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_body);

    uint32_t id_out_ptr = spirv_builder_alloc_id(b);
    emit_op(b, SpvOpAccessChain, 6); spirv_builder_emit(b, id_ptr_sb_f);
    spirv_builder_emit(b, id_out_ptr); spirv_builder_emit(b, id_var_out);
    spirv_builder_emit(b, id_const_0); spirv_builder_emit(b, id_gid_x);

    emit_op(b, SpvOpStore, 3);
    spirv_builder_emit(b, id_out_ptr); spirv_builder_emit(b, id_fill_val);

    emit_op(b, SpvOpBranch, 2); spirv_builder_emit(b, id_label_end);

    /* End label */
    emit_op(b, SpvOpLabel, 2); spirv_builder_emit(b, id_label_end);
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
