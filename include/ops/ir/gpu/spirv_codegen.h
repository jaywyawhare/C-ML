/*
 * SPIR-V binary code generation for Vulkan compute shaders.
 * Generates SPIR-V binary modules from UOps IR. Emits compute shaders with
 * storage buffer bindings and GlobalInvocationID-based thread indexing.
 */

#ifndef CML_GPU_SPIRV_CODEGEN_H
#define CML_GPU_SPIRV_CODEGEN_H

#include "ops/uops.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

typedef struct SPIRVBuilder {
    uint32_t* words;
    size_t    len;
    size_t    cap;
    uint32_t  next_id;    /* next SSA ID to allocate */

    /* Cached type IDs */
    uint32_t id_void;
    uint32_t id_bool;
    uint32_t id_uint;
    uint32_t id_int;
    uint32_t id_float;
    uint32_t id_uint3;     /* uvec3 for GlobalInvocationId */
    uint32_t id_ptr_uint3; /* input ptr to uvec3 */
    uint32_t id_ptr_storage_float;  /* storage buffer ptr to float */
    uint32_t id_ptr_storage_uint;   /* storage buffer ptr to uint */
    uint32_t id_ptr_uniform_uint;   /* uniform ptr to uint (for n param) */
    uint32_t id_void_fn;  /* void(void) function type */
    uint32_t id_glsl_ext; /* GLSL.std.450 extended instruction set */

    uint32_t id_global_invocation_id;
} SPIRVBuilder;

typedef struct CMLSPIRVCodegen {
    int  local_size_x;  /* workgroup size X (default: 256) */
    int  local_size_y;  /* workgroup size Y (default: 1) */
    int  local_size_z;  /* workgroup size Z (default: 1) */
    int  kernel_count;
    bool initialized;
} CMLSPIRVCodegen;

CMLSPIRVCodegen* cml_spirv_codegen_create(void);
void             cml_spirv_codegen_destroy(CMLSPIRVCodegen* cg);

/* Each function returns a heap-allocated uint32_t array of SPIR-V words.
   Caller must free(). out_size is set to the size in bytes. */
uint32_t* cml_spirv_gen_unary(CMLSPIRVCodegen* cg, UOpType op, const char* name,
                               size_t* out_size);
uint32_t* cml_spirv_gen_binary(CMLSPIRVCodegen* cg, UOpType op, const char* name,
                                size_t* out_size);
uint32_t* cml_spirv_gen_reduction(CMLSPIRVCodegen* cg, UOpType op, const char* name,
                                   size_t* out_size);
uint32_t* cml_spirv_gen_matmul(CMLSPIRVCodegen* cg, const char* name, size_t* out_size);
uint32_t* cml_spirv_gen_fill(CMLSPIRVCodegen* cg, float value, const char* name,
                              size_t* out_size);

SPIRVBuilder* spirv_builder_create(void);
void          spirv_builder_destroy(SPIRVBuilder* b);
void          spirv_builder_emit(SPIRVBuilder* b, uint32_t word);
uint32_t      spirv_builder_alloc_id(SPIRVBuilder* b);
uint32_t*     spirv_builder_finalize(SPIRVBuilder* b, size_t* out_size);

#ifdef __cplusplus
}
#endif

#endif /* CML_GPU_SPIRV_CODEGEN_H */
