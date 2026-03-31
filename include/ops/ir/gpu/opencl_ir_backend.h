/*
 * OpenCL compute backend for IR graph execution.
 * Compiles static OpenCL C kernels at init, executes IR graphs on GPU.
 */

#ifndef CML_GPU_OPENCL_IR_BACKEND_H
#define CML_GPU_OPENCL_IR_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Tensor;
typedef struct Tensor Tensor;
struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

#ifdef CML_HAS_OPENCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>
#endif

#define CML_OCL_MAX_TRACKED_BUFFERS 512
#define CML_OCL_MAX_GEMM_VARIANTS   128
#define CML_OCL_GEMM_CACHE_SIZE     64

typedef struct CMLGemmVariantParams {
    int tsm;            /* tile size M: 64 or 128 */
    int tsn;            /* tile size N: 64 or 128 */
    int tsk;            /* tile size K: 8 or 16 */
    int reg_m;          /* register block rows: 4 or 8 */
    int reg_n;          /* register block cols: 4 or 8 */
    int slm_pad;        /* SLM row padding: 0 or 1 */
    bool transpose_a;   /* store A transposed in SLM */
} CMLGemmVariantParams;

typedef struct CMLGemmVariant {
    CMLGemmVariantParams params;
    cl_program  program;
    cl_kernel   kernel;
    size_t      local_size[2];   /* precomputed workgroup size */
    bool        valid;
} CMLGemmVariant;

typedef struct CMLGemmCacheEntry {
    uint64_t key;           /* (M<<40)|(N<<20)|K */
    int      variant_idx;
    bool     occupied;
} CMLGemmCacheEntry;

typedef struct CMLOCLBufferEntry {
    Tensor*  tensor;      /* tensor pointer (for current graph) */
    void*    data_ptr;    /* CPU data pointer — used as cache key for inputs */
    cl_mem   gpu_buf;
    size_t   size;
    bool     valid;       /* GPU data is up-to-date */
    bool     is_input;    /* true = leaf input, persists across graph executions */
} CMLOCLBufferEntry;

typedef struct CMLOpenCLIRBackend {
    bool initialized;

    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue  queue;
    cl_program       program;

    char device_name[256];
    size_t total_memory;
    uint32_t max_work_group_size;
    uint32_t max_compute_units;

    /* Pre-compiled kernels */
    cl_kernel k_matmul_naive;
    cl_kernel k_matmul;
    cl_kernel k_matmul_fused_bias_relu;
    cl_kernel k_add;
    cl_kernel k_sub;
    cl_kernel k_mul;
    cl_kernel k_div;
    cl_kernel k_neg;
    cl_kernel k_relu;
    cl_kernel k_sigmoid;
    cl_kernel k_tanh;
    cl_kernel k_exp;
    cl_kernel k_log;
    cl_kernel k_sqrt;
    cl_kernel k_sum_reduce;
    cl_kernel k_max_reduce;
    cl_kernel k_mean_reduce;
    cl_kernel k_fill;

    CMLOCLBufferEntry buffers[CML_OCL_MAX_TRACKED_BUFFERS];
    int buffer_count;

    CMLGemmVariant    gemm_variants[CML_OCL_MAX_GEMM_VARIANTS];
    int               gemm_variant_count;
    CMLGemmCacheEntry gemm_cache[CML_OCL_GEMM_CACHE_SIZE];
    cl_command_queue  profiling_queue;
    int               beam_width;       /* 0=off, N=search top-N variants per shape */
} CMLOpenCLIRBackend;

bool                 cml_opencl_ir_available(void);
CMLOpenCLIRBackend*  cml_opencl_ir_backend_create(void);
int                  cml_opencl_ir_backend_init(CMLOpenCLIRBackend* backend);
void                 cml_opencl_ir_backend_free(CMLOpenCLIRBackend* backend);
int                  cml_opencl_execute_graph(CMLOpenCLIRBackend* backend, CMLGraph_t ir);

#else /* !CML_HAS_OPENCL */

typedef struct CMLOpenCLIRBackend {
    bool initialized;
} CMLOpenCLIRBackend;

static inline bool                cml_opencl_ir_available(void) { return false; }
static inline CMLOpenCLIRBackend* cml_opencl_ir_backend_create(void) { return NULL; }
static inline int                 cml_opencl_ir_backend_init(CMLOpenCLIRBackend* b) { (void)b; return -1; }
static inline void                cml_opencl_ir_backend_free(CMLOpenCLIRBackend* b) { (void)b; }
static inline int                 cml_opencl_execute_graph(CMLOpenCLIRBackend* b, CMLGraph_t ir) { (void)b; (void)ir; return -1; }

#endif /* CML_HAS_OPENCL */

#ifdef __cplusplus
}
#endif

#endif /* CML_GPU_OPENCL_IR_BACKEND_H */
