# Compiler & IR Pipeline

End-to-end reference for the CML compilation pipeline, from IR graph construction through scheduling, linearization, code generation, caching, and execution.

For related topics see: [Graph Mode](graph_mode.md), [Linearization & Fused Codegen](linearization.md), [Optimizations](optimizations.md).

---

## Pipeline Overview

```
IR Graph (CMLGraph_t)
    |
    v
[1] Pattern Matching & Rewrites  (pattern_matcher.h)
    |   - Algebraic simplifications
    |   - Dead code elimination
    |
    v
[2] Z3 Formal Verification       (z3_verify.h)  [optional]
    |   - Prove rewrite correctness
    |   - Bounds checking
    |
    v
[3] Scheduling & Fusion           (schedule.h)
    |   - Group ops into CMLFusionGroup[]
    |   - Detect buffer elimination
    |
    v
[4] Linearization                  (fused_codegen.h / linearize.c)
    |   - Convert groups to CMLLinearProgram
    |   - Virtual register allocation
    |   See: linearization.md
    |
    v
[5] Fused Code Generation         (fused_codegen.h)
    |   - Lower to C / PTX / SPIR-V / WGSL / Metal
    |   See: linearization.md
    |
    v
[6] Kernel Cache                   (kernel_cache.h)
    |   - LRU cache with FNV-1a hash
    |   - Thread-safe, memory-limited
    |
    v
[7] Execution
    |-- Runtime JIT   (runtime_compiler.h)
    |-- AOT compile   (aot.h)
    |-- TinyJit replay (tiny_jit.h)
    |-- Graph capture  (graph_capture.h)
```

---

## Stage 1: Pattern Matching & Rewrites

**Header:** `include/ops/ir/pattern_matcher.h`

Declarative `(pattern -> replacement)` rules that rewrite IR subgraphs until convergence. This is the primary optimization pass before scheduling.

### Key Types

| Type | Purpose |
|------|---------|
| `CMLPatternNode` | A pattern tree node: `CML_PAT_OP` (match opcode), `CML_PAT_CAPTURE` (bind by name), `CML_PAT_ANY` (wildcard) |
| `CMLRewriteRule` | A pattern + emitter function + priority |
| `CMLRewriteRegistry` | Collection of up to 64 rules |
| `CMLMatchResult` | Captured nodes from a successful match (up to 8 captures) |

### API

```c
// Build patterns
CMLPatternNode* cml_pattern_op(UOpType type, CMLPatternNode** inputs, int n);
CMLPatternNode* cml_pattern_capture(const char* name);
CMLPatternNode* cml_pattern_any(void);

// Registry
CMLRewriteRegistry* cml_rewrite_registry_create(void);
CMLRewriteRegistry* cml_rewrite_builtin_rules(void);   // all built-in algebraic rules
int  cml_rewrite_register(CMLRewriteRegistry* reg, CMLPatternNode* pat,
                           CMLEmitFn emit, int priority, const char* name);

// Apply rules to graph until fixpoint (max_iterations=0 for default of 16)
int  cml_rewrite_apply(CMLRewriteRegistry* reg, CMLGraph_t ir, int max_iterations);

// Dead code elimination
int  cml_rewrite_dce(CMLGraph_t ir);
```

---

## Stage 2: Z3 Formal Verification (Optional)

**Header:** `include/ops/ir/z3_verify.h`

Uses the Z3 SMT solver to prove that IR transformations preserve semantics. Primarily useful during development and testing.

### API

```c
bool cml_z3_available(void);
CMLZ3Verifier* cml_z3_verifier_create(int timeout_ms);
void cml_z3_verifier_free(CMLZ3Verifier* v);

CMLVerifyResult cml_z3_verify_equivalence(CMLZ3Verifier* v,
                                           CMLGraph_t original,
                                           CMLGraph_t optimized);
CMLVerifyResult cml_z3_verify_bounds(CMLZ3Verifier* v, CMLGraph_t ir);
CMLVerifyResult cml_z3_verify_schedule(CMLZ3Verifier* v, CMLGraph_t ir, void* schedule);
```

Results: `CML_VERIFY_PASS`, `CML_VERIFY_FAIL`, `CML_VERIFY_TIMEOUT`, `CML_VERIFY_UNSUPPORTED`.

---

## Stage 3: Scheduling & Fusion

**Header:** `include/ops/ir/schedule.h`

Analyzes the IR graph and groups compatible operations into fusion groups that become single kernels. Two scheduler versions exist.

### Schedule Item Types

| Type | Description |
|------|-------------|
| `SCHED_ELEMENTWISE` | Fused chain of elementwise ops |
| `SCHED_REDUCE` | Reduction (breaks fusion boundary) |
| `SCHED_MATMUL` | Matrix multiply kernel |
| `SCHED_CONV` | Convolution kernel |
| `SCHED_MOVEMENT` | View/reshape (zero-cost, no kernel) |
| `SCHED_COPY` | Device-to-device memory copy |
| `SCHED_CUSTOM` | Custom/unfuseable op |

### V1 Scheduler

```c
CMLScheduleOptions cml_schedule_default_options(void);
CMLSchedule* cml_schedule_create(CMLGraph_t graph, const CMLScheduleOptions* opts);
void cml_schedule_free(CMLSchedule* sched);
```

`CMLSchedule` contains topologically sorted `CMLScheduleItem[]` with dependency tracking, FLOP estimates, and arithmetic intensity.

### V2 Scheduler (Recommended)

```c
CMLScheduleV2* cml_schedule_v2_create(CMLGraph_t graph, const CMLScheduleOptions* opts);
void cml_schedule_v2_free(CMLScheduleV2* sched);
CMLFusionAnalysis cml_schedule_analyze_fusion(struct IRNode* a, struct IRNode* b);
```

V2 produces `CMLFusionGroup[]` with buffer elimination tracking. Each `CMLFusionGroup` records which intermediate buffers can stay in registers.

### Scheduling Options

```c
typedef struct {
    bool enable_fusion;          // default: true
    bool enable_movement_fold;   // fold movements into loads/stores, default: true
    int  max_fused_ops;          // default: 64
    bool estimate_costs;         // compute FLOP/memory estimates, default: true
    bool topological_sort;       // default: true
} CMLScheduleOptions;
```

### Fusion Queries

```c
bool cml_schedule_can_fuse(UOpType a, UOpType b);
bool cml_schedule_is_elementwise(UOpType type);
bool cml_schedule_is_reduction(UOpType type);
bool cml_schedule_is_movement(UOpType type);
```

---

## Stage 4: Linearization

**Header:** `include/ops/ir/fused_codegen.h`
**Detail:** [linearization.md](linearization.md)

Converts each `CMLFusionGroup` into a flat `CMLLinearProgram` with virtual register allocation.

```c
CMLLinearProgram* cml_linearize_group(const CMLFusionGroup* group);
void cml_linear_program_free(CMLLinearProgram* prog);
void cml_linear_program_print(const CMLLinearProgram* prog);
```

Instructions: `LINOP_LOAD`, `LINOP_COMPUTE`, `LINOP_STORE`. Eliminated intermediates stay in virtual registers, avoiding memory round-trips.

---

## Stage 5: Fused Code Generation

**Header:** `include/ops/ir/fused_codegen.h`
**Detail:** [linearization.md](linearization.md)

Lowers a `CMLLinearProgram` to backend-specific kernel source code.

### Backends

| Backend | Output | Header |
|---------|--------|--------|
| `CML_FUSED_BACKEND_C` | Portable C for-loop | `fused_codegen.h` |
| `CML_FUSED_BACKEND_PTX` | NVIDIA PTX assembly | `gpu/ptx_codegen.h` |
| `CML_FUSED_BACKEND_SPIRV` | Vulkan SPIR-V binary | `gpu/spirv_codegen.h` |
| `CML_FUSED_BACKEND_WGSL` | WebGPU shading language | `fused_codegen.h` |
| `CML_FUSED_BACKEND_METAL` | Apple Metal shaders | `fused_codegen.h` |

### API

```c
// Generic
CMLFusedKernel* cml_fused_codegen(const CMLLinearProgram* prog,
                                    CMLFusedBackend backend, size_t work_size);
CMLFusedKernel* cml_fused_codegen_group(const CMLFusionGroup* group,
                                          CMLFusedBackend backend);
void cml_fused_kernel_free(CMLFusedKernel* kernel);

// Backend-specific
char*     cml_ptx_gen_fused_kernel(const CMLLinearProgram* prog, size_t work_size);
uint32_t* cml_spirv_gen_fused_kernel(const CMLLinearProgram* prog,
                                      size_t work_size, int* out_num_words);
```

### PTX Codegen

**Header:** `include/ops/ir/gpu/ptx_codegen.h`

Standalone PTX assembly generation targeting sm_50+. Supports unary, binary, fill, where, reduction, matmul, tiled matmul, and conv2d kernels. All `.f32` precision.

```c
CMLPTXCodegen* cml_ptx_codegen_create(int sm_version, struct CMLCUDABackend* cuda);
void cml_ptx_codegen_destroy(CMLPTXCodegen* cg);

char* cml_ptx_gen_unary(CMLPTXCodegen* cg, UOpType op, const char* name);
char* cml_ptx_gen_binary(CMLPTXCodegen* cg, UOpType op, const char* name);
char* cml_ptx_gen_reduction(CMLPTXCodegen* cg, UOpType op, const char* name);
char* cml_ptx_gen_matmul(CMLPTXCodegen* cg, const char* name);
char* cml_ptx_gen_tiled_matmul(CMLPTXCodegen* cg, const char* name);  // 16x16 tiles via NVRTC
char* cml_ptx_gen_conv2d(CMLPTXCodegen* cg, const char* name);        // direct conv2d via NVRTC
int   cml_ptx_execute_graph(CMLPTXCodegen* cg, CMLGraph_t ir);
```

### SPIR-V Codegen

**Header:** `include/ops/ir/gpu/spirv_codegen.h`

Generates SPIR-V binary modules for Vulkan compute shaders. Uses `GlobalInvocationID` thread indexing and storage buffer bindings. Default workgroup size: 256x1x1.

```c
CMLSPIRVCodegen* cml_spirv_codegen_create(void);
void cml_spirv_codegen_destroy(CMLSPIRVCodegen* cg);

uint32_t* cml_spirv_gen_unary(CMLSPIRVCodegen* cg, UOpType op, const char* name, size_t* out_size);
uint32_t* cml_spirv_gen_binary(CMLSPIRVCodegen* cg, UOpType op, const char* name, size_t* out_size);
uint32_t* cml_spirv_gen_reduction(CMLSPIRVCodegen* cg, UOpType op, const char* name, size_t* out_size);
uint32_t* cml_spirv_gen_matmul(CMLSPIRVCodegen* cg, const char* name, size_t* out_size);
uint32_t* cml_spirv_gen_fill(CMLSPIRVCodegen* cg, float value, const char* name, size_t* out_size);

// Low-level builder
SPIRVBuilder* spirv_builder_create(void);
void          spirv_builder_emit(SPIRVBuilder* b, uint32_t word);
uint32_t      spirv_builder_alloc_id(SPIRVBuilder* b);
uint32_t*     spirv_builder_finalize(SPIRVBuilder* b, size_t* out_size);
```

---

## Stage 6: Kernel Cache

**Header:** `include/ops/ir/kernel_cache.h`

Thread-safe LRU cache for compiled kernels. Uses FNV-1a hashing of IR structure and tensor shapes. Supports per-entry and total memory limits.

### Supported Backends

`CML_KERNEL_CPU_FALLBACK`, `CML_KERNEL_CPU_LLVM`, `CML_KERNEL_CUDA`, `CML_KERNEL_ROCM`, `CML_KERNEL_METAL`, `CML_KERNEL_WEBGPU`.

### API

```c
// Lifecycle
CMLKernelCache* cml_kernel_cache_create(size_t max_entries);
CMLKernelCache* cml_kernel_cache_create_with_limits(size_t max_entries, size_t max_memory);
void cml_kernel_cache_free(CMLKernelCache* cache);
CMLKernelCache* cml_kernel_cache_get_default(void);  // global singleton

// Lookup
uint64_t       cml_kernel_cache_compute_hash(CMLGraph_t ir, Tensor** inputs,
                                              int num_inputs, CMLKernelBackend backend);
CMLKernelEntry* cml_kernel_cache_lookup(CMLKernelCache* cache, uint64_t hash);
CMLKernelEntry* cml_kernel_cache_lookup_ir(CMLKernelCache* cache, CMLGraph_t ir,
                                            Tensor** inputs, int num_inputs,
                                            CMLKernelBackend backend);

// Insert
int cml_kernel_cache_insert(CMLKernelCache* cache, uint64_t hash,
                             CMLKernelBackend backend, void* compiled, size_t memory_size);

// Eviction & stats
int    cml_kernel_cache_evict_lru(CMLKernelCache* cache);
double kernel_cache_hit_rate(CMLKernelCache* cache);

// Register per-backend free function
void cml_kernel_cache_set_free_fn(CMLKernelBackend backend, CMLKernelFreeFn free_fn);
```

---

## Stage 7: Execution

Three execution strategies exist depending on the use case.

### Runtime JIT Compiler

**Header:** `include/ops/ir/runtime_compiler.h`

Orchestrates the full pipeline: IR -> schedule -> linearize -> codegen -> compile -> cache -> launch. Includes a built-in 256-entry compiled kernel cache.

```c
CMLRuntimeCompiler* cml_runtime_compiler_create(void);
void cml_runtime_compiler_free(CMLRuntimeCompiler* rc);

// Compile (with caching)
const CMLCompiledKernel* cml_runtime_compile_group(CMLRuntimeCompiler* rc,
                                                     const CMLFusionGroup* group);
const CMLCompiledKernel* cml_runtime_compile_program(CMLRuntimeCompiler* rc,
                                                       const CMLLinearProgram* prog,
                                                       size_t work_size);

// Execute
int cml_runtime_execute_compiled(const CMLCompiledKernel* kernel,
                                  Tensor** inputs, int num_inputs,
                                  Tensor** outputs, int num_outputs);
int cml_runtime_execute_graph(CMLRuntimeCompiler* rc, CMLGraph_t ir);

// Configuration
void cml_runtime_compiler_set_backend(CMLRuntimeCompiler* rc, CMLFusedBackend backend);
void cml_runtime_compiler_clear_cache(CMLRuntimeCompiler* rc);
void cml_runtime_compiler_stats(const CMLRuntimeCompiler* rc,
                                 size_t* hits, size_t* misses, size_t* compilations);
```

### AOT (Ahead-of-Time) Compilation

**Header:** `include/ops/ir/aot.h`

Compiles IR graphs to native shared libraries with zero runtime compilation dependency. Supports cross-compilation, weight bundling, and C header generation.

```c
AOTCompileOptions cml_aot_default_options(void);

// Compile
int cml_aot_compile(CMLGraph_t ir, const char* output_path, const AOTCompileOptions* options);
int cml_aot_compile_module(struct Module* module, Tensor* sample_input,
                            const char* output_path, const AOTCompileOptions* options);

// Load & execute (runtime -- no compiler needed)
CMLAOTModel* cml_aot_load(const char* path);
int cml_aot_execute(CMLAOTModel* model, Tensor** inputs, int num_inputs,
                     Tensor** outputs, int num_outputs);
void cml_aot_free(CMLAOTModel* model);

// Header generation
int cml_aot_generate_header(CMLGraph_t ir, const char* header_path, const char* function_name);
```

**AOT Options:**

| Field | Description |
|-------|-------------|
| `target_triple` | e.g. `"x86_64-unknown-linux-gnu"` (NULL = host) |
| `cpu` | e.g. `"skylake"` (NULL = generic) |
| `features` | e.g. `"+avx2,+fma"` (NULL = default) |
| `opt_level` | `AOT_OPT_O0` through `AOT_OPT_O3`, plus `AOT_OPT_Os` |
| `format` | `AOT_FORMAT_OBJECT`, `SHARED_LIB`, `STATIC_LIB`, `HEADER_ONLY`, `LLVM_IR` |
| `include_weights` | Bundle weights into the binary |
| `generate_header` | Emit companion `.h` file |

### TinyJit (Capture-and-Replay)

**Header:** `include/ops/ir/tiny_jit.h`

On first execution, runs the graph normally and records the kernel launch trace. Subsequent calls with the same graph hash and shapes replay the cached trace, bypassing scheduling and codegen entirely.

```c
CMLTinyJit* cml_tinyjit_create(void);
void cml_tinyjit_free(CMLTinyJit* jit);
int  cml_tinyjit_execute(CMLTinyJit* jit, CMLGraph_t ir);
void cml_tinyjit_stats(const CMLTinyJit* jit,
                        size_t* hits, size_t* misses, size_t* invalidations);
```

Cache: 64 entries keyed by graph hash + shape signature.

### GPU Graph Capture

**Header:** `include/ops/ir/graph_capture.h`

Records GPU kernel launches into a replayable graph (CUDA Graph / Metal Command Buffer style). Eliminates per-kernel launch overhead for repeated patterns.

```c
CMLCapturedGraph* cml_graph_capture_create(void);
void cml_graph_capture_free(CMLCapturedGraph* graph);

// Record
int cml_graph_capture_begin(CMLCapturedGraph* graph);
int cml_graph_capture_record(CMLCapturedGraph* graph, UOpType op,
                              void* kernel_handle,
                              const size_t grid[3], const size_t block[3],
                              void** args, int num_args, size_t shared_mem);
int cml_graph_capture_end(CMLCapturedGraph* graph);

// Replay
int cml_graph_capture_replay(CMLCapturedGraph* graph);

// Rebind inputs/outputs for new data
int cml_graph_capture_bind_input(CMLCapturedGraph* graph, int index, Tensor* tensor);
int cml_graph_capture_bind_output(CMLCapturedGraph* graph, int index, Tensor* tensor);
```

States: `CML_CAPTURE_IDLE` -> `CML_CAPTURE_RECORDING` -> `CML_CAPTURE_READY`.

---

## Typical Usage

### JIT path (default)

```c
CMLRuntimeCompiler* rc = cml_runtime_compiler_create();
cml_runtime_compiler_set_backend(rc, CML_FUSED_BACKEND_PTX);

// Executes full pipeline: schedule -> linearize -> codegen -> cache -> launch
cml_runtime_execute_graph(rc, ir);

// Second call hits the cache
cml_runtime_execute_graph(rc, ir);

size_t hits, misses, compilations;
cml_runtime_compiler_stats(rc, &hits, &misses, &compilations);
cml_runtime_compiler_free(rc);
```

### AOT path (deploy without compiler)

```c
// Build time
AOTCompileOptions opts = cml_aot_default_options();
opts.opt_level = AOT_OPT_O3;
opts.include_weights = true;
opts.format = AOT_FORMAT_SHARED_LIB;
cml_aot_compile(ir, "model.so", &opts);

// Runtime (no compiler dependency)
CMLAOTModel* model = cml_aot_load("model.so");
cml_aot_execute(model, inputs, 1, outputs, 1);
cml_aot_free(model);
```

### TinyJit path (training loops)

```c
CMLTinyJit* jit = cml_tinyjit_create();
for (int step = 0; step < 1000; step++) {
    cml_tinyjit_execute(jit, ir);  // first call compiles, rest replay
}
cml_tinyjit_free(jit);
```
