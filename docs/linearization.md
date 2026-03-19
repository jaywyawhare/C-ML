# Linearization & Fused Code Generation

Linearization converts a `CMLFusionGroup` (a set of fused IR operations) into a flat instruction sequence with virtual register allocation, which is then lowered to backend-specific kernel code. This is the bridge between the IR graph optimizer and actual executable GPU/CPU kernels.


## Table of Contents

1. [Overview](#overview)
1. [Pipeline](#pipeline)
1. [Linearization](#linearization)
1. [Code Generation Backends](#code-generation-backends)
1. [API Reference](#api-reference)
1. [Buffer Elimination](#buffer-elimination)
1. [Runtime Compilation & Caching](#runtime-compilation--caching)
1. [Usage Example](#usage-example)


## Overview

After the IR graph scheduler groups operations into `CMLFusionGroup`s, each group needs to be compiled into an executable kernel. The linearization step converts the group's dependency graph into a sequential instruction stream, assigns virtual registers, and identifies intermediate buffers that can stay in registers instead of being written to memory.

**Files:** `src/ops/ir/linearize.c`, `src/ops/ir/fused_codegen.c`, `include/ops/ir/fused_codegen.h`


## Pipeline

```
IR Graph
     |
     v
Schedule V2  (cml_schedule_v2_create)
  topological walk, group ops by fusibility, detect buffer elimination
     |
     v  CMLFusionGroup[]
Linearization  (cml_linearize_group)
  map node outputs to virtual registers (v0-v63)
  emit LOAD / COMPUTE / STORE instructions
  mark eliminated intermediates
     |
     v  CMLLinearProgram
Code Generation  (cml_fused_codegen)
  C: for-loop with float registers
  PTX: NVIDIA CUDA assembly
  SPIR-V: Vulkan shader bytecode
  WGSL / Metal: WebGPU / Apple shaders
     |
     v  CMLFusedKernel
Runtime Compilation & Cache
  hash LinearProgram (FNV-1a), compile on cache miss, execute
```


## Linearization

The linearization pass walks a fusion group's nodes in topological order and emits three kinds of instructions:

| Instruction | Purpose |
|-------------|---------|
| `LINOP_LOAD` | Load a tensor from memory into a virtual register |
| `LINOP_COMPUTE` | Apply an operation (ADD, MUL, etc.) with source vregs, write to dest vreg |
| `LINOP_STORE` | Write a virtual register back to memory |

### Example

Given a fusion group: `a * b + c` where the multiply result is eliminated:

```
Input: FusionGroup [LOAD(a), LOAD(b), MUL, LOAD(c), ADD]

Output LinearProgram:
  [0] LOAD    v0 ← tensor(a)
  [1] LOAD    v1 ← tensor(b)
  [2] COMPUTE v2 ← MUL(v0, v1)    [ELIMINATED — stays in register]
  [3] LOAD    v3 ← tensor(c)
  [4] COMPUTE v4 ← ADD(v2, v3)
  [5] STORE   tensor(out) ← v4
```

The key optimization: `v2` (the MUL result) is never written to memory because all its consumers are within the same fusion group. It stays in a register, saving memory bandwidth.

### Data Structures

```c
typedef struct CMLLinearOp {
    CMLLinearOpKind kind;       /* LINOP_LOAD, LINOP_COMPUTE, LINOP_STORE */
    UOpType uop;                /* Operation type (ADD, MUL, etc.) */
    int dest_reg;               /* Destination virtual register */
    int src_regs[8];            /* Source virtual registers */
    int num_srcs;
    Tensor* tensor;             /* Associated tensor (for LOAD/STORE) */
    bool is_eliminated;         /* True if intermediate stays in register */
} CMLLinearOp;

typedef struct CMLLinearProgram {
    CMLLinearOp* ops;           /* Array of linear operations */
    int num_ops;
    int capacity;
    int next_vreg;              /* Next free virtual register */
} CMLLinearProgram;
```


## Code Generation Backends

The linear program is lowered to backend-specific code via `cml_fused_codegen()`:

```c
typedef enum {
    CML_FUSED_BACKEND_C = 0,       /* Portable C with for-loop */
    CML_FUSED_BACKEND_PTX,         /* NVIDIA CUDA PTX assembly */
    CML_FUSED_BACKEND_SPIRV,       /* Vulkan SPIR-V bytecode */
    CML_FUSED_BACKEND_WGSL,        /* WebGPU shaders */
    CML_FUSED_BACKEND_METAL,       /* Apple Metal shaders */
} CMLFusedBackend;
```

### C Backend

Emits a `for(int i=0; i<n; i++)` loop with float register variables:

```c
// Generated C kernel
void cml_fused_kernel(float* in0, float* in1, float* in2, float* out0, int n) {
    for (int i = 0; i < n; i++) {
        float v0 = in0[i];
        float v1 = in1[i];
        float v2 = v0 * v1;       // eliminated — register only
        float v3 = in2[i];
        float v4 = v2 + v3;
        out0[i] = v4;
    }
}
```

### PTX Backend

Emits NVIDIA CUDA PTX assembly with thread-parallel execution:

```
.reg .f32 v0, v1, v2, v3, v4;
ld.global.f32 v0, [in0 + tid*4];
ld.global.f32 v1, [in1 + tid*4];
mul.f32 v2, v0, v1;
ld.global.f32 v3, [in2 + tid*4];
add.f32 v4, v2, v3;
st.global.f32 [out0 + tid*4], v4;
```


## API Reference

### Linearization

| Function | Description |
|----------|-------------|
| `cml_linearize_group(group)` | Convert a fusion group to a linear program |
| `cml_linear_program_free(prog)` | Free a linear program |
| `cml_linear_program_print(prog)` | Print linear program for debugging |

### Code Generation

| Function | Description |
|----------|-------------|
| `cml_fused_codegen(prog, backend, work_size)` | Generate kernel from linear program |
| `cml_fused_codegen_group(group, backend)` | Linearize + codegen in one call |
| `cml_fused_kernel_free(kernel)` | Free a generated kernel |
| `cml_fused_kernel_print(kernel)` | Print generated source for debugging |

### Backend-Specific

| Function | Description |
|----------|-------------|
| `cml_ptx_gen_fused_kernel(prog, work_size)` | Generate PTX source |
| `cml_spirv_gen_fused_kernel(prog, work_size, out_words)` | Generate SPIR-V binary |

### Runtime Compilation

| Function | Description |
|----------|-------------|
| `cml_runtime_compiler_create()` | Create a runtime compiler with kernel cache |
| `cml_runtime_compiler_free(rc)` | Free the runtime compiler |
| `cml_runtime_compile_group(rc, group)` | Compile a fusion group (cached) |
| `cml_runtime_compile_program(rc, prog, work_size)` | Compile a linear program (cached) |


## Buffer Elimination

Buffer elimination is the primary optimization in this pipeline. During scheduling, if an intermediate tensor's **all consumers** are within the same fusion group, that buffer is marked as eliminated:

```
Without elimination:            With elimination:
  LOAD a                          LOAD a
  LOAD b                          LOAD b
  MUL tmp = a * b                 MUL v2 = a * b  (register)
  STORE tmp → memory              LOAD c
  LOAD tmp                        ADD out = v2 + c
  LOAD c                          STORE out
  ADD out = tmp + c
  STORE out

  Memory ops: 8                   Memory ops: 4
```

This reduces memory bandwidth by keeping intermediate values in registers, which is critical for GPU performance where memory access is often the bottleneck.


## Runtime Compilation & Caching

The runtime compiler hashes each `CMLLinearProgram` using FNV-1a and caches compiled kernels. Identical programs skip compilation entirely:

```c
CMLRuntimeCompiler* rc = cml_runtime_compiler_create();

/* First call: compiles and caches */
const CMLCompiledKernel* k1 = cml_runtime_compile_group(rc, group);

/* Same group again: cache hit, no recompilation */
const CMLCompiledKernel* k2 = cml_runtime_compile_group(rc, group);

cml_runtime_compiler_free(rc);
```


## Usage Example

```c
#include "ops/ir/fused_codegen.h"

CMLLinearProgram* prog = cml_linearize_group(group);
cml_linear_program_print(prog);

CMLFusedKernel* kernel = cml_fused_codegen(prog, CML_FUSED_BACKEND_PTX, work_size);
printf("Generated PTX:\n%s\n", kernel->source);
printf("Inputs: %d, Outputs: %d, VRegs: %d\n",
       kernel->num_inputs, kernel->num_outputs, kernel->num_vregs);

CMLFusedKernel* kernel2 = cml_fused_codegen_group(group, CML_FUSED_BACKEND_C);

cml_fused_kernel_free(kernel);
cml_fused_kernel_free(kernel2);
cml_linear_program_free(prog);
```
