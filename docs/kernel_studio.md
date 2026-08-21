# Kernel Studio - Code Generation & Optimization Visualization

## Overview

The **Kernel Studio** is an interactive visualization tool for analyzing generated code kernels, dead code elimination, and optimization passes in C-ML. It provides insights into how the IR (Intermediate Representation) compiler optimizes your computation graphs.

## Running the dashboard

```bash
VIZ=1 ./your_program        # exports the artifacts and launches the dashboard
```

`VIZ=1` writes these next to the working directory:

| File | Contents |
|------|----------|
| `graph.json` | IR graph after optimization — dead/fused flags, module scopes |
| `kernels.json` | Generated kernels, `unoptimized` and `optimized` side by side |
| `training.json` | Per-epoch loss/accuracy/LR, plus weight & gradient distributions |
| `model_architecture.json` | Layer summary and parameter counts |
| `flamegraph.json` | Per-kernel execution timings (see below) |

The graph and kernel exports are re-run only when the graph's structure changes,
not on every step: a training loop rebuilds the same graph each iteration and the
dashboard only ever shows the latest, so exporting per step rewrote identical
files hundreds of times. A shape change re-triggers it automatically.

### Turning everything off

`NO_EXPORT=1` emits **no** files and skips the work behind them — use it for
benchmarking (no I/O jitter in the numbers) and production training:

```bash
NO_EXPORT=1 ./your_program
```

Worth knowing: `training.json` is written even without `VIZ`, because the
auto-capture metrics path is not VIZ-gated — a 57-epoch run rewrote it ~199
times. `NO_EXPORT=1` is what silences that.

`VIZ=1` and `NO_EXPORT=1` are **mutually exclusive**; setting both fails
`cml_init()` with an explanatory error rather than silently picking a winner.

## Features

### 1. **Kernel Inspection**

- View all generated kernels with syntax-highlighted code
- Browse individual kernel implementations
- See input/output tensor mappings
- Understand operation types (UOps)

### 2. **Dead Code Elimination**

The optimizer automatically removes unused operations that don't contribute to final outputs:

```c
// Example: These operations are marked as dead code
Tensor* dead_mul = tensor_mul(b, c);  // Not used in any output
Tensor* dead_add = tensor_add(dead_mul, a);  // Not used
Tensor* dead_exp = tensor_exp(dead_add);  // Not used
```

**How it works:**

1. **Mark Phase**: Starting from output nodes, traverse backward to mark all reachable nodes
1. **Sweep Phase**: Remove nodes that weren't marked as reachable
1. **Result**: Reduced memory usage and faster execution

### 3. **Kernel Fusion**

The optimizer combines multiple operations into single, efficient kernels:

#### Fusion Types

**FMA (Fused Multiply-Add)**

```c
// Before optimization:
t1 = a * b;  // MUL kernel
t2 = t1 + c; // ADD kernel

// After optimization:
t2 = fmaf(a, b, c);  // Single FMA kernel (1.8x faster)
```

**Elementwise Chains**

```c
// Before: 5 separate kernels
t1 = a + b;
t2 = exp(t1);
t3 = log(t2);
t4 = sqrt(t3);
t5 = t4 * c;

// After: 1 fused kernel (4.2x faster)
for (int i = 0; i < n; i++) {
    float t0 = inputs[0][i] + inputs[1][i];
    float t1 = expf(t0);
    float t2 = logf(t1);
    float t3 = sqrtf(t2);
    outputs[0][i] = t3 * inputs[2][i];
}
```

**Identity Elimination**

```c
// Before:
t1 = exp(a);
t2 = log(t1);  // exp(log(x)) = x

// After:
t2 = a;  // Optimized to identity
```

**Other Fusion Patterns:**

- `NEG + ADD -> SUB`
- `SQRT + MUL -> sqrt_mul`
- `EXP + RECIP -> exp_recip`
- `MUL + DIV -> identity` (if same operand)

### 4. **Cache Locality Optimization**

Operations are reordered using topological sort to improve cache utilization:

- Reduces memory bandwidth requirements
- Improves data locality
- Minimizes cache misses

## Using the Kernel Studio

### Grouping the graph by module

IR nodes carry a `scope` — the module path of the layer that emitted them, e.g.
`Sequential/Linear` — so the graph view can collapse thousands of decomposed
primitives back into the layers they came from. Toggle **Group by module scope**
in the graph controls; nested containers render as nested boxes.

Scopes are recorded in `module_forward`, so every layer gets them without opting
in. They are only tracked under `VIZ` (the tagging costs a string copy per node).

### Viewing Kernels

1. **Toggle View**: Switch between "Unoptimized" and "Optimized" views
1. **Browse Kernels**: Click on kernels in the left panel to view details
1. **Inspect Code**: See generated C/CUDA code with syntax highlighting
1. **Check Status**: Look for badges:
   - **DEAD** - Will be eliminated
   - **FUSED** - Part of optimized kernel

### Understanding Statistics

**Unoptimized View:**

- **Total Nodes**: All operations in the graph
- **Kernels**: Number of individual kernels
- **Dead Nodes**: Operations that will be removed
- **Fusion Ops**: Opportunities for kernel fusion

**Optimized View:**

- **Total Nodes**: Operations after dead code elimination
- **Kernels**: Optimized kernel count
- **Fused**: Number of fused kernels created

### Execution Flamegraph

`VIZ=1` also times each executed kernel and writes `flamegraph.json`, shown at the
top of Kernel Studio (width = time). It answers the question the kernel listing
cannot: whether fusion actually collapsed a hot chain into one wide bar.

Timings are **aggregated at capture time** by kernel signature
`(phase, kind, op, work-size)` — the same key the view groups by — carrying total
time, an occurrence count and the slowest single execution:

| field | meaning |
|-------|---------|
| `ms` | total time across every execution of this signature |
| `count` | how many executions folded into this entry |
| `max_ms` | slowest single execution — catches an outlier step |

Recording one entry per *execution* would be storing an O(steps × nodes) log to
compute an O(distinct-kernels) view. Aggregation keeps the whole run bounded by
graph shape instead: a 300-step run went from 1.2 MB to 4.2 KB, with no cap and
no truncation. `max_ms` is often the most useful column — a signature with
`count=5` but `max=200 ms` is a one-off allocation, not a hot loop.

Capture is enabled by `VIZ=1`; set `FLAMEGRAPH=0` to decline it (the graph and
metrics panels still work). The file is written when the process exits, so finish
the run before hitting Refresh.

### Optimization Insights Tab

View detailed information about:

- Dead code elimination statistics
- Kernel fusion patterns applied
- Cache locality improvements
- Performance gains (speedup estimates)

## API Usage

### Enabling Auto-Capture

```c
CMLGraph_t ir = cml_ir_new(IR_TARGET_CUDA);
cml_ir_enable_auto_capture(ir);

Tensor* a = tensor_empty(shape, 1, &config);
Tensor* b = tensor_empty(shape, 1, &config);
Tensor* c = tensor_mul(a, b);  // Captured to IR
Tensor* d = tensor_add(c, a);  // Captured to IR

cml_ir_optimize(ir);

char* code = cml_ir_compile(ir, NULL);
printf("%s\n", code);
free(code);

char* analysis = cml_ir_export_kernel_analysis(ir, true);
free(analysis);

cml_ir_disable_auto_capture();
cml_ir_free(ir);
```

### Manual IR Construction

```c
CMLGraph_t ir = cml_ir_new(IR_TARGET_C_SIMD);

Tensor* inputs[] = {a, b};
cml_ir_add_uop(ir, UOP_MUL, inputs, 2, NULL);

// ... add more operations

cml_ir_optimize(ir);
char* code = cml_ir_compile(ir, "output.c");
free(code);
```

## Optimization Passes

The `cml_ir_optimize()` function runs multiple passes:

1. **Build Dependency Graph**: Analyze which operations depend on others
1. **Mark Reachable Nodes**: Identify operations contributing to outputs
1. **Remove Dead Nodes**: Eliminate unused operations
1. **Fuse Operations**: Combine compatible operations into optimized kernels
1. **Reorder for Cache**: Topologically sort for better memory access patterns

## Performance Impact

Typical improvements from optimization:

- **Dead Code Elimination**: 10-30% reduction in operations
- **Kernel Fusion**: 1.5-5x speedup for fused chains
- **Cache Optimization**: 15-25% reduction in memory bandwidth

## Example: Comprehensive Fusion

See `examples/demos/comprehensive_fusion_example.c` for a complete demonstration of all fusion types and optimizations.

```bash
# Compile and run the example
make comprehensive_fusion_example
./comprehensive_fusion_example
```

## Target Backends

The Kernel Studio supports code generation for:

- **IR_TARGET_C**: Plain C (scalar operations)
- **IR_TARGET_C_SIMD**: C with SIMD intrinsics (AVX, NEON)
- **IR_TARGET_CUDA**: CUDA kernels for NVIDIA GPUs
- **IR_TARGET_METAL**: Metal shaders for Apple GPUs
- **IR_TARGET_OPENCL**: OpenCL kernels (portable)
- **IR_TARGET_WGSL**: WebGPU shaders

## Future Enhancements

Planned features for Kernel Studio:

- [ ] Interactive kernel editing and testing
- [ ] Performance profiling integration
- [ ] Register allocation visualization
- [ ] Memory access pattern analysis
- [ ] Multi-backend comparison
- [ ] Auto-tuning suggestions
- [ ] Export to standalone kernels
- [ ] Benchmark generation

## Technical Details

### IR Node Structure

Each IR node contains:

- **Type**: UOpType (operation type)
- **Inputs**: Array of input tensor names
- **Output**: Output tensor name
- **Params**: Operation-specific parameters
- **Optimization Metadata**:
  - `is_used`: Marked during dead code elimination
  - `is_fused`: Part of a fused kernel
  - `fused_kernel`: Pointer to fused kernel structure
  - `use_count`: Number of operations using this output
  - `users`: Array of dependent operations

### Fusion Detection

Fusion opportunities are detected by:

1. Analyzing operation types and dependencies
1. Checking if operations can be safely combined
1. Verifying data flow patterns
1. Ensuring no side effects are violated

### Code Generation

Code generation process:

1. Traverse optimized IR graph
1. For each kernel (fused or individual):
   - Generate function signature
   - Emit operation code
   - Handle broadcasting and shape transformations
   - Add target-specific optimizations (SIMD, GPU)
1. Combine into complete program

## Debugging Tips

**Enable Debug Logging:**

```c
cml_set_log_level(LOG_LEVEL_DEBUG);
```

**Inspect IR Before/After Optimization:**

```c
char* before = cml_ir_to_string(ir);
printf("Before:\n%s\n", before);
free(before);

cml_ir_optimize(ir);

char* after = cml_ir_to_string(ir);
printf("After:\n%s\n", after);
free(after);
```

**Check Individual Nodes:**

```c
struct IRNode* node = cml_ir_get_tail(ir);
printf("Last node: %s, used: %d\n",
    uop_type_to_string(node->type),
    node->is_used);
```

## Contributing

To add new fusion patterns:

1. Add fusion type to `FusionType` enum in `ir_internal.h`
1. Implement detection in `can_fuse_operations()` in `ir.c`
1. Add code generation in `generate_c_code()` or backend-specific generators
1. Update Kernel Studio visualization to display new pattern

## References

- [Getting Started](getting_started.md) - General C-ML usage guide
- [API Reference](api_reference.md) - API reference
