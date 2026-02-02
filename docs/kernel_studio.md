# Kernel Studio - Code Generation & Optimization Visualization

## Overview

The **Kernel Studio** is an interactive visualization tool for analyzing generated code kernels, dead code elimination, and optimization passes in C-ML. It provides insights into how the IR (Intermediate Representation) compiler optimizes your computation graphs.

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

- `NEG + ADD → SUB`
- `SQRT + MUL → sqrt_mul`
- `EXP + RECIP → exp_recip`
- `MUL + DIV → identity` (if same operand)

### 4. **Cache Locality Optimization**

Operations are reordered using topological sort to improve cache utilization:

- Reduces memory bandwidth requirements
- Improves data locality
- Minimizes cache misses

## Using the Kernel Studio

### Viewing Kernels

1. **Toggle View**: Switch between "Unoptimized" and "Optimized" views
1. **Browse Kernels**: Click on kernels in the left panel to view details
1. **Inspect Code**: See generated C/CUDA code with syntax highlighting
1. **Check Status**: Look for badges:
   - 🔴 **DEAD** - Will be eliminated
   - 🟢 **FUSED** - Part of optimized kernel

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

### Optimization Insights Tab

View detailed information about:

- Dead code elimination statistics
- Kernel fusion patterns applied
- Cache locality improvements
- Performance gains (speedup estimates)

## API Usage

### Enabling Auto-Capture

```c
// Create IR context
CMLIR_t ir = cml_ir_new(IR_TARGET_CUDA);

// Enable automatic operation capture
cml_ir_enable_auto_capture(ir);

// Your tensor operations are automatically captured
Tensor* a = tensor_empty(shape, 1, &config);
Tensor* b = tensor_empty(shape, 1, &config);
Tensor* c = tensor_mul(a, b);  // Captured to IR
Tensor* d = tensor_add(c, a);  // Captured to IR

// Optimize the IR
cml_ir_optimize(ir);

// Generate code
char* code = cml_ir_compile(ir, NULL);
printf("%s\n", code);
free(code);

// Export kernel analysis for visualization
char* analysis = cml_ir_export_kernel_analysis(ir, true);
// Send to frontend...
free(analysis);

cml_ir_disable_auto_capture();
cml_ir_free(ir);
```

### Manual IR Construction

```c
CMLIR_t ir = cml_ir_new(IR_TARGET_C_SIMD);

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

See `examples/comprehensive_fusion_example.c` for a complete demonstration of all fusion types and optimizations.

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

- [USAGE.md](USAGE.md) - General C-ML usage guide
- [REFERENCE.md](REFERENCE.md) - API reference
- [examples/comprehensive_fusion_example.c](examples/comprehensive_fusion_example.c) - Complete example
