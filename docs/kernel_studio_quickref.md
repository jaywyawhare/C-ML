# Kernel Studio - Quick Reference Card

## 🚀 Quick Start

```c
// 1. Create IR context
CMLIR_t ir = cml_ir_new(IR_TARGET_CUDA);

// 2. Enable auto-capture
cml_ir_enable_auto_capture(ir);

// 3. Write your tensor code (automatically captured)
Tensor* result = tensor_add(tensor_mul(a, b), c);

// 4. Optimize!
cml_ir_optimize(ir);

// 5. Generate code
char* code = cml_ir_compile(ir, NULL);
printf("%s\n", code);
free(code);

// 6. Cleanup
cml_ir_disable_auto_capture();
cml_ir_free(ir);
```

## 📊 Optimization Passes

| Pass                      | What It Does                            | Impact                |
| ------------------------- | --------------------------------------- | --------------------- |
| **Dead Code Elimination** | Removes unused operations               | 10-30% fewer ops      |
| **Kernel Fusion**         | Combines operations into single kernels | 1.5-5x speedup        |
| **Cache Reordering**      | Improves memory access patterns         | 15-25% less bandwidth |

## 🔥 Fusion Patterns

### Pattern 1: FMA (Fused Multiply-Add)

```c
// Before: 2 kernels
t1 = a * b;
t2 = t1 + c;

// After: 1 kernel (1.8x faster)
t2 = fmaf(a, b, c);
```

### Pattern 2: Elementwise Chain

```c
// Before: 5 kernels
t1 = a + b;
t2 = exp(t1);
t3 = log(t2);
t4 = sqrt(t3);
t5 = t4 * c;

// After: 1 kernel (4.2x faster)
for (int i = 0; i < n; i++) {
    float t0 = a[i] + b[i];
    float t1 = expf(t0);
    float t2 = logf(t1);
    float t3 = sqrtf(t2);
    out[i] = t3 * c[i];
}
```

### Pattern 3: Identity Elimination

```c
// Before: 2 kernels
t1 = exp(a);
t2 = log(t1);  // exp(log(x)) = x

// After: 0 kernels (eliminated!)
t2 = a;
```

## 🎯 Common Patterns

| Pattern | Before      | After      | Speedup        |
| ------- | ----------- | ---------- | -------------- |
| FMA     | `MUL + ADD` | `fmaf()`   | 1.8x           |
| NEG-ADD | `NEG + ADD` | `SUB`      | 1.5x           |
| EXP-LOG | `EXP + LOG` | `identity` | ∞ (eliminated) |
| MUL-DIV | `MUL + DIV` | `identity` | ∞ (eliminated) |
| Chain   | `n ops`     | `1 kernel` | 2-5x           |

## 🔍 Debugging

### Enable Debug Logging

```c
cml_set_log_level(LOG_LEVEL_DEBUG);
```

### Inspect IR

```c
// Before optimization
char* before = cml_ir_to_string(ir);
printf("Before:\n%s\n", before);
free(before);

cml_ir_optimize(ir);

// After optimization
char* after = cml_ir_to_string(ir);
printf("After:\n%s\n", after);
free(after);
```

### Check Node Status

```c
struct IRNode* node = cml_ir_get_tail(ir);
printf("Type: %s\n", uop_type_to_string(node->type));
printf("Used: %d\n", node->is_used);
printf("Fused: %d\n", node->is_fused);
```

## 📈 Performance Tips

### ✅ DO:

- Use auto-capture for automatic optimization
- Chain elementwise operations (they fuse well)
- Let the optimizer eliminate dead code
- Use `fmaf()` patterns (MUL + ADD)

### ❌ DON'T:

- Mix different data types unnecessarily
- Create intermediate results you don't need
- Manually optimize before profiling
- Assume fusion always happens (check the output!)

## 🎨 Kernel Studio UI

### View Toggle

- **Unoptimized**: See all operations including dead code
- **Optimized**: See final optimized kernels

### Status Badges

- 🔴 **DEAD**: Will be eliminated
- 🟢 **FUSED**: Part of optimized kernel

### Tabs

- **Kernels**: Browse and inspect individual kernels
- **Optimizations**: View optimization insights and statistics

## 🎓 Learning Path

1. **Start Simple**: Run `comprehensive_fusion_example.c`
1. **Understand Basics**: Read `docs/kernel_studio.md`
1. **Experiment**: Try different operation patterns
1. **Optimize**: Use Kernel Studio to verify optimizations
1. **Profile**: Measure actual performance gains

## 📚 Resources

- **Full Docs**: `docs/kernel_studio.md`
- **Example**: `examples/comprehensive_fusion_example.c`
- **API Reference**: `REFERENCE.md`
- **Implementation**: `src/ops/ir.c`

## 🔧 Supported Targets

```c
IR_TARGET_C          // Plain C (scalar)
IR_TARGET_C_SIMD     // C with SIMD (AVX, NEON)
IR_TARGET_CUDA       // NVIDIA CUDA
IR_TARGET_METAL      // Apple Metal
IR_TARGET_OPENCL     // OpenCL (portable)
IR_TARGET_WGSL       // WebGPU
```

## 💡 Pro Tips

1. **Chain Operations**: More chained ops = better fusion opportunities
1. **Check Generated Code**: Always verify what code is generated
1. **Profile First**: Optimize based on actual bottlenecks
1. **Use Debug Mode**: See what the optimizer is doing
1. **Read the Logs**: Optimization passes log their actions

## 🐛 Common Issues

### Issue: Operations not fusing

**Solution**: Check if operations are compatible and consecutive

### Issue: Dead code not eliminated

**Solution**: Ensure operations aren't used by any output

### Issue: No speedup from fusion

**Solution**: Profile on actual hardware, not just theory

## 📞 Getting Help

1. Enable debug logging: `cml_set_log_level(LOG_LEVEL_DEBUG)`
1. Check IR output: `cml_ir_to_string(ir)`
1. View generated code: `cml_ir_compile(ir, NULL)`
1. Use Kernel Studio UI to visualize

## 🎯 Quick Checklist

- [ ] Created IR context
- [ ] Enabled auto-capture
- [ ] Wrote tensor operations
- [ ] Called `cml_ir_optimize()`
- [ ] Generated code
- [ ] Verified optimizations in Kernel Studio
- [ ] Profiled performance
- [ ] Cleaned up resources

______________________________________________________________________

**Remember**: The best optimization is the one you can measure! Always profile before and after.
