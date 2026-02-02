# C-ML MLIR Integration Plan

**Document Version:** 1.0
**Date:** 2025-11-27
**Status:** Planning Phase

## Executive Summary

This document outlines the comprehensive plan to integrate MLIR (Multi-Level Intermediate Representation) into C-ML to enable:

- **JIT Compilation** for runtime performance optimization
- **Production-Ready Deployment** with ahead-of-time (AOT) compilation
- **Multi-Backend Support** (CPU, CUDA, ROCm, Vulkan, Metal, WebGPU)
- **Advanced Optimizations** (fusion, vectorization, polyhedral)

## Table of Contents

1. [Architecture Overview](#architecture-overview)
1. [Current State Analysis](#current-state-analysis)
1. [MLIR Integration Strategy](#mlir-integration-strategy)
1. [Implementation Phases](#implementation-phases)
1. [Technical Specifications](#technical-specifications)
1. [Dependencies & Build System](#dependencies--build-system)
1. [Migration Path](#migration-path)
1. [Performance Benchmarks](#performance-benchmarks)
1. [Risk Assessment](#risk-assessment)
1. [Timeline & Milestones](#timeline--milestones)

______________________________________________________________________

## Architecture Overview

### Current Architecture

```
User APIs (nn.h, optim.h, autograd.h)
    ↓
Tensor Facade (tensor/tensor.h, autograd/forward_ops.h)
    ↓
UOps (ops/uops.h) - 27 micro-operations
    ↓
IR (ops/ir/ir.h) - Intermediate representation with optimization
    ↓
Backend Codegen (C, CUDA, OpenCL, Metal, WGSL)
    ↓
Execution (CPU, GPU)
```

### Target Architecture with MLIR

```
User APIs (nn.h, optim.h, autograd.h)
    ↓
Tensor Facade (tensor/tensor.h, autograd/forward_ops.h)
    ↓
UOps (ops/uops.h) - 27 micro-operations
    ↓
C-ML IR (ops/ir/ir.h) - High-level IR with fusion
    ↓
    ├─→ [Fast Path] Direct Codegen (for debugging/development)
    │
    └─→ [Optimized Path] MLIR Pipeline
            ↓
        MLIR High-Level Dialects (Linalg, Tensor)
            ↓
        MLIR Optimization Passes
            ↓
        MLIR Low-Level Dialects (SCF, Arith, Math)
            ↓
        Backend Selection
            ├─→ LLVM IR → CPU (x86, ARM, RISC-V)
            ├─→ GPU Dialect → CUDA/ROCm/Vulkan
            ├─→ SPIR-V → OpenCL/Vulcan
            └─→ WebGPU WGSL
            ↓
        JIT Compilation OR AOT Binary
            ↓
        Execution
```

______________________________________________________________________

## Current State Analysis

### Strengths

✅ **Clean abstraction layers** - Clear separation between user API, tensors, UOps, and IR
✅ **Complete UOp coverage** - All 27 operations implemented
✅ **IR optimization** - Fusion passes already implemented
✅ **Multiple backends** - C, CUDA, OpenCL, Metal, WGSL codegen
✅ **Pure C codebase** - No external runtime dependencies

### Weaknesses

❌ **Hand-rolled codegen** - High maintenance, code duplication across backends
❌ **No JIT compilation** - String-based code generation + compile step
❌ **Limited optimization** - Only basic fusion, no auto-vectorization
❌ **Performance gap** - Slower than PyTorch/TensorFlow on production workloads
❌ **Scalability** - Adding new backends requires reimplementing all 27 ops

### Opportunities with MLIR

🚀 **JIT compilation** - Compile IR to machine code at runtime
🚀 **Advanced optimizations** - Polyhedral optimization, auto-tuning
🚀 **Unified backend** - One MLIR dialect → all backends
🚀 **Industry standard** - Same stack as TensorFlow, PyTorch, JAX
🚀 **Future-proof** - TPU, NPU, custom accelerators supported

______________________________________________________________________

## MLIR Integration Strategy

### Hybrid Execution Model (Recommended)

#### Mode 1: Debug/Development Mode (Current Codegen)

```c
// Keep existing hand-rolled codegen for:
// - Fast iteration during development
// - Debugging with readable generated code
// - Platform without MLIR support
cml_set_execution_mode(CML_EXEC_INTERPRETED);
```

#### Mode 2: JIT Mode (MLIR Runtime Compilation)

```c
// Use MLIR JIT for:
// - Training workloads
// - Dynamic shapes
// - Exploratory research
cml_set_execution_mode(CML_EXEC_JIT);
```

#### Mode 3: AOT Mode (Ahead-of-Time Compilation)

```c
// Use MLIR AOT for:
// - Production inference
// - Embedded deployment
// - Mobile devices
cml_set_execution_mode(CML_EXEC_AOT);
```

### Backward Compatibility

**Guarantee:** All existing C-ML code continues to work without modification.

```c
// Existing code - no changes needed
Tensor* a = tensor_randn((int[]){2, 3}, 2, NULL);
Tensor* b = tensor_randn((int[]){2, 3}, 2, NULL);
Tensor* c = tensor_add(a, b);  // Automatically uses best execution mode

// Opt-in to MLIR features
cml_enable_jit(true);  // Now tensor_add uses JIT compilation
```

______________________________________________________________________

## Implementation Phases

See full plan at: [MLIR Integration Detailed Phases](./MLIR_PHASES.md)

### Phase 0: Preparation (Week 1-2)

- Set up MLIR build environment
- Create minimal "hello world" MLIR integration
- Validate toolchain

### Phase 1: Foundation (Week 3-6)

- Implement converter for 5 core operations
- Get JIT working end-to-end
- Establish testing infrastructure

### Phase 2: Full UOp Coverage (Week 7-10)

- All 27 UOps mapped to MLIR
- Complete test coverage
- Benchmarking suite

### Phase 3: Optimization Pipeline (Week 11-14)

- MLIR optimization passes
- Fusion at MLIR level
- Vectorization

### Phase 4: Multi-Backend Support (Week 15-18)

- CUDA/ROCm
- Vulkan/SPIR-V
- Metal, WebGPU

### Phase 5: JIT Engine (Week 19-22)

- Robust JIT compilation
- Kernel caching
- Dynamic shapes

### Phase 6: AOT Compilation (Week 23-26)

- Ahead-of-time compilation
- Model serialization
- Deployment tooling

______________________________________________________________________

## Key Milestones

| Milestone                   | Week | Deliverables                       |
| --------------------------- | ---- | ---------------------------------- |
| **M1: MLIR Hello World**    | 2    | MLIR builds, simple ADD works      |
| **M2: Core Operations**     | 6    | 5 ops end-to-end, 2x speedup       |
| **M3: Full Coverage**       | 10   | All 27 ops, >90% test coverage     |
| **M4: Optimized**           | 14   | Fusion + vectorization, 5x speedup |
| **M5: Production Backends** | 18   | CUDA + Vulkan working              |
| **M6: JIT Production**      | 22   | Caching, dynamic shapes, robust    |
| **M7: AOT Ready**           | 26   | AOT compilation, deployment ready  |

______________________________________________________________________

## Success Criteria

### Must Have

✅ All 27 UOps compile via MLIR
✅ JIT compilation functional
✅ 5x performance improvement over interpreted mode
✅ Backward compatibility maintained
✅ At least CUDA backend working

### Should Have

✅ AOT compilation working
✅ Multiple GPU backends (CUDA + Vulkan/ROCm)
✅ Comprehensive documentation
✅ Production-ready error handling

### Nice to Have

✅ Metal/WebGPU backends
✅ Auto-tuning for different hardware
✅ Distributed execution support

______________________________________________________________________

## Resources

### Documentation

- [MLIR Official Docs](https://mlir.llvm.org/)
- [MLIR Tutorials](https://mlir.llvm.org/docs/Tutorials/)
- [Linalg Dialect Guide](https://mlir.llvm.org/docs/Dialects/Linalg/)

### Example Projects

- [TensorFlow MLIR](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir)
- [PyTorch MLIR](https://github.com/llvm/torch-mlir)
- [IREE](https://github.com/openxla/iree) - End-to-end MLIR compiler

### Community

- [LLVM Discourse](https://discourse.llvm.org/c/mlir/)
- [MLIR Discord](https://discord.gg/xS7Z362)

______________________________________________________________________

## Next Steps

1. **Review this plan** with the team
1. **Set up development environment** (LLVM/MLIR build)
1. **Begin Phase 0** Preparation
1. **Regular progress reviews** every 2 weeks

For detailed implementation specifications, see the full plan document.

**Document Revision:** 1.0 | **Date:** 2025-11-27
