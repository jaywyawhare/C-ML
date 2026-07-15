# C-ML IR/Dispatch/Autograd Audit Report

Date: 2026-04-21
Scope: dispatch layer, autograd backward path, IR generation, IR optimization/scheduling

## Executive Summary

The current stack has good breadth (many backends and many UOps), but key control-plane pieces are inconsistent: backend selection is fragmented, backward graph construction is mostly stubbed, and optimizer/scheduler behavior is not consistently integrated into normal execution.

Most critical project-wide risks are:
1. Incorrect or incomplete gradient computation for a large portion of supported ops.
2. Backend routing inconsistencies that cause silent CPU fallback or behavior drift between execution entry points.
3. IR optimization behavior that can drop live nodes in multi-output scenarios and leak/free params inconsistently.

---

## Findings (Ordered by Severity)

### 1) Backward graph build is effectively a no-op (High)
- Evidence:
  - `src/ops/ir/backward.c:25-33` (`cml_ir_build_backward`) only validates args and sets `output_node->is_used = true`.
- What is missing / wrong:
  - No actual backward graph construction, dependency pruning, or explicit gradient graph structure.
- Project impact:
  - Gradient flow depends on generic full-graph reverse scan rather than an explicit, scoped backward graph.
  - Makes gradient correctness fragile as graph complexity grows (branching, reused tensors, multi-loss setups).

### 2) Backward execution runs reverse over full node list, not true dependency-pruned subgraph (High)
- Evidence:
  - `src/ops/ir/backward.c:1214-1252` builds array of all nodes.
  - `src/ops/ir/backward.c:1245-1247` iterates full reverse order.
- What is missing / wrong:
  - No traversal from target loss/output through only required predecessors.
- Project impact:
  - Unrelated graph regions can participate in backward traversal.
  - Higher runtime and elevated risk of accidental gradient side effects in shared/global IR contexts.

### 3) Gradient initialization strategy is internally inconsistent (High)
- Evidence:
  - `ensure_grad` explicitly warns against lazy `tensor_zeros` and uses eager allocation (`src/ops/ir/backward.c:39-49`).
  - `cml_ir_execute_backward` still initializes loss grad with `tensor_zeros` (`src/ops/ir/backward.c:1267-1274`).
  - `tensor_zeros` is lazy via `uop_fill_ex` (`src/tensor/tensor.c:259-267`).
- What is missing / wrong:
  - Backward path mixes eager and lazy grad tensors despite clear local comment that lazy zeros is unsafe mid-backward.
- Project impact:
  - Risk of re-entering IR execution during backward initialization.
  - Can produce incorrect execution order, extra graph growth, or hard-to-debug memory/liveness issues.

### 4) Backward rule coverage lags far behind UOp surface (High)
- Evidence:
  - Distinct UOP symbols: `140` (`include/ops/uops.h`).
  - Distinct backward switch cases: `50` (`src/ops/ir/backward.c`).
  - Many ops default to no gradient rule (`src/ops/ir/backward.c:1205-1208`).
  - Unary op builder broadly marks `requires_grad` (`src/ops/uops.c:124-140`) and exposes ops like `uop_abs`, `uop_cos`, `uop_tan` (`src/ops/uops.c:160-163`).
- What is missing / wrong:
  - Public op surface is larger than implemented gradient rules.
- Project impact:
  - Silent partial autograd: model trains “successfully” but with wrong/missing gradients for many ops.
  - This is a correctness risk, not just a performance issue.

### 5) Dispatch best-backend selection omits OpenCL despite initialized OpenCL support (High)
- Evidence:
  - OpenCL backend initialization exists (`src/ops/ir/dispatch.c:307-328`).
  - `cml_dispatch_get_best_backend` does not consider `CML_BACKEND_OPENCL` (`src/ops/ir/dispatch.c:454-489`).
- What is missing / wrong:
  - Best-backend heuristic is incomplete.
- Project impact:
  - Systems where OpenCL is the only viable accelerator can be misrouted to less optimal paths.
  - Makes backend behavior unpredictable relative to detected capabilities.

### 6) Execution entry points use inconsistent backend routing logic (High)
- Evidence:
  - `cml_ir_execute` only special-cases OpenCL env (`src/ops/ir/execution.c:3589-3599`).
  - `cml_ir_execute_up_to` only recognizes Metal/OpenCL (`src/ops/ir/execution.c:3629-3638`).
  - Dispatch layer supports many env values (`src/ops/ir/dispatch.c:911-936`) and generic fallback execution (`src/ops/ir/dispatch.c:703-733`).
- What is missing / wrong:
  - Multiple conflicting backend-selection implementations across files.
- Project impact:
  - Different execution APIs can choose different backends for same graph/env.
  - Hard to reason about performance and correctness regressions.

### 7) `cml_ir_execute_up_to` can ignore “up_to” semantics under schedule v2 / selected backend path (High)
- Evidence:
  - In `cml_ir_execute_up_to`, if backend path succeeds it returns immediately (`src/ops/ir/execution.c:3644-3647`).
  - If `CML_SCHEDULE_V2=1`, it returns `cml_ir_execute_v2(ir)` (`src/ops/ir/execution.c:3649-3658`), which executes ordered groups for the graph (`src/ops/ir/schedule_v2.c:533-550`).
- What is missing / wrong:
  - Partial-execution API can become full-graph execution depending on env.
- Project impact:
  - API contract mismatch; callers expecting prefix/target execution may see hidden extra work and side effects.

### 8) Matmul shape/autograd path appears 2D-centric while API accepts ndim>=2 (Medium-High)
- Evidence:
  - `tensor_matmul` accepts ndim>=2 but forces output shape to 2D `[M, N]` (`src/autograd/forward_ops.c:405-425`).
  - Backward matmul path uses 2D assumptions (`src/ops/ir/backward.c`, matmul gradient block around `UOP_MATMUL`).
- What is missing / wrong:
  - Batched matmul semantics are not represented end-to-end.
- Project impact:
  - Wrong shapes/gradients for higher-rank matmul use cases.
  - Limits transformer-class workloads and complicates interoperability expectations.

### 9) DCE reachability roots only at `ir->tail` (Medium-High)
- Evidence:
  - `mark_reachable_nodes` seeds traversal only from `ir->tail` (`src/ops/ir/optimization.c:87-90`).
- What is missing / wrong:
  - Multi-output/liveness roots are not modeled.
- Project impact:
  - Optimizer can mark still-needed nodes dead in graphs with retained non-tail outputs.

### 10) Dead-node free path in optimizer bypasses centralized param-free helper (Medium)
- Evidence:
  - Optimizer’s `remove_dead_nodes` manually frees many fields then `free(node)` (`src/ops/ir/optimization.c:147-185`).
  - Centralized per-op param cleanup exists (`cml_ir_free_node_params`, `src/ops/ir/ir.c:300+`) and is used in normal node free path (`src/ops/ir/ir.c:725`).
- What is missing / wrong:
  - DCE free path is not unified with canonical node destruction path.
- Project impact:
  - Leak or stale-free risk for op params added in future and not mirrored here.

### 11) Dispatch synchronize is partial (Medium)
- Evidence:
  - `cml_dispatch_synchronize` handles CUDA/ROCm/Vulkan only (`src/ops/ir/dispatch.c:832-843`).
  - Dispatch supports additional async-capable backends (OpenCL/Metal/WebGPU/NV/AM) in execution/init paths.
- What is missing / wrong:
  - Synchronization coverage does not match backend surface.
- Project impact:
  - Inconsistent completion semantics and timing behavior across backends.

### 12) `cml_dispatch_execute_on` ignores explicit input/output binding arguments (Medium)
- Evidence:
  - `(void)inputs; (void)nin; (void)outputs; (void)nout;` (`src/ops/ir/dispatch.c:498-500`).
- What is missing / wrong:
  - API signature implies data binding but implementation discards it.
- Project impact:
  - Misleading interface; limits future runtime integration (external buffers, IO pinning, launch contracts).

### 13) Schedule v2 currently executes via CPU node fallback (Medium)
- Evidence:
  - `cml_ir_execute_v2` runs each node with `cpu_execute_node` (`src/ops/ir/schedule_v2.c:540-545`).
- What is missing / wrong:
  - Grouping/fusion schedule does not dispatch to backend-specific kernels.
- Project impact:
  - `CML_SCHEDULE_V2=1` may regress acceleration expectations and create confusing perf profiles.

### 14) Global IR context initialization appears unsynchronized (Medium)
- Evidence:
  - `cml_ir_get_or_create_context` lazily initializes `g_global_ir_context` without mutex/once guard (`src/ops/ir/context.c:135-142`).
- What is missing / wrong:
  - Thread-safe initialization semantics are unclear.
- Project impact:
  - Potential races in multi-threaded usage; non-deterministic graph ownership/lifetime issues.

---

## What We Are Doing Wrong (Systemic)

1. We expose broad capability (many backends and ops) before making correctness/completeness contracts explicit.
2. We duplicate control logic (backend selection, partial execution behavior) across layers instead of having one source of truth.
3. We mix lazy/eager semantics in autograd-critical paths without strict invariants.
4. We treat optimizer/scheduler as optional side-paths instead of defining when they are authoritative in normal execution.
5. We maintain multiple node destruction/free paths, increasing long-term maintenance risk.

---

## Project-Wide Impact

- Training correctness risk: missing/partial gradients can silently degrade model quality.
- Performance unpredictability: backend selection differs by API path/env and may collapse to CPU unexpectedly.
- Debug complexity: same graph can behave differently under `execute`, `execute_up_to`, and schedule-v2 modes.
- Scalability limits: batched linear algebra and multi-output graph optimization are not robustly modeled.
- Reliability debt: memory lifecycle and synchronization inconsistencies become more severe as feature surface grows.

---

## Recommended Remediation Plan (Priority Order)

1. Unify execution routing around dispatch (single backend selection/override path used by `cml_ir_execute` and `cml_ir_execute_up_to`).
2. Implement real backward graph construction and dependency-pruned backward traversal.
3. Standardize gradient tensor allocation in backward to eager-safe primitives only.
4. Define and enforce autograd coverage policy:
   - Either implement gradient rules for all `requires_grad`-capable ops, or
   - fail loudly when an op in grad path lacks a rule.
5. Fix matmul semantics for batched tensors (forward shape inference + backward kernels).
6. Fix DCE rooting to include all required outputs/retained tensors, not only tail.
7. Refactor optimizer node deletion to use canonical node-free routine (including param free helper).
8. Expand `cml_dispatch_synchronize` coverage to all async backends.
9. Clarify/implement `inputs/outputs` contract in `cml_dispatch_execute_on` or remove parameters.
10. Make schedule-v2 backend-aware or gate it as experimental CPU-only mode.
11. Add thread-safe global IR context initialization.

---

## Suggested Verification Matrix After Fixes

- Gradient correctness tests:
  - Per-op finite difference checks for all ops marked `requires_grad`.
  - Multi-branch graph backward pruning correctness.
- Dispatch consistency tests:
  - Same graph, same env through `cml_ir_execute` vs `cml_ir_execute_up_to` should select consistent backend policy.
- Matmul shape/grad tests:
  - 2D and batched-ND cases, including broadcasting where supported.
- DCE safety tests:
  - Multi-output graph where non-tail outputs are retained and used.
- Synchronization tests:
  - Cross-backend async completion invariants.

