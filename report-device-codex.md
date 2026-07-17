# Device / Driver / Hardware Audit Report

Date: 2026-04-21
Repo: `C-ML`
Scope: device code, driver code, hardware-specific backend code, wiring, tests, and docs consistency.

## Executive Summary

The repository has two backend layers that are partially overlapping:
- Legacy layer: `src/backend/{device,backend}.c`
- Main execution layer: `src/ops/ir/dispatch.c` + backend-specific IR executors

Most advanced GPU/driver behavior is implemented in the dispatch layer, while the legacy layer exposes enums/APIs that suggest broader support than what is actually wired. The biggest issues are consistency and integration, not just missing files.

## High-Priority Findings

### 1) Device API advertises OpenCL/Vulkan-style priority, but implementation is incomplete

Evidence:
- `include/backend/device.h:18` defines `DEVICE_OPENCL`.
- `include/backend/device.h:35` and `:40` comments claim priority includes `Vulkan`/`OpenCL`/`oneAPI`.
- `src/backend/device.c:378-394` (`device_get_name`) has no `DEVICE_OPENCL` case.
- `src/backend/device.c:466-557` (`device_alloc`) has no `DEVICE_OPENCL` case.
- `src/backend/device.c:567-616` (`device_free`) has no `DEVICE_OPENCL` case.
- `src/backend/device.c:397-431` (`device_print_info`) only reports CUDA/Metal/ROCm/SimGPU.

Impact:
- API contract and runtime behavior diverge.
- Any call path using `DEVICE_OPENCL` will hit default/unknown logic instead of true OpenCL behavior.
- Comments set expectations for backends that are not represented in the enum or logic (e.g., Vulkan/oneAPI).

What is missing:
- Either full `DEVICE_OPENCL` implementation in `device.c`, or removal/deprecation of `DEVICE_OPENCL` from this API.
- Comment cleanup to match real selection policy.

---

### 2) Legacy `BackendType` GPU labels do not map to actual GPU execution

Evidence:
- `src/backend/backend.c:645-647` maps `BACKEND_CUDA` to `scalar_ops`.
- `src/backend/backend.c:579-643` maps METAL/ROCM mostly to `simd_ops` and host-side behavior.
- `src/backend/backend.c:764-781` reports CUDA/METAL/ROCM availability checks, implying true backend support.

Impact:
- Choosing `BACKEND_CUDA` does not give CUDA compute in this layer.
- Very high risk of false confidence and benchmark misinterpretation.

What is missing:
- Hard separation between “vectorized CPU backend” and “actual GPU backend”.
- If this layer is kept, rename semantics or route to dispatch layer.

---

### 3) `cml_dispatch_get_best_backend` ignores OpenCL despite detecting and executing it

Evidence:
- OpenCL is detected and initialized: `src/ops/ir/dispatch.c:307-326`.
- OpenCL execution exists: `src/ops/ir/dispatch.c:672-685`.
- Fallback chain includes OpenCL: `src/ops/ir/dispatch.c:137`.
- Best-backend chooser omits OpenCL: `src/ops/ir/dispatch.c:454-489`.

Impact:
- Auto selection can skip a viable OpenCL backend.
- Behavior differs between explicit preference/fallback and best-backend logic.

What is missing:
- OpenCL placement in `cml_dispatch_get_best_backend` decision order.

---

### 4) Synchronization API does not cover many initialized backends

Evidence:
- `src/ops/ir/dispatch.c:832-843` only synchronizes CUDA/ROCm/Vulkan.
- Missing explicit sync for Metal/WebGPU/OpenCL/NV/AM backends in this function.

Impact:
- Inconsistent completion guarantees across backends.
- Higher chance of timing/race-related issues in mixed backend workflows.

What is missing:
- Per-backend sync calls (or explicit “not required” guarantees documented and enforced).

---

### 5) `cml_ir_execute` path bypasses general backend selection logic

Evidence:
- `src/ops/ir/execution.c:3589-3599` special-cases only OpenCL from env.
- `src/ops/ir/execution.c:3623-3638` in `cml_ir_execute_up_to` only special-cases Metal/OpenCL.
- Dispatch env parser supports many values (`cuda`, `rocm`, `vulkan`, `nv`, `am`, `nir`, `webgpu`, `opencl`): `src/ops/ir/dispatch.c:901-941`.

Impact:
- `BACKEND=cuda` and others may not behave as expected on direct `cml_ir_execute` paths.
- Surprising differences between APIs depending on entry point.

What is missing:
- Single source of truth for backend selection; `execution.c` should delegate to dispatch consistently.

## Medium-Priority Findings

### 6) CMake backend source gating is inconsistent and includes duplicates

Evidence:
- `CMakeLists.txt:292-295` includes `nv_driver.c`/`am_driver.c` unconditionally in `OPS_SOURCES`, even though feature options exist.
- `CMakeLists.txt:314` appends `webgpu_backend.c` unconditionally, while `ENABLE_WEBGPU` defaults OFF.
- Duplicated source entries:
  - `schedule_v2.c` appears at `:272` and `:333`
  - `linearize.c` appears at `:273` and `:333`
  - `nir_compiler.c` / `hcq_nir.c` appear in base list (`:278-279`) and again under `ENABLE_NIR` (`:323`)

Impact:
- Build intent is unclear.
- Feature flags are less meaningful than they appear.
- Maintenance/debug overhead increases.

What is missing:
- A strict source gating strategy aligned to feature options and compile definitions.

---

### 7) Hardware-specific backends include placeholder or early-stage execution logic

Evidence:
- Adreno execution emits placeholder identity kernel per op: `src/ops/ir/gpu/adreno_backend.c:357-363`.
- Hexagon backend defaults DSP version and calls generic remote invoke per node with comments indicating incomplete shared-memory/serialization path: `src/ops/ir/gpu/hexagon_backend.c:86-90`, `:159-165`.
- Thunder executor has a finite op mapping and explicit unimplemented dispatch default: `src/backend/thunder_executor.c:189-191`.

Impact:
- “Backend available” does not always imply functionally complete operator support.
- Production readiness differs significantly by backend.

What is missing:
- Capability matrix enforcement at runtime (op coverage / dtype / shape constraints).
- Backend maturity tiers reflected in docs and API.

## Test & Validation Gaps

Evidence of missing focused tests:
- No dedicated tests for legacy `device.c` OpenCL handling (or lack thereof).
- No `test_thunder_executor.c`.
- No `test_usb_device.c` (there is `test_usb3_gpu.c`).
- Existing backend tests are strong on dispatch and selected drivers, but coverage is uneven for legacy APIs and adapter layers.

Impact:
- Regressions in legacy/backend glue code are likely to go unnoticed.

What is missing:
- Contract tests around backend/device API parity and env-driven backend selection across all entry points.

## Documentation Drift

Evidence:
- README claims broad hardware coverage and userspace driver support: `README.md:76`, `:98`.
- Current codebase includes several partial/stub/backend-specific caveats that are not reflected in top-level claims.

Impact:
- Users may assume feature-complete support where behavior is backend- and path-dependent.

What is missing:
- Clear support-status table (stable / experimental / partial / mock) per backend.

## What We Are Doing Wrong (Root Causes)

1. Maintaining two backend abstractions without strict ownership boundaries.
2. Advertising capabilities at enum/docs level before end-to-end wiring is complete.
3. Allowing backend selection logic to fragment across `dispatch.c`, `execution.c`, and legacy APIs.
4. Treating backend “available” as equivalent to “feature complete”.
5. Letting build flags and source inclusion drift out of sync.

## Recommended Fix Plan (Priority Order)

1. Unify backend selection and execution through dispatch layer only.
2. Decide fate of legacy `backend.c`/`device.c` GPU semantics:
   - either deprecate them,
   - or rewire them to dispatch-backed execution with explicit compatibility notes.
3. Fix device API contract mismatch (`DEVICE_OPENCL`, selection comments, print/info paths).
4. Add missing backend sync coverage in `cml_dispatch_synchronize`.
5. Update `cml_dispatch_get_best_backend` to include OpenCL (and verify intended order).
6. Refactor `execution.c` env/backend handling to call dispatch parser/selection once.
7. Clean CMake source gating and duplicate source entries.
8. Publish a backend maturity matrix in docs and expose capability queries in API.
9. Add targeted tests for:
   - backend/device contract consistency,
   - env-based backend selection across all execution entry points,
   - thunder/usb_device/adreno/hexagon capability boundaries.

## Suggested Immediate “Safety” Changes

- In legacy APIs, fail loudly when selecting unsupported GPU paths instead of silently falling back to scalar.
- Add runtime warnings whenever a backend marked as available is running placeholder/stub operators.
- Gate README hardware claims with maturity labels to reduce user-facing ambiguity.

## Conclusion

The repo has substantial hardware backend surface area, but the main issue is not lack of files; it is inconsistent wiring and mixed abstraction ownership. Fixing contract consistency (API/docs/build/runtime alignment) will produce bigger reliability gains than adding new backend code right now.
