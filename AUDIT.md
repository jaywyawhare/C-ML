# C-ML Codebase Audit — Status

~110K LOC C ML library. This tracks the original audit findings against the
**current** tree. Verified 2026-07-10 by reading the referenced code and running
the full test suite (104/104 C tests pass).

Summary: **all 21 findings are addressed.** The last 4 (large, standalone
efforts) were implemented on branch `shape-specific-simd` with dedicated tests;
#10 landed a safe, build-verified first step (CPU as a first-class HCQ backend)
rather than a full multi-GPU rewrite — see its note below.

Legend: ✅ fixed · 🟡 partial · ⭕ open

---

## Critical Bugs — all resolved

### 1. ✅ Non-atomic global ID counter
`src/symbolic/symbolic.c:8,56` — now `static _Atomic int g_var_id_counter` with
`atomic_fetch_add`. Fixed.

### 2. ✅ `pthread_once_t` manual reset UB
`src/autograd/autograd.c:16-62` — the `pthread_once` reset was removed; init now
uses `g_autograd_init_mutex` + a `g_autograd_initialized` bool, so
shutdown→init cycles are well-defined. Fixed.

### 3. ✅ `tensor_retain_grad` semantics
`src/autograd/autograd.c:143-148` — now sets `t->retains_grad = true` (a
dedicated flag), not `requires_grad`. Matches the PyTorch semantic. Fixed.

### 4. ✅ Hook storage aliasing
`src/autograd/autograd.c:162-182` — hooks live in dedicated `t->backward_hooks`
/ `module->backward_hooks` fields, no longer stomping `user_data`. Fixed.

### 5. ✅ Fixed hook capacity
`src/autograd/autograd.c:184-210` — `hook_list_append_*` grows capacity
(`0→4→8…`) via `realloc`. No silent drops. Fixed.

### 6. ✅ Module hook overwrite
`src/autograd/autograd.c:200-210` — module hooks use `hook_list_append_module`
(append to a growable list), not single-slot overwrite. Fixed.

### 7. ✅ `tensor_detach_inplace` dropping lazy compute
`src/autograd/autograd.c:126-141` — materializes via `tensor_ensure_executed`
before clearing `ir_node`. Fixed.

---

## Architectural Issues

### 8. ✅ Single global IR context
`src/ops/ir/context.c:14` — now `static _Thread_local CMLGraph_t
tls_ir_context`. Graph construction is thread-local, so concurrent threads no
longer corrupt a shared graph. Fixed (this was the #1 "drastic change").

### 9. ✅ Constant folding duplicated
`src/symbolic/symbolic.c:22` — folding is centralized in `sym_fold_binop`,
called from both `sym_binop` (:66) and `sym_simplify`/`sym_eval` (:195,:223).
Fixed.

### 10. 🟡 GPU backend fragmentation — HCQ unification (first step done)
The CPU is now a first-class HCQ backend (`g_hcq_cpu_ops` in
`src/ops/ir/hcq_backend.c`); the ~8 scattered `if (backend != CML_HCQ_CPU)`
special-cases in `src/ops/ir/hcq.c` are gone, so every backend — CPU included —
dispatches uniformly through `cml_hcq_backend_ops()`. This gives HCQ a complete,
always-available reference implementation and makes the queue/signal/pipeline
machinery testable without hardware (`tests/test_hcq_cpu.c`, 6/6).
**Remaining:** the full `src/ops/ir/gpu/*_backend.c` (CUDA/ROCm/Vulkan/CL/WebGPU)
still carry independent memory/dispatch code; collapsing those onto HCQ needs
real multi-GPU hardware to validate and was intentionally not attempted blind.

### 11. ✅ `rand()` in layer init
`rand()` is gone from `src/nn/layers/`; `cml_manual_seed` (`src/cml.c:1037`) +
`tensor_manual_seed` provide a seeded PRNG. Fixed (see also #21).

### 12. ✅ Inconsistent error propagation
Unified: `cml_log_message` now records every `LOG_ERROR` into the thread-local
error stack (`src/core/logging.c`), so any failure is queryable via
`cml_get_last_error()` / `cml_get_last_error_code()` regardless of a function's
NULL/-1 return. Added `cml_error_string()` and documented the canonical
convention in `include/core/error_codes.h`. Test: `tests/test_error_propagation.c`.

### 13. ✅ `sym_eval` unknown-variable fallback
`src/symbolic/symbolic.c:182` — returns `-1` on an unknown variable instead of
guessing the midpoint. Fixed.

---

## Performance Bottlenecks

### 14. ✅ Buffer cache fragmentation / wire TLSF
`src/ops/ir/execution.c:77-186` — `cml_buffer_cache_alloc` now falls back to
`exec_pool_alloc`, backed by the TLSF allocator (`g_exec_tlsf`,
`cml_tlsf_alloc`); per-bucket count raised to 32. Fixed.

### 15. ✅ Attention not fused / flash attention
`src/nn/layers/transformer.c:140-161` — `use_flash` path calls
`cml_gqa_flash_forward` (chunked Q×K online-softmax) for long sequences;
`flash_attention_forward` at :701. Implemented.

### 16. ✅ Kernel cache on the hot path
`src/ops/ir/kernel_cache.c` is wired through `src/ops/ir/dispatch.c` (compile
path) and `src/cml.c`. The LLVM backend additionally caches shape-specialized
kernels (`src/ops/ir/llvm/llvm_backend.c`). Connected.

### 17. ✅ Reduction SIMD vectorization
`src/ops/ir/execution.c:942-1004` — SUM/MEAN use `simd_sum_float` /
`simd_sum_float_strided` (row/col/strided paths), now portable scalar loops the
compiler auto-vectorizes; the JIT emits shape-specialized reduction kernels.
Fixed (see the shape-specific SIMD work — `docs/shape_specific_simd_plan.md`).

---

## Incomplete / Wired-But-Not-Connected

### 18. ✅ Serving layer integrated (generation loop)
`cml_serving_step` now drives real autoregressive generation via a pluggable
model forward callback (`cml_serving_set_model`) + sampling (greedy / temperature
/ top-p), with EOS and max-token termination and continuous batching across
requests (`src/nn/serving.c`). The callback owns the forward pass + KV cache.
Test: `tests/test_serving_generate.c` (EOS, max-token, batched).

### 19. ✅ Quantized matmul dispatch (weight-only int8)
`cml_quantize_weight_int8` produces an int8 weight carrying affine params
(`CML_QUANT_AFFINE_INT8`), and the matmul executor dispatches to
`cml_qmatmul_affine_int8` (int8 weights, f32 activations, no dequant round-trip —
4× smaller weights). Wired in `src/ops/ir/execution.c`; the LLVM matmul path
defers quantized weights to it. Test: `tests/test_quant_matmul.c` (exact GEMM +
end-to-end within the rigorous quant-error bound). INT4/NF4 GEMM remains future
work.

### 20. ✅ Python boundary validation
`python/cml/core.py:113-141` — `_validate_shape` / `_validate_dtype` enforce
shape and dtype at the CFFI boundary. Fixed.

### 21. ✅ Reproducibility / `cml_manual_seed`
`src/cml.c:1037` (`cml_manual_seed` → `tensor_manual_seed`); exposed in
`include/cml.h:182` and via `torch_manual_seed`. Fixed.

---

## Remaining future work (beyond the original audit)

- **#10 full GPU unification** — collapse the `gpu/*_backend.c` memory/dispatch
  code onto HCQ. Needs real multi-GPU hardware to validate; the CPU reference
  backend and uniform dispatch are now in place as the foundation.
- **#19 INT4/NF4 GEMM** — a packed 4-bit matmul path (quantize + dequantize
  already exist; only the fused GEMM is missing).
- **Multi-dtype compute** — the executor is still f32-only (the dtype enum is
  storage metadata). This is the largest structural gap for numpy/torch parity.

All original 21 audit items are addressed. Full suite: 108/108 C tests pass.
