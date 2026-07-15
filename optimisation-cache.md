# CML BLAS Optimisation Cache

Tracked optimisation ideas and their current status.

---

## Approach 2 — Cache-Blocked GEMM with B-Matrix Packing

**Status:** Implemented in `src/backend/blas.c`
**Actual gain:** fused_512: 1.82ms → 1.57ms (14%); primary benefit is the LP64 fallback path (no scipy_openblas64)

### Why the current AVX2 kernel is slow

`sgemm_avx_small` in `src/backend/blas.c` uses a naive three-loop layout:

```
for m in M:
  for k in K:
    a_mk = A[m, k]                   # scalar broadcast
    for n in N (vectorised, 8-wide):
      C[m, n] += a_mk * B[k, n]      # strided B read every k-step
```

Problems:
- **B is never packed.** B[k, n] strides by N floats between k iterations. For K=784,
  every inner loop re-loads B rows from RAM/L3, not L1.
- **No register blocking.** Each (m,k) pair occupies one AVX register pair for C.
  A good microkernel holds a 6×16 tile of C in AVX registers across all K iterations
  so C is never written back until the tile is complete.
- **No A packing.** A's rows are read once each but cache conflicts occur when M×K
  overflows L2.

Measured: 0.36 ms for (64, 784) × (784, 128).  
scipy_openblas64 (Haswell kernel): 0.09 ms — same hardware, same AVX2+FMA.

### What a proper implementation looks like

Three-level tiling matching the L1/L2/L3 cache hierarchy (BLIS-style):

```
Macro-kernel tiles: (MC × KC) from A, (KC × NC) from B
  MC ≈ 120 (rows of A that fit alongside a packed B panel in L2)
  KC ≈ 256 (reduction depth that fits a B panel in L2: KC×NC×4 bytes ≤ 256 KB)
  NC ≈ 2040 (cols of B; entire packed B panel fits in L3)

Micro-kernel: 6 × 16 register tile (6 AVX2 registers for C, 2 for B, 1 for A)
  Processes MR=6 rows of A × NR=16 cols of B across all KC k-steps.
  Inner loop (unrolled × 4):
    vbroadcastss  ymm_a0, A[m+0, k]
    vfmadd231ps   ymm_c00, ymm_a0, ymm_b0    # C[m+0, n+0..7]
    vfmadd231ps   ymm_c01, ymm_a0, ymm_b1    # C[m+0, n+8..15]
    ... repeat for m+1..m+5
```

**B packing** (the key step): before the macro-kernel, copy the KC×NC panel of B into a
contiguous buffer in column-major NC×KC layout. All subsequent B accesses are sequential
with no stride. This is what eliminates the cache misses in the current kernel.

### Implementation plan

1. Add `sgemm_pack_B(const float* B, float* B_packed, int K, int N)` — copies B into
   column-panel layout with NR=16 alignment.
2. Add `sgemm_pack_A(const float* A, float* A_packed, int M, int K)` — row-panel copy.
3. Add `sgemm_ukr_6x16(float* C, const float* A_packed, const float* B_packed, int K)`
   — the hot inner kernel: 6 accumulators × 2 B vectors × K FMA pairs, fully unrolled
   by 4 with `__m256` intrinsics.
4. Wire into `cml_blas_sgemm` as a third path between the current AVX micro-kernel
   and the OpenBLAS path: `M*N*K ∈ [8M, threshold_for_openblas]`.
5. Pack B once in `fast_path_build` (since weights are fixed at inference time) —
   zero packing cost at inference.

### Expected outcome

| Shape | Current (AVX2) | Packed kernel | scipy_openblas64 |
|---|---|---|---|
| (64, 784)→128 | 0.36 ms | ~0.10 ms | 0.09 ms |
| (512, 512)→512 | — (OpenBLAS) | ~1.8 ms | — |

Full MLP forward target: **~0.13 ms** (down from 0.5 ms).
