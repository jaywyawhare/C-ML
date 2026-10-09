# C-ML Distributed + I/O + Streaming Audit Report

Date: 2026-04-21
Scope: distributed runtime/backends, data loading/prefetch streaming, model/tensor/optimizer I/O, disk I/O backend, paged/streaming inference path

## Executive Summary

The codebase exposes broad distributed and I/O APIs, but several paths are partially implemented or internally inconsistent. The most severe issues are correctness bugs (not just missing optimizations): async completion semantics are wrong, loader batch APIs return freed tensors, and serialization formats can become misaligned.

Top project risks:
1. Silent incorrect behavior in distributed async APIs and multi-rank initialization.
2. Use-after-free/double-free risk in DataLoader helper APIs.
3. Serialization/deserialization stream corruption in edge paths.
4. “Streaming/async” APIs that are sync-only in practice, which can mislead higher layers and hide bottlenecks.

---

## Findings (Ordered by Severity)

### 1) `dataloader_get_batch_tensors` returns pointers to tensors that are freed before return (Critical)
- Evidence:
  - It captures pointers from `Batch` (`src/core/dataset.c:1354-1355`), then immediately frees the batch (`src/core/dataset.c:1358`).
- Why this is wrong:
  - Returned `Tensor*` are dangling pointers.
- Project impact:
  - Immediate UAF/double-free risk in training loops using this helper.
  - `dataloader_for_each` then frees them again (`src/core/dataset.c:1385-1390`), compounding memory corruption risk.

### 2) Gloo ring all-reduce uses blocking send-then-recv on both sides (deadlock-prone) (Critical)
- Evidence:
  - In both ring phases each rank does `send` then `recv` (`src/distributed/ring_allreduce.c:89-93`, `133-137`).
- Why this is wrong:
  - Symmetric blocking send/recv can deadlock (especially with large payloads / limited socket buffers).
- Project impact:
  - Multi-process training can hang nondeterministically under load.

### 3) NCCL communicator initialization is incorrect across ranks (Critical)
- Evidence:
  - Each rank calls `ncclGetUniqueId` locally (`src/distributed/nccl_backend.c:268-270`) and immediately `ncclCommInitRank` with its local ID (`272-274`).
- Why this is wrong:
  - NCCL requires one shared unique ID broadcast to all ranks.
- Project impact:
  - NCCL backend cannot reliably initialize real multi-process jobs.

### 4) Distributed wait semantics hide async errors and can report success incorrectly (High)
- Evidence:
  - `cml_dist_wait` returns `0` whenever `work->completed` is true (`src/distributed/distributed.c:199-200`), ignoring `work->error_code`.
  - Sync fallback in `cml_dist_allreduce_async` marks completed but does not propagate `cml_dist_allreduce` result into `error_code` (`src/distributed/distributed.c:186-190`).
- Project impact:
  - Callers can observe success on failed collectives.
  - Hard-to-debug training divergence and stale gradients.

### 5) DataLoader worker prefetch duplicates work across threads (High)
- Evidence:
  - Each worker starts with `next_batch = 0` and independently enqueues from the same sequence (`src/core/dataset.c:223, 227-234`).
- Why this is wrong:
  - No shared atomic cursor / coordinated work partitioning.
- Project impact:
  - Duplicate batches, missing batches, unstable epoch semantics.

### 6) Pipeline parallel forward leaks cached micro-batch outputs across invocations (High)
- Evidence:
  - Previous cached pointers are overwritten with `NULL` without freeing (`src/distributed/pipeline_parallel.c:161-165`).
- Project impact:
  - Repeated forward passes leak tensors; long runs will bloat memory.

### 7) `module_save_stream` writes declared parameter count but silently skips invalid entries (High)
- Evidence:
  - Writes total `num_params` first (`src/core/serialization.c:129-134`), then `continue`s on invalid params without writing placeholder/record (`139-142`).
- Why this is wrong:
  - Stream layout no longer matches declared count.
- Project impact:
  - Loader desync / parse failure for affected models.

### 8) Optimizer state serialization can emit state type without required tensor payload (High)
- Evidence:
  - For SGD/Adam/RMSprop/Adagrad, save path still writes typed state marker when state tensors are absent (`src/core/serialization.c:711-725`, `726-754`, `755-784`).
  - Load path then expects tensors for typed states (`src/core/serialization.c:954`, `986-991`, `1035`, `1066`).
- Project impact:
  - Stream corruption and load failures or truncated state restore.

### 9) Dataset download helper returns static path buffers reused across calls (High)
- Evidence:
  - `cml_dataset_download` returns `static char path[1024]` (`src/datasets/datasets.c:83-85`).
  - `download_and_gunzip` returns `static char final_path[1024]` (`src/datasets/datasets.c:123-125`).
  - MNIST loader stores multiple returned pointers (`305-308`) and later uses them (`317-336`).
- Project impact:
  - Path aliasing/overwrite bugs when multiple files are downloaded in sequence.
  - Incorrect file reads or flaky dataset loading.

### 10) Distributed backend destroy path is inconsistent and double-invokes destroy callback (Medium-High)
- Evidence:
  - `cml_dist_destroy` calls `ops->destroy(backend_ctx)` (`src/distributed/distributed.c:132-133`), then `cml_dist_free_backend` which again calls `ops->destroy(NULL)` (`src/distributed/comm_backend.c:10-12`).
- Project impact:
  - Redundant lifecycle calls; backend-specific undefined behavior risk.

### 11) Gloo backend only has real multi-rank logic for allreduce path; other collectives are effectively single-process stubs (Medium-High)
- Evidence:
  - `gloo_broadcast`/`gloo_allgather`/`gloo_reduce_scatter`/`gloo_barrier` are no-op or local-copy style (`src/distributed/gloo_backend.c:226-269`, `237-263`).
- Project impact:
  - API appears feature-complete but behavior is incomplete for true distributed training.

### 12) MPI backend uses hard-coded MPI constants and ignores runtime rank/size validation (Medium-High)
- Evidence:
  - Hard-coded constants like `CML_MPI_COMM_WORLD ((void*)0x44000000)` and datatypes/ops (`src/distributed/mpi_backend.c:9-14`).
  - `MPI_Comm_rank`/`MPI_Comm_size` function pointers loaded but not used for consistency checks.
- Project impact:
  - Portability and correctness risk across MPI implementations.

### 13) Pipeline parallel config fields are not honored (`interleaved`, stage devices) (Medium)
- Evidence:
  - `PipelineConfig.interleaved` exists (`include/distributed/pipeline_parallel.h:21`) but forward path hardcodes GPipe schedule (`src/distributed/pipeline_parallel.c:188`).
  - Stage `device/device_id` and `group` are stored but not used for communication/placement.
- Project impact:
  - Users expect distributed pipeline behavior but receive local sequential execution.

### 14) DataParallel config surface is only partially implemented (Medium)
- Evidence:
  - Config fields `broadcast_buffers`, `find_unused_parameters`, `gradient_as_bucket_view` exist (`include/distributed/data_parallel.h:14-16`) but are not used in implementation.
  - `bucket_ready` allocated but never used (`src/distributed/data_parallel.c:70`, freed at `217`).
- Project impact:
  - API contract overpromises behavior; tuning knobs appear ineffective.

### 15) Disk backend I/O mode contract does not match implementation (Medium)
- Evidence:
  - Header advertises async/io_uring (`include/backend/disk_backend.h:21, 29-31, 65-68`).
  - Implementation sets `has_io_uring=false` always (`src/backend/disk_backend.c:48`) and `cml_disk_async_read` is synchronous fallback (`241-266`).
- Project impact:
  - Streaming I/O throughput assumptions are invalid; overlap is unavailable.

### 16) Disk backend tensor save/load path is float32-centric and can ignore dtype semantics (Medium)
- Evidence:
  - Save writes `data_size = numel * sizeof(float)` (`src/backend/disk_backend.c:77`) and writes as float elements (`83`).
  - Load reads element count as `hdr.data_size/sizeof(float)` (`122`), creates tensor with zeroed config (`117-119`) instead of header dtype/device.
- Project impact:
  - Incorrect persistence for non-float tensors and potential data interpretation bugs.

### 17) `model_io` format handling is weakly validated (Medium)
- Evidence:
  - Version read but not enforced in load paths (`src/nn/model_io.c:79-83`, `212-214`).
  - Most `fread/fwrite` calls are unchecked for short read/write.
- Project impact:
  - Corrupt/truncated checkpoints can be accepted partially and fail later in subtle ways.

### 18) Paged streaming attention path is CPU-only and restricted to batch=1 (Medium)
- Evidence:
  - Hard failure for `batch != 1` (`src/nn/paged_attention.c:377-380`).
  - Output tensor always CPU float32 (`485-487`).
- Project impact:
  - Streaming inference feature exists but is constrained for production batched serving / device placement.

---

## Systemic Problems

1. API surface is ahead of implementation maturity (many knobs/features are declared but stubbed).
2. Async/streaming naming is used for sync implementations, causing semantic mismatch.
3. Distributed correctness contracts (rank/ID exchange, collective completeness, work completion) are not centralized or rigorously enforced.
4. Serialization code paths are duplicated (`model_io` and `core/serialization`) with different guarantees and validation quality.
5. Ownership/lifetime boundaries in data-loading paths are not consistently respected.

---

## Project-Wide Impact

- Reliability risk: training/inference can hang or silently proceed with incorrect assumptions.
- Correctness risk: gradients/state/data batches may be wrong without obvious hard failures.
- Performance risk: “async/streaming” pathways do not provide expected overlap, reducing scalability.
- Maintainability risk: duplicated serialization stacks and partially wired distributed configs raise regression probability.

---

## Recommended Remediation Plan (Priority Order)

1. Fix DataLoader ownership bug (`dataloader_get_batch_tensors`) and add ASan tests for batch lifecycle.
2. Fix distributed async contract:
   - propagate real error codes,
   - make `cml_dist_wait` return `work->error_code` when completed.
3. Correct NCCL initialization with shared unique-id bootstrap.
4. Rework ring allreduce transport order (or use nonblocking I/O) to remove send/recv deadlock risk.
5. Make Gloo collectives either fully implemented for multi-rank or explicitly return “unsupported”.
6. Repair serialization invariants:
   - write exact record counts,
   - encode absent optimizer state as `NONE`,
   - enforce robust read/write checks.
7. Resolve static-path reuse in dataset download helpers.
8. Implement worker prefetch with shared atomic batch cursor and queue reset semantics.
9. Either implement real async/io_uring disk path or downgrade API contract/docs to sync.
10. Align pipeline/data-parallel config surface with actual behavior (or mark experimental/stubbed explicitly).

---

## Verification Matrix

- Distributed:
  - multi-rank integration tests for NCCL/MPI/Gloo allreduce/broadcast/allgather/barrier.
  - hang tests with large tensors for ring transport.
- DataLoader:
  - sanitizer tests for `dataloader_get_batch_tensors`/`dataloader_for_each`.
  - deterministic coverage test for worker prefetch (no duplicates/misses).
- I/O/serialization:
  - corrupt-file fuzz tests for model/tensor/optimizer load.
  - roundtrip tests across dtypes/devices.
- Streaming:
  - benchmark proving async disk read overlap (or explicit sync behavior assertions).
  - paged attention tests for batch>1 behavior expectations (supported/unsupported explicitly).

