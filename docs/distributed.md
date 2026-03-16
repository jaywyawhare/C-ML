# Distributed Training

CML provides distributed training primitives for multi-GPU and multi-node
setups.  The stack includes process group management, collective operations,
Distributed Data Parallel (DDP), pipeline parallelism, and Megatron-LM style
tensor parallelism.  Both a C API and Python bindings are available.

## Table of Contents

- [Process Group Initialization](#process-group-initialization)
- [Communication Backends](#communication-backends)
- [Collective Operations](#collective-operations)
- [Distributed Data Parallel (DDP)](#distributed-data-parallel-ddp)
- [Pipeline Parallelism](#pipeline-parallelism)
- [Tensor Parallelism](#tensor-parallelism)
- [Python API](#python-api)
- [C API Reference](#c-api-reference)
- [Example: DDP Training Loop](#example-ddp-training-loop)

---

## Process Group Initialization

Before using any distributed functionality you must initialize a process group.
CML reads environment variables for auto-configuration when `world_size` or
`rank` are set to `-1`:

| Variable           | Fallback        | Description               |
|--------------------|-----------------|---------------------------|
| `CML_WORLD_SIZE`   | `WORLD_SIZE`    | Total number of processes |
| `CML_RANK`         | `RANK`, `LOCAL_RANK` | This process's rank  |

### C API

```c
#include "distributed/distributed.h"

// Initialize with NCCL backend, auto-detect rank/world_size from env
int ret = cml_dist_init(DIST_BACKEND_NCCL, /*world_size=*/-1, /*rank=*/-1);

// Query state
int rank       = cml_dist_get_rank();
int world_size = cml_dist_get_world_size();
bool ready     = cml_dist_is_initialized();

// Access the default process group directly
DistProcessGroup* group = cml_dist_get_default_group();

// Shutdown
cml_dist_destroy();
```

### Python API

```python
import cml.distributed as dist

dist.init_process_group(backend="nccl")   # or "mpi", "gloo"

rank       = dist.get_rank()
world_size = dist.get_world_size()
ready      = dist.is_initialized()

# When finished
dist.destroy_process_group()
```

---

## Communication Backends

CML supports three communication backends, loaded at runtime via `dlopen` so
the library can compile without any of these as hard dependencies.

| Backend | Enum constant        | Best for            | Notes                                         |
|---------|----------------------|---------------------|-----------------------------------------------|
| NCCL    | `DIST_BACKEND_NCCL`  | Multi-GPU (NVIDIA)  | Falls back to Gloo if `libnccl.so` is absent  |
| MPI     | `DIST_BACKEND_MPI`   | CPU / multi-node    | Falls back to Gloo if `libmpi.so` is absent   |
| Gloo    | `DIST_BACKEND_GLOO`  | CPU fallback        | Always available                               |

Each backend implements the `DistCommOps` vtable (defined in
`include/distributed/comm_backend.h`):

```c
DistCommOps* cml_dist_create_nccl_backend(void);
DistCommOps* cml_dist_create_mpi_backend(void);
DistCommOps* cml_dist_create_gloo_backend(void);
void         cml_dist_free_backend(DistCommOps* ops);
```

---

## Collective Operations

All collectives operate on the default process group.

### All-Reduce

Reduces a tensor **in-place** across all ranks.

```c
// Synchronous
int cml_dist_allreduce(Tensor* tensor, DistReduceOp op);

// Asynchronous (returns a handle; falls back to sync if backend lacks async)
DistWork* cml_dist_allreduce_async(Tensor* tensor, DistReduceOp op);
int       cml_dist_wait(DistWork* work);
void      cml_dist_work_free(DistWork* work);
```

Supported reduction operations:

| Operation            | Enum constant         |
|----------------------|-----------------------|
| Sum                  | `DIST_REDUCE_SUM`     |
| Product              | `DIST_REDUCE_PRODUCT` |
| Max                  | `DIST_REDUCE_MAX`     |
| Min                  | `DIST_REDUCE_MIN`     |
| Average              | `DIST_REDUCE_AVG`     |

### Broadcast

Copies a tensor from `src_rank` to all other ranks (in-place on receivers).

```c
int cml_dist_broadcast(Tensor* tensor, int src_rank);
```

### All-Gather

Gathers tensors from every rank into an array of tensors.

```c
int cml_dist_allgather(Tensor** output, Tensor* input);
```

`output` must be a pre-allocated array of `world_size` tensor pointers.

### Barrier

Blocks until all ranks reach the barrier.

```c
int cml_dist_barrier(void);
```

### Ring All-Reduce

A bandwidth-optimal ring algorithm is available for float data in
`include/distributed/ring_allreduce.h`:

```c
int cml_ring_allreduce(float* data, size_t count,
                       int world_size, int rank,
                       DistReduceOp op,
                       DistCommOps* ops, void* ctx);
```

The algorithm runs in two phases:
1. **Reduce-scatter ring** -- each rank ends up with one reduced chunk.
2. **All-gather ring** -- each rank collects all reduced chunks.

### Point-to-Point

The `DistCommOps` vtable also exposes raw send/recv:

```c
int (*send)(Tensor* tensor, int dst_rank, int tag, void* ctx);
int (*recv)(Tensor* tensor, int src_rank, int tag, void* ctx);
```

---

## Distributed Data Parallel (DDP)

DDP replicates the model on each rank and synchronizes gradients after the
backward pass.  The implementation follows PyTorch conventions:

1. Parameters are **broadcast from rank 0** during `cml_ddp_create`.
2. Gradients are packed into **25 MB buckets** (configurable).
3. Each bucket is **all-reduced (sum)** then **averaged** by `world_size`.

### DDPConfig

```c
typedef struct {
    size_t bucket_size_bytes;       // default: 25 MB
    bool   broadcast_buffers;       // broadcast non-parameter buffers
    bool   find_unused_parameters;  // skip unused params
    int    gradient_as_bucket_view; // memory-efficient gradient views
} DDPConfig;

DDPConfig cml_ddp_default_config(void);
```

### C API

```c
#include "distributed/data_parallel.h"

// Wrap a module
CMLDataParallel* ddp = cml_ddp_create(model, NULL);  // NULL = default config

// Training loop
Tensor* output = cml_ddp_forward(ddp, input);
// ... compute loss, call tensor_backward(loss) ...
cml_ddp_sync_gradients(ddp);   // all-reduce + average gradients
optimizer_step(optimizer);
optimizer_zero_grad(optimizer);

// Cleanup
cml_ddp_free(ddp);  // does NOT free the underlying module
```

### Python API

```python
import cml
import cml.distributed as dist

dist.init_process_group(backend="nccl")

model = cml.Sequential()
model.add(cml.Linear(784, 256))
model.add(cml.Linear(256, 10))

ddp_model = dist.DistributedDataParallel(model, bucket_size_mb=25)

for epoch in range(num_epochs):
    output = ddp_model(input_data)
    loss = cml.mse_loss(output, target)
    cml.backward(loss)
    ddp_model.sync_gradients()  # all-reduce + average
    optimizer.step()

dist.destroy_process_group()
```

---

## Pipeline Parallelism

GPipe-style pipeline parallelism splits a model into stages, each assigned to
a different device.  Input batches are split into micro-batches that flow
through the pipeline, reducing the bubble overhead.

### Configuration

```c
typedef struct {
    int  num_micro_batches;  // default: 4
    int  num_stages;         // number of pipeline stages
    bool interleaved;        // use 1F1B interleaved schedule
} PipelineConfig;
```

### Pipeline Stages

Each stage wraps a `Module` and is pinned to a device:

```c
typedef struct PipelineStage {
    Module*    module;
    int        device_id;
    DeviceType device;
    int        stage_id;
} PipelineStage;
```

### C API

```c
#include "distributed/pipeline_parallel.h"

PipelineStage stages[2] = {
    { .module = encoder, .device_id = 0, .device = DEVICE_GPU, .stage_id = 0 },
    { .module = decoder, .device_id = 1, .device = DEVICE_GPU, .stage_id = 1 },
};

PipelineConfig cfg = { .num_micro_batches = 8, .num_stages = 2, .interleaved = true };
CMLPipelineParallel* pipe = cml_pipeline_create(stages, 2, &cfg);

Tensor* output = cml_pipeline_forward(pipe, input);   // full batch in, full batch out
int ret = cml_pipeline_backward(pipe, grad_output);

cml_pipeline_free(pipe);
```

### Python API

```python
import cml.distributed as dist

pipe = dist.PipelineParallel(
    modules=[encoder, decoder],
    num_micro_batches=8,
)
output = pipe(input_tensor)
```

---

## Tensor Parallelism

CML implements Megatron-LM style tensor parallelism with two layer types:

### Column-Parallel Linear

Shards the weight along the **output** dimension.  Each rank holds
`[out_features / tp_size, in_features]` and produces a local output of shape
`[batch, out_features / tp_size]`.  Used for QKV, gate, and up projections.

```c
CMLColumnParallelLinear* cml_column_parallel_create(
    Tensor* full_weight,   // [out_features, in_features]
    Tensor* full_bias,     // [out_features] or NULL
    int tp_size,
    int tp_rank
);

Tensor* cml_column_parallel_forward(CMLColumnParallelLinear* cp, Tensor* input);
void    cml_column_parallel_free(CMLColumnParallelLinear* cp);
```

### Row-Parallel Linear

Shards the weight along the **input** dimension.  Each rank holds
`[out_features, in_features / tp_size]` and produces a partial output of shape
`[batch, out_features]`.  An **all-reduce sum** across ranks is required after
the forward pass.  Used for output and down projections.

```c
CMLRowParallelLinear* cml_row_parallel_create(
    Tensor* full_weight,   // [out_features, in_features]
    Tensor* full_bias,     // [out_features] or NULL (only rank 0 keeps bias)
    int tp_size,
    int tp_rank
);

Tensor* cml_row_parallel_forward(CMLRowParallelLinear* rp, Tensor* input);
void    cml_row_parallel_free(CMLRowParallelLinear* rp);
```

### Utilities

```c
// Shard a 2-D weight tensor along dim 0 (rows) or dim 1 (columns)
Tensor* cml_tp_shard_weight(Tensor* weight, int dim, int tp_size, int tp_rank);

// Simulate an all-reduce sum over an array of partial-result tensors
Tensor* cml_tp_all_reduce_sum(Tensor** partials, int num_parts);
```

### Typical Transformer Column/Row Pairing

In a transformer MLP block the pattern is:

```
input -----> ColumnParallel (gate_proj) -----> activation -----> RowParallel (down_proj) -----> all-reduce
```

Each rank independently computes its column-parallel shard, applies the
activation, then feeds the result through the row-parallel layer.  The
all-reduce after the row-parallel layer produces the final output.

---

## Python API

The Python distributed module lives at `python/cml/distributed.py` and exposes:

| Symbol                       | Description                          |
|------------------------------|--------------------------------------|
| `init_process_group(backend, world_size, rank)` | Initialize distributed      |
| `get_rank()`                 | Current rank                         |
| `get_world_size()`           | Total processes                      |
| `is_initialized()`           | Check init status                    |
| `destroy_process_group()`    | Shutdown                             |
| `barrier()`                  | Synchronize all processes            |
| `DistributedDataParallel`    | DDP wrapper class                    |
| `PipelineParallel`           | Pipeline parallel wrapper class      |

Backend name strings: `"nccl"`, `"mpi"`, `"gloo"`.

---

## C API Reference

### Headers

| Header                                   | Contents                                    |
|------------------------------------------|---------------------------------------------|
| `include/distributed/distributed.h`      | Process group, collectives, async ops       |
| `include/distributed/comm_backend.h`     | Backend creation (NCCL, MPI, Gloo)          |
| `include/distributed/data_parallel.h`    | DDP wrapper and gradient sync               |
| `include/distributed/pipeline_parallel.h`| Pipeline stages and micro-batching          |
| `include/distributed/tensor_parallel.h`  | Column/row parallel layers, weight sharding |
| `include/distributed/ring_allreduce.h`   | Ring all-reduce algorithm                   |

### Sources

| Source                                   | Contents                                    |
|------------------------------------------|---------------------------------------------|
| `src/distributed/distributed.c`          | Process group init, collective dispatch      |
| `src/distributed/data_parallel.c`        | DDP bucketed gradient sync                   |
| `src/distributed/pipeline_parallel.c`    | Pipeline forward/backward                    |
| `src/distributed/tensor_parallel.c`      | Column/row parallel, weight sharding, matmul |
| `src/distributed/ring_allreduce.c`       | Ring all-reduce implementation               |
| `src/distributed/nccl_backend.c`         | NCCL backend (dlopen)                        |
| `src/distributed/mpi_backend.c`          | MPI backend (dlopen)                         |
| `src/distributed/gloo_backend.c`         | Gloo backend                                 |

---

## Example: DDP Training Loop

### C

```c
#include "distributed/distributed.h"
#include "distributed/data_parallel.h"
#include "nn.h"
#include "optim.h"

int main(int argc, char** argv) {
    // 1. Initialize distributed (reads CML_RANK / CML_WORLD_SIZE from env)
    if (cml_dist_init(DIST_BACKEND_NCCL, -1, -1) != 0) {
        fprintf(stderr, "Failed to init distributed\n");
        return 1;
    }

    int rank       = cml_dist_get_rank();
    int world_size = cml_dist_get_world_size();
    printf("Rank %d/%d\n", rank, world_size);

    // 2. Build model
    Module* model = module_create("mlp");
    module_add_layer(model, linear_create(784, 256));
    module_add_layer(model, linear_create(256, 10));

    // 3. Wrap in DDP (broadcasts params from rank 0, sets up buckets)
    DDPConfig cfg = cml_ddp_default_config();
    cfg.bucket_size_bytes = 25 * 1024 * 1024;  // 25 MB
    CMLDataParallel* ddp = cml_ddp_create(model, &cfg);

    // 4. Create optimizer
    Optimizer* opt = optimizer_adam_create(model, 1e-3f);

    // 5. Training loop
    for (int epoch = 0; epoch < 10; epoch++) {
        // Each rank processes its own data shard
        Tensor* input  = load_data_shard(rank, world_size, epoch);
        Tensor* target = load_label_shard(rank, world_size, epoch);

        Tensor* output = cml_ddp_forward(ddp, input);
        Tensor* loss   = mse_loss(output, target);

        tensor_backward(loss);

        // Synchronize gradients (all-reduce + average)
        cml_ddp_sync_gradients(ddp);

        optimizer_step(opt);
        optimizer_zero_grad(opt);

        // Optional: barrier to keep ranks in sync for logging
        cml_dist_barrier();

        if (rank == 0) {
            printf("Epoch %d complete\n", epoch);
        }
    }

    // 6. Cleanup
    cml_ddp_free(ddp);
    optimizer_free(opt);
    module_free(model);
    cml_dist_destroy();

    return 0;
}
```

### Python

```python
import cml
import cml.distributed as dist

# 1. Initialize
dist.init_process_group(backend="nccl")
rank       = dist.get_rank()
world_size = dist.get_world_size()
print(f"Rank {rank}/{world_size}")

# 2. Build model
model = cml.Sequential()
model.add(cml.Linear(784, 256))
model.add(cml.Linear(256, 10))

# 3. Wrap in DDP
ddp_model = dist.DistributedDataParallel(model, bucket_size_mb=25)

# 4. Optimizer
optimizer = cml.Adam(model.parameters(), lr=1e-3)

# 5. Training loop
for epoch in range(10):
    input_data, target = load_data_shard(rank, world_size, epoch)

    output = ddp_model(input_data)
    loss = cml.mse_loss(output, target)

    cml.backward(loss)
    ddp_model.sync_gradients()

    optimizer.step()
    optimizer.zero_grad()

    dist.barrier()

    if rank == 0:
        print(f"Epoch {epoch} complete")

# 6. Cleanup
dist.destroy_process_group()
```

### Launching

Use `torchrun`-style launching or set environment variables manually:

```bash
# 2-GPU single-node
CML_WORLD_SIZE=2 CML_RANK=0 ./train &
CML_WORLD_SIZE=2 CML_RANK=1 ./train &
wait
```

Or with MPI:

```bash
mpirun -np 4 ./train
```
