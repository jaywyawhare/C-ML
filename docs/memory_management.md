# Memory Management

C-ML provides a layered memory management system designed for high-performance tensor computation. The system includes a TLSF allocator for O(1) allocation, a graph allocator for computation graph memory planning, memory pools for fast fixed-size allocation, a timeline planner for optimal memory scheduling, and safe allocation wrappers with debugging support.

## Architecture Overview

```
Application / Graph Execution
     |
     v
Graph Allocator              Context Allocator
  liveness analysis,           pre-allocated or
  per-graph buffers            dynamic allocation
     |                              |
     v                              v
Memory Pools                 TLSF Allocator
  fixed-size blocks,           general-purpose,
  tensor pools                 O(1) alloc/free
     |                              |
     +----------+-------------------+
                |
                v
     Timeline Planner (offline memory scheduling)
                |
                v
     Safe wrappers / device-aware allocation
                |
                v
     malloc / device_alloc (CPU, GPU, etc.)
```

**Header files:**
- `include/alloc/tlsf_alloc.h` -- TLSF allocator and timeline planner
- `include/alloc/graph_allocator.h` -- Graph-based allocator and context
- `include/alloc/memory_pools.h` -- Memory and tensor pools
- `include/alloc/memory_management.h` -- Safe allocation helpers

**Source files:**
- `src/alloc/tlsf_alloc.c`
- `src/alloc/graph_allocator.c`
- `src/alloc/memory_pools.c`
- `src/alloc/memory_management.c`

---

## TLSF Allocator (Two-Level Segregated Fit)

The TLSF allocator provides **O(1) time** allocation and deallocation with bounded fragmentation. It is suitable as a general-purpose allocator for tensor data and intermediate buffers.

### How It Works

TLSF uses a two-level bitmap index to locate free blocks:

- **First level (FL):** `floor(log2(size))` -- groups blocks by power-of-two size ranges.
- **Second level (SL):** subdivides each FL range into 16 equal parts for finer granularity.
- Bitmaps at each level allow O(1) lookup of the best-fit free block using hardware bit-scan instructions (`__builtin_clz`, `__builtin_ctz`).

Memory layout within the pool:

```
[Block Header | User Data] [Block Header | User Data] ... [Sentinel Header]
```

Each block header stores a `prev_phys` pointer and a `size` field. The two least-significant bits of `size` encode flags: `FLAG_FREE` and `FLAG_PREVFREE`. Free blocks store `next_free` and `prev_free` pointers in the user data area, forming doubly-linked segregated free lists.

### Configuration Constants

| Constant | Value | Meaning |
|---|---|---|
| `TLSF_FL_INDEX_COUNT` | 32 | Number of first-level buckets |
| `TLSF_SL_INDEX_COUNT` | 16 | Subdivisions per first-level bucket |
| `TLSF_MIN_BLOCK_SIZE` | 16 | Minimum allocation size (bytes) |
| `TLSF_ALIGN` | 16 | Allocation alignment (bytes) |

### API

```c
CMLTLSFAllocator* cml_tlsf_create(size_t pool_size);
CMLTLSFAllocator* cml_tlsf_create_with_pool(void* pool, size_t pool_size);

// Destroy
void cml_tlsf_destroy(CMLTLSFAllocator* alloc);

void* cml_tlsf_alloc(CMLTLSFAllocator* alloc, size_t size);
void* cml_tlsf_alloc_aligned(CMLTLSFAllocator* alloc, size_t size, size_t alignment);
void cml_tlsf_free(CMLTLSFAllocator* alloc, void* ptr);  // O(1), coalesces adjacent free blocks

// Realloc (tries in-place expansion before falling back to alloc+copy+free)
void* cml_tlsf_realloc(CMLTLSFAllocator* alloc, void* ptr, size_t new_size);

// Query allocated size for a pointer
size_t cml_tlsf_alloc_size(CMLTLSFAllocator* alloc, void* ptr);

void cml_tlsf_stats(const CMLTLSFAllocator* alloc,
                     size_t* used, size_t* peak,
                     size_t* num_allocs, size_t* num_frees);

// Debug: verify allocator integrity (bitmap consistency, block chain)
bool cml_tlsf_check(const CMLTLSFAllocator* alloc);
```

### Usage Example

```c
CMLTLSFAllocator* alloc = cml_tlsf_create(64 * 1024 * 1024);
float* data = (float*)cml_tlsf_alloc(alloc, 1024 * sizeof(float));
float* aligned = (float*)cml_tlsf_alloc_aligned(alloc, 4096, 64);  // 64-byte alignment for SIMD

// Free
cml_tlsf_free(alloc, data);
cml_tlsf_free(alloc, aligned);

size_t used, peak;
cml_tlsf_stats(alloc, &used, &peak, NULL, NULL);

cml_tlsf_destroy(alloc);
```

---

## Timeline Planner

The timeline planner solves an offline memory scheduling problem: given a set of tensors with known lifetimes (allocation step and free step), assign memory offsets so that non-overlapping lifetimes can share the same physical memory. This minimizes peak memory usage for computation graph execution.

### Algorithm

1. Records are sorted by allocation time (ascending), then by size (descending, so large tensors are placed first).
2. A greedy first-fit-by-offset algorithm places each tensor at the lowest available offset that does not conflict with already-placed tensors whose lifetimes overlap.
3. After solving, the planner reports the total contiguous memory required and the peak concurrent usage.

### API

```c
CMLTimelinePlanner* cml_timeline_planner_create(int initial_capacity);

// Destroy planner
void cml_timeline_planner_destroy(CMLTimelinePlanner* planner);

// Register a tensor lifetime
int cml_timeline_planner_add(CMLTimelinePlanner* planner,
                              int tensor_id, size_t size,
                              int alloc_time, int free_time);

// Solve: assign offsets to all registered tensors
int cml_timeline_planner_solve(CMLTimelinePlanner* planner);

// Query the assigned record for a specific tensor
const CMLTimelineRecord* cml_timeline_planner_get(
    const CMLTimelinePlanner* planner, int tensor_id);

size_t cml_timeline_planner_total_memory(const CMLTimelinePlanner* planner);
size_t cml_timeline_planner_peak_usage(const CMLTimelinePlanner* planner);
void cml_timeline_planner_print(const CMLTimelinePlanner* planner);
```

### Timeline Record

Each record contains:

| Field | Type | Description |
|---|---|---|
| `tensor_id` | `int` | Unique tensor identifier |
| `size` | `size_t` | Tensor size in bytes (aligned to 16) |
| `offset` | `size_t` | Assigned offset within the memory pool |
| `alloc_time` | `int` | Computation step when tensor is first needed |
| `free_time` | `int` | Computation step when tensor is last used |

### Usage Example

```c
CMLTimelinePlanner* planner = cml_timeline_planner_create(16);

// Tensor 0: 4KB, alive during steps 0-3
cml_timeline_planner_add(planner, 0, 4096, 0, 3);
// Tensor 1: 8KB, alive during steps 1-5
cml_timeline_planner_add(planner, 1, 8192, 1, 5);
// Tensor 2: 4KB, alive during steps 4-7 (can reuse tensor 0's memory)
cml_timeline_planner_add(planner, 2, 4096, 4, 7);

cml_timeline_planner_solve(planner);
cml_timeline_planner_print(planner);

size_t total = cml_timeline_planner_total_memory(planner);
// total will be 12KB (tensors 0 and 2 share the same offset)

cml_timeline_planner_destroy(planner);
```

---

## Graph Allocator

The graph allocator manages memory for computation graph execution. It performs liveness analysis on the graph to determine peak memory requirements, allocates backend buffers of the right size, and supports memory pooling for repeated executions.

### How It Works

1. **Liveness analysis:** When `cml_graph_allocator_reserve()` is called with a graph, the allocator performs a full liveness analysis using topological sort (Kahn's algorithm). It tracks which tensors are alive at each execution step and computes peak memory. A simpler adaptive heuristic (30-50% liveness factor based on graph complexity) is used as a fallback.

2. **Buffer allocation:** `cml_graph_allocator_alloc_graph()` allocates backend buffers sized to the peak memory requirement. It uses the backend buffer type system to allocate on the appropriate device (CPU, GPU, etc.).

3. **Pooling:** When enabled, the allocator can attach memory pools to individual buffers for block-level reuse.

### API

```c
CMLGraphAllocator_t cml_graph_allocator_new(CMLBackendBufferType_t buft);
CMLGraphAllocator_t cml_graph_allocator_new_n(
    CMLBackendBufferType_t* bufts, int n_bufs);

// Free
void cml_graph_allocator_free(CMLGraphAllocator_t galloc);

// Reserve: analyze graph and pre-compute buffer sizes
bool cml_graph_allocator_reserve(CMLGraphAllocator_t galloc, void* graph);

// Reserve with explicit buffer ID assignments per node
bool cml_graph_allocator_reserve_n(CMLGraphAllocator_t galloc, void* graph,
                                    const int* node_buffer_ids,
                                    const int* leaf_buffer_ids);

bool cml_graph_allocator_alloc_graph(CMLGraphAllocator_t galloc, void* graph);

// Query buffer size
size_t cml_graph_allocator_get_buffer_size(CMLGraphAllocator_t galloc, int buffer_id);

// Enable/disable pooling
void cml_graph_allocator_enable_pooling(CMLGraphAllocator_t galloc, bool enable);

// Dynamically grow a buffer
bool cml_graph_allocator_realloc_buffer(CMLGraphAllocator_t galloc,
                                         int buffer_id, size_t new_size);

// Attach a memory pool to a buffer
bool cml_graph_allocator_init_pool(CMLGraphAllocator_t galloc, int buffer_id,
                                    size_t block_size, int num_blocks, DType dtype);
```

### Tensor Allocator

A lightweight bump allocator that hands out sequential offsets within a backend buffer:

```c
typedef struct {
    CMLBackendBuffer_t buffer;
    void* base;
    size_t alignment;
    size_t offset;  // current watermark
} CMLTensorAllocator;

void cml_tensor_allocator_new(CMLTensorAllocator* talloc, CMLBackendBuffer_t buffer);
int  cml_tensor_allocator_alloc(CMLTensorAllocator* talloc, Tensor* tensor);
```

### Computation Context

A higher-level interface (similar to ggml_context) that wraps buffer allocation and tensor creation:

```c
typedef struct {
    size_t mem_size;   // 0 = dynamic allocation
    void* mem_buffer;  // NULL = allocate internally
    bool no_alloc;     // true = measure only, don't allocate
} CMLContextParams;

CMLContext_t cml_context_new(CMLContextParams params);
void         cml_context_free(CMLContext_t ctx);
size_t       cml_context_used_mem(CMLContext_t ctx);
size_t       cml_context_total_mem(CMLContext_t ctx);

Tensor* cml_context_alloc_tensor(CMLContext_t ctx, int* shape, int ndim,
                                  DType dtype, DeviceType device);

// Mark tensor as trainable parameter
void cml_context_set_param(CMLContext_t ctx, Tensor* tensor);
```

The context supports three modes:
- **Pre-allocated buffer:** Pass `mem_buffer` and `mem_size` to use existing memory.
- **Fixed-size allocation:** Set `mem_size > 0` with `mem_buffer = NULL` to allocate a fixed pool.
- **Dynamic allocation:** Set `mem_size = 0` to allocate each tensor individually via the backend buffer system.
- **Measurement mode:** Set `no_alloc = true` to only track memory usage without allocating.

---

## Memory Pools

Memory pools provide fast, fragmentation-free allocation for fixed-size blocks. Two pool types are available.

### MemoryPool (Raw Blocks)

Pre-allocates a set of identically-sized memory blocks. Allocation scans for the first unused block (O(n) in pool size, but pools are typically small).

```c
MemoryPool* memory_pool_create(size_t block_size, int num_blocks, DType dtype);
void memory_pool_free(MemoryPool* pool);
void* memory_pool_alloc(MemoryPool* pool);
int memory_pool_free_block(MemoryPool* pool, void* block);
```

### TensorPool (Pre-allocated Tensors)

Pre-allocates tensors of a fixed shape, dtype, and device. Useful for workloads that repeatedly create and destroy tensors of the same size.

```c
TensorPool* tensor_pool_create(int* shape, int ndim, size_t num_tensors,
                                DType dtype, DeviceType device);
void tensor_pool_free(TensorPool* pool);
```

### Usage Example

```c
// Pool of 32 blocks, each 4096 bytes
MemoryPool* pool = memory_pool_create(4096, 32, DTYPE_FLOAT32);

void* block = memory_pool_alloc(pool);
// ... use block ...
memory_pool_free_block(pool, block);

memory_pool_free(pool);
```

---

## Safe Allocation Wrappers

The `memory_management.h` module provides safe wrappers around standard `malloc`, `calloc`, `realloc`, and `free` with error logging and debugging support. These are intended for **CPU-side struct allocations** (not tensor data).

```c
// Safe malloc with file/line tracking
void* cml_safe_malloc(size_t size, const char* file, int line);

// Safe calloc (zeroed memory)
void* cml_safe_calloc(size_t nmemb, size_t size, const char* file, int line);

// Safe realloc
void* cml_safe_realloc(void* ptr, size_t size, const char* file, int line);

// Safe free (sets pointer to NULL to prevent double-free)
void cml_safe_free(void** ptr);
```

### Device-Aware Allocation

For tensor data that may reside on GPU or other accelerator memory:

```c
void* cml_device_alloc(size_t size);
void cml_device_free(void* ptr, DeviceType device);
```

These delegate to the device abstraction layer (`device_alloc` / `device_free`) which routes to the appropriate backend (CPU `malloc`, CUDA `cuMemAlloc`, etc.).

---

## Choosing the Right Allocator

| Scenario | Recommended Allocator |
|---|---|
| General-purpose tensor allocation with minimal fragmentation | TLSF allocator |
| Computation graph execution with known topology | Graph allocator with liveness analysis |
| Offline memory planning when all tensor lifetimes are known | Timeline planner |
| Repeated allocation/deallocation of same-size buffers | Memory pools |
| Pre-allocated tensors of fixed shape for hot loops | Tensor pools |
| CPU-side struct/metadata allocation | Safe allocation wrappers |
| Device-specific memory (GPU VRAM, etc.) | `cml_device_alloc` / backend buffer system |
