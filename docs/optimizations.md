# C-ML Optimization Guide

This document provides a comprehensive overview of all optimization techniques implemented in C-ML, how they work internally, and how to leverage them for maximum performance.

______________________________________________________________________

## Table of Contents

1. [Overview](#overview)
1. [SIMD Vectorization](#1-simd-vectorization)
1. [Memory Optimizations](#2-memory-optimizations)
1. [Parallelization](#3-parallelization)
1. [BLAS Integration](#4-blas-integration)
1. [IR Graph Optimizations](#5-ir-graph-optimizations)
1. [Operation Fusion](#6-operation-fusion)
1. [LLVM JIT Backend](#7-llvm-jit-backend)
1. [Caching](#8-caching)
1. [GPU Backends](#9-gpu-backends)
1. [Gradient Checkpointing](#10-gradient-checkpointing)
1. [Compiler Optimizations](#11-compiler-optimizations)
1. [Profiling](#12-profiling)
1. [Best Practices](#best-practices)

______________________________________________________________________

## Overview

C-ML employs a multi-level optimization strategy that operates at different stages of the computation pipeline:

```
User Code
    v
┌─────────────────────────────────────────────────────┐
│  Level 1: C-ML IR Optimizations                     │
│  - Dead Code Elimination (DCE)                      │
│  - Operation Fusion (11+ patterns)                  │
│  - Cache Locality Reordering                        │
└─────────────────────────────────────────────────────┘
    v
┌─────────────────────────────────────────────────────┐
│  Level 2: LLVM JIT Compilation (optional)           │
│  - IR to LLVM IR lowering                           │
│  - LLVM optimization passes                         │
│  - Native machine code generation                   │
│  - Kernel caching (LRU)                             │
└─────────────────────────────────────────────────────┘
    v
┌─────────────────────────────────────────────────────┐
│  Level 3: Runtime Optimizations                     │
│  - SIMD Vectorization (SSE/AVX/AVX-512/NEON)        │
│  - Multi-threading (Thread Pool)                    │
│  - BLAS Acceleration (OpenBLAS/MKL/Accelerate)      │
│  - Kernel Caching (LRU)                             │
│  - Memory Pooling                                   │
└─────────────────────────────────────────────────────┘
    v
Hardware (CPU/GPU)
```

______________________________________________________________________

## 1. SIMD Vectorization

**Files:** `include/ops/simd_math.h`, `src/ops/simd_math.c`

SIMD (Single Instruction, Multiple Data) allows processing multiple data elements with a single instruction, providing significant speedup for element-wise operations.

### 1.1 Runtime Detection

C-ML detects SIMD capabilities at runtime using CPUID (x86) or compile-time detection (ARM):

```c
typedef struct {
    bool has_sse;      // SSE/SSE2 (128-bit, 4 floats)
    bool has_sse4;     // SSE4.1/4.2
    bool has_avx;      // AVX (256-bit, 8 floats)
    bool has_avx2;     // AVX2 + FMA
    bool has_avx512;   // AVX-512F (512-bit, 16 floats)
    bool has_neon;     // ARM NEON (128-bit, 4 floats)
    bool has_sleef;    // SLEEF library available
} CMLSimdCaps;

// Get capabilities at runtime
const CMLSimdCaps* caps = cml_get_simd_caps();
```

### 1.2 Detection Implementation (x86)

```c
static CMLSimdCaps cml_detect_simd_caps(void) {
    CMLSimdCaps caps = {0};
    int info[4];
    cpuid(info, 0);
    int max_leaf = info[0];

    if (max_leaf >= 1) {
        cpuid(info, 1);
        caps.has_sse = (info[3] & (1 << 25)) != 0;   // SSE
        caps.has_sse4 = (info[2] & (1 << 19)) != 0;  // SSE4.1

        // Check OS support for AVX via XSAVE
        bool os_uses_xsave = (info[2] & (1 << 27)) != 0;
        bool cpu_has_avx = (info[2] & (1 << 28)) != 0;

        if (os_uses_xsave && cpu_has_avx) {
            unsigned long long xcr0 = xgetbv(0);
            bool os_saves_ymm = (xcr0 & 6) == 6;

            if (os_saves_ymm) {
                caps.has_avx = true;
                if (max_leaf >= 7) {
                    cpuid_count(info, 7, 0);
                    caps.has_avx2 = (info[1] & (1 << 5)) != 0;
                    caps.has_avx512 = (info[1] & (1 << 16)) != 0;
                }
            }
        }
    }
    return caps;
}
```

### 1.3 Vectorized Operations

| Category    | Operations                                                                                   |
| ----------- | -------------------------------------------------------------------------------------------- |
| **Unary**   | `exp`, `log`, `sqrt`, `rsqrt`, `recip`, `abs`, `sin`, `cos`, `tan`, `tanh`, `sigmoid`, `neg` |
| **Binary**  | `add`, `sub`, `mul`, `div`, `pow`, `min`, `max`, `cmplt`, `cmpgt`                            |
| **Ternary** | `where` (conditional select)                                                                 |
| **Matrix**  | `transpose` (cache-blocked)                                                                  |

### 1.4 AVX-512 Example: Exponential

```c
static inline __m512 exp_poly_avx512(__m512 x) {
    // Clamp to prevent overflow
    x = _mm512_max_ps(_mm512_min_ps(x, _mm512_set1_ps(88.0f)),
                      _mm512_set1_ps(-88.0f));

    // Range reduction: x = k*ln(2) + r
    __m512 k = _mm512_roundscale_ps(
        _mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f)),
        _MM_FROUND_TO_NEAREST_INT
    );
    __m512 r = _mm512_fnmadd_ps(k, _mm512_set1_ps(0.6931471805599453f), x);

    // 6th-order polynomial approximation for exp(r)
    __m512 result = _mm512_fmadd_ps(_mm512_set1_ps(0.001388889f), r,
                                     _mm512_set1_ps(0.008333333f));
    result = _mm512_fmadd_ps(result, r, _mm512_set1_ps(0.041666667f));
    result = _mm512_fmadd_ps(result, r, _mm512_set1_ps(0.166666667f));
    result = _mm512_fmadd_ps(result, r, _mm512_set1_ps(0.5f));
    result = _mm512_fmadd_ps(result, r, _mm512_set1_ps(1.0f));
    result = _mm512_fmadd_ps(result, r, _mm512_set1_ps(1.0f));

    // Scale by 2^k using integer manipulation
    __m512i ki = _mm512_cvtps_epi32(k);
    ki = _mm512_add_epi32(ki, _mm512_set1_epi32(127));
    ki = _mm512_slli_epi32(ki, 23);
    __m512 scale = _mm512_castsi512_ps(ki);

    return _mm512_mul_ps(result, scale);
}
```

### 1.5 SLEEF Integration

SLEEF (SIMD Library for Evaluating Elementary Functions) provides high-accuracy vectorized math. C-ML dynamically loads SLEEF at runtime:

```c
static int try_load_sleef(void) {
    const char* sleef_paths[] = {
        "libsleef.so", "libsleef.so.3",
        "/usr/lib/libsleef.so", "/usr/local/lib/libsleef.so",
        NULL
    };

    for (int i = 0; sleef_paths[i]; i++) {
        g_sleef_handle = dlopen(sleef_paths[i], RTLD_LAZY);
        if (g_sleef_handle) {
            sleef_expf8 = dlsym(g_sleef_handle, "Sleef_expf8_u10");
            sleef_logf8 = dlsym(g_sleef_handle, "Sleef_logf8_u10");
            // ... more functions
            return 1;
        }
    }
    return 0;
}
```

### 1.6 Parallel SIMD Operations

For large arrays (>10,000 elements), operations run in parallel:

```c
void simd_set_parallel_threshold(size_t threshold);

void simd_add_f32_parallel(const float* a, const float* b, float* out, size_t n);
void simd_mul_f32_parallel(const float* a, const float* b, float* out, size_t n);
void simd_exp_f32_parallel(const float* in, float* out, size_t n);
float simd_sum_f32_parallel(const float* data, size_t n);  // Parallel reduction
```

______________________________________________________________________

## 2. Memory Optimizations

### 2.1 Memory Pooling

**Files:** `include/alloc/memory_pools.h`, `src/alloc/memory_pools.c`

Pre-allocated memory pools reduce allocation overhead and fragmentation:

```c
// Memory Pool: Raw memory blocks
typedef struct MemoryPool {
    void** blocks;       // Array of memory blocks
    size_t* block_sizes; // Size of each block
    size_t* used;        // Usage flags
    int num_blocks;
    size_t block_size;
    DType dtype;
} MemoryPool;

// Create pool with 100 blocks of 4KB each
MemoryPool* pool = memory_pool_create(4096, 100, DTYPE_F32);

// Allocate from pool (O(n) first-fit)
void* block = memory_pool_alloc(pool);

// Return to pool
memory_pool_free_block(pool, block);
```

### 2.2 Tensor Pooling

Pre-allocated tensors for specific shapes:

```c
typedef struct TensorPool {
    Tensor** tensors;   // Pre-allocated tensors
    int* shape;         // Fixed shape
    int ndim;
    bool* in_use;       // Usage flags
} TensorPool;

// Create pool of 50 tensors with shape [64, 784]
int shape[] = {64, 784};
TensorPool* tpool = tensor_pool_create(shape, 2, 50, DTYPE_F32, DEVICE_CPU);

// Get pre-allocated tensor
Tensor* t = tensor_pool_get(tpool);

// Return to pool
tensor_pool_return(tpool, t);
```

### 2.3 Tensor Reuse

Reuse existing tensor memory when shapes are compatible:

```c
bool tensor_can_reuse(Tensor* tensor, int* new_shape, int new_ndim) {
    if (tensor->ndim != new_ndim) return false;

    size_t new_numel = 1;
    for (int i = 0; i < new_ndim; i++) {
        new_numel *= new_shape[i];
    }

    // Can reuse if same number of elements and contiguous
    return tensor->numel == new_numel && tensor->is_contiguous;
}

Tensor* tensor_reuse(Tensor* tensor, int* new_shape, int new_ndim);
```

### 2.4 Graph Allocator (ggml-style)

**Files:** `include/alloc/graph_allocator.h`, `src/alloc/graph_allocator.c`

Pre-allocates memory based on computation graph structure using liveness analysis:

```c
// Create graph allocator
CMLBackendBufferType_t buft = cml_backend_buffer_type_for_device(DEVICE_CPU);
CMLGraphAllocator_t galloc = cml_graph_allocator_new(buft);

// Reserve memory based on graph (performs liveness analysis)
cml_graph_allocator_reserve(galloc, computation_graph);

// Allocate buffers
cml_graph_allocator_alloc_graph(galloc, computation_graph);
```

#### Liveness Analysis Algorithm

The graph allocator performs full liveness analysis to calculate peak memory:

```c
static size_t calculate_peak_memory(CMLComputationGraph_t graph) {
    // Step 1: Build dependency graph and calculate tensor sizes
    for (size_t i = 0; i < node_count; i++) {
        tensor_sizes[i] = calculate_tensor_size(node->tensor);
        is_leaf[i] = cml_graph_node_is_leaf(node);
        // Count consumers for each node
    }

    // Step 2: Topological sort using Kahn's algorithm
    // Nodes with in_degree=0 go in queue first
    while (queue_front < queue_back) {
        current = queue[queue_front++];
        execution_order[current] = order++;
        // Decrease in-degree of dependent nodes
    }

    // Step 3: Calculate when each tensor can be freed
    // Tensor alive until all consumers finish
    for (size_t i = 0; i < node_count; i++) {
        if (is_leaf[i]) {
            alive_until[i] = order;  // Leaves never freed
        } else {
            alive_until[i] = last_consumer_order + 1;
        }
    }

    // Step 4: Find peak = max(sum of alive tensors) at each step
    for (int step = 0; step < order; step++) {
        size_t current_memory = 0;
        for (size_t i = 0; i < node_count; i++) {
            if (exec_order[i] <= step && step < alive_until[i]) {
                current_memory += tensor_sizes[i];
            }
        }
        peak_memory = max(peak_memory, current_memory);
    }
    return peak_memory;
}
```

______________________________________________________________________

## 3. Parallelization

**Files:** `include/backend/threadpool.h`, `src/backend/threadpool.c`

### 3.1 Thread Pool

C-ML uses a global thread pool for parallel operations:

```c
typedef struct ThreadPool {
    Worker* workers;
    size_t num_threads;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    TaskNode* task_queue_head;
    bool shutdown;
} ThreadPool;

// Create thread pool (0 = auto-detect CPU count)
ThreadPool* pool = threadpool_create(0);

// Get global singleton
ThreadPool* global = threadpool_get_global();
```

### 3.2 Parallel For

```c
typedef void (*TaskFunc)(void* data, size_t start, size_t end);

void threadpool_parallel_for(ThreadPool* pool, TaskFunc func, void* data, size_t n) {
    if (pool->num_threads == 1 || n < 1000) {
        // Sequential for small tasks
        func(data, 0, n);
        return;
    }

    // Distribute work: each thread gets n/num_threads elements
    for (size_t i = 0; i < pool->num_threads; i++) {
        size_t chunk_size = n / pool->num_threads;
        size_t start = i * chunk_size;
        size_t end = (i == pool->num_threads - 1) ? n : start + chunk_size;
        // Submit chunk to worker
    }

    // Wait for all workers
    threadpool_wait(pool);
}
```

### 3.3 Worker Thread Implementation

```c
static void* worker_thread(void* arg) {
    Worker* worker = (Worker*)arg;
    ThreadPool* pool = worker->pool;

    while (1) {
        pthread_mutex_lock(&pool->mutex);

        // Wait for tasks
        while (!pool->shutdown && pool->task_queue_head == NULL) {
            pthread_cond_wait(&pool->task_cond, &pool->mutex);
        }

        if (pool->shutdown) break;

        TaskNode* task_node = pool->task_queue_head;
        pthread_mutex_unlock(&pool->mutex);

        // Execute assigned chunk
        size_t chunk_size = task.total_size / pool->num_threads;
        size_t start = worker->id * chunk_size;
        size_t end = (worker->id == pool->num_threads - 1)
                    ? task.total_size : start + chunk_size;

        task.func(task.data, start, end);

        // Signal completion
        pthread_cond_broadcast(&pool->cond);
    }
    return NULL;
}
```

______________________________________________________________________

## 4. BLAS Integration

**Files:** `include/backend/blas.h`, `src/backend/blas.c`

BLAS (Basic Linear Algebra Subprograms) provides highly optimized matrix operations.

### 4.1 Dynamic Library Loading

C-ML dynamically loads BLAS at runtime, trying multiple libraries in order of performance:

```c
static const char* blas_library_paths[] = {
#ifdef __linux__
    "libmkl_rt.so",       // Intel MKL (fastest)
    "libopenblas.so.0",   // OpenBLAS
    "libopenblas.so",
    "libatlas.so",        // ATLAS
    "libcblas.so.3",      // Reference CBLAS
#elif defined(__APPLE__)
    "/System/Library/Frameworks/Accelerate.framework/Accelerate",
    "libopenblas.dylib",
#endif
    NULL
};

// Environment variable override
const char* env_blas = getenv("CML_BLAS_LIB");
```

### 4.2 BLAS Operations

```c
// GEMM: C = alpha * A @ B + beta * C
int cml_blas_sgemm(CMLBlasContext* ctx,
                   const float* A, const float* B, float* C,
                   int M, int N, int K,
                   float alpha, float beta);

// GEMV: y = alpha * A @ x + beta * y
int cml_blas_sgemv(CMLBlasContext* ctx,
                   const float* A, const float* x, float* y,
                   int M, int N, float alpha, float beta);

// AXPY: y = alpha * x + y
int cml_blas_saxpy(CMLBlasContext* ctx,
                   const float* x, float* y, int n, float alpha);

// DOT: result = x · y
float cml_blas_sdot(CMLBlasContext* ctx, const float* x, const float* y, int n);

// NRM2: result = ||x||₂
float cml_blas_snrm2(CMLBlasContext* ctx, const float* x, int n);
```

### 4.3 SGEMM Implementation

```c
int cml_blas_sgemm(CMLBlasContext* ctx,
                   const float* A, const float* B, float* C,
                   int M, int N, int K,
                   float alpha, float beta) {
    if (ctx->cblas_sgemm) {
        // CBLAS interface (row-major)
        ctx->cblas_sgemm(
            CML_BLAS_ROW_MAJOR, CML_BLAS_NO_TRANS, CML_BLAS_NO_TRANS,
            M, N, K,
            alpha, A, K,    // A is M x K
            B, N,           // B is K x N
            beta, C, N      // C is M x N
        );
        return 0;
    } else if (ctx->sgemm_) {
        // Fortran BLAS (column-major) - compute C^T = B^T @ A^T
        char transA = 'N', transB = 'N';
        ctx->sgemm_(&transA, &transB, &N, &M, &K,
                    &alpha, B, &N, A, &K, &beta, C, &N);
        return 0;
    }
    return -1;
}
```

### 4.4 Fallback: Blocked Matrix Multiplication

When BLAS is unavailable, C-ML uses cache-blocked multiplication:

```c
// Cache-blocked matmul (32x32 blocks)
const int BLOCK = 32;
for (int i0 = 0; i0 < M; i0 += BLOCK) {
    for (int j0 = 0; j0 < N; j0 += BLOCK) {
        for (int k0 = 0; k0 < K; k0 += BLOCK) {
            int i_end = min(i0 + BLOCK, M);
            int j_end = min(j0 + BLOCK, N);
            int k_end = min(k0 + BLOCK, K);

            for (int i = i0; i < i_end; i++) {
                for (int k = k0; k < k_end; k++) {
                    float a_ik = A[i * K + k];
                    for (int j = j0; j < j_end; j++) {
                        C[i * N + j] += a_ik * B[k * N + j];
                    }
                }
            }
        }
    }
}
```

______________________________________________________________________

## 5. IR Graph Optimizations

**Files:** `include/ops/ir/optimization.h`, `src/ops/ir/optimization.c`

### 5.1 Optimization Pipeline

```c
int cml_ir_optimize(CMLGraph_t ir) {
    // Pass 1: Build dependency graph
    build_dependency_graph(ir);

    // Pass 2: Mark reachable nodes (for DCE)
    mark_reachable_nodes(ir);

    // Pass 3: Remove dead nodes
    remove_dead_nodes(ir);

    // Pass 4: Rebuild dependencies after DCE
    build_dependency_graph(ir);

    // Pass 5: Fuse operations
    fuse_operations(ir);

    // Pass 6: Reorder for cache locality
    reorder_for_cache_locality(ir);

    return 0;
}
```

### 5.2 Dead Code Elimination (DCE)

Uses DFS from outputs to mark reachable nodes:

```c
static void mark_reachable_nodes(CMLGraph_t ir) {
    // Mark all as unused initially
    struct IRNode* node = ir->head;
    while (node) {
        node->is_used = false;
        node = node->next;
    }

    // DFS from tail (output) node
    struct IRNode* stack[256];
    int stack_top = 0;

    if (ir->tail) {
        stack[stack_top++] = ir->tail;
        ir->tail->is_used = true;
    }

    while (stack_top > 0) {
        struct IRNode* current = stack[--stack_top];

        // Mark all input producers as used
        for (int i = 0; i < current->num_inputs; i++) {
            struct IRNode* producer = find_node_by_output(ir, current->input_names[i]);
            if (producer && !producer->is_used) {
                producer->is_used = true;
                stack[stack_top++] = producer;
            }
        }
    }
}

static int remove_dead_nodes(CMLGraph_t ir) {
    struct IRNode* prev = NULL;
    struct IRNode* node = ir->head;
    int removed = 0;

    while (node) {
        struct IRNode* next = node->next;

        if (!node->is_used && node->use_count == 0) {
            // Remove from linked list
            if (prev) prev->next = next;
            else ir->head = next;

            // Clear tensor references to prevent dangling pointers
            if (node->output) {
                node->output->ir_node = NULL;
                node->output->ir_context = NULL;
            }

            // Free node and resources
            free(node);
            ir->node_count--;
            removed++;
        } else {
            prev = node;
        }
        node = next;
    }
    return 0;
}
```

### 5.3 Dependency Graph Construction

```c
static int build_dependency_graph(CMLGraph_t ir) {
    // Pass 1: Initialize use counts
    struct IRNode* node = ir->head;
    while (node) {
        node->use_count = 0;
        node = node->next;
    }

    // Pass 2: For each node, find its producers
    node = ir->head;
    while (node) {
        for (int i = 0; i < node->num_inputs; i++) {
            struct IRNode* producer = find_node_by_output(ir, node->input_names[i]);
            if (producer) {
                // Add this node to producer's users list
                if (producer->use_count >= producer->users_capacity) {
                    int new_cap = producer->users_capacity == 0 ? 4 : producer->users_capacity * 2;
                    producer->users = realloc(producer->users, new_cap * sizeof(struct IRNode*));
                    producer->users_capacity = new_cap;
                }
                producer->users[producer->use_count++] = node;
            }
        }
        node = node->next;
    }
    return 0;
}
```

### 5.4 Cache Locality Reordering

Uses Kahn's algorithm for topological sort:

```c
static int reorder_for_cache_locality(CMLGraph_t ir) {
    // Calculate in-degrees
    int* in_degree = calloc(ir->node_count, sizeof(int));
    for (int i = 0; i < ir->node_count; i++) {
        for (int j = 0; j < all_nodes[i]->num_inputs; j++) {
            struct IRNode* producer = find_node_by_output(ir, all_nodes[i]->input_names[j]);
            if (producer) in_degree[i]++;
        }
    }

    // Queue nodes with no dependencies
    struct IRNode** queue = malloc(ir->node_count * sizeof(struct IRNode*));
    int queue_front = 0, queue_back = 0;

    for (int i = 0; i < ir->node_count; i++) {
        if (in_degree[i] == 0) {
            queue[queue_back++] = all_nodes[i];
        }
    }

    // Topological sort
    struct IRNode** sorted = malloc(ir->node_count * sizeof(struct IRNode*));
    int sorted_count = 0;

    while (queue_front < queue_back) {
        struct IRNode* current = queue[queue_front++];
        sorted[sorted_count++] = current;

        // Decrease in-degree of dependent nodes
        for (int i = 0; i < ir->node_count; i++) {
            if (depends_on(all_nodes[i], current)) {
                in_degree[i]--;
                if (in_degree[i] == 0) {
                    queue[queue_back++] = all_nodes[i];
                }
            }
        }
    }

    // Rebuild linked list in sorted order
    ir->head = sorted[0];
    ir->tail = sorted[sorted_count - 1];
    for (int i = 0; i < sorted_count - 1; i++) {
        sorted[i]->next = sorted[i + 1];
    }
    sorted[sorted_count - 1]->next = NULL;

    return 0;
}
```

______________________________________________________________________

## 6. Operation Fusion

**Files:** `src/ops/ir/optimization.c`

### 6.1 Fusion Patterns

C-ML recognizes and fuses 11+ patterns:

| Pattern                 | Transformation           | Benefit                             |
| ----------------------- | ------------------------ | ----------------------------------- |
| MUL + ADD               | FMA (Fused Multiply-Add) | Single instruction, better accuracy |
| NEG + ADD               | SUB                      | Eliminate negation                  |
| EXP + LOG               | Identity                 | Eliminate both ops                  |
| MUL + DIV               | Identity                 | (a\*b)/a -> b                       |
| SQRT + MUL              | sqrt_mul                 | Combined kernel                     |
| EXP + RECIP             | exp_recip                | 1/exp(x) kernel                     |
| NEG + EXP               | exp(-x)                  | Common in sigmoid                   |
| LOG + MUL               | log(x)\*y                | Common in losses                    |
| Elementwise chain       | Fused kernel             | Reduce memory traffic               |
| ADD + MUL               | Chained kernel           | Cache locality                      |
| Reduction + Elementwise | Combined                 | SUM + DIV for mean                  |

### 6.2 Fusion Detection

```c
static bool can_fuse_operations(struct IRNode* node1, struct IRNode* node2,
                                FusionType* fusion_type) {
    // Must be producer-consumer relationship
    if (!node_uses_output(node1, node2)) return false;

    // Pattern 1: MUL + ADD -> FMA
    if (node1->type == UOP_MUL && node2->type == UOP_ADD) {
        *fusion_type = FUSION_FMA;
        return true;
    }

    // Pattern 2: NEG + ADD -> SUB
    if (node1->type == UOP_NEG && node2->type == UOP_ADD) {
        *fusion_type = FUSION_NEG_ADD;
        return true;
    }

    // Pattern 7: Elementwise chain (general)
    bool node1_is_elementwise = (node1->type == UOP_ADD || node1->type == UOP_MUL ||
                                  node1->type == UOP_EXP || node1->type == UOP_LOG || ...);
    bool node2_is_elementwise = (node2->type == UOP_ADD || node2->type == UOP_MUL || ...);

    if (node1_is_elementwise && node2_is_elementwise && node1->use_count == 1) {
        *fusion_type = FUSION_CHAIN_ELEMENTWISE;
        return true;
    }

    return false;
}
```

### 6.3 Chain Fusion

Finds longest chains of fusable operations:

```c
static int find_fusable_chain(struct IRNode* start, struct IRNode** chain, int max_chain) {
    chain[0] = start;
    int chain_len = 1;
    struct IRNode* current = start;

    // Follow chain while operations can be fused
    while (chain_len < max_chain && current->use_count == 1) {
        struct IRNode* next = current->users[0];
        if (!next || next->is_fused) break;

        // Verify producer-consumer relationship
        if (!node_uses_output(current, next)) break;

        FusionType fusion_type;
        if (can_fuse_operations(current, next, &fusion_type)) {
            // Don't fuse identical ops (e.g., ADD -> ADD)
            if (current->type != next->type) {
                chain[chain_len++] = next;
                current = next;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    return chain_len;
}
```

### 6.4 Fusion Application

```c
static int apply_fusion(struct IRNode* node1, struct IRNode* node2, FusionType fusion_type) {
    switch (fusion_type) {
    case FUSION_FMA: {
        // MUL + ADD -> FMA: a * b + c
        char* other_input = find_other_input(node1, node2);
        LOG_DEBUG("Fused MUL+ADD -> FMA: %s * %s + %s",
                 node1->input_names[0], node1->input_names[1], other_input);

        struct IRNode* ops[] = {node1, node2};
        FusedKernel* kernel = create_fused_kernel(ops, 2, FUSION_FMA);
        node1->fused_kernel = kernel;
        node2->fused_kernel = kernel;
        node1->is_fused = true;
        node2->is_fused = true;
        break;
    }

    case FUSION_NEG_ADD: {
        // NEG + ADD -> SUB: -a + b -> b - a
        // Transform ADD to SUB and swap inputs
        node2->type = UOP_SUB;
        // ... swap input order
        break;
    }
    // ... other patterns
    }
    return 0;
}
```

______________________________________________________________________

## 7. LLVM JIT Backend

**Files:** `include/ops/ir/llvm/`, `src/ops/ir/llvm/`

The LLVM JIT backend compiles IR graphs to native machine code at runtime using LLVM's ORC JIT engine, providing significant performance improvements over the CPU interpreter for repeated computations.

### 7.1 Compilation Pipeline

```
IR Graph -> LLVM IR Generation -> Optimization Passes -> Native Code -> Execution
```

### 7.2 Kernel Caching

Compiled kernels are cached with LRU eviction (256 entry limit) to avoid recompilation:

```c
void cml_kernel_cache_clear(void);
void cml_kernel_cache_stats(size_t* hits, size_t* misses, size_t* count, size_t* memory);
double cml_kernel_cache_hit_rate(void);
void cml_kernel_cache_print_stats(void);
```

### 7.3 Two-Level Optimization Pipeline

```c
// Step 1: C-ML IR-level optimization (pattern matching)
cml_ir_optimize(ir);  // DCE, fusion, reordering

// Step 2: LLVM-level optimization (machine code generation)
// Happens automatically during execution
```

______________________________________________________________________

## 8. Caching

### 8.1 Graph Cache

**Files:** `include/ops/ir/graph_cache.h`, `src/ops/ir/graph_cache.c`

Caches execution plans for repeated graph executions:

```c
typedef struct CMLExecutionPlan {
    struct IRNode** nodes;    // Topologically sorted nodes
    float** buffers;          // Pre-allocated output buffers
    size_t* buffer_sizes;
    size_t num_nodes;
    uint64_t signature;       // Graph hash
    bool valid;
} CMLExecutionPlan;

typedef struct CMLGraphCache {
    CMLGraphCacheEntry** buckets;
    size_t num_buckets;
    size_t max_entries;
    size_t count;
    uint64_t timestamp;
    size_t hits;
    size_t misses;
} CMLGraphCache;
```

#### Graph Signature Hashing (FNV-1a)

```c
uint64_t cml_graph_compute_signature(CMLGraph_t ir) {
    uint64_t hash = FNV_OFFSET;  // 0xcbf29ce484222325ULL

    struct IRNode* node = ir->head;
    while (node) {
        // Hash operation type
        hash ^= (uint64_t)node->type;
        hash *= FNV_PRIME;  // 0x100000001b3ULL

        // Hash output shape
        if (node->output) {
            hash ^= (uint64_t)node->output->ndim;
            hash *= FNV_PRIME;
            for (int i = 0; i < node->output->ndim; i++) {
                hash ^= (uint64_t)node->output->shape[i];
                hash *= FNV_PRIME;
            }
        }

        // Hash number of inputs
        hash ^= (uint64_t)node->num_inputs;
        hash *= FNV_PRIME;

        node = node->next;
    }
    return hash;
}
```

#### LRU Eviction

```c
static void evict_lru_entry(CMLGraphCache* cache) {
    uint64_t oldest_time = UINT64_MAX;
    CMLGraphCacheEntry* oldest_entry = NULL;

    // Find LRU entry across all buckets
    for (size_t b = 0; b < cache->num_buckets; b++) {
        CMLGraphCacheEntry* entry = cache->buckets[b];
        while (entry) {
            if (entry->last_used < oldest_time) {
                oldest_time = entry->last_used;
                oldest_entry = entry;
            }
            entry = entry->next;
        }
    }

    // Remove and free
    if (oldest_entry) {
        // ... remove from bucket linked list
        cml_free_execution_plan(oldest_entry->plan);
        free(oldest_entry);
        cache->count--;
    }
}
```

### 8.2 Kernel Cache

**Files:** `include/ops/ir/kernel_cache.h`, `src/ops/ir/kernel_cache.c`

Caches compiled JIT kernels:

```c
typedef struct CMLKernelEntry {
    uint64_t hash;
    CMLKernelBackend backend;
    void* compiled;           // Compiled kernel
    size_t memory_size;
    uint64_t last_used;
    int num_ops;
    int num_inputs;
    int num_outputs;
    struct CMLKernelEntry* next;
} CMLKernelEntry;

typedef struct CMLKernelCache {
    CMLKernelEntry** buckets;
    size_t num_buckets;
    size_t max_entries;
    size_t max_memory;
    size_t count;
    size_t total_memory;
    size_t hits;
    size_t misses;
    size_t evictions;
    pthread_mutex_t lock;     // Thread-safe
} CMLKernelCache;
```

#### Kernel Hash Computation

```c
uint64_t cml_kernel_cache_compute_hash(CMLGraph_t ir, Tensor** inputs,
                                        int num_inputs, CMLKernelBackend backend) {
    uint64_t hash = fnv1a_hash_init();

    // Hash backend type
    hash = fnv1a_hash_int(hash, (int)backend);

    // Hash number of inputs
    hash = fnv1a_hash_int(hash, num_inputs);

    // Hash input shapes and dtypes
    for (int i = 0; i < num_inputs; i++) {
        if (inputs && inputs[i]) {
            Tensor* t = inputs[i];
            hash = fnv1a_hash_int(hash, t->ndim);
            for (int d = 0; d < t->ndim; d++) {
                hash = fnv1a_hash_int(hash, t->shape[d]);
            }
            hash = fnv1a_hash_int(hash, (int)t->dtype);
        }
    }

    // Hash IR structure
    if (ir) {
        hash = fnv1a_hash_size(hash, ir->node_count);
        struct IRNode* node = ir->head;
        while (node) {
            hash = fnv1a_hash_int(hash, (int)node->type);
            // ... hash output shapes
            node = node->next;
        }
    }
    return hash;
}
```

#### Thread-Safe Operations

```c
CMLKernelEntry* cml_kernel_cache_lookup(CMLKernelCache* cache, uint64_t hash) {
    pthread_mutex_lock(&cache->lock);

    size_t bucket_idx = hash % cache->num_buckets;
    CMLKernelEntry* entry = cache->buckets[bucket_idx];

    while (entry) {
        if (entry->hash == hash) {
            entry->last_used = ++cache->timestamp;
            cache->hits++;
            pthread_mutex_unlock(&cache->lock);
            return entry;
        }
        entry = entry->next;
    }

    cache->misses++;
    pthread_mutex_unlock(&cache->lock);
    return NULL;
}
```

______________________________________________________________________

## 9. GPU Backends

C-ML supports multiple GPU backends through a unified dispatch system.

### 9.1 Backend Dispatch

```c
typedef enum {
    CML_KERNEL_BACKEND_CPU_FALLBACK = 0,
    CML_KERNEL_BACKEND_CPU_LLVM,
    CML_KERNEL_BACKEND_CUDA,
    CML_KERNEL_BACKEND_ROCM,
    CML_KERNEL_BACKEND_METAL,
    CML_KERNEL_BACKEND_VULKAN,
    CML_KERNEL_BACKEND_COUNT
} CMLKernelBackend;
```

### 9.2 CUDA Backend

**Files:** `include/ops/ir/gpu/`, `src/ops/ir/gpu/`

- PTX code compilation via `cuModuleLoadData`
- Kernel execution via `cuLaunchKernel`
- Device memory: `cuMemAlloc`, `cuMemFree`
- Memory transfers: H2D, D2H, D2D
- NVRTC runtime compilation support
- Stream management for async operations

### 9.3 ROCm/HIP Backend

- HIP module loading and kernel functions
- Multi-device support
- Stream synchronization

### 9.4 Metal Backend (macOS)

- Metal framework integration (Objective-C)
- Metal Shading Language (MSL) kernels

### 9.5 Vulkan Backend

- Cross-platform GPU support
- SPIR-V code generation

______________________________________________________________________

## 10. Gradient Checkpointing

**Files:** `include/autograd/checkpointing.h`, `src/autograd/checkpointing.c`

Trades compute for memory by recomputing activations during backward pass.

### 10.1 Checkpointing API

```c
// Enable/disable globally
void autograd_set_checkpointing(bool enabled);
bool autograd_is_checkpointing_enabled(void);

// Checkpoint a tensor (save inputs, discard activations)
int autograd_checkpoint(Tensor* tensor);

// Recompute during backward pass
Tensor* autograd_recompute(Tensor* tensor);

// Cleanup at end of training
void autograd_checkpointing_cleanup(void);
```

### 10.2 Checkpoint Implementation

```c
typedef struct CheckpointedTensor {
    Tensor* tensor;
    struct IRNode* saved_ir_node;    // Save computation graph
    CMLGraph_t saved_ir_context;
    Tensor** saved_inputs;           // Save input tensors
    int num_inputs;
} CheckpointedTensor;

int autograd_checkpoint(Tensor* tensor) {
    CheckpointedTensor* checkpoint = malloc(sizeof(CheckpointedTensor));

    checkpoint->tensor = tensor;
    checkpoint->saved_ir_node = tensor->ir_node;
    checkpoint->saved_ir_context = tensor->ir_context;

    // Save input references
    if (tensor->ir_node && tensor->ir_node->inputs) {
        checkpoint->num_inputs = tensor->ir_node->num_inputs;
        checkpoint->saved_inputs = malloc(checkpoint->num_inputs * sizeof(Tensor*));
        for (int i = 0; i < checkpoint->num_inputs; i++) {
            checkpoint->saved_inputs[i] = tensor->ir_node->inputs[i];
        }
    }

    // Clear IR node to force recomputation
    tensor->ir_node = NULL;
    tensor->ir_context = NULL;

    // Add to registry
    checkpointed_tensors[num_checkpointed++] = checkpoint;
    return 0;
}
```

### 10.3 Recomputation

```c
Tensor* autograd_recompute(Tensor* tensor) {
    // Find checkpoint
    CheckpointedTensor* checkpoint = find_checkpoint(tensor);
    if (!checkpoint) return tensor;

    // Recursively recompute inputs if checkpointed
    for (int i = 0; i < checkpoint->num_inputs; i++) {
        if (!checkpoint->saved_inputs[i]->ir_node) {
            autograd_recompute(checkpoint->saved_inputs[i]);
        }
    }

    // Recompute forward pass based on operation type
    Tensor* recomputed = NULL;
    switch (checkpoint->saved_ir_node->type) {
    case UOP_ADD:
        recomputed = uop_add(checkpoint->saved_inputs[0], checkpoint->saved_inputs[1]);
        break;
    case UOP_MUL:
        recomputed = uop_mul(checkpoint->saved_inputs[0], checkpoint->saved_inputs[1]);
        break;
    case UOP_MATMUL:
        recomputed = uop_matmul(checkpoint->saved_inputs[0], checkpoint->saved_inputs[1]);
        break;
    // ... other operations
    }

    // Copy recomputed data and restore IR node
    if (recomputed) {
        memcpy(tensor->data, recomputed->data, tensor->numel * sizeof(float));
        tensor->ir_node = checkpoint->saved_ir_node;
        tensor_free(recomputed);
    }

    return tensor;
}
```

______________________________________________________________________

## 11. Compiler Optimizations

### 11.1 Makefile Flags

```makefile
# Base flags
BASE_CFLAGS = -Wall -Wextra -Wpedantic -fPIC -std=c11

# SIMD flags (auto-detected)
SIMD_FLAGS = -msse -msse2 -mfpmath=sse
ifeq ($(AVX_SUPPORT),yes)
    SIMD_FLAGS += -mavx
endif
ifeq ($(AVX2_SUPPORT),yes)
    SIMD_FLAGS += -mavx2 -mfma
endif
ifeq ($(AVX512_SUPPORT),yes)
    SIMD_FLAGS += -mavx512f
endif

# Build configurations
# Default: -O2 -g
CFLAGS = -O2 -g $(BASE_CFLAGS)
```

### 11.2 Build Targets

```makefile
# Debug build (sanitizers enabled)
debug: CFLAGS = -O0 -g3 -fsanitize=address,undefined $(BASE_CFLAGS)

# Release build (maximum optimization)
release: CFLAGS = -O3 -DNDEBUG $(BASE_CFLAGS)

# Fast build (CPU-specific optimizations)
fast: CFLAGS = -O3 -DNDEBUG -march=native -mtune=native \
               -ffast-math -funroll-loops $(BASE_CFLAGS)

# Static analysis
analyze: CFLAGS += -fanalyzer
```

### 11.3 Optimization Flag Descriptions

| Flag             | Description                        |
| ---------------- | ---------------------------------- |
| `-O2`            | Standard optimization (default)    |
| `-O3`            | Aggressive optimization            |
| `-march=native`  | Optimize for current CPU           |
| `-mtune=native`  | Tune for current CPU               |
| `-ffast-math`    | Fast floating-point (less precise) |
| `-funroll-loops` | Unroll loops                       |
| `-mavx2 -mfma`   | Enable AVX2 + FMA instructions     |
| `-mavx512f`      | Enable AVX-512 instructions        |

______________________________________________________________________

## 12. Profiling

**Files:** `include/backend/profiling.h`, `src/backend/profiling.c`

### 12.1 Timer API

```c
typedef struct Timer {
    struct timespec start;
    struct timespec end;
} Timer;

void timer_start(Timer* timer);
void timer_stop(Timer* timer);
double timer_elapsed_ms(Timer* timer);
double timer_elapsed_us(Timer* timer);
```

### 12.2 Cache Statistics

```c
// Graph cache stats
void cml_graph_cache_print_stats(CMLGraphCache* cache);
// Output:
//   Entries: 15 / 32
//   Hits: 1247
//   Misses: 23
//   Hit rate: 98.2%

// Kernel cache stats
void cml_kernel_cache_print_stats(CMLKernelCache* cache);
double cml_kernel_cache_hit_rate(CMLKernelCache* cache);

// BLAS status
void cml_blas_print_status(CMLBlasContext* ctx);

// SIMD capabilities
void cml_print_simd_caps(void);
```

______________________________________________________________________

## Best Practices

### Memory Management

1. **Use tensor pools for fixed-shape tensors** (training batches)
1. **Enable graph allocator** for computation graphs
1. **Call cleanup functions** after training epochs

### Performance Tuning

1. **Check SIMD support**: `cml_print_simd_caps()`
1. **Verify BLAS loading**: `cml_blas_print_status(NULL)`
1. **Monitor cache hit rates**: `cml_kernel_cache_hit_rate()`
1. **Use `make fast`** for CPU-specific builds

### Training

1. **Enable gradient checkpointing** for large models:

   ```c
   autograd_set_checkpointing(true);
   autograd_checkpoint(large_activation);
   ```

1. **Reuse computation graphs** when possible to benefit from caching

1. **Batch operations** to amortize kernel launch overhead

### Debugging Performance

```c
// Print all optimization status
cml_print_simd_caps();
cml_blas_print_status(NULL);
cml_graph_cache_print_stats(NULL);
cml_kernel_cache_print_stats(NULL);
```

______________________________________________________________________

## Known Limitations

### LLVM JIT Memory Growth

When using LLVM JIT execution, memory can grow over time because LLVM's internal memory pools don't always release memory back to the OS.

**Workarounds:**

- Call `cml_reset_ir_context()` after each batch to limit IR node growth per epoch
- Use larger batch sizes to reduce total JIT compilations
- The kernel cache (LRU, 256 entries) automatically limits compiled kernel accumulation

______________________________________________________________________

## Summary Table

| Optimization     | Location          | Speedup Factor | Memory Impact         |
| ---------------- | ----------------- | -------------- | --------------------- |
| AVX-512 SIMD     | `simd_math.c`     | 8-16x          | None                  |
| AVX2 SIMD        | `simd_math.c`     | 4-8x           | None                  |
| BLAS GEMM        | `blas.c`          | 10-100x        | None                  |
| Operation Fusion | `optimization.c`  | 1.5-3x         | Reduced               |
| Graph Caching    | `graph_cache.c`   | 2-10x          | Increased             |
| Kernel Caching   | `kernel_cache.c`  | 2-5x           | Increased             |
| Memory Pooling   | `memory_pools.c`  | 1.2-2x         | Reduced fragmentation |
| Checkpointing    | `checkpointing.c` | 1x             | 50-80% reduction      |
| Thread Pool      | `threadpool.c`    | Nx (N = cores) | Minimal               |

______________________________________________________________________

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [SLEEF Library](https://sleef.org/)
- [OpenBLAS](https://www.openblas.net/)
- [LLVM Documentation](https://llvm.org/docs/)
