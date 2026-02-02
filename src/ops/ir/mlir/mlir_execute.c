#include "ops/ir/mlir/mlir_internal.h"
#include "ops/ir/mlir/mlir_backend.h"
#include "ops/ir/mlir/mlir_codegen.h"
#include "ops/ir/mlir/mlir_context.h"
#include "ops/ir/mlir/mlir_execute.h"
#include "ops/ir/mlir/mlir_cpp_bridge.h"
#include "ops/ir/mlir/mlir_kernel_cache.h"
#include "ops/ir/mlir/mlir_internal.h"
#include "ops/ir/internal.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <ffi.h>

// ============================================================================
// LLVM Engine Cache Support
// ============================================================================

#ifdef CML_HAS_MLIR
// Free function for cached LLVM execution engines
static void llvm_engine_free_fn(void* compiled) {
    if (compiled) {
        MlirExecutionEngine engine = {compiled};
        mlirExecutionEngineDestroy(engine);
        LOG_DEBUG("Freed cached LLVM execution engine");
    }
}

// Flag to track if we've registered the free function
static bool g_llvm_free_fn_registered = false;

// Ensure the LLVM free function is registered
static void ensure_llvm_free_fn_registered(void) {
    if (!g_llvm_free_fn_registered) {
        cml_kernel_cache_set_free_fn(CML_KERNEL_CPU_LLVM, llvm_engine_free_fn);
        g_llvm_free_fn_registered = true;
        LOG_DEBUG("Registered LLVM engine free function for kernel cache");
    }
}

// Compute a hash based on tensor configurations (shapes and dtypes)
// This identifies equivalent computation patterns for caching
static uint64_t compute_execution_hash(CMLMLIRContext* ctx) {
    // FNV-1a hash constants
    const uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    const uint64_t FNV_PRIME  = 0x100000001b3ULL;

    uint64_t hash = FNV_OFFSET;

    // Hash number of inputs and outputs
    hash ^= (uint64_t)ctx->num_inputs;
    hash *= FNV_PRIME;
    hash ^= (uint64_t)ctx->num_outputs;
    hash *= FNV_PRIME;

    // Hash input tensor shapes and dtypes
    for (int i = 0; i < ctx->num_inputs; i++) {
        Tensor* t = ctx->inputs[i];
        if (t) {
            hash ^= (uint64_t)t->ndim;
            hash *= FNV_PRIME;
            for (int d = 0; d < t->ndim; d++) {
                hash ^= (uint64_t)t->shape[d];
                hash *= FNV_PRIME;
            }
            hash ^= (uint64_t)t->dtype;
            hash *= FNV_PRIME;
        }
    }

    // Hash output tensor shapes
    for (int i = 0; i < ctx->num_outputs; i++) {
        Tensor* t = ctx->outputs[i];
        if (t) {
            hash ^= (uint64_t)t->ndim;
            hash *= FNV_PRIME;
            for (int d = 0; d < t->ndim; d++) {
                hash ^= (uint64_t)t->shape[d];
                hash *= FNV_PRIME;
            }
        }
    }

    return hash;
}
#endif

#ifdef CML_HAS_MLIR
// JIT Engine structure
struct CMLJITEngine {
    void* engine;          // MlirExecutionEngine stored as void*
    CMLKernelCache* cache; // Kernel cache for compiled kernels
};
#endif

// ============================================================================
// JIT Engine Management
// ============================================================================

// ============================================================================
// JIT Engine Management
// ============================================================================

CMLJITEngine* cml_jit_engine_create(void) {
#ifdef CML_HAS_MLIR
    CMLJITEngine* engine = (CMLJITEngine*)malloc(sizeof(CMLJITEngine));
    if (!engine)
        return NULL;

    // Engine is created lazily during kernel compilation
    engine->engine = NULL;

    // Create kernel cache with 256 max entries (can be configured)
    engine->cache = cml_kernel_cache_create(256);
    if (!engine->cache) {
        LOG_WARNING("Failed to create kernel cache, running without caching");
    }

    LOG_INFO("JIT Engine created with kernel cache");
    return engine;
#else
    LOG_WARNING("JIT engine requires MLIR");
    return NULL;
#endif
}

void cml_jit_engine_destroy(CMLJITEngine* engine) {
#ifdef CML_HAS_MLIR
    if (!engine)
        return;

    if (engine->engine) {
        MlirExecutionEngine exec_engine = {engine->engine};
        mlirExecutionEngineDestroy(exec_engine);
    }

    // Free kernel cache
    if (engine->cache) {
        kernel_cache_print_stats(engine->cache);
        cml_kernel_cache_free(engine->cache);
    }

    free(engine);
    LOG_INFO("JIT Engine destroyed");
#else
    (void)engine;
#endif
}

// ============================================================================
// Compilation & Execution
// ============================================================================

CMLJITKernelFunction cml_jit_compile_kernel(CMLJITEngine* engine, CMLIR_t ir) {
#ifdef CML_HAS_MLIR
    if (!engine || !ir) {
        LOG_ERROR("Invalid arguments to cml_jit_compile_kernel");
        return NULL;
    }

    // Check kernel cache first
    if (engine->cache) {
        uint64_t ir_hash       = cml_kernel_cache_compute_hash(ir, NULL, 0, CML_KERNEL_CPU_LLVM);
        CMLKernelEntry* cached = cml_kernel_cache_lookup(engine->cache, ir_hash);
        if (cached && cached->compiled) {
            LOG_INFO("JIT kernel found in cache (hash: 0x%llx)", (unsigned long long)ir_hash);
            return (CMLJITKernelFunction)cached->compiled;
        }
    }

    LOG_INFO("JIT compiling kernel from IR graph (cache miss)...");

    // 1. Initialize MLIR context
    CMLMLIRContext* ctx = cml_mlir_init();
    if (!ctx) {
        LOG_ERROR("Failed to initialize MLIR context for JIT");
        return NULL;
    }

    // 2. Convert IR to MLIR
    if (!cml_ir_to_mlir(ctx, ir)) {
        LOG_ERROR("Failed to convert IR to MLIR for JIT");
        cml_mlir_destroy(ctx);
        return NULL;
    }

    // 3. Apply optimization passes
    LOG_INFO("Applying optimization passes (O3)...");
    if (cml_mlir_optimize(ctx->module.ptr, ctx->context.ptr) != 0) {
        LOG_WARNING("Optimization passes had issues, continuing anyway");
    }

    // 4. CRITICAL: Lower module to LLVM before creating execution engine
    MlirModule mod = ctx->module;
    LOG_INFO("Lowering MLIR module to LLVM dialect...");
    int lower_result = cml_mlir_lower_module_to_llvm(mod);
    if (lower_result != 0) {
        LOG_ERROR("Failed to lower MLIR module to LLVM (error: %d)", lower_result);
        LOG_WARNING("JIT compilation disabled - falling back to interpreter");
        cml_mlir_destroy(ctx);
        return NULL;
    }
    LOG_INFO("LLVM lowering successful");

    // 5. Create ExecutionEngine with optimization
    MlirExecutionEngine jit_engine = mlirExecutionEngineCreate(mod,
                                                               3, // O3 optimization
                                                               0, // No shared library paths
                                                               NULL, false);

    if (mlirExecutionEngineIsNull(jit_engine)) {
        LOG_ERROR("Failed to create JIT execution engine");
        cml_mlir_destroy(ctx);
        return NULL;
    }

    // 5. Store engine in wrapper
    if (engine->engine) {
        MlirExecutionEngine old_engine = {engine->engine};
        mlirExecutionEngineDestroy(old_engine);
    }
    engine->engine = jit_engine.ptr;

    CMLJITKernelFunction kernel_fn = (CMLJITKernelFunction)(uintptr_t)jit_engine.ptr;

    // 6. Cache the compiled kernel
    if (engine->cache) {
        uint64_t ir_hash      = cml_kernel_cache_compute_hash(ir, NULL, 0, CML_KERNEL_CPU_LLVM);
        size_t estimated_size = ir->node_count * 1024; // Rough estimate
        cml_kernel_cache_insert(engine->cache, ir_hash, CML_KERNEL_CPU_LLVM, (void*)kernel_fn,
                                estimated_size);
        LOG_INFO("JIT compilation successful (cached for reuse, hash: 0x%llx)",
                 (unsigned long long)ir_hash);
    } else {
        LOG_INFO("JIT compilation successful");
    }

    return kernel_fn;

#else
    (void)engine;
    (void)ir;
    LOG_ERROR("MLIR support not compiled in");
    return NULL;
#endif
}

CMLJITKernelFunction cml_mlir_jit_compile(CMLMLIRContext* ctx, void* mlir_module) {
#ifdef CML_HAS_MLIR
    if (!ctx || !mlir_module) {
        LOG_ERROR("Invalid arguments to cml_mlir_jit_compile");
        return NULL;
    }

    LOG_INFO("JIT compiling MLIR module...");

    // Apply optimization passes
    if (cml_mlir_optimize(mlir_module, ctx->context.ptr) != 0) {
        LOG_WARNING("Optimization failed, continuing anyway");
    }

    // CRITICAL: Lower module to LLVM before creating execution engine
    // The execution engine expects LLVM dialect, not linalg/memref
    MlirModule mod = {mlir_module};
    LOG_INFO("Lowering MLIR module to LLVM dialect...");
    int lower_result = cml_mlir_lower_module_to_llvm(mod);
    if (lower_result != 0) {
        LOG_ERROR("Failed to lower MLIR module to LLVM (error: %d)", lower_result);
        LOG_WARNING("JIT compilation disabled - falling back to interpreter");
        return NULL;
    }
    LOG_INFO("LLVM lowering successful");

    // Create execution engine
    MlirExecutionEngine jit_engine = mlirExecutionEngineCreate(mod,
                                                               3, // O3 optimization
                                                               0, // No shared library paths
                                                               NULL, false);

    if (mlirExecutionEngineIsNull(jit_engine)) {
        LOG_ERROR("Failed to create JIT execution engine");
        return NULL;
    }

    LOG_INFO("JIT compilation successful");
    return (CMLJITKernelFunction)(uintptr_t)jit_engine.ptr;

#else
    (void)ctx;
    (void)mlir_module;
    LOG_ERROR("MLIR support not compiled in");
    return NULL;
#endif
}

void cml_jit_execute(CMLJITKernelFunction fn, float** inputs, int num_inputs, float* output) {
#ifdef CML_HAS_MLIR
    // Legacy interface - converts float** to Tensor** for MLIR backend
    // Modern code should use cml_mlir_execute directly
    LOG_WARNING("cml_jit_execute (legacy float**) not fully implemented for MLIR backend");
    (void)fn;
    (void)inputs;
    (void)num_inputs;
    (void)output;
#else
    (void)fn;
    (void)inputs;
    (void)num_inputs;
    (void)output;
#endif
}

// Memref descriptor for 1D memref - matches MLIR's StridedMemRefType<float, 1>
typedef struct {
    float* allocated;   // Base allocation pointer
    float* aligned;     // Aligned data pointer (may equal allocated)
    int64_t offset;     // Offset from aligned
    int64_t sizes[1];   // Size in each dimension
    int64_t strides[1]; // Stride in each dimension
} MemRefDescriptor1D;

int cml_mlir_execute(void* engine, Tensor** inputs, int num_inputs, Tensor** outputs,
                     int num_outputs) {
#ifdef CML_HAS_MLIR
    if (!engine) {
        LOG_ERROR("Invalid arguments to cml_mlir_execute: engine is NULL");
        return -1;
    }

    CMLMLIRContext* ctx = (CMLMLIRContext*)engine;

    // Use provided inputs/outputs if available, otherwise use context's arrays
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    // Note: We use ctx->inputs and ctx->outputs which are set during IR construction

    if (!ctx->inputs || !ctx->outputs) {
        LOG_ERROR("ctx->inputs or outputs is NULL");
        return -1;
    }

    // Ensure LLVM free function is registered for cache eviction
    ensure_llvm_free_fn_registered();

    // Get the global kernel cache
    CMLKernelCache* cache = cml_kernel_cache_get_default();

    // Compute hash for this execution pattern (based on tensor shapes)
    uint64_t exec_hash = compute_execution_hash(ctx);

    // Check if we have a cached execution engine for this pattern
    CMLKernelEntry* cached_entry = cml_kernel_cache_lookup(cache, exec_hash);
    MlirExecutionEngine jit_engine;
    bool from_cache = false;

    if (cached_entry && cached_entry->compiled) {
        // Cache hit - reuse the compiled engine
        jit_engine.ptr = cached_entry->compiled;
        from_cache     = true;
        LOG_DEBUG("Kernel cache HIT (hash: 0x%016llx)", (unsigned long long)exec_hash);
    } else {
        // Cache miss - need to compile
        LOG_DEBUG("Kernel cache MISS (hash: 0x%016llx) - compiling...",
                  (unsigned long long)exec_hash);

        // CRITICAL: Lower module to LLVM before creating execution engine
        // The execution engine expects LLVM dialect, not linalg/memref
        LOG_INFO("Lowering MLIR module to LLVM dialect for execution...");
        int lower_result = cml_mlir_lower_module_to_llvm(ctx->module);
        if (lower_result != 0) {
            LOG_ERROR("Failed to lower MLIR module to LLVM (error: %d)", lower_result);
            LOG_WARNING("MLIR JIT execution failed - graph will use CPU interpreter fallback");
            return -1;
        }
        LOG_INFO("LLVM lowering successful");

        // Create execution engine with O3 optimization
        jit_engine = mlirExecutionEngineCreate(ctx->module,
                                               3, // O3 optimization
                                               0, NULL, false);
        if (mlirExecutionEngineIsNull(jit_engine)) {
            LOG_ERROR("Failed to create MLIR JIT execution engine");
            return -1;
        }

        // Cache the newly compiled engine for future reuse
        // Estimate memory: ~1KB per operation is a rough heuristic
        size_t estimated_mem = (ctx->num_inputs + ctx->num_outputs) * 4096;
        int insert_result    = cml_kernel_cache_insert(cache, exec_hash, CML_KERNEL_CPU_LLVM,
                                                       jit_engine.ptr, estimated_mem);
        if (insert_result == 0) {
            LOG_INFO("Cached JIT engine (hash: 0x%016llx, est. %zu bytes)",
                     (unsigned long long)exec_hash, estimated_mem);
        } else {
            LOG_WARNING("Failed to cache JIT engine - will destroy after use");
        }
    }

    int total_args = ctx->num_inputs + ctx->num_outputs;

    // Allocate output tensor data if needed (lazy allocation)
    for (int i = 0; i < ctx->num_outputs; i++) {
        Tensor* out = ctx->outputs[i];
        if (out && !out->data && out->numel > 0) {
            size_t size = out->numel * sizeof(float); // Assuming float for now
            out->data   = calloc(1, size);
            if (!out->data) {
                LOG_ERROR("Failed to allocate output tensor %d data (%zu bytes)", i, size);
                if (!from_cache) {
                    mlirExecutionEngineDestroy(jit_engine);
                }
                return -1;
            }
            out->owns_data = true;
            LOG_DEBUG("Allocated %zu bytes for output tensor %d", size, i);
        }
    }

    // Allocate memref descriptors and argument pointers
    MemRefDescriptor1D* descriptors =
        (MemRefDescriptor1D*)malloc(sizeof(MemRefDescriptor1D) * total_args);
    void** packed_args = (void**)malloc(sizeof(void*) * total_args);

    if (!descriptors || !packed_args) {
        free(descriptors);
        free(packed_args);
        if (!from_cache) {
            mlirExecutionEngineDestroy(jit_engine);
        }
        return -1;
    }

    // Setup memref descriptors for all tensors
    for (int i = 0; i < total_args; i++) {
        Tensor* t = i < ctx->num_inputs ? ctx->inputs[i] : ctx->outputs[i - ctx->num_inputs];
        if (!t || !t->data) {
            LOG_ERROR("Tensor %d is NULL or has no data", i);
            free(descriptors);
            free(packed_args);
            if (!from_cache) {
                mlirExecutionEngineDestroy(jit_engine);
            }
            return -1;
        }

        // Fill memref descriptor (for 1D memref)
        descriptors[i].allocated  = (float*)t->data;
        descriptors[i].aligned    = (float*)t->data;
        descriptors[i].offset     = 0;
        descriptors[i].sizes[0]   = t->numel;
        descriptors[i].strides[0] = 1;

        // InvokePacked expects pointers to each descriptor
        packed_args[i] = &descriptors[i];
    }

    // Look up the C-interface function directly by its mangled name
    MlirStringRef ciface_name = mlirStringRefCreateFromCString("_mlir_ciface_main");
    void* fn_ptr              = mlirExecutionEngineLookup(jit_engine, ciface_name);

    if (!fn_ptr) {
        // Fall back to looking up with LookupPacked
        MlirStringRef func_name = mlirStringRefCreateFromCString("main");
        fn_ptr                  = mlirExecutionEngineLookupPacked(jit_engine, func_name);
    }

    if (!fn_ptr) {
        LOG_ERROR("Failed to lookup function 'main' or '_mlir_ciface_main' in JIT engine");
        free(descriptors);
        free(packed_args);
        // Only destroy if we failed to cache it
        if (!from_cache) {
            // Remove from cache since we can't use it
            cml_kernel_cache_remove(cache, exec_hash);
            mlirExecutionEngineDestroy(jit_engine);
        }
        return -1;
    }

    // The C-interface wrapper expects: void fn(MemRefDescriptor1D*, MemRefDescriptor1D*,
    // MemRefDescriptor1D*) For 3 memrefs (2 inputs + 1 output)
    typedef void (*MainFn3)(MemRefDescriptor1D*, MemRefDescriptor1D*, MemRefDescriptor1D*);
    MainFn3 main_fn = (MainFn3)fn_ptr;

    // Call the JIT-compiled function
    main_fn(&descriptors[0], &descriptors[1], &descriptors[2]);

    free(descriptors);
    free(packed_args);

    // Mark outputs as executed
    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i]) {
            outputs[i]->is_executed = true;
        }
    }

    // NOTE: We do NOT destroy the engine here anymore!
    // It's either cached for reuse, or was already in the cache.
    // The cache will handle cleanup via LRU eviction or explicit cache_free().

    return 0;
#else
    (void)engine;
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    return -1;
#endif
}

#ifdef CML_HAS_MLIR
// String stream structure for printing
typedef struct {
    char* data;
    size_t size;
    size_t capacity;
} StringStream;

static void print_callback(MlirStringRef string, void* userdata) {
    StringStream* stream = (StringStream*)userdata;
    size_t len           = string.length;
    if (stream->size + len + 1 > stream->capacity) {
        stream->capacity = (stream->size + len + 1) * 2;
        stream->data     = (char*)realloc(stream->data, stream->capacity);
    }
    memcpy(stream->data + stream->size, string.data, len);
    stream->size += len;
    stream->data[stream->size] = '\0';
}

// ...

/*
void cml_jit_cache_stats(size_t* hits, size_t* misses, size_t* size_bytes) {
    if (hits) *hits = jit_cache.hits;
    if (misses) *misses = jit_cache.misses;
    if (size_bytes) *size_bytes = jit_cache.current_size;
}

void cml_jit_cache_clear(void) {
    CacheEntry* entry = jit_cache.head;
    while (entry) {
        CacheEntry* next = entry->next;
        free(entry);
        entry = next;
    }
    jit_cache.head = NULL;
    jit_cache.current_size = 0;
    jit_cache.hits = 0;
    jit_cache.misses = 0;
    LOG_INFO("JIT cache cleared");
}
*/
#endif

char* cml_mlir_dump_module(void* mlir_module) {
#ifdef CML_HAS_MLIR
    if (!mlir_module)
        return NULL;
    MlirModule module = {mlir_module};
    MlirOperation op  = mlirModuleGetOperation(module);

    StringStream stream;
    stream.size     = 0;
    stream.capacity = 1024;
    stream.data     = (char*)malloc(stream.capacity);
    stream.data[0]  = '\0';

    mlirOperationPrint(op, print_callback, &stream);

    return stream.data;
#else
    (void)mlir_module;
    return NULL;
#endif
}

int cml_mlir_compile_to_object(const void* mlir_module, const char* output_path) {
#ifdef CML_HAS_MLIR
    if (!mlir_module || !output_path) {
        LOG_ERROR("Invalid arguments to cml_mlir_compile_to_object");
        return -1;
    }

    LOG_INFO("AOT compilation to object file: %s", output_path);

    // Step 1: Generate LLVM IR using codegen function
    char* llvm_ir = cml_mlir_gen_llvm_ir(mlir_module);
    if (!llvm_ir) {
        LOG_ERROR("Failed to generate LLVM IR for AOT compilation");
        return -1;
    }

    // Step 2: Write LLVM IR to temporary file
    char temp_ir_path[512];
    snprintf(temp_ir_path, sizeof(temp_ir_path), "%s.ll", output_path);

    FILE* ir_file = fopen(temp_ir_path, "w");
    if (!ir_file) {
        LOG_ERROR("Failed to create temporary IR file: %s", temp_ir_path);
        free(llvm_ir);
        return -1;
    }

    fprintf(ir_file, "%s", llvm_ir);
    fclose(ir_file);
    free(llvm_ir);

    LOG_INFO("Wrote LLVM IR to: %s", temp_ir_path);

    // Step 3: Compile LLVM IR to object file using llc
    char compile_cmd[1024];
    snprintf(compile_cmd, sizeof(compile_cmd), "llc -filetype=obj -O3 %s -o %s 2>&1", temp_ir_path,
             output_path);

    LOG_INFO("Compiling with: %s", compile_cmd);

    FILE* compile_output = popen(compile_cmd, "r");
    if (!compile_output) {
        LOG_ERROR("Failed to execute llc compiler");
        remove(temp_ir_path);
        return -1;
    }

    // Read compilation output
    char line[512];
    bool has_errors = false;
    while (fgets(line, sizeof(line), compile_output)) {
        if (strstr(line, "error") || strstr(line, "Error")) {
            has_errors = true;
            LOG_ERROR("Compilation error: %s", line);
        } else if (strstr(line, "warning") || strstr(line, "Warning")) {
            LOG_WARNING("Compilation warning: %s", line);
        } else {
            LOG_DEBUG("llc: %s", line);
        }
    }

    int status = pclose(compile_output);

    // Clean up temporary IR file
    remove(temp_ir_path);

    if (status != 0 || has_errors) {
        LOG_ERROR("AOT compilation failed (status: %d)", status);
        return -1;
    }

    LOG_INFO("Successfully compiled to object file: %s", output_path);
    return 0;

#else
    (void)mlir_module;
    (void)output_path;
    LOG_ERROR("MLIR support not compiled in");
    return -1;
#endif
}

// ============================================================================
// JIT Cache Management
// ============================================================================

typedef struct CacheEntry {
    uint64_t hash;
    CMLJITKernelFunction kernel;
    size_t size_bytes;
    uint64_t last_used;
    struct CacheEntry* next;
} CacheEntry;

static struct {
    CacheEntry* head;
    size_t max_size;
    size_t current_size;
    size_t hits;
    size_t misses;
    uint64_t access_counter;
} jit_cache = {NULL, 100 * 1024 * 1024, 0, 0, 0, 0}; // 100MB default

/*
static uint64_t hash_ir(CMLIR_t ir) {
    // Simple hash based on node count and types
    uint64_t hash = 0x123456789ABCDEF0ULL;
    struct IRNode* node = ir->head;
    while (node) {
        hash ^= (uint64_t)node->type;
        hash = (hash << 5) | (hash >> 59);
        node = node->next;
    }
    return hash;
}

static CMLJITKernelFunction cache_lookup(uint64_t hash) {
    CacheEntry* entry = jit_cache.head;
    while (entry) {
        if (entry->hash == hash) {
            entry->last_used = ++jit_cache.access_counter;
            jit_cache.hits++;
            return entry->kernel;
        }
        entry = entry->next;
    }
    jit_cache.misses++;
    return NULL;
}
*/

/*
static void cache_insert(uint64_t hash, CMLJITKernelFunction kernel, size_t size) {
    // ...
}

void cml_jit_cache_set_size(size_t max_bytes) {
    jit_cache.max_size = max_bytes;
    LOG_INFO("JIT cache size set to %zu bytes", max_bytes);
}
*/

void cml_jit_cache_stats(size_t* hits, size_t* misses, size_t* size_bytes) {
    if (hits)
        *hits = jit_cache.hits;
    if (misses)
        *misses = jit_cache.misses;
    if (size_bytes)
        *size_bytes = jit_cache.current_size;
}

void cml_jit_cache_clear(void) {
    CacheEntry* entry = jit_cache.head;
    while (entry) {
        CacheEntry* next = entry->next;
        free(entry);
        entry = next;
    }
    jit_cache.head         = NULL;
    jit_cache.current_size = 0;
    jit_cache.hits         = 0;
    jit_cache.misses       = 0;
    LOG_INFO("JIT cache cleared");
}
