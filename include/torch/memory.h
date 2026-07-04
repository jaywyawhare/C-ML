/*
 * memory.h — User-provided memory arenas for embedded / edge deployment
 *
 * ExecuTorch MemoryManager analogue: all runtime allocations come from a
 * caller-supplied buffer, enabling placement in SRAM, DRAM, or mmap'd PTE segments.
 */

#ifndef CML_TORCH_MEMORY_H
#define CML_TORCH_MEMORY_H

#include "core/export.h"
#include "alloc/graph_allocator.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

typedef struct TorchMemoryManager {
    void*              arena_buffer;
    size_t             arena_size;
    size_t             used_bytes;
    size_t             peak_bytes;
    CMLContext_t       context;
    CMLGraphAllocator_t graph_allocator;
    bool               owns_buffer;
    uint8_t*           bump_ptr;
} TorchMemoryManager;

typedef struct TorchMemoryOptions {
    size_t alignment; /* default 64 */
    bool   track_peak;
} TorchMemoryOptions;

CML_API TorchMemoryOptions torch_memory_default_options(void);

/* Allocate an owned arena of `size` bytes. */
CML_API TorchMemoryManager* torch_memory_create(size_t size);

/* Wrap a caller-supplied buffer (not freed on destroy). */
CML_API TorchMemoryManager* torch_memory_from_buffer(void* buffer, size_t size);

CML_API void   torch_memory_free(TorchMemoryManager* mgr);

/* Bump-allocate `size` bytes from the arena.
 * Returns NULL when mgr is NULL, size is 0, or the arena is exhausted
 * (used + aligned(size) > arena_size). Does not grow the arena. */
CML_API void*  torch_memory_alloc(TorchMemoryManager* mgr, size_t size);

CML_API size_t torch_memory_used(const TorchMemoryManager* mgr);
CML_API size_t torch_memory_peak(const TorchMemoryManager* mgr);
CML_API size_t torch_memory_remaining(const TorchMemoryManager* mgr);

/* Rewind the bump pointer to the start of the arena.
 * Pointers previously returned by torch_memory_alloc() are invalidated and
 * must not be dereferenced after this call. peak_bytes is preserved. */
CML_API void   torch_memory_reset(TorchMemoryManager* mgr);

/* Reserve graph buffers from a worst-case IR graph. */
CML_API bool torch_memory_reserve_graph(TorchMemoryManager* mgr, CMLGraph_t graph);

CML_API CMLContext_t torch_memory_get_context(TorchMemoryManager* mgr);

#ifdef __cplusplus
}
#endif

#endif /* CML_TORCH_MEMORY_H */
