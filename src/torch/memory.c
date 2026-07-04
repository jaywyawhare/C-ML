/*
 * memory.c — User-provided memory arenas for edge deployment
 */

#include "torch/memory.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

#define TORCH_MEMORY_ALIGN 64

static size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

TorchMemoryOptions torch_memory_default_options(void) {
    TorchMemoryOptions opts = {.alignment = TORCH_MEMORY_ALIGN, .track_peak = true};
    return opts;
}

TorchMemoryManager* torch_memory_create(size_t size) {
    if (size == 0)
        return NULL;

    TorchMemoryManager* mgr = calloc(1, sizeof(TorchMemoryManager));
    if (!mgr)
        return NULL;

    size_t aligned_size = align_up(size, TORCH_MEMORY_ALIGN);
    mgr->arena_buffer = aligned_alloc(TORCH_MEMORY_ALIGN, aligned_size);
    if (!mgr->arena_buffer) {
        free(mgr);
        return NULL;
    }

    mgr->arena_size   = aligned_size;
    mgr->bump_ptr     = (uint8_t*)mgr->arena_buffer;
    mgr->owns_buffer  = true;

    CMLContextParams params = {.mem_size = size, .mem_buffer = mgr->arena_buffer, .no_alloc = false};
    mgr->context = cml_context_new(params);
    if (!mgr->context) {
        free(mgr->arena_buffer);
        free(mgr);
        return NULL;
    }
    mgr->graph_allocator = cml_graph_allocator_new(cml_backend_buffer_type_for_device(DEVICE_CPU));
    if (!mgr->graph_allocator) {
        cml_context_free(mgr->context);
        free(mgr->arena_buffer);
        free(mgr);
        return NULL;
    }

    return mgr;
}

TorchMemoryManager* torch_memory_from_buffer(void* buffer, size_t size) {
    if (!buffer || size == 0)
        return NULL;

    TorchMemoryManager* mgr = calloc(1, sizeof(TorchMemoryManager));
    if (!mgr)
        return NULL;

    mgr->arena_buffer  = buffer;
    mgr->arena_size    = size;
    mgr->bump_ptr      = (uint8_t*)buffer;
    mgr->owns_buffer   = false;

    CMLContextParams params = {.mem_size = size, .mem_buffer = buffer, .no_alloc = false};
    mgr->context = cml_context_new(params);
    if (!mgr->context) {
        free(mgr);
        return NULL;
    }
    mgr->graph_allocator = cml_graph_allocator_new(cml_backend_buffer_type_for_device(DEVICE_CPU));
    if (!mgr->graph_allocator) {
        cml_context_free(mgr->context);
        free(mgr);
        return NULL;
    }

    return mgr;
}

void torch_memory_free(TorchMemoryManager* mgr) {
    if (!mgr)
        return;
    if (mgr->graph_allocator)
        cml_graph_allocator_free(mgr->graph_allocator);
    if (mgr->context)
        cml_context_free(mgr->context);
    if (mgr->owns_buffer && mgr->arena_buffer)
        free(mgr->arena_buffer);
    free(mgr);
}

void* torch_memory_alloc(TorchMemoryManager* mgr, size_t size) {
    if (!mgr || size == 0)
        return NULL;

    size_t aligned = align_up(size, TORCH_MEMORY_ALIGN);
    if (mgr->used_bytes + aligned > mgr->arena_size)
        return NULL;

    void* ptr = mgr->bump_ptr;
    mgr->bump_ptr += aligned;
    mgr->used_bytes += aligned;
    if (mgr->used_bytes > mgr->peak_bytes)
        mgr->peak_bytes = mgr->used_bytes;
    return ptr;
}

size_t torch_memory_used(const TorchMemoryManager* mgr) {
    return mgr ? mgr->used_bytes : 0;
}

size_t torch_memory_peak(const TorchMemoryManager* mgr) {
    return mgr ? mgr->peak_bytes : 0;
}

size_t torch_memory_remaining(const TorchMemoryManager* mgr) {
    if (!mgr)
        return 0;
    return mgr->arena_size > mgr->used_bytes ? mgr->arena_size - mgr->used_bytes : 0;
}

void torch_memory_reset(TorchMemoryManager* mgr) {
    if (!mgr)
        return;
    mgr->bump_ptr   = (uint8_t*)mgr->arena_buffer;
    mgr->used_bytes = 0;
}

bool torch_memory_reserve_graph(TorchMemoryManager* mgr, CMLGraph_t graph) {
    if (!mgr || !mgr->graph_allocator)
        return false;
    return cml_graph_allocator_reserve(mgr->graph_allocator, graph);
}

CMLContext_t torch_memory_get_context(TorchMemoryManager* mgr) {
    return mgr ? mgr->context : NULL;
}
