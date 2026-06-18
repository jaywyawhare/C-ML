#ifndef CML_ALLOCATOR_H
#define CML_ALLOCATOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Ultra fast custom allocator replacing malloc/calloc/realloc/free/strdup.
 * Design: thread-local caches + size-class segregated freelists + slab carving.
 * Small/medium allocations are O(1) with almost no lock contention on fast path.
 * Large allocations fall back to mmap-backed direct paths (zero syscall amortization).
 * Allocations are at least 16-byte aligned (64-byte for >=64B requests when possible).
 */

/* Core replacements */
void*  cml_malloc(size_t size);
void*  cml_calloc(size_t nmemb, size_t size);
void*  cml_realloc(void* ptr, size_t new_size);
void   cml_free(void* ptr);

/* String helper */
char*  cml_strdup(const char* s);

/* Aligned allocation (alignment must be power of two, >= 16) */
void*  cml_aligned_alloc(size_t size, size_t alignment);
void   cml_aligned_free(void* ptr); /* safe to mix with cml_free in most cases */

/* Stats (approximate, racy) */
void   cml_allocator_get_stats(size_t* bytes_allocated, size_t* peak_bytes, size_t* alloc_count);

/* Optional: flush this thread's caches back to central (call before thread exit if desired) */
void   cml_allocator_flush_thread_cache(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ALLOCATOR_H */
