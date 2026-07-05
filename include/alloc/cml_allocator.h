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

/* Aligned allocation (alignment must be power of two, >= 16).
 *
 * IMPORTANT: pointers returned by cml_aligned_alloc() MUST be freed with
 * cml_aligned_free(), not cml_free().
 *
 * cml_aligned_alloc() returns an address that may not point at the normal
 * AllocHeader layout used by cml_malloc(); it stores a small delta prefix
 * before the returned pointer so the real backing block can be recovered.
 * Passing that pointer directly to cml_free() will read a bogus header
 * (magic mismatch / wrong class_idx) and either leak the block or corrupt
 * the allocator state.
 *
 * Conversely, cml_malloc()/cml_calloc()/cml_realloc() pointers must be freed
 * with cml_free() (or cml_realloc(ptr, 0)), never cml_aligned_free().
 *
 * Also never mix system malloc/calloc/aligned_alloc/posix_memalign pointers
 * with cml_free() or cml_aligned_free() — those must go through the matching
 * system free() (or cml_malloc/cml_aligned_alloc equivalents end-to-end).
 */
void*  cml_aligned_alloc(size_t size, size_t alignment);
void   cml_aligned_free(void* ptr);

/* Stats (approximate, racy) */
void   cml_allocator_get_stats(size_t* bytes_allocated, size_t* peak_bytes, size_t* alloc_count);

/* Optional: flush this thread's caches back to central (call before thread exit if desired) */
void   cml_allocator_flush_thread_cache(void);

/* Fault injection for simulation / OOM testing.
 *
 * cml_malloc_fault_after(n): the next n successful allocations succeed normally;
 *   the (n+1)-th call returns NULL (simulates OOM).  n=0 → fail immediately.
 *   Call cml_malloc_fault_reset() to disable.
 *
 * cml_malloc_fault_reset(): disable fault injection (default state).
 *
 * cml_malloc_alloc_index(): returns how many allocations have been made since
 *   the last reset — useful to binary-search for the failing site.
 */
void   cml_malloc_fault_after(int n);
void   cml_malloc_fault_reset(void);
long   cml_malloc_alloc_index(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ALLOCATOR_H */
