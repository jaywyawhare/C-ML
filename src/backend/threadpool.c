#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#include "backend/threadpool.h"
#include "core/logging.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdatomic.h>
#include <stdint.h>
#include "alloc/cml_allocator.h"

/*
 * Fork/join thread pool built around a single "generation" broadcast plus
 * atomic chunk claiming. The previous implementation had two independent,
 * mutually-inconsistent code paths (a task queue the workers serviced, and a
 * parallel_for that signalled a condition the workers never waited on) — its
 * fork path deadlocked and its queue path could double-process a chunk. It only
 * ever appeared to work because callers fell back to running serially.
 *
 * Contract (relied on by simd_sum_f32_parallel's slot math): parallel_for splits
 * [0,n) into exactly `num_threads` contiguous chunks of size
 * chunk = ceil(n / num_threads); chunk i covers [i*chunk, min(n,(i+1)*chunk)).
 * So a chunk's start is always i*chunk, and start/chunk recovers i.
 *
 * A claim carries the generation it belongs to. When a batch finished, a worker
 * did not leave `drain_chunks` at the same instant the submitter stopped waiting
 * -- it kept probing the claim counter. Since the counters were reset in place,
 * those stragglers consumed the *next* batch's chunk indices and counted them
 * against the old batch, so the next batch was claimed but never completed:
 * every worker slept while the submitter waited forever on a batch whose
 * done_chunks stayed 0 (observed: next_chunk=26, done_chunks=0, num_chunks=12).
 * Packing the generation into the claim word makes a straggler's claim fail its
 * generation test and stop, so it cannot consume work it will not account for.
 */

/* claim word: high 32 bits = generation, low 32 bits = next chunk index */
#define CLAIM_MAKE(gen, idx) (((uint64_t)(uint32_t)(gen) << 32) | (uint32_t)(idx))
#define CLAIM_GEN(c)         ((uint32_t)((c) >> 32))
#define CLAIM_IDX(c)         ((uint32_t)((c) & 0xffffffffu))

struct ThreadPool {
    pthread_t*      threads;
    size_t          num_threads;
    pthread_mutex_t mutex;
    pthread_cond_t  work_ready;   /* workers wait here for a new batch     */
    pthread_cond_t  work_done;    /* the submitter waits here for the batch */

    /* Current batch (published under mutex, then read lock-free). */
    TaskFunc          func;
    void*             data;
    size_t            total;      /* number of items                        */
    size_t            chunk;      /* ceil(total / num_threads)              */
    size_t            num_chunks; /* == num_threads for a live batch        */
    _Atomic uint64_t  claim;      /* (generation << 32) | next chunk index  */
    /* (generation << 32) | completed-chunk count for THAT generation. Packing
     * matters: a straggler finishing a chunk of batch N while batch N+1 is
     * live must not count toward N+1 — an unpacked counter let stale
     * completions release the submitter before every chunk had run
     * (use-after-free / hang). */
    _Atomic uint64_t  done;       /* (generation << 32) | chunks finished   */
    uint64_t          generation; /* bumped per batch; workers track last   */
    bool              shutdown;
};

static pthread_mutex_t g_pool_lock;
static bool            g_pool_lock_initialized = false;
static ThreadPool*     g_global_pool           = NULL;

static inline void pool_lock(void)   { if (g_pool_lock_initialized) pthread_mutex_lock(&g_pool_lock); }
static inline void pool_unlock(void) { if (g_pool_lock_initialized) pthread_mutex_unlock(&g_pool_lock); }

/* A thread's private copy of the batch descriptor, taken while it holds the
 * mutex. The pool's own fields are recycled by the next submit, and a thread can
 * still be inside a chunk when that happens, so reading them lock-free is a data
 * race on every one of them. Copying once per batch costs a few words. */
typedef struct {
    TaskFunc func;
    void*    data;
    size_t   total;
    size_t   chunk;
    size_t   num_chunks;
    uint32_t gen;
} Batch;

/* Snapshot the live batch. Caller must hold pool->mutex. */
static Batch batch_snapshot(const ThreadPool* pool) {
    Batch b;
    b.func       = pool->func;
    b.data       = pool->data;
    b.total      = pool->total;
    b.chunk      = pool->chunk;
    b.num_chunks = pool->num_chunks;
    b.gen        = (uint32_t)pool->generation;
    return b;
}

/* Run one chunk index against `b` (no-op for empty tail chunks), then record
 * the completion generation-tagged. A straggler whose batch has moved on finds
 * a done-word with a different generation and drops the increment — it must
 * not release a submitter waiting on a newer batch. */
static void run_chunk(ThreadPool* pool, const Batch* b, size_t c) {
    size_t start = c * b->chunk;
    size_t end   = start + b->chunk;
    if (end > b->total) end = b->total;
    if (start < end)
        b->func(b->data, start, end);

    for (;;) {
        uint64_t cur = atomic_load(&pool->done);
        if (CLAIM_GEN(cur) != b->gen)
            return; /* batch already superseded; our result was accounted for */
        if (atomic_compare_exchange_weak(&pool->done, &cur,
                                         CLAIM_MAKE(b->gen, CLAIM_IDX(cur) + 1)))
            break;
    }
    if (CLAIM_IDX(atomic_load(&pool->done)) == b->num_chunks) {
        pthread_mutex_lock(&pool->mutex);
        pthread_cond_signal(&pool->work_done);
        pthread_mutex_unlock(&pool->mutex);
    }
}

/* Claim and run chunks of batch `b` until it is exhausted.
 *
 * The generation test is what keeps a straggler from eating the next batch's
 * work: once the pool has moved on, this thread's claims no longer match and it
 * leaves without consuming an index. Every index it does claim is one it runs
 * and accounts for, so done_chunks always reaches num_chunks. */
static void drain_chunks(ThreadPool* pool, const Batch* b) {
    uint64_t cur = atomic_load(&pool->claim);
    for (;;) {
        if (CLAIM_GEN(cur) != b->gen) return;           /* batch moved on */
        uint32_t idx = CLAIM_IDX(cur);
        if (idx >= (uint32_t)b->num_chunks) return;     /* batch exhausted */
        if (atomic_compare_exchange_weak(&pool->claim, &cur,
                                         CLAIM_MAKE(b->gen, idx + 1))) {
            run_chunk(pool, b, idx);
            cur = atomic_load(&pool->claim);
        }
        /* CAS failure refreshes `cur`; retry against the new value. */
    }
}

static void* worker_thread(void* arg) {
    ThreadPool* pool = (ThreadPool*)arg;
    uint64_t last_gen = 0;
    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (!pool->shutdown && pool->generation == last_gen)
            pthread_cond_wait(&pool->work_ready, &pool->mutex);
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        /* Advance one batch at a time rather than jumping to the current
         * generation. A batch cannot complete without this worker's chunk, so
         * the submitter cannot publish the next one until we have run this one;
         * stepping by one makes that invariant explicit. */
        last_gen++;
        Batch b = batch_snapshot(pool);
        b.gen   = (uint32_t)last_gen;
        pthread_mutex_unlock(&pool->mutex);

        drain_chunks(pool, &b);
    }
    return NULL;
}

ThreadPool* threadpool_create(size_t num_threads) {
    if (num_threads == 0) {
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        num_threads = si.dwNumberOfProcessors > 0 ? (size_t)si.dwNumberOfProcessors : 1;
#else
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        num_threads = (n > 0) ? (size_t)n : 1;
#endif
    }

    ThreadPool* pool = cml_calloc(1, sizeof(ThreadPool));
    if (!pool) {
        LOG_ERROR("Failed to allocate thread pool");
        return NULL;
    }
    pool->num_threads = num_threads;
    pool->generation  = 0;
    pool->shutdown    = false;
    atomic_store(&pool->claim, CLAIM_MAKE(0, 0));
    atomic_store(&pool->done, CLAIM_MAKE(0, 0));
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->work_ready, NULL);
    pthread_cond_init(&pool->work_done, NULL);

    /* One fewer OS thread than num_threads: the submitting thread participates
     * in every batch, so `num_threads` total workers execute chunks. */
    size_t spawn = num_threads > 0 ? num_threads - 1 : 0;
    pool->threads = spawn ? cml_calloc(spawn, sizeof(pthread_t)) : NULL;
    if (spawn && !pool->threads) {
        cml_free(pool);
        return NULL;
    }
    for (size_t i = 0; i < spawn; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_thread, pool) != 0) {
            /* Shrink to the threads we actually created; still correct. */
            pool->num_threads = i + 1;  /* +1 for the submitter */
            break;
        }
    }
    LOG_DEBUG("Created thread pool with %zu workers", pool->num_threads);
    return pool;
}

void threadpool_destroy(ThreadPool* pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);

    size_t spawn = pool->num_threads > 0 ? pool->num_threads - 1 : 0;
    for (size_t i = 0; i < spawn && pool->threads; i++)
        pthread_join(pool->threads[i], NULL);

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->work_ready);
    pthread_cond_destroy(&pool->work_done);
    cml_free(pool->threads);
    cml_free(pool);
}

void threadpool_parallel_for(ThreadPool* pool, TaskFunc func, void* data, size_t n) {
    if (!pool) pool = threadpool_get_global();
    if (n == 0 || !func) return;
    if (!pool || pool->num_threads <= 1) {
        func(data, 0, n);   /* single chunk == slot 0; matches sum's slot math */
        return;
    }

    pthread_mutex_lock(&pool->mutex);
    pool->func       = func;
    pool->data       = data;
    pool->total      = n;
    pool->chunk      = (n + pool->num_threads - 1) / pool->num_threads;  /* ceil */
    pool->num_chunks = pool->num_threads;
    pool->generation++;
    atomic_store(&pool->done, CLAIM_MAKE((uint32_t)pool->generation, 0));
    atomic_store(&pool->claim, CLAIM_MAKE(pool->generation, 0));
    Batch b = batch_snapshot(pool);
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);

    /* The submitting thread is one of the workers. */
    drain_chunks(pool, &b);

    /* Wait until every chunk of THIS generation has completed. The done word
     * is generation-tagged, so stragglers from older batches cannot release
     * us early (and our own wait cannot be satisfied by their work). */
    pthread_mutex_lock(&pool->mutex);
    while (CLAIM_GEN(atomic_load(&pool->done)) != (uint32_t)pool->generation ||
           CLAIM_IDX(atomic_load(&pool->done)) < pool->num_chunks)
        pthread_cond_wait(&pool->work_done, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

/* Legacy API kept for source compatibility (no current external caller). submit
 * runs the task synchronously via the parallel machinery; wait is then a no-op. */
int threadpool_submit(ThreadPool* pool, Task* task) {
    if (!task) return -1;
    threadpool_parallel_for(pool, task->func, task->data, task->total_size);
    return 0;
}
void threadpool_wait(ThreadPool* pool) { (void)pool; }

size_t threadpool_get_num_threads(ThreadPool* pool) { return pool ? pool->num_threads : 0; }

ThreadPool* threadpool_get_global(void) {
    if (!g_pool_lock_initialized) {
        pthread_mutex_init(&g_pool_lock, NULL);
        g_pool_lock_initialized = true;
    }
    pool_lock();
    if (!g_global_pool)
        g_global_pool = threadpool_create(0);
    ThreadPool* result = g_global_pool;
    pool_unlock();
    return result;
}

void threadpool_set_global(ThreadPool* pool) {
    if (!g_pool_lock_initialized) {
        pthread_mutex_init(&g_pool_lock, NULL);
        g_pool_lock_initialized = true;
    }
    pool_lock();
    if (g_global_pool && g_global_pool != pool)
        threadpool_destroy(g_global_pool);
    g_global_pool = pool;
    pool_unlock();
}
