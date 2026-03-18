#ifndef CML_CORE_THREADPOOL_H
#define CML_CORE_THREADPOOL_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ThreadPool ThreadPool;

typedef void (*TaskFunc)(void* data, size_t start, size_t end);

typedef struct {
    TaskFunc func;
    void* data;
    size_t total_size;
} Task;

/* @param num_threads Number of threads (0 for auto-detect) */
ThreadPool* threadpool_create(size_t num_threads);
void threadpool_destroy(ThreadPool* pool);
int threadpool_submit(ThreadPool* pool, Task* task);
void threadpool_wait(ThreadPool* pool);
size_t threadpool_get_num_threads(ThreadPool* pool);
ThreadPool* threadpool_get_global(void);
void threadpool_set_global(ThreadPool* pool);

/* @param pool Thread pool (NULL for global) */
void threadpool_parallel_for(ThreadPool* pool, TaskFunc func, void* data, size_t n);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_THREADPOOL_H
