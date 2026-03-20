#define _POSIX_C_SOURCE 200809L
#include "backend/threadpool.h"
#include "core/logging.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

typedef struct TaskNode {
    Task task;
    struct TaskNode* next;
    volatile int completed_chunks; // Atomic counter for completed chunks
    int total_chunks;
    bool completed;
} TaskNode;

typedef struct {
    Task task;
    pthread_t thread;
    bool active;
    bool completed;
    size_t id;
    ThreadPool* pool;
} Worker;

struct ThreadPool {
    Worker* workers;
    size_t num_threads;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_cond_t task_cond; // Condition for task availability
    TaskNode* task_queue_head;
    TaskNode* task_queue_tail;
    size_t queue_size;
    size_t active_tasks;
    bool shutdown;
};

static pthread_mutex_t g_pool_lock;
static bool g_pool_lock_initialized = false;

static inline void pool_lock(void) {
    if (g_pool_lock_initialized) pthread_mutex_lock(&g_pool_lock);
}
static inline void pool_unlock(void) {
    if (g_pool_lock_initialized) pthread_mutex_unlock(&g_pool_lock);
}

static ThreadPool* g_global_pool = NULL;

static void* worker_thread(void* arg) {
    Worker* worker   = (Worker*)arg;
    ThreadPool* pool = worker->pool;

    while (1) {
        pthread_mutex_lock(&pool->mutex);

        while (!pool->shutdown && pool->task_queue_head == NULL) {
            pthread_cond_wait(&pool->task_cond, &pool->mutex);
        }

        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        // Get current task (all workers work on same task)
        TaskNode* task_node = pool->task_queue_head;
        if (task_node && !task_node->completed) {
            Task task = task_node->task;
            pool->active_tasks++;
            pthread_mutex_unlock(&pool->mutex);

            size_t chunk_size = task.total_size / pool->num_threads;
            size_t start      = worker->id * chunk_size;
            size_t end =
                (worker->id == pool->num_threads - 1) ? task.total_size : start + chunk_size;

            task.func(task.data, start, end);

            pthread_mutex_lock(&pool->mutex);
            task_node->completed_chunks++;
            pool->active_tasks--;

            // If all chunks completed, mark task as done and notify waiters
            if (task_node->completed_chunks >= task_node->total_chunks) {
                task_node->completed = true;
                pool->task_queue_head = task_node->next;
                if (!pool->task_queue_head) {
                    pool->task_queue_tail = NULL;
                }
                pool->queue_size--;
                free(task_node);
                pthread_cond_broadcast(&pool->cond);
            }
            pthread_mutex_unlock(&pool->mutex);
        } else {
            pthread_mutex_unlock(&pool->mutex);
        }
    }

    return NULL;
}

ThreadPool* threadpool_create(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = (size_t)sysconf(_SC_NPROCESSORS_ONLN);
        if (num_threads == 0) {
            num_threads = 1;
        }
    }

    ThreadPool* pool = malloc(sizeof(ThreadPool));
    if (!pool) {
        LOG_ERROR("Failed to allocate thread pool");
        return NULL;
    }

    pool->num_threads = num_threads;
    pool->workers     = calloc(num_threads, sizeof(Worker));
    if (!pool->workers) {
        free(pool);
        return NULL;
    }

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond, NULL);
    pthread_cond_init(&pool->task_cond, NULL);
    pool->task_queue_head = NULL;
    pool->task_queue_tail = NULL;
    pool->queue_size      = 0;
    pool->active_tasks    = 0;
    pool->shutdown        = false;

    for (size_t i = 0; i < num_threads; i++) {
        pool->workers[i].pool      = pool;
        pool->workers[i].id        = i;
        pool->workers[i].active    = false;
        pool->workers[i].completed = false;
        pthread_create(&pool->workers[i].thread, NULL, worker_thread, &pool->workers[i]);
    }

    LOG_DEBUG("Created thread pool with %zu threads", num_threads);
    return pool;
}

void threadpool_destroy(ThreadPool* pool) {
    if (!pool) {
        return;
    }

    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);

    for (size_t i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->workers[i].thread, NULL);
    }

    TaskNode* node = pool->task_queue_head;
    while (node) {
        TaskNode* next = node->next;
        free(node);
        node = next;
    }

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond);
    pthread_cond_destroy(&pool->task_cond);
    free(pool->workers);
    free(pool);
}

int threadpool_submit(ThreadPool* pool, Task* task) {
    if (!pool || !task) {
        return -1;
    }

    TaskNode* task_node = malloc(sizeof(TaskNode));
    if (!task_node) {
        LOG_ERROR("Failed to allocate task node");
        return -1;
    }

    task_node->task             = *task;
    task_node->next             = NULL;
    task_node->completed_chunks = 0;
    task_node->total_chunks     = (int)pool->num_threads;
    task_node->completed        = false;

    pthread_mutex_lock(&pool->mutex);

    if (pool->task_queue_tail) {
        pool->task_queue_tail->next = task_node;
    } else {
        pool->task_queue_head = task_node;
    }
    pool->task_queue_tail = task_node;
    pool->queue_size++;

    pthread_cond_broadcast(&pool->task_cond);
    pthread_mutex_unlock(&pool->mutex);

    return 0;
}

void threadpool_wait(ThreadPool* pool) {
    if (!pool) {
        return;
    }

    pthread_mutex_lock(&pool->mutex);

    while (pool->queue_size > 0 || pool->active_tasks > 0) {
        pthread_cond_wait(&pool->cond, &pool->mutex);
    }

    pthread_mutex_unlock(&pool->mutex);
}

size_t threadpool_get_num_threads(ThreadPool* pool) { return pool ? pool->num_threads : 0; }

ThreadPool* threadpool_get_global(void) {
    if (!g_pool_lock_initialized) {
        pthread_mutex_init(&g_pool_lock, NULL);
        g_pool_lock_initialized = true;
    }
    pool_lock();
    if (!g_global_pool) {
        g_global_pool = threadpool_create(0);
    }
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
    if (g_global_pool && g_global_pool != pool) {
        threadpool_destroy(g_global_pool);
    }
    g_global_pool = pool;
    pool_unlock();
}

void threadpool_parallel_for(ThreadPool* pool, TaskFunc func, void* data, size_t n) {
    if (!pool) {
        pool = threadpool_get_global();
    }

    if (pool->num_threads == 1 || n < 1000) {
        func(data, 0, n);
        return;
    }

    for (size_t i = 0; i < pool->num_threads; i++) {
        Task task               = {.func = func, .data = data, .total_size = n};
        pool->workers[i].task   = task;
        pool->workers[i].active = true;
    }

    pthread_cond_broadcast(&pool->cond);

    pthread_mutex_lock(&pool->mutex);
    bool all_completed = false;
    while (!all_completed) {
        all_completed = true;
        for (size_t i = 0; i < pool->num_threads; i++) {
            if (pool->workers[i].active && !pool->workers[i].completed) {
                all_completed = false;
                break;
            }
        }

        if (!all_completed) {
            pthread_cond_wait(&pool->cond, &pool->mutex);
        }
    }
    pthread_mutex_unlock(&pool->mutex);
}
