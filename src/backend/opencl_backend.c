#include "backend/opencl_backend.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

#ifdef CML_HAS_OPENCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>
#endif

static pthread_mutex_t g_opencl_lock;
static bool g_lock_initialized = false;

static inline void opencl_lock(void) {
    if (g_lock_initialized) pthread_mutex_lock(&g_opencl_lock);
}
static inline void opencl_unlock(void) {
    if (g_lock_initialized) pthread_mutex_unlock(&g_opencl_lock);
}

#define GPU_POOL_MAX_ENTRIES 128

typedef struct {
    cl_mem  buffer;
    size_t  size;
    bool    in_use;
} GPUPoolEntry;

typedef struct {
    GPUPoolEntry entries[GPU_POOL_MAX_ENTRIES];
    int          count;
} GPUBufferPool;

static GPUBufferPool g_gpu_pool = { .count = 0 };

static cl_mem gpu_pool_alloc(cl_context ctx, cl_mem_flags flags, size_t size, void* host_ptr,
                             cl_int* errcode) {
    /* Best-fit reuse: find smallest free buffer >= size */
    int best = -1;
    size_t best_size = (size_t)-1;
    for (int i = 0; i < g_gpu_pool.count; i++) {
        GPUPoolEntry* e = &g_gpu_pool.entries[i];
        if (!e->in_use && e->size >= size && e->size < best_size) {
            best = i;
            best_size = e->size;
        }
    }
    if (best >= 0) {
        g_gpu_pool.entries[best].in_use = true;
        cl_mem buf = g_gpu_pool.entries[best].buffer;
        /* If caller supplied host data, write it into the reused buffer */
        if (host_ptr && (flags & CL_MEM_COPY_HOST_PTR)) {
            /* Caller must enqueue a write after this; for simplicity we
               create a fresh buffer when host_ptr is given. */
        } else {
            if (errcode) *errcode = CL_SUCCESS;
            return buf;
        }
    }

    /* No suitable buffer found — allocate a new one */
    cl_int err;
    cl_mem buf = clCreateBuffer(ctx, flags, size, host_ptr, &err);
    if (errcode) *errcode = err;
    if (err != CL_SUCCESS) return NULL;

    /* Register in pool */
    if (g_gpu_pool.count < GPU_POOL_MAX_ENTRIES) {
        GPUPoolEntry* e = &g_gpu_pool.entries[g_gpu_pool.count++];
        e->buffer = buf;
        e->size   = size;
        e->in_use = true;
    }
    return buf;
}

static void gpu_pool_release(cl_mem buf) {
    for (int i = 0; i < g_gpu_pool.count; i++) {
        if (g_gpu_pool.entries[i].buffer == buf) {
            g_gpu_pool.entries[i].in_use = false;
            return;
        }
    }
    /* Buffer not in pool — release it directly */
    clReleaseMemObject(buf);
}

static void gpu_pool_cleanup(void) {
    for (int i = 0; i < g_gpu_pool.count; i++) {
        if (g_gpu_pool.entries[i].buffer) {
            clReleaseMemObject(g_gpu_pool.entries[i].buffer);
            g_gpu_pool.entries[i].buffer = NULL;
        }
    }
    g_gpu_pool.count = 0;
}

static cl_platform_id   g_platform   = NULL;
static cl_device_id     g_device     = NULL;
static cl_context        g_context   = NULL;
static cl_command_queue  g_queue     = NULL;
static cl_program        g_program   = NULL;
static bool              g_initialized = false;

static const char* g_kernels_src =
    "__kernel void cl_add(__global const float* a, __global const float* b,\n"
    "                     __global float* out, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) out[i] = a[i] + b[i];\n"
    "}\n"
    "__kernel void cl_mul(__global const float* a, __global const float* b,\n"
    "                     __global float* out, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) out[i] = a[i] * b[i];\n"
    "}\n"
    "__kernel void cl_relu(__global const float* x, __global float* out, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) out[i] = x[i] > 0.0f ? x[i] : 0.0f;\n"
    "}\n"
    "__kernel void cl_sigmoid(__global const float* x, __global float* out, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) out[i] = 1.0f / (1.0f + exp(-x[i]));\n"
    "}\n"
    "__kernel void cl_matmul(__global const float* a, __global const float* b,\n"
    "                        __global float* out, int m, int n, int k) {\n"
    "    int row = get_global_id(0);\n"
    "    int col = get_global_id(1);\n"
    "    if (row < m && col < n) {\n"
    "        float sum = 0.0f;\n"
    "        for (int i = 0; i < k; i++)\n"
    "            sum += a[row * k + i] * b[i * n + col];\n"
    "        out[row * n + col] = sum;\n"
    "    }\n"
    "}\n"
    "__kernel void cl_sum(__global const float* x, __global float* out, int n) {\n"
    "    float s = 0.0f;\n"
    "    for (int i = 0; i < n; i++) s += x[i];\n"
    "    out[0] = s;\n"
    "}\n";

static cl_kernel g_k_add     = NULL;
static cl_kernel g_k_mul     = NULL;
static cl_kernel g_k_relu    = NULL;
static cl_kernel g_k_sigmoid = NULL;
static cl_kernel g_k_matmul  = NULL;
static cl_kernel g_k_sum     = NULL;

static void opencl_elementwise(cl_kernel kernel, const void* a, const void* b,
                                void* out, size_t n, DType dtype) {
    if (dtype != DTYPE_FLOAT32 || !g_initialized) return;
    (void)b; // b may be NULL for unary ops

    opencl_lock();

    cl_int err;
    size_t byte_size = n * sizeof(float);

    cl_mem buf_a   = gpu_pool_alloc(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     byte_size, (void*)a, &err);
    cl_mem buf_out = gpu_pool_alloc(g_context, CL_MEM_WRITE_ONLY, byte_size, NULL, &err);
    cl_mem buf_b   = NULL;

    if (b) {
        buf_b = gpu_pool_alloc(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                byte_size, (void*)b, &err);
    }

    int n_int = (int)n;
    int arg = 0;
    clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buf_a);
    if (b) clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buf_b);
    clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buf_out);
    clSetKernelArg(kernel, arg++, sizeof(int), &n_int);

    size_t global_size = n;
    clEnqueueNDRangeKernel(g_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, buf_out, CL_TRUE, 0, byte_size, out, 0, NULL, NULL);

    gpu_pool_release(buf_a);
    gpu_pool_release(buf_out);
    if (buf_b) gpu_pool_release(buf_b);

    opencl_unlock();
}

static void opencl_add(const void* a, const void* b, void* out, size_t n, DType dtype) {
    opencl_elementwise(g_k_add, a, b, out, n, dtype);
}

static void opencl_mul(const void* a, const void* b, void* out, size_t n, DType dtype) {
    opencl_elementwise(g_k_mul, a, b, out, n, dtype);
}

static void opencl_relu(const void* x, void* out, size_t n, DType dtype) {
    opencl_elementwise(g_k_relu, x, NULL, out, n, dtype);
}

static void opencl_sigmoid(const void* x, void* out, size_t n, DType dtype) {
    opencl_elementwise(g_k_sigmoid, x, NULL, out, n, dtype);
}

static void opencl_matmul(const void* a, const void* b, void* out, int m, int n, int k,
                           DType dtype) {
    if (dtype != DTYPE_FLOAT32 || !g_initialized) return;

    opencl_lock();

    cl_int err;
    cl_mem buf_a   = gpu_pool_alloc(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     (size_t)(m * k) * sizeof(float), (void*)a, &err);
    cl_mem buf_b   = gpu_pool_alloc(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     (size_t)(k * n) * sizeof(float), (void*)b, &err);
    cl_mem buf_out = gpu_pool_alloc(g_context, CL_MEM_WRITE_ONLY,
                                     (size_t)(m * n) * sizeof(float), NULL, &err);

    clSetKernelArg(g_k_matmul, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(g_k_matmul, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(g_k_matmul, 2, sizeof(cl_mem), &buf_out);
    clSetKernelArg(g_k_matmul, 3, sizeof(int), &m);
    clSetKernelArg(g_k_matmul, 4, sizeof(int), &n);
    clSetKernelArg(g_k_matmul, 5, sizeof(int), &k);

    size_t global_size[2] = {(size_t)m, (size_t)n};
    clEnqueueNDRangeKernel(g_queue, g_k_matmul, 2, NULL, global_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, buf_out, CL_TRUE, 0,
                        (size_t)(m * n) * sizeof(float), out, 0, NULL, NULL);

    gpu_pool_release(buf_a);
    gpu_pool_release(buf_b);
    gpu_pool_release(buf_out);

    opencl_unlock();
}

static void opencl_sum(const void* x, void* out, size_t n, DType dtype) {
    if (dtype != DTYPE_FLOAT32 || !g_initialized) return;

    opencl_lock();

    cl_int err;
    cl_mem buf_x   = gpu_pool_alloc(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     n * sizeof(float), (void*)x, &err);
    cl_mem buf_out = gpu_pool_alloc(g_context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);

    int n_int = (int)n;
    clSetKernelArg(g_k_sum, 0, sizeof(cl_mem), &buf_x);
    clSetKernelArg(g_k_sum, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(g_k_sum, 2, sizeof(int), &n_int);

    size_t global_size = 1;
    clEnqueueNDRangeKernel(g_queue, g_k_sum, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, buf_out, CL_TRUE, 0, sizeof(float), out, 0, NULL, NULL);

    gpu_pool_release(buf_x);
    gpu_pool_release(buf_out);

    opencl_unlock();
}

static void opencl_mean(const void* x, void* out, size_t n, DType dtype) {
    opencl_sum(x, out, n, dtype);
    if (dtype == DTYPE_FLOAT32 && n > 0) {
        ((float*)out)[0] /= (float)n;
    }
}

static void opencl_matmul_add(const void* a, const void* b, const void* bias, void* out,
                               int m, int n, int k, DType dtype) {
    opencl_matmul(a, b, out, m, n, k, dtype);
    if (dtype == DTYPE_FLOAT32 && bias) {
        float* o = (float*)out;
        const float* bv = (const float*)bias;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                o[i * n + j] += bv[j];
    }
}

int opencl_backend_init(void) {
    if (!g_lock_initialized) {
        pthread_mutex_init(&g_opencl_lock, NULL);
        g_lock_initialized = true;
    }

    opencl_lock();

    if (g_initialized) {
        opencl_unlock();
        return 0;
    }

    cl_int err;

    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        LOG_ERROR("OpenCL: no platforms found");
        opencl_unlock();
        return -1;
    }

    cl_platform_id* platforms = malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    g_platform = platforms[0];
    free(platforms);

    err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_ALL, 1, &g_device, NULL);
        if (err != CL_SUCCESS) {
            LOG_ERROR("OpenCL: no devices found");
            opencl_unlock();
            return -1;
        }
    }

    g_context = clCreateContext(NULL, 1, &g_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL: failed to create context");
        opencl_unlock();
        return -1;
    }

#ifdef CL_VERSION_2_0
    g_queue = clCreateCommandQueueWithProperties(g_context, g_device, NULL, &err);
#else
    g_queue = clCreateCommandQueue(g_context, g_device, 0, &err);
#endif
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL: failed to create command queue");
        clReleaseContext(g_context);
        opencl_unlock();
        return -1;
    }

    size_t src_len = strlen(g_kernels_src);
    g_program = clCreateProgramWithSource(g_context, 1, &g_kernels_src, &src_len, &err);
    if (err != CL_SUCCESS) {
        LOG_ERROR("OpenCL: failed to create program");
        opencl_backend_cleanup();
        opencl_unlock();
        return -1;
    }

    err = clBuildProgram(g_program, 1, &g_device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        LOG_ERROR("OpenCL build error: %s", log);
        opencl_backend_cleanup();
        opencl_unlock();
        return -1;
    }

    g_k_add     = clCreateKernel(g_program, "cl_add", &err);
    g_k_mul     = clCreateKernel(g_program, "cl_mul", &err);
    g_k_relu    = clCreateKernel(g_program, "cl_relu", &err);
    g_k_sigmoid = clCreateKernel(g_program, "cl_sigmoid", &err);
    g_k_matmul  = clCreateKernel(g_program, "cl_matmul", &err);
    g_k_sum     = clCreateKernel(g_program, "cl_sum", &err);

    g_initialized = true;
    LOG_INFO("OpenCL backend initialized");

    opencl_unlock();
    return 0;
}

void opencl_backend_cleanup(void) {
    opencl_lock();

    gpu_pool_cleanup();

    if (g_k_add)     clReleaseKernel(g_k_add);
    if (g_k_mul)     clReleaseKernel(g_k_mul);
    if (g_k_relu)    clReleaseKernel(g_k_relu);
    if (g_k_sigmoid) clReleaseKernel(g_k_sigmoid);
    if (g_k_matmul)  clReleaseKernel(g_k_matmul);
    if (g_k_sum)     clReleaseKernel(g_k_sum);
    if (g_program)   clReleaseProgram(g_program);
    if (g_queue)     clReleaseCommandQueue(g_queue);
    if (g_context)   clReleaseContext(g_context);

    g_k_add = g_k_mul = g_k_relu = g_k_sigmoid = g_k_matmul = g_k_sum = NULL;
    g_program = NULL;
    g_queue = NULL;
    g_context = NULL;
    g_initialized = false;

    opencl_unlock();

    if (g_lock_initialized) {
        pthread_mutex_destroy(&g_opencl_lock);
        g_lock_initialized = false;
    }
}

bool opencl_backend_is_available(void) {
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    return (err == CL_SUCCESS && num_platforms > 0);
}

BackendOps opencl_backend_get_ops(void) {
    BackendOps ops = {0};
    ops.matmul     = opencl_matmul;
    ops.matmul_add = opencl_matmul_add;
    ops.add        = opencl_add;
    ops.mul        = opencl_mul;
    ops.relu       = opencl_relu;
    ops.sigmoid    = opencl_sigmoid;
    ops.sum        = opencl_sum;
    ops.mean       = opencl_mean;
    return ops;
}

int opencl_backend_get_device_info(char* buffer, size_t buffer_size) {
    opencl_lock();

    if (!g_initialized || !g_device) {
        opencl_unlock();
        return -1;
    }

    char name[256] = {0};
    char vendor[256] = {0};
    cl_ulong mem_size = 0;

    clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(g_device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    clGetDeviceInfo(g_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);

    snprintf(buffer, buffer_size, "OpenCL Device: %s (%s), Memory: %lu MB",
             name, vendor, (unsigned long)(mem_size / (1024 * 1024)));

    opencl_unlock();
    return 0;
}

#else /* !CML_HAS_OPENCL */

int opencl_backend_init(void) {
    LOG_ERROR("OpenCL backend not compiled. Rebuild with -DENABLE_OPENCL=ON");
    return -1;
}

void opencl_backend_cleanup(void) {}

bool opencl_backend_is_available(void) {
    return false;
}

BackendOps opencl_backend_get_ops(void) {
    BackendOps ops = {0};
    return ops;
}

int opencl_backend_get_device_info(char* buffer, size_t buffer_size) {
    if (buffer && buffer_size > 0) buffer[0] = '\0';
    return -1;
}

#endif /* CML_HAS_OPENCL */
