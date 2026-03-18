#include "ops/ir/gpu/adreno_backend.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>

#define CL_SUCCESS                  0
#define CL_DEVICE_TYPE_GPU          (1 << 2)
#define CL_PLATFORM_NAME            0x0902
#define CL_DEVICE_NAME              0x102B
#define CL_DEVICE_VENDOR            0x102C
#define CL_DEVICE_VERSION           0x102F
#define CL_DEVICE_GLOBAL_MEM_SIZE   0x101F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_PROGRAM_BUILD_LOG        0x1183

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_device_type;
typedef uint32_t cl_platform_info;
typedef uint32_t cl_device_info;
typedef uint32_t cl_program_build_info;

typedef cl_int (*clGetPlatformIDs_fn)(cl_uint, void**, cl_uint*);
typedef cl_int (*clGetDeviceIDs_fn)(void*, cl_device_type, cl_uint, void**, cl_uint*);
typedef cl_int (*clGetPlatformInfo_fn)(void*, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (*clGetDeviceInfo_fn)(void*, cl_device_info, size_t, void*, size_t*);
typedef void*  (*clCreateContext_fn)(void*, cl_uint, void**, void*, void*, cl_int*);
typedef void*  (*clCreateCommandQueue_fn)(void*, void*, uint64_t, cl_int*);
typedef void*  (*clCreateProgramWithSource_fn)(void*, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*clBuildProgram_fn)(void*, cl_uint, void**, const char*, void*, void*);
typedef void*  (*clCreateKernel_fn)(void*, const char*, cl_int*);
typedef cl_int (*clSetKernelArg_fn)(void*, cl_uint, size_t, const void*);
typedef cl_int (*clEnqueueNDRangeKernel_fn)(void*, void*, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, void*, void*);
typedef cl_int (*clFinish_fn)(void*);
typedef cl_int (*clReleaseKernel_fn)(void*);
typedef cl_int (*clReleaseProgram_fn)(void*);
typedef cl_int (*clReleaseCommandQueue_fn)(void*);
typedef cl_int (*clReleaseContext_fn)(void*);
typedef void*  (*clCreateBuffer_fn)(void*, uint64_t, size_t, void*, cl_int*);
typedef cl_int (*clEnqueueReadBuffer_fn)(void*, void*, cl_uint, size_t, size_t, void*, cl_uint, void*, void*);
typedef cl_int (*clEnqueueWriteBuffer_fn)(void*, void*, cl_uint, size_t, size_t, const void*, cl_uint, void*, void*);
typedef cl_int (*clReleaseMemObject_fn)(void*);
typedef cl_int (*clGetProgramBuildInfo_fn)(void*, void*, cl_program_build_info, size_t, void*, size_t*);

static void* s_cl_lib = NULL;

static clGetPlatformIDs_fn          fn_clGetPlatformIDs          = NULL;
static clGetDeviceIDs_fn            fn_clGetDeviceIDs            = NULL;
static clGetPlatformInfo_fn         fn_clGetPlatformInfo         = NULL;
static clGetDeviceInfo_fn           fn_clGetDeviceInfo           = NULL;
static clCreateContext_fn           fn_clCreateContext            = NULL;
static clCreateCommandQueue_fn      fn_clCreateCommandQueue      = NULL;
static clCreateProgramWithSource_fn fn_clCreateProgramWithSource = NULL;
static clBuildProgram_fn            fn_clBuildProgram            = NULL;
static clCreateKernel_fn            fn_clCreateKernel            = NULL;
static clSetKernelArg_fn            fn_clSetKernelArg            = NULL;
static clEnqueueNDRangeKernel_fn    fn_clEnqueueNDRangeKernel    = NULL;
static clFinish_fn                  fn_clFinish                  = NULL;
static clReleaseKernel_fn           fn_clReleaseKernel           = NULL;
static clReleaseProgram_fn          fn_clReleaseProgram          = NULL;
static clReleaseCommandQueue_fn     fn_clReleaseCommandQueue     = NULL;
static clReleaseContext_fn          fn_clReleaseContext           = NULL;
static clCreateBuffer_fn            fn_clCreateBuffer            = NULL;
static clEnqueueReadBuffer_fn       fn_clEnqueueReadBuffer       = NULL;
static clEnqueueWriteBuffer_fn      fn_clEnqueueWriteBuffer      = NULL;
static clReleaseMemObject_fn        fn_clReleaseMemObject        = NULL;
static clGetProgramBuildInfo_fn     fn_clGetProgramBuildInfo     = NULL;

static char s_device_info_buf[512];

static void* open_opencl_lib(void) {
    void* h = dlopen("libOpenCL.so", RTLD_LAZY);
    if (!h) {
        h = dlopen("libOpenCL.so.1", RTLD_LAZY);
    }
    return h;
}

#define LOAD_CL_SYM(name) do { \
    fn_##name = (name##_fn)dlsym(lib, #name); \
    if (!fn_##name) { \
        LOG_ERROR("Adreno: failed to load %s: %s", #name, dlerror()); \
        return -1; \
    } \
} while (0)

static int load_opencl_symbols(void* lib) {
    LOAD_CL_SYM(clGetPlatformIDs);
    LOAD_CL_SYM(clGetDeviceIDs);
    LOAD_CL_SYM(clGetPlatformInfo);
    LOAD_CL_SYM(clGetDeviceInfo);
    LOAD_CL_SYM(clCreateContext);
    LOAD_CL_SYM(clCreateCommandQueue);
    LOAD_CL_SYM(clCreateProgramWithSource);
    LOAD_CL_SYM(clBuildProgram);
    LOAD_CL_SYM(clCreateKernel);
    LOAD_CL_SYM(clSetKernelArg);
    LOAD_CL_SYM(clEnqueueNDRangeKernel);
    LOAD_CL_SYM(clFinish);
    LOAD_CL_SYM(clReleaseKernel);
    LOAD_CL_SYM(clReleaseProgram);
    LOAD_CL_SYM(clReleaseCommandQueue);
    LOAD_CL_SYM(clReleaseContext);
    LOAD_CL_SYM(clCreateBuffer);
    LOAD_CL_SYM(clEnqueueReadBuffer);
    LOAD_CL_SYM(clEnqueueWriteBuffer);
    LOAD_CL_SYM(clReleaseMemObject);
    LOAD_CL_SYM(clGetProgramBuildInfo);
    return 0;
}

static bool find_adreno_device(void** out_platform, void** out_device) {
    cl_uint num_platforms = 0;
    if (fn_clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        return false;
    }

    void** platforms = (void**)calloc(num_platforms, sizeof(void*));
    if (!platforms) return false;

    if (fn_clGetPlatformIDs(num_platforms, platforms, NULL) != CL_SUCCESS) {
        free(platforms);
        return false;
    }

    for (cl_uint i = 0; i < num_platforms; i++) {
        char name[256] = {0};
        fn_clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);

        cl_uint num_devices = 0;
        if (fn_clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices) != CL_SUCCESS
            || num_devices == 0) {
            continue;
        }

        void** devices = (void**)calloc(num_devices, sizeof(void*));
        if (!devices) continue;

        if (fn_clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL) != CL_SUCCESS) {
            free(devices);
            continue;
        }

        for (cl_uint j = 0; j < num_devices; j++) {
            char dev_name[256] = {0};
            fn_clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);

            if (strstr(dev_name, "Adreno") || strstr(name, "Qualcomm") || strstr(name, "QUALCOMM")) {
                *out_platform = platforms[i];
                *out_device = devices[j];
                free(devices);
                free(platforms);
                return true;
            }
        }
        free(devices);
    }

    free(platforms);
    return false;
}

bool cml_adreno_available(void) {
    void* lib = open_opencl_lib();
    if (!lib) return false;

    clGetPlatformIDs_fn get_plat = (clGetPlatformIDs_fn)dlsym(lib, "clGetPlatformIDs");
    clGetDeviceIDs_fn get_dev = (clGetDeviceIDs_fn)dlsym(lib, "clGetDeviceIDs");
    clGetPlatformInfo_fn get_plat_info = (clGetPlatformInfo_fn)dlsym(lib, "clGetPlatformInfo");
    clGetDeviceInfo_fn get_dev_info = (clGetDeviceInfo_fn)dlsym(lib, "clGetDeviceInfo");

    if (!get_plat || !get_dev || !get_plat_info || !get_dev_info) {
        dlclose(lib);
        return false;
    }

    cl_uint num_platforms = 0;
    if (get_plat(0, NULL, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        dlclose(lib);
        return false;
    }

    void** platforms = (void**)calloc(num_platforms, sizeof(void*));
    if (!platforms) { dlclose(lib); return false; }

    bool found = false;
    if (get_plat(num_platforms, platforms, NULL) == CL_SUCCESS) {
        for (cl_uint i = 0; i < num_platforms && !found; i++) {
            char name[256] = {0};
            get_plat_info(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);

            cl_uint num_devices = 0;
            if (get_dev(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices) != CL_SUCCESS
                || num_devices == 0) {
                continue;
            }

            void** devices = (void**)calloc(num_devices, sizeof(void*));
            if (!devices) continue;

            if (get_dev(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL) == CL_SUCCESS) {
                for (cl_uint j = 0; j < num_devices; j++) {
                    char dev_name[256] = {0};
                    get_dev_info(devices[j], CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
                    if (strstr(dev_name, "Adreno") || strstr(name, "Qualcomm") || strstr(name, "QUALCOMM")) {
                        found = true;
                        break;
                    }
                }
            }
            free(devices);
        }
    }

    free(platforms);
    dlclose(lib);
    return found;
}

CMLAdrenoBackend* cml_adreno_backend_create(void) {
    CMLAdrenoBackend* b = (CMLAdrenoBackend*)calloc(1, sizeof(CMLAdrenoBackend));
    if (!b) {
        LOG_ERROR("Failed to allocate CMLAdrenoBackend");
    }
    return b;
}

int cml_adreno_backend_init(CMLAdrenoBackend* backend) {
    if (!backend) return -1;

    if (backend->initialized) {
        LOG_WARNING("Adreno backend already initialized");
        return 0;
    }

    void* lib = open_opencl_lib();
    if (!lib) {
        LOG_ERROR("Adreno: failed to open libOpenCL.so");
        return -1;
    }

    if (load_opencl_symbols(lib) != 0) {
        dlclose(lib);
        return -1;
    }

    s_cl_lib = lib;

    void* platform = NULL;
    void* device = NULL;
    if (!find_adreno_device(&platform, &device)) {
        LOG_ERROR("Adreno: no Adreno GPU device found");
        goto fail;
    }

    cl_ulong global_mem = 0;
    cl_ulong max_alloc = 0;
    size_t max_wg = 0;
    cl_uint compute_units = 0;
    char dev_version[128] = {0};

    fn_clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    fn_clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
    fn_clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    fn_clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    fn_clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(dev_version), dev_version, NULL);

    int gpu_ver = 0;
    const char* adreno_pos = strstr(dev_version, "Adreno");
    if (adreno_pos) {
        const char* p = adreno_pos;
        while (*p && (*p < '0' || *p > '9')) p++;
        if (*p) gpu_ver = atoi(p);
    }

    cl_int err = 0;
    void* context = fn_clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !context) {
        LOG_ERROR("Adreno: clCreateContext failed (err=%d)", err);
        goto fail;
    }

    void* queue = fn_clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS || !queue) {
        LOG_ERROR("Adreno: clCreateCommandQueue failed (err=%d)", err);
        fn_clReleaseContext(context);
        goto fail;
    }

    backend->gpu_version = gpu_ver;
    backend->global_mem_size = (size_t)global_mem;
    backend->max_alloc_size = (size_t)max_alloc;
    backend->max_work_group_size = (int)max_wg;
    backend->compute_units = (int)compute_units;
    backend->cl_context = context;
    backend->cl_queue = queue;
    backend->initialized = true;

    LOG_INFO("Adreno backend initialized: GPU %d, %zu MB global mem, %d CUs, max WG %d",
             backend->gpu_version,
             backend->global_mem_size / (1024 * 1024),
             backend->compute_units,
             backend->max_work_group_size);
    return 0;

fail:
    dlclose(lib);
    s_cl_lib = NULL;
    return -1;
}

void cml_adreno_backend_free(CMLAdrenoBackend* backend) {
    if (!backend) return;

    if (backend->cl_queue && fn_clReleaseCommandQueue) {
        fn_clReleaseCommandQueue(backend->cl_queue);
        backend->cl_queue = NULL;
    }
    if (backend->cl_context && fn_clReleaseContext) {
        fn_clReleaseContext(backend->cl_context);
        backend->cl_context = NULL;
    }
    if (s_cl_lib) {
        dlclose(s_cl_lib);
        s_cl_lib = NULL;
    }

    backend->initialized = false;
    free(backend);
}

int cml_adreno_execute(CMLAdrenoBackend* backend, CMLGraph_t ir) {
    if (!backend || !backend->initialized) {
        LOG_ERROR("Adreno backend not initialized");
        return -1;
    }
    if (!ir) {
        LOG_ERROR("Adreno execute: NULL IR graph");
        return -1;
    }

    struct IRNode* node = ir->head;
    int node_idx = 0;
    int status = 0;

    while (node) {
        const char* op_name = uop_type_to_string(node->type);
        LOG_DEBUG("Adreno execute: node %d, op=%s, inputs=%d", node_idx, op_name, node->num_inputs);

        char kernel_src[1024];
        snprintf(kernel_src, sizeof(kernel_src),
                 "__kernel void op_%d(__global float* out, __global const float* in, uint n) {\n"
                 "    uint gid = get_global_id(0);\n"
                 "    if (gid < n) out[gid] = in[gid]; /* placeholder for op %s */\n"
                 "}\n",
                 node_idx, op_name);

        const char* src_ptr = kernel_src;
        size_t src_len = strlen(kernel_src);
        cl_int err = 0;

        void* program = fn_clCreateProgramWithSource(backend->cl_context, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS || !program) {
            LOG_ERROR("Adreno execute: clCreateProgramWithSource failed for node %d (err=%d)", node_idx, err);
            status = -1;
            break;
        }

        err = fn_clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
        if (err != CL_SUCCESS) {
            LOG_ERROR("Adreno execute: clBuildProgram failed for node %d (err=%d)", node_idx, err);
            fn_clReleaseProgram(program);
            status = -1;
            break;
        }

        char kernel_name[32];
        snprintf(kernel_name, sizeof(kernel_name), "op_%d", node_idx);

        void* kernel = fn_clCreateKernel(program, kernel_name, &err);
        if (err != CL_SUCCESS || !kernel) {
            LOG_ERROR("Adreno execute: clCreateKernel failed for node %d (err=%d)", node_idx, err);
            fn_clReleaseProgram(program);
            status = -1;
            break;
        }

        LOG_DEBUG("Adreno execute: kernel compiled for node %d (op=%s)", node_idx, op_name);

        fn_clReleaseKernel(kernel);
        fn_clReleaseProgram(program);

        node = node->next;
        node_idx++;
    }

    if (fn_clFinish(backend->cl_queue) != CL_SUCCESS) {
        LOG_WARNING("Adreno execute: clFinish returned error");
    }

    LOG_DEBUG("Adreno execute: %d nodes processed", node_idx);
    return status;
}

const char* cml_adreno_device_info(const CMLAdrenoBackend* backend) {
    if (!backend || !backend->initialized) {
        return "Adreno GPU (not available)";
    }

    snprintf(s_device_info_buf, sizeof(s_device_info_buf),
             "Adreno %d | %zu MB global | %d CUs | max WG %d | max alloc %zu MB",
             backend->gpu_version,
             backend->global_mem_size / (1024 * 1024),
             backend->compute_units,
             backend->max_work_group_size,
             backend->max_alloc_size / (1024 * 1024));

    return s_device_info_buf;
}
