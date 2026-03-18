#include "ops/ir/gpu/webgpu_backend.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef CML_HAS_WEBGPU

#if defined(__linux__)
#include <dlfcn.h>
#define WGPU_LIB_NAME "libwgpu_native.so"
#elif defined(__APPLE__)
#include <dlfcn.h>
#define WGPU_LIB_NAME "libwgpu_native.dylib"
#elif defined(_WIN32)
#include <windows.h>
#define WGPU_LIB_NAME "wgpu_native.dll"
#endif


#if defined(__linux__) || defined(__APPLE__)
static void* wgpu_load_library(const char* name) {
    void* lib = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        LOG_DEBUG("Failed to load %s: %s", name, dlerror());
    }
    return lib;
}

static void* wgpu_get_symbol(void* lib, const char* name) {
    return dlsym(lib, name);
}

static void wgpu_unload_library(void* lib) {
    if (lib) dlclose(lib);
}
#elif defined(_WIN32)
static void* wgpu_load_library(const char* name) {
    HMODULE lib = LoadLibraryA(name);
    if (!lib) {
        LOG_DEBUG("Failed to load %s: error %lu", name, GetLastError());
    }
    return (void*)lib;
}

static void* wgpu_get_symbol(void* lib, const char* name) {
    return (void*)GetProcAddress((HMODULE)lib, name);
}

static void wgpu_unload_library(void* lib) {
    if (lib) FreeLibrary((HMODULE)lib);
}
#else
static void* wgpu_load_library(const char* name) { (void)name; return NULL; }
static void* wgpu_get_symbol(void* lib, const char* name) {
    (void)lib; (void)name; return NULL;
}
static void wgpu_unload_library(void* lib) { (void)lib; }
#endif


typedef void*    WGPUInstance;
typedef void*    WGPUAdapter;
typedef void*    WGPUDevice;
typedef void*    WGPUQueue;
typedef void*    WGPUShaderModule;
typedef void*    WGPUComputePipeline;
typedef void*    WGPUBindGroupLayout;
typedef void*    WGPUBindGroup;
typedef void*    WGPUPipelineLayout;
typedef void*    WGPUBuffer;
typedef void*    WGPUCommandEncoder;
typedef void*    WGPUComputePassEncoder;
typedef void*    WGPUCommandBuffer;
typedef uint32_t WGPUBufferUsageFlags;
typedef uint64_t WGPUMapModeFlags;

#define WGPU_BUFFER_USAGE_STORAGE  0x0080
#define WGPU_BUFFER_USAGE_COPY_SRC 0x0004
#define WGPU_BUFFER_USAGE_COPY_DST 0x0008
#define WGPU_BUFFER_USAGE_MAP_READ 0x0001

typedef enum {
    WGPURequestAdapterStatus_Success = 0,
} WGPURequestAdapterStatus;

typedef enum {
    WGPURequestDeviceStatus_Success = 0,
} WGPURequestDeviceStatus;

typedef enum {
    WGPUBufferMapAsyncStatus_Success = 0,
} WGPUBufferMapAsyncStatus;


typedef struct {
    WGPUAdapter adapter;
    bool done;
} AdapterUserData;

static void adapter_request_cb(WGPURequestAdapterStatus status,
                                WGPUAdapter adapter,
                                const char* message,
                                void* userdata) {
    AdapterUserData* ud = (AdapterUserData*)userdata;
    if (status == WGPURequestAdapterStatus_Success) {
        ud->adapter = adapter;
    } else {
        LOG_ERROR("WebGPU adapter request failed: %s", message ? message : "unknown");
        ud->adapter = NULL;
    }
    ud->done = true;
}


typedef struct {
    WGPUDevice device;
    bool done;
} DeviceUserData;

static void device_request_cb(WGPURequestDeviceStatus status,
                               WGPUDevice device,
                               const char* message,
                               void* userdata) {
    DeviceUserData* ud = (DeviceUserData*)userdata;
    if (status == WGPURequestDeviceStatus_Success) {
        ud->device = device;
    } else {
        LOG_ERROR("WebGPU device request failed: %s", message ? message : "unknown");
        ud->device = NULL;
    }
    ud->done = true;
}


typedef struct {
    bool success;
    bool done;
} MapUserData;

static void buffer_map_cb(WGPUBufferMapAsyncStatus status, void* userdata) {
    MapUserData* ud = (MapUserData*)userdata;
    ud->success = (status == WGPUBufferMapAsyncStatus_Success);
    ud->done = true;
}


bool cml_webgpu_available(void) {
#ifndef WGPU_LIB_NAME
    return false;
#else
    void* lib = wgpu_load_library(WGPU_LIB_NAME);
    if (!lib) return false;

    void* sym = wgpu_get_symbol(lib, "wgpuCreateInstance");
    wgpu_unload_library(lib);
    return sym != NULL;
#endif
}


CMLWebGPUBackend* cml_webgpu_backend_create(void) {
    CMLWebGPUBackend* backend = (CMLWebGPUBackend*)calloc(1, sizeof(CMLWebGPUBackend));
    if (!backend) {
        LOG_ERROR("Failed to allocate WebGPU backend");
        return NULL;
    }

#ifndef WGPU_LIB_NAME
    LOG_ERROR("WebGPU not supported on this platform");
    free(backend);
    return NULL;
#else
    backend->lib_handle = wgpu_load_library(WGPU_LIB_NAME);
    if (!backend->lib_handle) {
        LOG_ERROR("Failed to load wgpu-native library: %s", WGPU_LIB_NAME);
        free(backend);
        return NULL;
    }

#define WGPU_LOAD(field, sym_name)                                        \
    backend->field = wgpu_get_symbol(backend->lib_handle, sym_name);      \
    if (!backend->field) {                                                \
        LOG_WARNING("WebGPU: missing symbol %s", sym_name);               \
    }

    WGPU_LOAD(fn_create_instance,                "wgpuCreateInstance");
    WGPU_LOAD(fn_instance_request_adapter,       "wgpuInstanceRequestAdapter");
    WGPU_LOAD(fn_adapter_request_device,         "wgpuAdapterRequestDevice");
    WGPU_LOAD(fn_device_get_queue,               "wgpuDeviceGetQueue");
    WGPU_LOAD(fn_device_create_shader_module,    "wgpuDeviceCreateShaderModule");
    WGPU_LOAD(fn_device_create_compute_pipeline, "wgpuDeviceCreateComputePipeline");
    WGPU_LOAD(fn_device_create_bind_group_layout,"wgpuDeviceCreateBindGroupLayout");
    WGPU_LOAD(fn_device_create_bind_group,       "wgpuDeviceCreateBindGroup");
    WGPU_LOAD(fn_device_create_pipeline_layout,  "wgpuDeviceCreatePipelineLayout");
    WGPU_LOAD(fn_device_create_buffer,           "wgpuDeviceCreateBuffer");
    WGPU_LOAD(fn_device_create_command_encoder,  "wgpuDeviceCreateCommandEncoder");
    WGPU_LOAD(fn_command_encoder_begin_compute_pass,
                                                  "wgpuCommandEncoderBeginComputePass");
    WGPU_LOAD(fn_compute_pass_set_pipeline,      "wgpuComputePassEncoderSetPipeline");
    WGPU_LOAD(fn_compute_pass_set_bind_group,    "wgpuComputePassEncoderSetBindGroup");
    WGPU_LOAD(fn_compute_pass_dispatch_workgroups,
                                                  "wgpuComputePassEncoderDispatchWorkgroups");
    WGPU_LOAD(fn_compute_pass_end,               "wgpuComputePassEncoderEnd");
    WGPU_LOAD(fn_command_encoder_finish,         "wgpuCommandEncoderFinish");
    WGPU_LOAD(fn_queue_submit,                   "wgpuQueueSubmit");
    WGPU_LOAD(fn_queue_write_buffer,             "wgpuQueueWriteBuffer");
    WGPU_LOAD(fn_buffer_map_async,               "wgpuBufferMapAsync");
    WGPU_LOAD(fn_buffer_get_mapped_range,        "wgpuBufferGetMappedRange");
    WGPU_LOAD(fn_buffer_unmap,                   "wgpuBufferUnmap");
    WGPU_LOAD(fn_buffer_destroy,                 "wgpuBufferDestroy");
    WGPU_LOAD(fn_device_poll,                    "wgpuDevicePoll");
    WGPU_LOAD(fn_instance_release,               "wgpuInstanceRelease");

#undef WGPU_LOAD

    return backend;
#endif /* WGPU_LIB_NAME */
}

int cml_webgpu_backend_init(CMLWebGPUBackend* backend) {
    if (!backend) return -1;
    if (backend->initialized) {
        return 0;
    }

    typedef WGPUInstance (*PFN_wgpuCreateInstance)(const void*);
    typedef void (*PFN_wgpuInstanceRequestAdapter)(
        WGPUInstance, const void*, void (*)(WGPURequestAdapterStatus, WGPUAdapter, const char*, void*), void*);
    typedef void (*PFN_wgpuAdapterRequestDevice)(
        WGPUAdapter, const void*, void (*)(WGPURequestDeviceStatus, WGPUDevice, const char*, void*), void*);
    typedef WGPUQueue (*PFN_wgpuDeviceGetQueue)(WGPUDevice);
    typedef bool (*PFN_wgpuDevicePoll)(WGPUDevice, bool, const void*);

    if (!backend->fn_create_instance) {
        LOG_ERROR("WebGPU: wgpuCreateInstance not loaded");
        return -1;
    }

    PFN_wgpuCreateInstance createInst =
        (PFN_wgpuCreateInstance)backend->fn_create_instance;
    backend->instance = createInst(NULL);
    if (!backend->instance) {
        LOG_ERROR("wgpuCreateInstance failed");
        return -1;
    }

    AdapterUserData adapter_ud = {NULL, false};
    PFN_wgpuInstanceRequestAdapter reqAdapter =
        (PFN_wgpuInstanceRequestAdapter)backend->fn_instance_request_adapter;
    if (!reqAdapter) {
        LOG_ERROR("WebGPU: wgpuInstanceRequestAdapter not loaded");
        return -1;
    }
    reqAdapter(backend->instance, NULL, adapter_request_cb, &adapter_ud);

    if (backend->fn_device_poll) {
        int attempts = 0;
        while (!adapter_ud.done && attempts < 1000) {
            attempts++;
        }
    }
    if (!adapter_ud.adapter) {
        LOG_ERROR("WebGPU adapter request failed");
        return -1;
    }
    backend->adapter = adapter_ud.adapter;

    DeviceUserData device_ud = {NULL, false};
    PFN_wgpuAdapterRequestDevice reqDevice =
        (PFN_wgpuAdapterRequestDevice)backend->fn_adapter_request_device;
    if (!reqDevice) {
        LOG_ERROR("WebGPU: wgpuAdapterRequestDevice not loaded");
        return -1;
    }
    reqDevice(backend->adapter, NULL, device_request_cb, &device_ud);

    int attempts = 0;
    while (!device_ud.done && attempts < 1000) {
        attempts++;
    }
    if (!device_ud.device) {
        LOG_ERROR("WebGPU device request failed");
        return -1;
    }
    backend->device = device_ud.device;

    PFN_wgpuDeviceGetQueue getQueue =
        (PFN_wgpuDeviceGetQueue)backend->fn_device_get_queue;
    if (getQueue) {
        backend->queue = getQueue(backend->device);
    }
    if (!backend->queue) {
        LOG_ERROR("wgpuDeviceGetQueue returned NULL");
        return -1;
    }

    strncpy(backend->device_name, "WebGPU (wgpu-native)",
            sizeof(backend->device_name) - 1);
    backend->initialized = true;

    LOG_INFO("WebGPU backend initialized: %s", backend->device_name);
    return 0;
}

void cml_webgpu_backend_free(CMLWebGPUBackend* backend) {
    if (!backend) return;

    /*
     * WebGPU objects are reference-counted internally; we release only what
     * we explicitly acquired.  The instance is the root object.
     */
    if (backend->initialized) {
        typedef void (*PFN_release)(void*);

        if (backend->instance && backend->fn_instance_release) {
            PFN_release rel = (PFN_release)backend->fn_instance_release;
            rel(backend->instance);
        }
        /* adapter, device, queue are owned by instance/adapter lifetime */
        backend->instance = NULL;
        backend->adapter  = NULL;
        backend->device   = NULL;
        backend->queue    = NULL;
        backend->initialized = false;
    }

    if (backend->lib_handle) {
        wgpu_unload_library(backend->lib_handle);
        backend->lib_handle = NULL;
    }

    free(backend);
}


CMLWebGPUKernel* cml_webgpu_compile_wgsl(CMLWebGPUBackend* backend,
                                           const char* wgsl_source,
                                           const char* entry_point) {
    if (!backend || !backend->initialized || !wgsl_source || !entry_point) {
        LOG_ERROR("Invalid arguments to cml_webgpu_compile_wgsl");
        return NULL;
    }

    /*
     * The wgpu-native C-API uses descriptor structs.  We construct them
     * manually to avoid pulling in the full webgpu.h header.
     *
     * WGPUShaderModuleWGSLDescriptor contains:
     *   - WGPUChainedStruct chain  (sType = 5 = WGPUSType_ShaderModuleWGSLDescriptor)
     *   - const char* code
     *
     * WGPUShaderModuleDescriptor contains:
     *   - const WGPUChainedStruct* nextInChain
     *   - const char* label
     */

    /* Build packed descriptor structs on the stack.
     * This is ABI-compatible with the webgpu.h definitions used by wgpu-native.
     */
    struct {
        const void* next;   /* WGPUChainedStruct.next */
        uint32_t sType;     /* WGPUChainedStruct.sType */
        uint32_t pad;       /* alignment padding */
        const char* code;
    } wgsl_desc;

    memset(&wgsl_desc, 0, sizeof(wgsl_desc));
    wgsl_desc.next  = NULL;
    wgsl_desc.sType = 5; /* WGPUSType_ShaderModuleWGSLDescriptor */
    wgsl_desc.code  = wgsl_source;

    struct {
        const void* nextInChain;
        const char* label;
    } shader_desc;

    memset(&shader_desc, 0, sizeof(shader_desc));
    shader_desc.nextInChain = &wgsl_desc;
    shader_desc.label       = entry_point;

    typedef WGPUShaderModule (*PFN_createShaderModule)(WGPUDevice, const void*);
    PFN_createShaderModule createSM =
        (PFN_createShaderModule)backend->fn_device_create_shader_module;
    if (!createSM) {
        LOG_ERROR("WebGPU: wgpuDeviceCreateShaderModule not loaded");
        return NULL;
    }

    WGPUShaderModule shaderMod = createSM(backend->device, &shader_desc);
    if (!shaderMod) {
        LOG_ERROR("wgpuDeviceCreateShaderModule failed");
        return NULL;
    }

    /*
     * Create compute pipeline.
     * WGPUComputePipelineDescriptor {
     *   nextInChain, label,
     *   layout (NULL = auto),
     *   compute: { nextInChain, module, entryPoint, ... }
     * }
     *
     * We build this as a flat byte buffer because the exact padding depends
     * on pointer size.  Using a simple struct with explicit fields:
     */
    struct {
        const void* nextInChain;   /* NULL */
        const char* label;
        WGPUPipelineLayout layout; /* NULL = auto */
        /* WGPUProgrammableStageDescriptor compute: */
        const void* compute_next;
        WGPUShaderModule compute_module;
        const char* compute_entryPoint;
        size_t constant_count;
        const void* constants;
    } pipeline_desc;

    memset(&pipeline_desc, 0, sizeof(pipeline_desc));
    pipeline_desc.nextInChain       = NULL;
    pipeline_desc.label             = entry_point;
    pipeline_desc.layout            = NULL; /* auto layout */
    pipeline_desc.compute_next      = NULL;
    pipeline_desc.compute_module    = shaderMod;
    pipeline_desc.compute_entryPoint = entry_point;
    pipeline_desc.constant_count    = 0;
    pipeline_desc.constants         = NULL;

    typedef WGPUComputePipeline (*PFN_createPipeline)(WGPUDevice, const void*);
    PFN_createPipeline createPL =
        (PFN_createPipeline)backend->fn_device_create_compute_pipeline;
    if (!createPL) {
        LOG_ERROR("WebGPU: wgpuDeviceCreateComputePipeline not loaded");
        return NULL;
    }

    WGPUComputePipeline pipeline = createPL(backend->device, &pipeline_desc);
    if (!pipeline) {
        LOG_ERROR("wgpuDeviceCreateComputePipeline failed");
        return NULL;
    }

    CMLWebGPUKernel* kernel = (CMLWebGPUKernel*)calloc(1, sizeof(CMLWebGPUKernel));
    if (!kernel) {
        LOG_ERROR("Failed to allocate CMLWebGPUKernel");
        return NULL;
    }

    kernel->pipeline      = pipeline;
    kernel->shader_module = shaderMod;
    /* bind_group_layout / pipeline_layout are auto-managed when layout is NULL */
    kernel->bind_group_layout = NULL;
    kernel->pipeline_layout   = NULL;
    strncpy(kernel->name, entry_point, sizeof(kernel->name) - 1);
    kernel->name[sizeof(kernel->name) - 1] = '\0';

    return kernel;
}

void cml_webgpu_kernel_free(CMLWebGPUKernel* kernel) {
    if (!kernel) return;
    /* The pipeline and shader module are reference-counted internally by
     * wgpu-native; dropping our handle is sufficient.  We do not call
     * explicit release here to avoid double-free in case the caller
     * still holds references through bind groups.  */
    free(kernel);
}


int cml_webgpu_launch_kernel(CMLWebGPUBackend* backend,
                             CMLWebGPUKernel* kernel,
                             size_t workgroup_count[3],
                             void** buffers,
                             size_t* buffer_sizes,
                             int num_buffers) {
    if (!backend || !backend->initialized || !kernel || !kernel->pipeline) {
        LOG_ERROR("Invalid arguments to cml_webgpu_launch_kernel");
        return -1;
    }

    (void)buffer_sizes; /* sizes are embedded in the WGPUBuffer objects */

    typedef WGPUCommandEncoder (*PFN_createCmdEnc)(WGPUDevice, const void*);
    typedef WGPUComputePassEncoder (*PFN_beginPass)(WGPUCommandEncoder, const void*);
    typedef void (*PFN_setPipeline)(WGPUComputePassEncoder, WGPUComputePipeline);
    typedef void (*PFN_setBindGroup)(WGPUComputePassEncoder, uint32_t, WGPUBindGroup,
                                      size_t, const uint32_t*);
    typedef void (*PFN_dispatch)(WGPUComputePassEncoder, uint32_t, uint32_t, uint32_t);
    typedef void (*PFN_endPass)(WGPUComputePassEncoder);
    typedef WGPUCommandBuffer (*PFN_finish)(WGPUCommandEncoder, const void*);
    typedef void (*PFN_submit)(WGPUQueue, size_t, const WGPUCommandBuffer*);
    typedef bool (*PFN_poll)(WGPUDevice, bool, const void*);

    PFN_createCmdEnc createEnc = (PFN_createCmdEnc)backend->fn_device_create_command_encoder;
    PFN_beginPass    beginPass = (PFN_beginPass)backend->fn_command_encoder_begin_compute_pass;
    PFN_setPipeline  setPipe   = (PFN_setPipeline)backend->fn_compute_pass_set_pipeline;
    PFN_setBindGroup setBG     = (PFN_setBindGroup)backend->fn_compute_pass_set_bind_group;
    PFN_dispatch     dispatch  = (PFN_dispatch)backend->fn_compute_pass_dispatch_workgroups;
    PFN_endPass      endPass   = (PFN_endPass)backend->fn_compute_pass_end;
    PFN_finish       finish    = (PFN_finish)backend->fn_command_encoder_finish;
    PFN_submit       submit    = (PFN_submit)backend->fn_queue_submit;
    PFN_poll         poll      = (PFN_poll)backend->fn_device_poll;

    if (!createEnc || !beginPass || !setPipe || !dispatch || !endPass ||
        !finish || !submit) {
        LOG_ERROR("WebGPU: required function pointers are NULL");
        return -1;
    }

    typedef WGPUBindGroupLayout (*PFN_getBindGroupLayout)(WGPUComputePipeline, uint32_t);
    typedef WGPUBindGroup (*PFN_createBindGroup)(WGPUDevice, const void*);

    PFN_getBindGroupLayout getBGL = (PFN_getBindGroupLayout)
        wgpu_get_symbol(backend->lib_handle, "wgpuComputePipelineGetBindGroupLayout");
    PFN_createBindGroup createBG = (PFN_createBindGroup)backend->fn_device_create_bind_group;

    WGPUBindGroup bind_group = NULL;

    if (getBGL && createBG && num_buffers > 0) {
        WGPUBindGroupLayout layout = getBGL((WGPUComputePipeline)kernel->pipeline, 0);
        if (layout) {
            struct BindGroupEntry {
                const void* nextInChain;
                uint32_t binding;
                uint32_t pad0;
                WGPUBuffer buffer;
                uint64_t offset;
                uint64_t size;
                void* sampler;
                void* textureView;
            };

            struct BindGroupEntry* entries = (struct BindGroupEntry*)calloc(
                (size_t)num_buffers, sizeof(struct BindGroupEntry));

            if (entries) {
                for (int i = 0; i < num_buffers; i++) {
                    entries[i].nextInChain = NULL;
                    entries[i].binding = (uint32_t)i;
                    entries[i].buffer = (WGPUBuffer)buffers[i];
                    entries[i].offset = 0;
                    entries[i].size = buffer_sizes ? buffer_sizes[i] : 0;
                    entries[i].sampler = NULL;
                    entries[i].textureView = NULL;
                }

                struct {
                    const void* nextInChain;
                    const char* label;
                    WGPUBindGroupLayout layout;
                    size_t entryCount;
                    const void* entries;
                } bg_desc;
                memset(&bg_desc, 0, sizeof(bg_desc));
                bg_desc.label = "cml_bind_group";
                bg_desc.layout = layout;
                bg_desc.entryCount = (size_t)num_buffers;
                bg_desc.entries = entries;

                bind_group = createBG(backend->device, &bg_desc);
                free(entries);
            }
        }
    }

    WGPUCommandEncoder encoder = createEnc(backend->device, NULL);
    if (!encoder) {
        LOG_ERROR("Failed to create command encoder");
        return -1;
    }

    WGPUComputePassEncoder pass = beginPass(encoder, NULL);
    if (!pass) {
        LOG_ERROR("Failed to begin compute pass");
        return -1;
    }

    setPipe(pass, (WGPUComputePipeline)kernel->pipeline);

    if (bind_group && setBG) {
        setBG(pass, 0, bind_group, 0, NULL);
    } else if (num_buffers > 0 && setBG) {
        WGPUBindGroup bg = (WGPUBindGroup)buffers[num_buffers - 1];
        setBG(pass, 0, bg, 0, NULL);
    }

    dispatch(pass,
             (uint32_t)workgroup_count[0],
             (uint32_t)workgroup_count[1],
             (uint32_t)workgroup_count[2]);

    endPass(pass);

    WGPUCommandBuffer cmdBuf = finish(encoder, NULL);
    if (!cmdBuf) {
        LOG_ERROR("Failed to finish command encoder");
        return -1;
    }

    submit(backend->queue, 1, &cmdBuf);

    if (poll) {
        poll(backend->device, true, NULL);
    }

    return 0;
}


void* cml_webgpu_alloc(CMLWebGPUBackend* backend, size_t size) {
    if (!backend || !backend->initialized || size == 0) return NULL;

    typedef WGPUBuffer (*PFN_createBuffer)(WGPUDevice, const void*);
    PFN_createBuffer createBuf =
        (PFN_createBuffer)backend->fn_device_create_buffer;
    if (!createBuf) {
        LOG_ERROR("WebGPU: wgpuDeviceCreateBuffer not loaded");
        return NULL;
    }

    struct {
        const void* nextInChain;
        const char* label;
        WGPUBufferUsageFlags usage;
        uint64_t size;
        bool mappedAtCreation;
    } desc;

    memset(&desc, 0, sizeof(desc));
    desc.label = "cml_buffer";
    desc.usage = WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_SRC |
                 WGPU_BUFFER_USAGE_COPY_DST;
    desc.size  = (uint64_t)size;
    desc.mappedAtCreation = false;

    WGPUBuffer buffer = createBuf(backend->device, &desc);
    if (!buffer) {
        LOG_ERROR("wgpuDeviceCreateBuffer failed for %zu bytes", size);
        return NULL;
    }
    return buffer;
}

void cml_webgpu_free(CMLWebGPUBackend* backend, void* buffer) {
    if (!backend || !buffer) return;

    typedef void (*PFN_bufferDestroy)(WGPUBuffer);
    PFN_bufferDestroy destroy = (PFN_bufferDestroy)backend->fn_buffer_destroy;
    if (destroy) {
        destroy((WGPUBuffer)buffer);
    }
}

int cml_webgpu_upload(CMLWebGPUBackend* backend,
                      void* dst_buffer,
                      const void* src_host,
                      size_t size) {
    if (!backend || !backend->initialized || !dst_buffer || !src_host || size == 0)
        return -1;

    typedef void (*PFN_writeBuffer)(WGPUQueue, WGPUBuffer, uint64_t,
                                     const void*, size_t);
    PFN_writeBuffer writeBuf =
        (PFN_writeBuffer)backend->fn_queue_write_buffer;
    if (!writeBuf) {
        LOG_ERROR("WebGPU: wgpuQueueWriteBuffer not loaded");
        return -1;
    }

    writeBuf(backend->queue, (WGPUBuffer)dst_buffer, 0, src_host, size);
    return 0;
}

int cml_webgpu_download(CMLWebGPUBackend* backend,
                        void* dst_host,
                        void* src_buffer,
                        size_t size) {
    if (!backend || !backend->initialized || !dst_host || !src_buffer || size == 0)
        return -1;

    /* To download we need to:
     *  1. Create a staging buffer with MAP_READ | COPY_DST
     *  2. Copy from src_buffer to staging (via command encoder)
     *  3. Map the staging buffer
     *  4. memcpy to dst_host
     *  5. Unmap staging buffer
     *  6. Destroy staging buffer
     *
     * For this implementation we use an inline staging approach.
     */

    typedef WGPUBuffer (*PFN_createBuffer)(WGPUDevice, const void*);
    typedef WGPUCommandEncoder (*PFN_createCmdEnc)(WGPUDevice, const void*);
    typedef void (*PFN_copyBufToBuf)(WGPUCommandEncoder, WGPUBuffer, uint64_t,
                                      WGPUBuffer, uint64_t, uint64_t);
    typedef WGPUCommandBuffer (*PFN_finish)(WGPUCommandEncoder, const void*);
    typedef void (*PFN_submit)(WGPUQueue, size_t, const WGPUCommandBuffer*);
    typedef void (*PFN_mapAsync)(WGPUBuffer, WGPUMapModeFlags, size_t, size_t,
                                  void (*)(WGPUBufferMapAsyncStatus, void*), void*);
    typedef void* (*PFN_getMappedRange)(WGPUBuffer, size_t, size_t);
    typedef void (*PFN_unmap)(WGPUBuffer);
    typedef void (*PFN_bufferDestroy)(WGPUBuffer);
    typedef bool (*PFN_poll)(WGPUDevice, bool, const void*);

    PFN_createBuffer  createBuf = (PFN_createBuffer)backend->fn_device_create_buffer;
    PFN_createCmdEnc  createEnc = (PFN_createCmdEnc)backend->fn_device_create_command_encoder;
    PFN_finish        finishEnc = (PFN_finish)backend->fn_command_encoder_finish;
    PFN_submit        submitQ   = (PFN_submit)backend->fn_queue_submit;
    PFN_mapAsync      mapAsync  = (PFN_mapAsync)backend->fn_buffer_map_async;
    PFN_getMappedRange getMapped = (PFN_getMappedRange)backend->fn_buffer_get_mapped_range;
    PFN_unmap         unmap     = (PFN_unmap)backend->fn_buffer_unmap;
    PFN_bufferDestroy destroyBuf = (PFN_bufferDestroy)backend->fn_buffer_destroy;
    PFN_poll          poll       = (PFN_poll)backend->fn_device_poll;

    if (!createBuf || !createEnc || !finishEnc || !submitQ ||
        !mapAsync || !getMapped || !unmap) {
        LOG_ERROR("WebGPU: missing required symbols for download");
        return -1;
    }

    struct {
        const void* nextInChain;
        const char* label;
        WGPUBufferUsageFlags usage;
        uint64_t size;
        bool mappedAtCreation;
    } stg_desc;

    memset(&stg_desc, 0, sizeof(stg_desc));
    stg_desc.label = "cml_staging";
    stg_desc.usage = WGPU_BUFFER_USAGE_MAP_READ | WGPU_BUFFER_USAGE_COPY_DST;
    stg_desc.size  = (uint64_t)size;
    stg_desc.mappedAtCreation = false;

    WGPUBuffer staging = createBuf(backend->device, &stg_desc);
    if (!staging) {
        LOG_ERROR("Failed to create WebGPU staging buffer");
        return -1;
    }

    WGPUCommandEncoder enc = createEnc(backend->device, NULL);
    if (!enc) {
        if (destroyBuf) destroyBuf(staging);
        return -1;
    }

    PFN_copyBufToBuf copyBuf =
        (PFN_copyBufToBuf)wgpu_get_symbol(backend->lib_handle,
                                            "wgpuCommandEncoderCopyBufferToBuffer");
    if (!copyBuf) {
        LOG_ERROR("WebGPU: wgpuCommandEncoderCopyBufferToBuffer not found");
        if (destroyBuf) destroyBuf(staging);
        return -1;
    }
    copyBuf(enc, (WGPUBuffer)src_buffer, 0, staging, 0, (uint64_t)size);

    WGPUCommandBuffer cmdBuf = finishEnc(enc, NULL);
    submitQ(backend->queue, 1, &cmdBuf);

    if (poll) poll(backend->device, true, NULL);

    MapUserData map_ud = {false, false};
    mapAsync(staging, 0x0001 /* MAP_READ */, 0, size, buffer_map_cb, &map_ud);

    if (poll) {
        int attempts = 0;
        while (!map_ud.done && attempts < 10000) {
            poll(backend->device, false, NULL);
            attempts++;
        }
    }

    if (!map_ud.success) {
        LOG_ERROR("WebGPU buffer map failed");
        if (destroyBuf) destroyBuf(staging);
        return -1;
    }

    void* mapped = getMapped(staging, 0, size);
    if (mapped) {
        memcpy(dst_host, mapped, size);
    } else {
        LOG_ERROR("wgpuBufferGetMappedRange returned NULL");
        unmap(staging);
        if (destroyBuf) destroyBuf(staging);
        return -1;
    }

    unmap(staging);
    if (destroyBuf) destroyBuf(staging);

    return 0;
}

#else /* !CML_HAS_WEBGPU */

bool cml_webgpu_available(void) { return false; }

CMLWebGPUBackend* cml_webgpu_backend_create(void) { return NULL; }

int cml_webgpu_backend_init(CMLWebGPUBackend* backend) {
    (void)backend; return -1;
}

void cml_webgpu_backend_free(CMLWebGPUBackend* backend) {
    (void)backend;
}

CMLWebGPUKernel* cml_webgpu_compile_wgsl(CMLWebGPUBackend* backend,
                                           const char* wgsl_source,
                                           const char* entry_point) {
    (void)backend; (void)wgsl_source; (void)entry_point;
    return NULL;
}

void cml_webgpu_kernel_free(CMLWebGPUKernel* kernel) {
    (void)kernel;
}

int cml_webgpu_launch_kernel(CMLWebGPUBackend* backend,
                             CMLWebGPUKernel* kernel,
                             size_t workgroup_count[3],
                             void** buffers,
                             size_t* buffer_sizes,
                             int num_buffers) {
    (void)backend; (void)kernel; (void)workgroup_count;
    (void)buffers; (void)buffer_sizes; (void)num_buffers;
    return -1;
}

void* cml_webgpu_alloc(CMLWebGPUBackend* backend, size_t size) {
    (void)backend; (void)size; return NULL;
}

void cml_webgpu_free(CMLWebGPUBackend* backend, void* buffer) {
    (void)backend; (void)buffer;
}

int cml_webgpu_upload(CMLWebGPUBackend* backend,
                      void* dst_buffer,
                      const void* src_host,
                      size_t size) {
    (void)backend; (void)dst_buffer; (void)src_host; (void)size;
    return -1;
}

int cml_webgpu_download(CMLWebGPUBackend* backend,
                        void* dst_host,
                        void* src_buffer,
                        size_t size) {
    (void)backend; (void)dst_host; (void)src_buffer; (void)size;
    return -1;
}

#endif /* CML_HAS_WEBGPU */
