# GPU Backends

C-ML supports multiple GPU and accelerator backends for compute kernel execution. All backends use dynamic library loading (dlopen/dlsym) at runtime, so no compile-time SDK dependencies are required -- only the appropriate runtime libraries must be present on the system.

## Backend Summary

| Backend | Hardware | Shading Language | CMake Flag | Default |
|---|---|---|---|---|
| [CUDA](#cuda-backend) | NVIDIA GPUs | PTX / CUDA C | `ENABLE_CUDA` | ON |
| [NV Driver](#nvidia-direct-driver) | NVIDIA GPUs | PTX (via ptxas) | (built with CUDA) | -- |
| [ROCm/HIP](#rocmhip-backend) | AMD GPUs | HIP / HSACO | `ENABLE_ROCM` | ON |
| [AM Driver](#amd-direct-driver) | AMD RDNA3/4 GPUs | AMDGPU ELF (via LLVM) | (built with ROCm) | -- |
| [Vulkan](#vulkan-backend) | Any Vulkan GPU | SPIR-V | `ENABLE_VULKAN` | ON |
| [SPIR-V Codegen](#spir-v-codegen) | (Vulkan GPUs) | SPIR-V binary | (built with Vulkan) | -- |
| [WebGPU](#webgpu-backend) | Any wgpu-native GPU | WGSL | `ENABLE_WEBGPU` | OFF |
| [Metal](#metal-backend) | Apple GPUs (macOS) | MSL | `ENABLE_METAL` | ON (macOS) |
| [OpenCL](#opencl-backend) | Any OpenCL device | OpenCL C | `ENABLE_OPENCL` | ON |
| [Adreno](#adreno-backend) | Qualcomm Adreno GPUs | OpenCL / Vulkan | -- | -- |
| [Hexagon](#hexagon-backend) | Qualcomm Hexagon DSP | HVX intrinsics | -- | -- |

**Header directory:** `include/ops/ir/gpu/` (GPU-specific), `include/backend/` (OpenCL)

---

## CUDA Backend

**Header:** `include/ops/ir/gpu/cuda_backend.h`
**Hardware:** NVIDIA GPUs (Kepler and newer, sm_50+)
**CMake:** `-DENABLE_CUDA=ON` (default)
**Runtime dependency:** `libcuda.so` (NVIDIA driver), optionally `libnvrtc.so` (for CUDA C compilation)

The CUDA backend dynamically loads the CUDA driver API at runtime. It supports two compilation paths: loading pre-compiled PTX via `cuModuleLoadData`, and runtime compilation of CUDA C source via NVRTC.

### Key Types

- `CMLCUDABackend` -- Backend context holding device state, stream, and dlsym'd function pointers.
- `CMLCUDAKernel` -- Compiled kernel with module, function handle, and launch configuration.

### API

```c
// Availability
bool cml_cuda_available(void);

// Lifecycle
CMLCUDABackend* cml_cuda_backend_create(void);
int  cml_cuda_backend_init(CMLCUDABackend* backend, int device_ordinal);
void cml_cuda_backend_free(CMLCUDABackend* backend);
int  cml_cuda_get_device_count(CMLCUDABackend* backend);

// Kernel compilation
CMLCUDAKernel* cml_cuda_compile_ptx(CMLCUDABackend* backend,
                                     const char* ptx_code,
                                     const char* kernel_name);
CMLCUDAKernel* cml_cuda_compile_source(CMLCUDABackend* backend,
                                        const char* cuda_code,
                                        const char* kernel_name);
void cml_cuda_kernel_free(CMLCUDABackend* backend, CMLCUDAKernel* kernel);

// Execution
void cml_cuda_kernel_set_launch_config(CMLCUDAKernel* kernel,
                                        int grid_x, int grid_y, int grid_z,
                                        int block_x, int block_y, int block_z);
int  cml_cuda_launch_kernel(CMLCUDABackend* backend, CMLCUDAKernel* kernel,
                             void** args, int num_args);
int  cml_cuda_synchronize(CMLCUDABackend* backend);

// Memory
CUdeviceptr cml_cuda_malloc(CMLCUDABackend* backend, size_t size);
void cml_cuda_free(CMLCUDABackend* backend, CUdeviceptr ptr);
int  cml_cuda_memcpy_h2d(CMLCUDABackend* backend, CUdeviceptr dst, const void* src, size_t size);
int  cml_cuda_memcpy_d2h(CMLCUDABackend* backend, void* dst, CUdeviceptr src, size_t size);

// Tensor transfer
int cml_cuda_upload_tensor(CMLCUDABackend* backend, Tensor* tensor);
int cml_cuda_download_tensor(CMLCUDABackend* backend, Tensor* tensor);
```

### PTX Codegen

**Header:** `include/ops/ir/gpu/ptx_codegen.h`

Generates PTX assembly text from UOps IR without requiring LLVM. Targets sm_50+ with `.f32` precision.

```c
CMLPTXCodegen* cml_ptx_codegen_create(int sm_version, CMLCUDABackend* cuda);
void           cml_ptx_codegen_destroy(CMLPTXCodegen* cg);

// Generate PTX for various operations
char* cml_ptx_gen_unary(CMLPTXCodegen* cg, UOpType op, const char* name);
char* cml_ptx_gen_binary(CMLPTXCodegen* cg, UOpType op, const char* name);
char* cml_ptx_gen_reduction(CMLPTXCodegen* cg, UOpType op, const char* name);
char* cml_ptx_gen_matmul(CMLPTXCodegen* cg, const char* name);
char* cml_ptx_gen_fill(CMLPTXCodegen* cg, float value, const char* name);
char* cml_ptx_gen_where(CMLPTXCodegen* cg, const char* name);
char* cml_ptx_gen_tiled_matmul(CMLPTXCodegen* cg, const char* name);   // CUDA C, via NVRTC
char* cml_ptx_gen_conv2d(CMLPTXCodegen* cg, const char* name);         // CUDA C, via NVRTC

// Execute an IR graph end-to-end
int cml_ptx_execute_graph(CMLPTXCodegen* cg, CMLGraph_t ir);
```

All `cml_ptx_gen_*` functions return heap-allocated strings. The caller must `free()` them.

---

## NVIDIA Direct Driver

**Header:** `include/ops/ir/gpu/nv_driver.h`
**Hardware:** NVIDIA GPUs
**Runtime dependency:** `/dev/nvidiactl` and `/dev/nvidia0` device files, `ptxas` for CUBIN compilation

This backend bypasses `libcuda.so` entirely. It opens NVIDIA device files directly and uses RM ioctls for resource management, GPFIFO ring buffers for kernel dispatch, and semaphore polling for synchronization. This provides lower-latency kernel launch at the cost of less portability.

### Key Types

- `CMLNVDriver` -- Driver context with file descriptors, RM object hierarchy (client/device/subdevice/channel), GPFIFO, and VA allocator.
- `CMLNVBuffer` -- GPU buffer with virtual address, optional CPU mapping, and RM handle.
- `CMLNVKernel` -- Compiled CUBIN blob loaded to GPU memory.
- `CMLNVGPFIFO` -- Ring buffer of GPU FIFO entries for command submission.

### API

```c
// Availability
bool cml_nv_driver_available(void);

// Lifecycle
CMLNVDriver* cml_nv_driver_create(void);
int          cml_nv_driver_init(CMLNVDriver* drv);
void         cml_nv_driver_free(CMLNVDriver* drv);

// Buffer management
CMLNVBuffer* cml_nv_buffer_create(CMLNVDriver* drv, size_t size, bool host_visible);
void         cml_nv_buffer_free(CMLNVDriver* drv, CMLNVBuffer* buf);
int          cml_nv_buffer_upload(CMLNVDriver* drv, CMLNVBuffer* dst,
                                   const void* src, size_t n);
int          cml_nv_buffer_download(CMLNVDriver* drv, CMLNVBuffer* src,
                                     void* dst, size_t n);

// Kernel management (compiles PTX to CUBIN via ptxas, then loads)
CMLNVKernel* cml_nv_kernel_compile_ptx(CMLNVDriver* drv, const char* ptx_code,
                                         const char* kernel_name);
void         cml_nv_kernel_free(CMLNVDriver* drv, CMLNVKernel* kernel);
int          cml_nv_kernel_launch(CMLNVDriver* drv, CMLNVKernel* kernel,
                                   uint32_t grid[3], uint32_t block[3],
                                   void** args, int num_args);

// Synchronization
int cml_nv_synchronize(CMLNVDriver* drv);

// Graph execution
int cml_nv_execute_graph(CMLNVDriver* drv, CMLGraph_t ir);
```

---

## ROCm/HIP Backend

**Header:** `include/ops/ir/gpu/rocm_backend.h`
**Hardware:** AMD GPUs (GCN, RDNA, CDNA)
**CMake:** `-DENABLE_ROCM=ON` (default)
**Runtime dependency:** `libamdhip64.so`, optionally `libhiprtc.so`

The ROCm backend dynamically loads the HIP runtime. It provides kernel compilation from HSACO (HSA Code Object) format, kernel launch, and device memory management.

### Key Types

- `CMLROCmBackend` -- Backend context with HIP device, context, stream, and dlsym'd function pointers.
- `CMLROCmKernel` -- Compiled kernel with module, function handle, and launch dimensions.

### API

```c
// Availability
bool cml_rocm_available(void);

// Lifecycle
CMLROCmBackend* cml_rocm_backend_create(void);
int  cml_rocm_backend_init(CMLROCmBackend* backend, int device_ordinal);
void cml_rocm_backend_free(CMLROCmBackend* backend);

// Kernel compilation
CMLROCmKernel* cml_rocm_compile_hsaco(CMLROCmBackend* backend,
                                       const char* hsaco_code,
                                       const char* kernel_name);
void cml_rocm_kernel_free(CMLROCmBackend* backend, CMLROCmKernel* kernel);
int  cml_rocm_launch_kernel(CMLROCmBackend* backend, CMLROCmKernel* kernel,
                             void** args, int num_args);
int  cml_rocm_synchronize(CMLROCmBackend* backend);

// Memory
hipDeviceptr_t cml_rocm_malloc(CMLROCmBackend* backend, size_t size);
void cml_rocm_free(CMLROCmBackend* backend, hipDeviceptr_t ptr);
int  cml_rocm_memcpy_h2d(CMLROCmBackend* backend, hipDeviceptr_t dst,
                          const void* src, size_t size);
int  cml_rocm_memcpy_d2h(CMLROCmBackend* backend, void* dst,
                          hipDeviceptr_t src, size_t size);

// Tensor transfer
int cml_rocm_upload_tensor(CMLROCmBackend* backend, Tensor* tensor);
int cml_rocm_download_tensor(CMLROCmBackend* backend, Tensor* tensor);
```

---

## AMD Direct Driver

**Header:** `include/ops/ir/gpu/am_driver.h`
**Hardware:** AMD RDNA3/RDNA4 GPUs (gfx11.x)
**Runtime dependency:** `/dev/kfd` and `/dev/dri/renderD128` device files, LLVM AMDGPU codegen for kernel compilation

This backend bypasses `libamdhip64.so` entirely. It opens `/dev/kfd` directly and uses KFD ioctls for queue creation, memory allocation, and AQL (Architected Queuing Language) packet submission. This follows the HSA architecture specification.

### Key Types

- `CMLAMDriver` -- Driver context with KFD/DRM file descriptors, GPU info, AQL queue, and VA allocator.
- `CMLAMBuffer` -- GPU buffer with virtual address, optional CPU mapping, KFD handle. Can be VRAM or GTT (system memory).
- `CMLAMKernel` -- ELF code object loaded to GPU memory.
- `CMLAMQueue` -- AQL hardware queue with ring buffer of 64-byte dispatch packets and a doorbell register.

### API

```c
// Availability
bool cml_am_driver_available(void);

// Lifecycle
CMLAMDriver* cml_am_driver_create(void);
int          cml_am_driver_init(CMLAMDriver* drv);
void         cml_am_driver_free(CMLAMDriver* drv);

// Buffer management
CMLAMBuffer* cml_am_buffer_create(CMLAMDriver* drv, size_t size, bool vram);
void         cml_am_buffer_free(CMLAMDriver* drv, CMLAMBuffer* buf);
int          cml_am_buffer_upload(CMLAMDriver* drv, CMLAMBuffer* dst,
                                   const void* src, size_t n);
int          cml_am_buffer_download(CMLAMDriver* drv, CMLAMBuffer* src,
                                     void* dst, size_t n);

// Kernel management (load AMDGPU ELF code objects)
CMLAMKernel* cml_am_kernel_load(CMLAMDriver* drv, const void* code_object,
                                  size_t code_size, const char* kernel_name);
void         cml_am_kernel_free(CMLAMDriver* drv, CMLAMKernel* kernel);
int          cml_am_kernel_launch(CMLAMDriver* drv, CMLAMKernel* kernel,
                                   uint32_t grid[3], uint32_t block[3],
                                   void* kernarg, uint32_t kernarg_size);

// Synchronization
int cml_am_synchronize(CMLAMDriver* drv);

// Graph execution
int cml_am_execute_graph(CMLAMDriver* drv, CMLGraph_t ir);
```

---

## Vulkan Backend

**Header:** `include/ops/ir/gpu/vulkan_backend.h`
**Hardware:** Any GPU with Vulkan compute support
**CMake:** `-DENABLE_VULKAN=ON` (default)
**Runtime dependency:** `libvulkan.so.1`

A cross-vendor compute backend that dynamically loads all Vulkan functions at runtime. It creates an instance, enumerates physical devices, sets up a compute queue, and manages command buffer recording and submission.

### Key Types

- `CMLVulkanBackend` -- Full Vulkan state: instance, device, queue, command pool, device info, and all function pointers.
- `CMLVulkanBuffer` -- Buffer with device memory. Can be device-local (GPU-only) or host-visible (staging).
- `CMLVulkanKernel` -- Compiled compute pipeline from SPIR-V: shader module, pipeline layout, descriptor set.

### API

```c
// Availability
bool cml_vulkan_available(void);

// Lifecycle
CMLVulkanBackend* cml_vulkan_backend_create(void);
int               cml_vulkan_backend_init(CMLVulkanBackend* backend);
void              cml_vulkan_backend_free(CMLVulkanBackend* backend);

// Buffer management
CMLVulkanBuffer* cml_vulkan_buffer_create(CMLVulkanBackend* backend,
                                            VkDeviceSize size, bool device_local);
void             cml_vulkan_buffer_free(CMLVulkanBackend* backend, CMLVulkanBuffer* buf);
int              cml_vulkan_buffer_upload(CMLVulkanBackend* backend,
                                          CMLVulkanBuffer* dst,
                                          const void* src, size_t size);
int              cml_vulkan_buffer_download(CMLVulkanBackend* backend,
                                             CMLVulkanBuffer* src,
                                             void* dst, size_t size);

// Kernel management (from SPIR-V binary)
CMLVulkanKernel* cml_vulkan_kernel_create(CMLVulkanBackend* backend,
                                            const uint32_t* spirv,
                                            size_t spirv_size,
                                            const char* entry_point,
                                            int num_buffers);
void             cml_vulkan_kernel_free(CMLVulkanBackend* backend, CMLVulkanKernel* kernel);
int              cml_vulkan_kernel_bind_buffer(CMLVulkanBackend* backend,
                                                CMLVulkanKernel* kernel,
                                                int binding,
                                                CMLVulkanBuffer* buffer);
int              cml_vulkan_kernel_dispatch(CMLVulkanBackend* backend,
                                             CMLVulkanKernel* kernel,
                                             uint32_t gx, uint32_t gy, uint32_t gz);

// Graph execution
int cml_vulkan_execute_graph(CMLVulkanBackend* backend, CMLGraph_t ir);
int cml_vulkan_synchronize(CMLVulkanBackend* backend);
```

---

## SPIR-V Codegen

**Header:** `include/ops/ir/gpu/spirv_codegen.h`

Generates SPIR-V binary modules from UOps IR for use with the Vulkan backend. Emits compute shaders with storage buffer bindings and `GlobalInvocationID`-based thread indexing. Default workgroup size is 256x1x1.

### API

```c
CMLSPIRVCodegen* cml_spirv_codegen_create(void);
void             cml_spirv_codegen_destroy(CMLSPIRVCodegen* cg);

// Generate SPIR-V binary for specific operations
// Returns heap-allocated uint32_t array; caller must free(). out_size is byte count.
uint32_t* cml_spirv_gen_unary(CMLSPIRVCodegen* cg, UOpType op, const char* name, size_t* out_size);
uint32_t* cml_spirv_gen_binary(CMLSPIRVCodegen* cg, UOpType op, const char* name, size_t* out_size);
uint32_t* cml_spirv_gen_reduction(CMLSPIRVCodegen* cg, UOpType op, const char* name, size_t* out_size);
uint32_t* cml_spirv_gen_matmul(CMLSPIRVCodegen* cg, const char* name, size_t* out_size);
uint32_t* cml_spirv_gen_fill(CMLSPIRVCodegen* cg, float value, const char* name, size_t* out_size);
```

### SPIR-V Builder Utilities

Low-level helpers for constructing SPIR-V binary modules:

```c
SPIRVBuilder* spirv_builder_create(void);
void          spirv_builder_destroy(SPIRVBuilder* b);
void          spirv_builder_emit(SPIRVBuilder* b, uint32_t word);
uint32_t      spirv_builder_alloc_id(SPIRVBuilder* b);
uint32_t*     spirv_builder_finalize(SPIRVBuilder* b, size_t* out_size);
```

---

## WebGPU Backend

**Header:** `include/ops/ir/gpu/webgpu_backend.h`
**Hardware:** Any GPU supported by wgpu-native (Vulkan, Metal, DX12)
**CMake:** `-DENABLE_WEBGPU=ON` (default: OFF)
**Runtime dependency:** `libwgpu_native.so` / `libwgpu_native.dylib`

A cross-platform backend that uses the WebGPU API via wgpu-native. All wgpu functions are loaded dynamically. Kernels are written in WGSL (WebGPU Shading Language).

### Key Types

- `CMLWebGPUBackend` -- Backend context with wgpu instance, adapter, device, queue, and all dlsym'd function pointers.
- `CMLWebGPUKernel` -- Compiled compute pipeline from WGSL source.

### API

```c
// Availability
bool cml_webgpu_available(void);

// Lifecycle
CMLWebGPUBackend* cml_webgpu_backend_create(void);
int               cml_webgpu_backend_init(CMLWebGPUBackend* backend);
void              cml_webgpu_backend_free(CMLWebGPUBackend* backend);

// Compilation (WGSL source)
CMLWebGPUKernel* cml_webgpu_compile_wgsl(CMLWebGPUBackend* backend,
                                           const char* wgsl_source,
                                           const char* entry_point);
void cml_webgpu_kernel_free(CMLWebGPUKernel* kernel);

// Execution
int cml_webgpu_launch_kernel(CMLWebGPUBackend* backend, CMLWebGPUKernel* kernel,
                              size_t workgroup_count[3],
                              void** buffers, size_t* buffer_sizes, int num_buffers);

// Memory
void* cml_webgpu_alloc(CMLWebGPUBackend* backend, size_t size);
void  cml_webgpu_free(CMLWebGPUBackend* backend, void* buffer);
int   cml_webgpu_upload(CMLWebGPUBackend* backend, void* dst_buffer,
                         const void* src_host, size_t size);
int   cml_webgpu_download(CMLWebGPUBackend* backend, void* dst_host,
                           void* src_buffer, size_t size);

// WGSL codegen from IR
char* cml_wgsl_generate(struct IRNode* node);

// Graph execution
int cml_webgpu_execute_graph(CMLWebGPUBackend* backend, CMLGraph_t graph);
```

---

## Metal Backend

**Header:** `include/ops/ir/gpu/metal_backend.h`
**Hardware:** Apple GPUs (macOS, M-series and Intel Macs with discrete GPUs)
**CMake:** `-DENABLE_METAL=ON` (default on macOS)
**Runtime dependency:** Metal framework (macOS system library)

Uses the Metal API (`MTLDevice`, `MTLCommandQueue`, `MTLComputePipelineState`) for GPU compute on macOS. Kernels are written in MSL (Metal Shading Language).

### API

```c
// Availability
bool cml_metal_available(void);

// Lifecycle
CMLMetalBackend* cml_metal_backend_create(void);
int              cml_metal_backend_init(CMLMetalBackend* backend);
void             cml_metal_backend_free(CMLMetalBackend* backend);

// Compilation (MSL source)
CMLMetalKernel* cml_metal_compile_msl(CMLMetalBackend* backend,
                                       const char* msl_source,
                                       const char* function_name);
void cml_metal_kernel_free(CMLMetalKernel* kernel);

// Execution
int cml_metal_launch_kernel(CMLMetalBackend* backend, CMLMetalKernel* kernel,
                             size_t grid[3], size_t block[3],
                             void** buffers, int num_buffers);

// Memory
void* cml_metal_alloc(CMLMetalBackend* backend, size_t size);
void  cml_metal_free(CMLMetalBackend* backend, void* buffer);
int   cml_metal_upload(CMLMetalBackend* backend, void* dst_buffer,
                        const void* src_host, size_t size);
int   cml_metal_download(CMLMetalBackend* backend, void* dst_host,
                          const void* src_buffer, size_t size);

// MSL codegen from IR
char* cml_metal_generate_msl(struct IRNode* node);

// Graph execution
int cml_metal_execute_graph(CMLMetalBackend* backend, CMLGraph_t graph);
```

---

## OpenCL Backend

**Header:** `include/backend/opencl_backend.h`
**Hardware:** Any device with an OpenCL ICD (GPUs from NVIDIA, AMD, Intel, ARM, Qualcomm, etc.)
**CMake:** `-DENABLE_OPENCL=ON` (default)
**Runtime dependency:** OpenCL ICD loader (`libOpenCL.so`)

A cross-vendor backend that discovers OpenCL platforms/devices and provides tensor operation implementations through the `BackendOps` interface.

### API

```c
// Check availability
bool opencl_backend_is_available(void);

// Lifecycle
int  opencl_backend_init(void);
void opencl_backend_cleanup(void);

// Get backend operations (implements BackendOps interface)
BackendOps opencl_backend_get_ops(void);

// Device info
int opencl_backend_get_device_info(char* buffer, size_t buffer_size);
```

---

## Adreno Backend

**Header:** `include/ops/ir/gpu/adreno_backend.h`
**Hardware:** Qualcomm Adreno GPUs (Snapdragon SoCs -- Adreno 730, 740, 750, etc.)
**Target:** Mobile inference on Android devices

Uses OpenCL (or Vulkan) under the hood for Adreno GPU compute. Reports GPU version, global memory size, max allocation size, compute units, and max work group size.

### API

```c
bool              cml_adreno_available(void);
CMLAdrenoBackend* cml_adreno_backend_create(void);
int               cml_adreno_backend_init(CMLAdrenoBackend* backend);
void              cml_adreno_backend_free(CMLAdrenoBackend* backend);
int               cml_adreno_execute(CMLAdrenoBackend* backend, CMLGraph_t ir);
const char*       cml_adreno_device_info(const CMLAdrenoBackend* backend);
```

---

## Hexagon Backend

**Header:** `include/ops/ir/gpu/hexagon_backend.h`
**Hardware:** Qualcomm Hexagon DSP (V68, V69, V73, etc.)
**Target:** Quantized inference on Snapdragon SoCs

Provides DSP acceleration using Hexagon Vector eXtensions (HVX, 128-byte vectors) and optionally Hexagon Matrix eXtensions (HMX) for efficient quantized inference on mobile devices.

### API

```c
bool               cml_hexagon_available(void);
CMLHexagonBackend* cml_hexagon_backend_create(void);
int                cml_hexagon_backend_init(CMLHexagonBackend* backend);
void               cml_hexagon_backend_free(CMLHexagonBackend* backend);
int                cml_hexagon_execute(CMLHexagonBackend* backend, CMLGraph_t ir);
```

---

## Enabling Backends at Build Time

Backends are controlled via CMake options:

```bash
cmake -B build \
  -DENABLE_CUDA=ON \
  -DENABLE_ROCM=ON \
  -DENABLE_VULKAN=ON \
  -DENABLE_OPENCL=ON \
  -DENABLE_METAL=ON \
  -DENABLE_WEBGPU=OFF \
  ..
```

All backends except WebGPU are enabled by default (Metal only on macOS). Each backend is compiled conditionally and uses dynamic loading, so the build will succeed even if the corresponding SDK/drivers are not installed -- the `cml_*_available()` check will simply return `false` at runtime.

## Common Pattern

All backends follow the same lifecycle pattern:

```c
// 1. Check availability
if (!cml_<backend>_available()) {
    // Backend not available on this system
}

// 2. Create + initialize
CML<Backend>* backend = cml_<backend>_backend_create();
cml_<backend>_backend_init(backend);

// 3. Compile kernels, allocate buffers, dispatch, synchronize
// ...

// 4. Execute an IR graph (high-level)
cml_<backend>_execute_graph(backend, graph);

// 5. Cleanup
cml_<backend>_backend_free(backend);
```
