/**
 * @file mlir_multi_backend.c
 * @brief Multi-backend support for MLIR (CPU, CUDA, Vulkan, Metal)
 */

#include "ops/ir/mlir/mlir_backend.h"
#include "ops/ir/mlir/mlir_internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#ifdef CML_HAS_MLIR
#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>
#include <mlir-c/Conversion.h>
// #include <mlir-c/Dialect/GPU.h> // Uncomment if available in your MLIR version
#include <mlir-c/Target/LLVMIR.h>
#endif

// ============================================================================
// Backend Management
// ============================================================================

int cml_mlir_set_target(CMLMLIRContext* ctx, MLIRTargetBackend target) {
#ifdef CML_HAS_MLIR
    if (!ctx)
        return -1;

    const char* target_names[] = {"CPU", "CUDA", "ROCm", "Vulkan", "Metal", "WebGPU"};

    LOG_INFO("Setting MLIR target backend to: %s", target_names[target]);
    ctx->target = target;

    return 0;
#else
    (void)ctx;
    (void)target;
    return -1;
#endif
}

MLIRTargetBackend cml_mlir_get_target(CMLMLIRContext* ctx) {
#ifdef CML_HAS_MLIR
    if (!ctx)
        return MLIR_TARGET_CPU;
    return ctx->target;
#else
    (void)ctx;
    return MLIR_TARGET_CPU;
#endif
}

// ============================================================================
// Backend-Specific Compilation
// ============================================================================

void* cml_mlir_compile_for_target(CMLMLIRContext* ctx, void* module, MLIRTargetBackend target) {
#ifdef CML_HAS_MLIR
    if (!ctx || !module)
        return NULL;

    MlirModule mlir_module = {module};
    MlirContext mlir_ctx   = ctx->context;

    LOG_INFO("Compiling MLIR module for target backend");

    // Create pass manager for target-specific lowering
    MlirPassManager pm = mlirPassManagerCreate(mlir_ctx);

    switch (target) {
    case MLIR_TARGET_CPU: {
        LOG_INFO("Target: CPU (LLVM)");

        // Lower to LLVM dialect (required for CPU execution)
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertMathToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionArithToLLVMConversionPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertControlFlowToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionReconcileUnrealizedCastsPass());

        LOG_DEBUG("Added LLVM conversion passes for CPU target");
        break;
    }

    case MLIR_TARGET_CUDA: {
        LOG_INFO("Target: CUDA (NVVM)");

        // For CUDA, we first lower to LLVM, then use NVPTX backend
        // Lower to LLVM dialect first
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertMathToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionArithToLLVMConversionPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertControlFlowToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionReconcileUnrealizedCastsPass());

        // Note: NVPTX-specific passes would be applied during codegen
        LOG_DEBUG("Lowered to LLVM for CUDA (NVPTX codegen handled separately)");
        break;
    }

    case MLIR_TARGET_VULKAN: {
        LOG_INFO("Target: Vulkan (SPIR-V)");

        // Lower to LLVM first, then SPIR-V conversion would happen
        // For now, we lower to LLVM and SPIR-V generation happens in codegen
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertMathToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionArithToLLVMConversionPass());

        LOG_DEBUG("Lowered to LLVM for Vulkan (SPIR-V codegen handled separately)");
        break;
    }

    case MLIR_TARGET_METAL: {
        LOG_INFO("Target: Metal (MSL)");

        // Lower to LLVM for Metal (MSL generation happens in codegen)
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertMathToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionArithToLLVMConversionPass());

        LOG_DEBUG("Lowered to LLVM for Metal (MSL codegen handled separately)");
        break;
    }

    case MLIR_TARGET_ROCM: {
        LOG_INFO("Target: ROCm (ROCDL)");

        // Lower to LLVM for ROCm (ROCDL generation happens in codegen)
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertMathToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionArithToLLVMConversionPass());

        LOG_DEBUG("Lowered to LLVM for ROCm (ROCDL codegen handled separately)");
        break;
    }

    case MLIR_TARGET_WEBGPU: {
        LOG_INFO("Target: WebGPU (WGSL)");

        // Lower to LLVM for WebGPU (WGSL generation happens in codegen)
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertMathToLLVMPass());
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionArithToLLVMConversionPass());

        LOG_DEBUG("Lowered to LLVM for WebGPU (WGSL codegen handled separately)");
        break;
    }

    default:
        LOG_ERROR("Unknown target backend: %d", target);
        mlirPassManagerDestroy(pm);
        return NULL;
    }

    // Run the lowering passes
    MlirOperation module_op  = mlirModuleGetOperation(mlir_module);
    MlirLogicalResult result = mlirPassManagerRunOnOp(pm, module_op);
    mlirPassManagerDestroy(pm);

    if (mlirLogicalResultIsFailure(result)) {
        LOG_ERROR("Failed to lower MLIR module for target");
        return NULL;
    }

    LOG_INFO("Successfully compiled for target backend");
    return module;

#else
    (void)ctx;
    (void)module;
    (void)target;
    return NULL;
#endif
}

// ============================================================================
// Target-Specific Code Generation
// ============================================================================

// ============================================================================
// Runtime Device Selection
// ============================================================================

typedef struct {
    MLIRTargetBackend backend;
    bool available;
    const char* name;
    int compute_capability; // For CUDA
} MLIRDeviceInfo;

static MLIRDeviceInfo available_devices[6] = {
    {MLIR_TARGET_CPU, true, "CPU", 0},      {MLIR_TARGET_CUDA, false, "CUDA", 0},
    {MLIR_TARGET_ROCM, false, "ROCm", 0},   {MLIR_TARGET_VULKAN, false, "Vulkan", 0},
    {MLIR_TARGET_METAL, false, "Metal", 0}, {MLIR_TARGET_WEBGPU, false, "WebGPU", 0}};

void cml_mlir_detect_devices(void) {
    LOG_INFO("Detecting available compute devices...");

    // CPU is always available
    available_devices[0].available = true;
    LOG_INFO("  [✓] CPU");

    // Detect CUDA
#ifdef __NVCC__
    int cuda_device_count = 0;
    cudaError_t err       = cudaGetDeviceCount(&cuda_device_count);
    if (err == cudaSuccess && cuda_device_count > 0) {
        available_devices[1].available = true;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        available_devices[1].compute_capability = prop.major * 10 + prop.minor;
        LOG_INFO("  [✓] CUDA (Device: %s, Compute: %d.%d)", prop.name, prop.major, prop.minor);
    } else {
        LOG_INFO("  [✗] CUDA (not available)");
    }
#else
    // Try to detect via system check
    FILE* fp = popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r");
    if (fp) {
        char gpu_name[256];
        if (fgets(gpu_name, sizeof(gpu_name), fp)) {
            available_devices[1].available = true;
            LOG_INFO("  [✓] CUDA (Device detected via nvidia-smi)");
        } else {
            LOG_INFO("  [✗] CUDA (not available)");
        }
        pclose(fp);
    } else {
        LOG_INFO("  [✗] CUDA (not available)");
    }
#endif

    // Detect ROCm
#ifdef __HIP_PLATFORM_AMD__
    int rocm_device_count = 0;
    hipError_t hip_err    = hipGetDeviceCount(&rocm_device_count);
    if (hip_err == hipSuccess && rocm_device_count > 0) {
        available_devices[2].available = true;
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        LOG_INFO("  [✓] ROCm (Device: %s)", prop.name);
    } else {
        LOG_INFO("  [✗] ROCm (not available)");
    }
#else
    FILE* rocm_fp = popen("rocm-smi --showproductname 2>/dev/null", "r");
    if (rocm_fp) {
        char line[256];
        if (fgets(line, sizeof(line), rocm_fp)) {
            available_devices[2].available = true;
            LOG_INFO("  [✓] ROCm (Device detected)");
        } else {
            LOG_INFO("  [✗] ROCm (not available)");
        }
        pclose(rocm_fp);
    } else {
        LOG_INFO("  [✗] ROCm (not available)");
    }
#endif

    // Detect Vulkan
#ifdef VK_VERSION_1_0
    VkInstance instance;
    VkInstanceCreateInfo createInfo = {0};
    createInfo.sType                = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    if (vkCreateInstance(&createInfo, NULL, &instance) == VK_SUCCESS) {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
        if (deviceCount > 0) {
            available_devices[3].available = true;
            LOG_INFO("  [✓] Vulkan (%d device(s))", deviceCount);
        } else {
            LOG_INFO("  [✗] Vulkan (no devices)");
        }
        vkDestroyInstance(instance, NULL);
    } else {
        LOG_INFO("  [✗] Vulkan (not available)");
    }
#else
    // Check for Vulkan loader
    FILE* vk_fp = popen("vulkaninfo --summary 2>/dev/null | grep -i 'device name'", "r");
    if (vk_fp) {
        char line[256];
        if (fgets(line, sizeof(line), vk_fp)) {
            available_devices[3].available = true;
            LOG_INFO("  [✓] Vulkan (Device detected)");
        } else {
            LOG_INFO("  [✗] Vulkan (not available)");
        }
        pclose(vk_fp);
    } else {
        LOG_INFO("  [✗] Vulkan (not available)");
    }
#endif

    // Detect Metal (macOS only)
#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_MAC
    // Metal is available on all modern Macs
    available_devices[4].available = true;
    LOG_INFO("  [✓] Metal (macOS)");
#else
    LOG_INFO("  [✗] Metal (not macOS)");
#endif
#else
    LOG_INFO("  [✗] Metal (not available - macOS only)");
#endif

    // Detect WebGPU
    // WebGPU is typically available in browsers, not native
    // For native, we'd need Dawn or wgpu-native
    LOG_INFO("  [✗] WebGPU (browser-only, not available for native)");
}

MLIRTargetBackend cml_mlir_select_best_device(void) {
    // Priority: CUDA > Metal > Vulkan > ROCm > CPU

    if (available_devices[1].available)
        return MLIR_TARGET_CUDA;
    if (available_devices[4].available)
        return MLIR_TARGET_METAL;
    if (available_devices[3].available)
        return MLIR_TARGET_VULKAN;
    if (available_devices[2].available)
        return MLIR_TARGET_ROCM;

    return MLIR_TARGET_CPU;
}
