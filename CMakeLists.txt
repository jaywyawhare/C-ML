cmake_minimum_required(VERSION 3.16)
project(C-ML VERSION 0.0.3 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckCCompilerFlag)

# Warnings in every build type (not just Debug); opt-in -Werror for CI.
option(CML_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall> $<$<COMPILE_LANGUAGE:C>:-Wextra>)
if(CML_WARNINGS_AS_ERRORS)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Werror>)
endif()

# Drop unreferenced code/data at link time (--gc-sections). Requires function/
# data sections so the linker can see section granularity; only meaningful for
# the shared lib and executables (static archives keep everything anyway).
option(CML_LINK_GC_SECTIONS "Link with --gc-sections to drop unreferenced code" OFF)
if(CML_LINK_GC_SECTIONS AND NOT MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
        $<$<COMPILE_LANGUAGE:C>:-fdata-sections>)
    add_link_options(-Wl,--gc-sections)
endif()

# Host-native codegen is a local-build convenience, not something a
# distributed binary should ship with — turn off for packaging/CI.
option(CML_NATIVE_OPTIMIZATIONS "Optimize for the build host CPU (-march=native)" ON)
if(CML_NATIVE_OPTIMIZATIONS)
    # Probe with -Werror so a flag that is merely accepted-with-a-warning
    # (e.g. -march=native on Apple clang / arm64) is treated as unsupported
    # rather than poisoning every subsequent compile under -Werror.
    set(_save_req_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror")
    check_c_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(NOT HAS_MARCH_NATIVE)
        # Apple silicon uses -mcpu=native; x86 uses -march=native.
        check_c_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
    endif()
    set(CMAKE_REQUIRED_FLAGS "${_save_req_flags}")
    if(HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
    elseif(HAS_MCPU_NATIVE)
        add_compile_options(-mcpu=native)
    endif()
endif()

option(ENABLE_SANITIZERS "AddressSanitizer + LeakSanitizer (Linux) + UndefinedBehaviorSanitizer (GCC/Clang)" OFF)
if(ENABLE_SANITIZERS)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        message(FATAL_ERROR "ENABLE_SANITIZERS requires GCC or Clang")
    endif()
    add_compile_options(
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer
        -fno-common
        -g
    )
    add_link_options(
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer
        -fno-common
    )
    message(STATUS "Sanitizers: address (leak detection on Linux), undefined behavior")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(CheckCCompilerFlag)

option(ENABLE_LLVM_BACKEND "Enable direct LLVM IR JIT backend" ON)

if(ENABLE_LLVM_BACKEND)
    find_package(LLVM QUIET CONFIG)
    if(LLVM_PACKAGE_VERSION)
        include(CheckIncludeFile)
        set(CMAKE_REQUIRED_INCLUDES "${LLVM_INCLUDE_DIRS}")
        check_include_file("llvm-c/LLJIT.h" HAS_LLVM_LLJIT)

        if(HAS_LLVM_LLJIT)
            add_definitions(-DCML_HAS_LLVM_BACKEND ${LLVM_DEFINITIONS})
            # The ORC C-API for pairing an externally-built module with a
            # ThreadSafeContext differs across LLVM versions: LLVM 19+ has
            # LLVMOrcCreateNewThreadSafeContextFromLLVMContext, older ones use
            # LLVMOrcThreadSafeContextGetContext. Detect which actually compiles
            # (with -Werror, so a deprecated/absent decl is rejected) rather than
            # guessing from the version number.
            include(CheckCSourceCompiles)
            set(CMAKE_REQUIRED_INCLUDES ${LLVM_INCLUDE_DIRS})
            # Prefer the modern adopt call; require it to compile clean (-Werror
            # rejects an absent/implicit declaration).
            set(CMAKE_REQUIRED_FLAGS "-Werror")
            check_c_source_compiles(
                "#include <llvm-c/Orc.h>\nint main(void){(void)LLVMOrcCreateNewThreadSafeContextFromLLVMContext;return 0;}"
                CML_HAVE_ORC_FROM_LLVM_CONTEXT)
            # GetContext is deprecated on some LLVMs (18) but still usable; probe
            # it WITHOUT -Werror so a deprecation doesn't hide it. The call site
            # suppresses the deprecation warning locally.
            unset(CMAKE_REQUIRED_FLAGS)
            check_c_source_compiles(
                "#include <llvm-c/Orc.h>\nint main(void){(void)LLVMOrcThreadSafeContextGetContext;return 0;}"
                CML_HAVE_ORC_TSC_GET_CONTEXT)
            unset(CMAKE_REQUIRED_INCLUDES)
            add_definitions(-DCML_ORC_DETECTED)
            if(CML_HAVE_ORC_FROM_LLVM_CONTEXT)
                add_definitions(-DCML_HAVE_ORC_FROM_LLVM_CONTEXT)
            endif()
            if(CML_HAVE_ORC_TSC_GET_CONTEXT)
                add_definitions(-DCML_HAVE_ORC_TSC_GET_CONTEXT)
            endif()
            # Resolve the link libraries with `llvm-config --libs`, which adapts
            # to the install layout: a single -lLLVM where only the monolithic
            # shared lib exists, or the component archives otherwise. The plain
            # monolithic-target shortcut / llvm_map_components shortcuts each
            # worked in only one of {local, CI} -- some libLLVM.so builds don't
            # export the ORC C-API (LLVMOrcCreateNewThreadSafeContextFromLLVMContext),
            # and some installs ship no component archives. llvm-config gets both.
            find_program(LLVM_CONFIG_EXE
                NAMES llvm-config llvm-config-22 llvm-config-21 llvm-config-20 llvm-config-19 llvm-config-18
                HINTS ${LLVM_TOOLS_BINARY_DIR})
            if(LLVM_CONFIG_EXE)
                # Link via llvm-config --libs: the shared libLLVM (or component
                # archives, whichever the install provides) carries the ORC C-API
                # and its C++ runtime. The code uses feature detection (above) to
                # avoid API calls absent from the target's headers, so no static
                # link is needed -- and static archives would drag in unresolved
                # libstdc++ symbols from LLVM's C++ objects.
                execute_process(COMMAND ${LLVM_CONFIG_EXE} --ldflags
                    OUTPUT_VARIABLE _llvm_ldflags OUTPUT_STRIP_TRAILING_WHITESPACE)
                execute_process(COMMAND ${LLVM_CONFIG_EXE} --libs core orcjit native nvptx amdgpu
                    OUTPUT_VARIABLE _llvm_libs OUTPUT_STRIP_TRAILING_WHITESPACE)
                execute_process(COMMAND ${LLVM_CONFIG_EXE} --system-libs
                    OUTPUT_VARIABLE _llvm_syslibs OUTPUT_STRIP_TRAILING_WHITESPACE)
                separate_arguments(LLVM_BACKEND_LIBS_LIST UNIX_COMMAND
                    "${_llvm_ldflags} ${_llvm_libs} ${_llvm_syslibs}")
            elseif(TARGET LLVM)
                set(LLVM_BACKEND_LIBS_LIST LLVM)
            else()
                include(${LLVM_CMAKE_DIR}/LLVM-Config.cmake)
                llvm_map_components_to_libnames(LLVM_BACKEND_LIBS_LIST core orcjit native nvptx amdgpu)
            endif()
            message(STATUS "LLVM backend enabled (LLVM ${LLVM_PACKAGE_VERSION})")
            set(LLVM_BACKEND_FOUND TRUE)
        else()
            message(STATUS "LLVM backend: LLJIT header not found, disabling")
            set(LLVM_BACKEND_FOUND FALSE)
        endif()
    else()
        message(STATUS "LLVM CMake package not found — LLVM backend disabled")
        set(LLVM_BACKEND_FOUND FALSE)
    endif()
else()
    message(STATUS "LLVM backend disabled")
    set(LLVM_BACKEND_FOUND FALSE)
endif()

option(ENABLE_CUDA "Enable CUDA backend" ON)
if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
        add_definitions(-DCML_HAS_CUDA)
        set(CUDA_LIBS "")
    else()
        message(STATUS "CUDA Toolkit not found - CUDA backend uses dynamic loading")
    endif()
endif()

option(ENABLE_ROCM "Enable ROCm/HIP backend" ON)
if(ENABLE_ROCM)
    find_package(hip QUIET CONFIG)
    if(hip_FOUND)
        message(STATUS "ROCm/HIP found: ${hip_VERSION}")
        add_definitions(-DCML_HAS_ROCM)
        set(ROCM_LIBS hip::host)
    else()
        message(STATUS "ROCm/HIP not found - ROCm backend uses dynamic loading")
    endif()
endif()

option(ENABLE_OPENCL "Enable OpenCL backend" ON)
if(ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found: ${OpenCL_VERSION_STRING}")
        add_definitions(-DCML_HAS_OPENCL)
        set(OPENCL_LIBS ${OpenCL_LIBRARIES})
    else()
        message(STATUS "OpenCL not found - OpenCL backend disabled")
        set(OPENCL_LIBS "")
    endif()
endif()

option(ENABLE_IO_URING "Enable io_uring async disk I/O (Linux, requires liburing)" ON)
set(IOURING_LIBS "")
if(ENABLE_IO_URING AND UNIX AND NOT APPLE)
    find_path(LIBURING_INCLUDE_DIR liburing.h)
    find_library(LIBURING_LIBRARY uring)
    if(LIBURING_INCLUDE_DIR AND LIBURING_LIBRARY)
        message(STATUS "liburing found: ${LIBURING_LIBRARY}")
        add_definitions(-DCML_HAS_IO_URING)
        set(IOURING_LIBS ${LIBURING_LIBRARY})
    else()
        message(STATUS "liburing not found - async disk I/O falls back to synchronous")
    endif()
endif()

option(ENABLE_METAL "Enable Metal backend (macOS only)" ON)
set(METAL_FOUND FALSE)
if(ENABLE_METAL AND APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        enable_language(OBJC)
        add_definitions(-DCML_HAS_METAL)
        set(METAL_LIBS ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
        set(METAL_FOUND TRUE)
        message(STATUS "Metal backend enabled")
    else()
        message(STATUS "Metal framework not found - Metal backend disabled")
    endif()
else()
    message(STATUS "Metal backend disabled (not macOS or disabled)")
endif()

option(ENABLE_WEBGPU "Enable WebGPU backend via wgpu-native" OFF)
if(ENABLE_WEBGPU)
    add_definitions(-DCML_HAS_WEBGPU)
    message(STATUS "WebGPU backend enabled (dynamic loading)")
endif()

option(ENABLE_VULKAN "Enable Vulkan/SPIR-V compute backend" ON)
if(ENABLE_VULKAN)
    add_definitions(-DCML_HAS_VULKAN)
    message(STATUS "Vulkan backend enabled (dynamic loading)")
endif()

option(ENABLE_NV_DRIVER "Enable NVIDIA userspace driver (direct ioctl)" OFF)
if(ENABLE_NV_DRIVER)
    add_definitions(-DCML_HAS_NV_DRIVER)
    message(STATUS "NV userspace driver enabled")
endif()

option(ENABLE_AM_DRIVER "Enable AMD userspace driver (KFD ioctl)" OFF)
if(ENABLE_AM_DRIVER)
    add_definitions(-DCML_HAS_AM_DRIVER)
    message(STATUS "AM userspace driver enabled")
endif()

option(ENABLE_NIR "Enable Mesa NIR multi-vendor GPU backend" OFF)
if(ENABLE_NIR)
    add_definitions(-DCML_HAS_NIR)
    message(STATUS "NIR/Mesa backend enabled (dynamic loading)")
endif()

option(ENABLE_ONNX "Enable ONNX model loading and execution" ON)
if(ENABLE_ONNX)
    add_definitions(-DCML_HAS_ONNX)
    message(STATUS "ONNX runtime enabled")
endif()

set(CORE_SOURCES
    src/cml.c
    src/core/logging.c
    src/core/cml_flags.c
    src/core/config.c
    src/core/dataset.c
    src/core/augmentation.c
    src/core/model_architecture.c
    src/core/training_loop.c
    src/core/training_metrics.c
    src/core/experiment.c
    src/core/serialization.c
    src/core/gguf.c
    src/core/safetensors.c
    src/core/cleanup.c
    src/core/error_stack.c
    src/core/computation_graph.c
    src/core/graph_context.c
    src/core/quantization.c
    src/core/gguf_quant.c
    src/core/protobuf_mini.c
    src/core/pth_loader.c
    src/core/tinyfs.c
    src/core/threefry.c
    src/core/mlperf_logging.c
    src/core/hevc.c
)

if(ENABLE_ONNX)
    list(APPEND CORE_SOURCES src/core/onnx.c src/core/onnx_ops.c src/core/onnx_export.c)
endif()

set(ALLOC_SOURCES
    src/alloc/memory_management.c
    src/alloc/memory_pools.c
    src/alloc/graph_allocator.c
    src/alloc/tlsf_alloc.c
    src/alloc/cml_allocator.c
)

set(BACKEND_SOURCES
    src/backend/backend.c
    src/backend/backend_buffer.c
    src/backend/device.c
    src/backend/threadpool.c
    src/backend/profiling.c
    src/backend/blas.c
    src/backend/opencl_backend.c
    src/backend/null_device.c
    src/backend/disk_backend.c
    src/backend/usb_device.c
    src/backend/remote_device.c
    src/backend/usb3_gpu.c
    src/backend/thunder_executor.c
)

set(OPS_SOURCES
    src/ops/uops.c
    src/ops/uops_optim_steps.c
    src/ops/uops_conv.c
    src/ops/winograd.c
    src/ops/ir/ir.c
    src/ops/ir/intern.c
    src/ops/ir/context.c
    src/ops/ir/execution.c
    src/ops/ir/execution_typed.c
    src/ops/ir/flamegraph.c
    src/ops/ir/cpu_lazy_materialize.c
    src/ops/ir/backward.c
    src/ops/ir/export.c
    src/ops/ir/optimization.c
    src/ops/ir/graph_cache.c
    src/ops/ir/aot.c
    src/ops/ir/kernel_cache.c
    src/ops/ir/disk_cache.c
    src/ops/ir/fusion_patterns.c
    src/ops/ir/dispatch.c
    src/ops/ir/decompose.c
    src/ops/ir/autodiff.c
    src/ops/ir/pattern_matcher.c
    src/ops/ir/tree_automaton.c
    src/ops/ir/hcq.c
    src/ops/ir/hcq_backend.c
    src/ops/ir/hcq_cuda.c
    src/ops/ir/hcq_opencl.c
    src/ops/ir/hcq_vulkan.c
    src/ops/ir/hcq_nv.c
    src/ops/ir/hcq_am.c
    src/ops/ir/hcq_rocm.c
    src/ops/ir/hcq_webgpu.c
    src/ops/ir/beam_search.c
    src/ops/ir/opt_transforms.c
    src/ops/ir/tc_opt.c
    src/ops/ir/heuristic_opt.c
    src/ops/ir/trace.c
    src/ops/ir/tiny_jit.c
    src/ops/ir/beam_cuda_timing.c
    src/ops/ir/schedule.c
    src/ops/ir/fusion_schedule.c
    src/ops/ir/fuse_elementwise.c
    src/ops/ir/linearize.c
    src/ops/ir/rangeify.c
    src/ops/ir/fused_codegen.c
    src/ops/ir/runtime_compiler.c
    src/ops/ir/graph_capture.c
    src/ops/ir/nir_compiler.c
    src/ops/ir/hcq_nir.c
    src/ops/ir/z3_verify.c
    src/ops/ir/compiler_viz.c
    src/ops/ir/process_replay.c
    src/ops/ir/memory_planner.c
    src/ops/ir/gpu/cuda_backend.c
    src/ops/ir/gpu/cuda_graph_replay.c
    src/ops/ir/multi_output.c
    src/ops/ir/cross_boundary_fusion.c
    src/ops/ir/gpu/rocm_backend.c
    src/ops/ir/gpu/wmma.c
    src/ops/ir/gpu/wgsl_codegen.c
    src/ops/ir/gpu/ptx_codegen.c
    src/ops/ir/gpu/nv_driver.c
    src/ops/ir/gpu/nv_qmd.c
    src/ops/ir/gpu/am_driver.c
    src/ops/ir/gpu/adreno_backend.c
    src/ops/ir/gpu/hexagon_backend.c
    src/ops/ir/gpu/hexagon_sim.c
    src/ops/ir/gpu/hip_mock.c
    src/ops/ir/gpu/amd_profiling.c
    src/ops/ir/gpu/amx.c
    src/ops/ir/gpu/xmx.c
    src/ops/ir/gpu/rdna3_emu.c
    src/ops/simd_utils.c
    src/ops/simd_math.c
    src/ops/simd_views.c
)

if(METAL_FOUND)
    list(APPEND OPS_SOURCES src/ops/ir/gpu/metal_backend.m)
    list(APPEND OPS_SOURCES src/ops/ir/gpu/metal_codegen.m)
    list(APPEND OPS_SOURCES src/ops/ir/gpu/metal_mps.m)
    list(APPEND METAL_LIBS "-framework MetalPerformanceShaders")
endif()

list(APPEND OPS_SOURCES src/ops/ir/gpu/webgpu_backend.c)

list(APPEND OPS_SOURCES
    src/ops/ir/gpu/vulkan_backend.c
    src/ops/ir/gpu/spirv_codegen.c
    src/ops/ir/gpu/opencl_ir_backend.c
)

if(ENABLE_NIR)
    # nir_compiler.c / hcq_nir.c are always compiled (test_nir_compiler links
    # them unconditionally); the option activates their dispatch integration
    # via CML_HAS_NIR. The old `list(APPEND ...)` here double-compiled both TUs.
    add_definitions(-DCML_HAS_NIR)
endif()

option(ENABLE_NAK "Enable NAK (Nouveau Architecture Kompiler) backend" OFF)
if(ENABLE_NAK)
    add_definitions(-DCML_HAS_NAK)
    message(STATUS "NAK backend enabled (dynamic loading)")
endif()
list(APPEND OPS_SOURCES src/ops/ir/gpu/nak_backend.c)

# fusion_schedule.c / linearize.c are already in the OPS_SOURCES base list.
list(APPEND OPS_SOURCES src/ops/ir/spec.c src/ops/ir/late_passes.c)
list(APPEND OPS_SOURCES
    src/ops/ir/validate.c
    src/ops/ir/viz_server.c
    src/ops/ir/schedule_indexing.c
    src/ops/ir/schedule_multi.c
    src/ops/ir/schedule_allreduce.c
)

if(LLVM_BACKEND_FOUND)
    list(APPEND OPS_SOURCES
        src/ops/ir/llvm/llvm_backend.c
        src/ops/ir/gpu/gpu_codegen.c
        src/ops/ir/gpu/amd_llvm.c
    )
endif()

set(TENSOR_SOURCES
    src/tensor/tensor.c
    src/tensor/tensor_views.c
    src/tensor/tensor_manipulation.c
    src/tensor/tensor_ops_extra.c
    src/tensor/sparse_tensor.c
    src/tensor/image_dtype.c
    src/tensor/shard.c
    src/tensor/tensor_hash.c
    src/tensor/shape_tracker.c
    src/tensor/dtype_vec.c
    src/tensor/realize.c
)

set(AUTOGRAD_SOURCES
    src/autograd/autograd.c
    src/autograd/forward_ops.c
    src/autograd/loss_functions.c
    src/autograd/checkpointing.c
    src/autograd/amp.c
)

set(NN_SOURCES
    src/nn.c
    src/nn/model_io.c
    src/nn/layers/linear.c
    src/nn/layers/activations.c
    src/nn/layers/dropout.c
    src/nn/layers/conv2d.c
    src/nn/layers/norm.c
    src/nn/layers/layernorm.c
    src/nn/layers/pooling.c
    src/nn/layers/sequential.c
    src/nn/layers/conv1d.c
    src/nn/layers/conv3d.c
    src/nn/layers/groupnorm.c
    src/nn/layers/embedding.c
    src/nn/layers/containers.c
    src/nn/layers/rnn.c
    src/nn/layers/transformer.c
    src/nn/layers/rmsnorm.c
    src/nn/layers/conv_transpose2d.c
    src/nn/layers/instancenorm.c
    src/nn/layers/conv_transpose1d.c
    src/nn/layers/layernorm2d.c
    src/nn/layers/conv_transpose3d.c
    src/nn/layers/upsample.c
    src/nn/layers/pixel_shuffle.c
    src/nn/layers/flatten.c
    src/nn/layers/identity.c
    src/nn/layers/prelu.c
    src/nn/llm_ops.c
    src/nn/llama.c
    src/nn/lora.c
    src/nn/qlora.c
    src/nn/paged_attention.c
    src/nn/serving.c
    src/nn/init.c
    src/nn/speculative.c
    src/nn/openai_api.c
    src/nn/state.c
)

set(SYMBOLIC_SOURCES
    src/symbolic/symbolic.c
    src/symbolic/divandmod.c
)

set(OPTIM_SOURCES src/optim.c src/optim/lr_scheduler.c)

set(TORCH_SOURCES
    src/torch/torch_c.c
    src/torch/torch_eager.c
    src/torch/pte.c
    src/torch/selective_build.c
    src/torch/memory.c
    src/torch/delegate.c
)

option(CML_TORCH_SELECTIVE_OPS "Comma-separated UOp names for selective kernel build (empty=all)" "")

function(cml_apply_torch_selective_ops target)
    if(CML_TORCH_SELECTIVE_OPS)
        target_compile_definitions(${target} PRIVATE
            CML_TORCH_SELECTIVE_OPS="${CML_TORCH_SELECTIVE_OPS}")
    endif()
endfunction()

if(CML_TORCH_SELECTIVE_OPS)
    message(STATUS "Torch selective build ops: ${CML_TORCH_SELECTIVE_OPS}")
endif()


option(ENABLE_DISTRIBUTED "Enable distributed training" OFF)
option(ENABLE_NCCL "Enable NCCL backend (requires libnccl)" ON)
option(ENABLE_MPI "Enable MPI backend (requires libmpi)" ON)

option(ENABLE_IB_TRANSPORT "Enable InfiniBand RDMA transport (dynamic loading)" OFF)
if(ENABLE_IB_TRANSPORT)
    add_definitions(-DCML_HAS_IB_TRANSPORT)
    message(STATUS "InfiniBand RDMA transport enabled (dynamic loading)")
endif()

set(DISTRIBUTED_SOURCES
    src/distributed/distributed.c
    src/distributed/comm_backend.c
    src/distributed/gloo_backend.c
    src/distributed/nccl_backend.c
    src/distributed/mpi_backend.c
    src/distributed/ring_allreduce.c
    src/distributed/data_parallel.c
    src/distributed/pipeline_parallel.c
    src/distributed/tensor_parallel.c
    src/distributed/ib_transport.c
)
if(ENABLE_DISTRIBUTED)
    add_definitions(-DCML_HAS_DISTRIBUTED)
endif()

set(DATASET_SOURCES
    src/datasets/datasets.c
    src/datasets/csv_parser.c
    src/datasets/idx_parser.c
    src/datasets/builtin_data.c
    src/datasets/loaders.c
)

option(BUILD_MODEL_ZOO "Build model zoo" ON)
if(BUILD_MODEL_ZOO)
    set(ZOO_SOURCES
        src/zoo/zoo.c
        src/zoo/resnet.c
        src/zoo/gpt2.c
        src/zoo/bert.c
        src/zoo/efficientnet.c
        src/zoo/yolov8.c
        src/zoo/whisper.c
        src/zoo/stable_diffusion.c
        src/zoo/vit.c
        src/zoo/clip.c
        src/zoo/t5.c
        src/zoo/unet.c
        src/zoo/convnext.c
        src/zoo/inception.c
        src/zoo/retinanet.c
        src/zoo/mask_rcnn.c
        src/zoo/unet3d.c
        src/zoo/rnnt.c
    )
    message(STATUS "Model zoo enabled")
else()
    set(ZOO_SOURCES "")
    message(STATUS "Model zoo disabled")
endif()

set(LIBRARY_SOURCES
    ${CORE_SOURCES}
    ${ALLOC_SOURCES}
    ${BACKEND_SOURCES}
    ${OPS_SOURCES}
    ${TENSOR_SOURCES}
    ${AUTOGRAD_SOURCES}
    ${NN_SOURCES}
    ${OPTIM_SOURCES}
    ${TORCH_SOURCES}
    ${SYMBOLIC_SOURCES}
    ${DISTRIBUTED_SOURCES}
    ${ZOO_SOURCES}
    ${DATASET_SOURCES}
)

function(cml_link_all target)
    # Scoped instead of a global include_directories(): BUILD_INTERFACE keeps
    # the source-tree path out of the installed CMLTargets export.
    target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
    # Third-party backend headers are SYSTEM so their warnings never trip our
    # -Werror builds; only targets compiling library sources need them.
    if(LLVM_INCLUDE_DIRS)
        target_include_directories(${target} SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
    endif()
    if(CUDAToolkit_INCLUDE_DIRS)
        target_include_directories(${target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    if(OpenCL_INCLUDE_DIRS)
        target_include_directories(${target} SYSTEM PRIVATE ${OpenCL_INCLUDE_DIRS})
    endif()
    target_link_libraries(${target} m dl pthread)
    if(LLVM_BACKEND_FOUND)
        target_link_libraries(${target} ${LLVM_BACKEND_LIBS_LIST})
    endif()
    if(CUDA_LIBS)
        target_link_libraries(${target} ${CUDA_LIBS})
    endif()
    if(ROCM_LIBS)
        target_link_libraries(${target} ${ROCM_LIBS})
    endif()
    if(OPENCL_LIBS)
        target_link_libraries(${target} ${OPENCL_LIBS})
    endif()
    if(METAL_FOUND)
        target_link_libraries(${target} ${METAL_LIBS})
    endif()
    if(IOURING_LIBS)
        target_link_libraries(${target} ${IOURING_LIBS})
    endif()
endfunction()

add_executable(cml main.c ${LIBRARY_SOURCES})
set_target_properties(cml PROPERTIES
    OUTPUT_NAME "cml"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
cml_link_all(cml)
cml_apply_torch_selective_ops(cml)

add_library(cml_static STATIC ${LIBRARY_SOURCES})
set_target_properties(cml_static PROPERTIES
    OUTPUT_NAME "cml"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    # -fPIC so the archive can be linked into the Python CFFI shared module.
    POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(cml_static PUBLIC CML_STATIC_DEFINE)
cml_link_all(cml_static)
cml_apply_torch_selective_ops(cml_static)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
if(BUILD_SHARED_LIBS)
    add_library(cml_shared SHARED ${LIBRARY_SOURCES})
    set_target_properties(cml_shared PROPERTIES
        OUTPUT_NAME "cml"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        C_VISIBILITY_PRESET hidden
    )
    target_compile_definitions(cml_shared PRIVATE CML_BUILDING_DLL)
    cml_link_all(cml_shared)
    cml_apply_torch_selective_ops(cml_shared)
endif()

# backtrace() can only name a function if its symbol is in the dynamic table.
# Without this the flame graph's stacks resolve to bare addresses.
if(NOT WIN32)
    set(CMAKE_ENABLE_EXPORTS ON)
    add_link_options(-rdynamic)
endif()

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    # Release (the default build type) defines NDEBUG, which turns every
    # assert() into a no-op -- 18 test files once relied on assert() as their
    # only check and so verified nothing here. Those now use REQUIRE() from
    # tests/test_require.h, which does not depend on NDEBUG; this flag is the
    # backstop for any plain assert() added later.
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-UNDEBUG>)
    file(GLOB TEST_SOURCES "tests/*.c")
    # Mock-driver suites need special definitions/libraries; added below.
    list(FILTER TEST_SOURCES EXCLUDE REGEX "tests/test_nv_driver.c$")
    list(FILTER TEST_SOURCES EXCLUDE REGEX "tests/test_am_driver.c$")
    list(FILTER TEST_SOURCES EXCLUDE REGEX "tests/test_hexagon_rpc.c$")
    list(FILTER TEST_SOURCES EXCLUDE REGEX "tests/test_hip_mock.c$")
    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} cml_static)
        cml_link_all(${test_name})
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()

    # The mock-driver suites simulate Linux kernel GPU/DSP driver interfaces
    # (NVIDIA ioctls, AMD KFD, HIP, Hexagon FastRPC) and link Linux-only
    # backend sources, so they only build on Linux.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_nv_mock
        tests/test_nv_driver.c
        src/ops/ir/gpu/nv_mock.c
        ${LIBRARY_SOURCES}
    )
    target_compile_definitions(test_nv_mock PRIVATE CML_NV_MOCK_GPU CML_HAS_NV_DRIVER)
    cml_link_all(test_nv_mock)
    cml_apply_torch_selective_ops(test_nv_mock)
    add_test(NAME test_nv_mock COMMAND test_nv_mock)

    add_executable(test_am_mock
        tests/test_am_driver.c
        src/ops/ir/gpu/am_mock.c
        ${LIBRARY_SOURCES}
    )
    target_compile_definitions(test_am_mock PRIVATE CML_AM_MOCK_GPU CML_HAS_AM_DRIVER)
    cml_link_all(test_am_mock)
    cml_apply_torch_selective_ops(test_am_mock)
    add_test(NAME test_am_mock COMMAND test_am_mock)

    # --- mock hardware drivers (no physical device needed) -----------------

    # FastRPC transport mock: shared library the Hexagon backend dlopen's via
    # the CML_DSP_RPC_LIB override.
    add_library(fastrpc_mock SHARED tools/fastrpc_mock.c)
    set_target_properties(fastrpc_mock PROPERTIES OUTPUT_NAME fastrpc_mock)

    add_executable(test_hexagon_rpc tests/test_hexagon_rpc.c)
    target_link_libraries(test_hexagon_rpc cml_static)
    cml_link_all(test_hexagon_rpc)
    target_compile_definitions(test_hexagon_rpc PRIVATE
        FASTRPC_MOCK_LIB="$<TARGET_FILE:fastrpc_mock>")
    cml_apply_torch_selective_ops(test_hexagon_rpc)
    add_test(NAME test_hexagon_rpc COMMAND test_hexagon_rpc)

    # HIP driver mock: exercises the ROCm backend + HCQ adapter end to end.
    # The library already contains hip_mock.c; enable its hook in this test.
    add_executable(test_hip_mock tests/test_hip_mock.c ${LIBRARY_SOURCES})
    target_compile_definitions(test_hip_mock PRIVATE CML_HAS_HIP_MOCK)
    cml_link_all(test_hip_mock)
    cml_apply_torch_selective_ops(test_hip_mock)
    add_test(NAME test_hip_mock COMMAND test_hip_mock)
    endif() # CMAKE_SYSTEM_NAME STREQUAL "Linux"

    if(ENABLE_SANITIZERS)
        set(CML_LSAN_SUPP "${CMAKE_SOURCE_DIR}/cmake/lsan_suppressions.txt")
        foreach(test_source ${TEST_SOURCES})
            get_filename_component(test_name ${test_source} NAME_WE)
            set_tests_properties(${test_name} PROPERTIES
                ENVIRONMENT "LSAN_OPTIONS=suppressions=${CML_LSAN_SUPP}:print_suppressions=0")
        endforeach()
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set_tests_properties(test_nv_mock test_am_mock PROPERTIES
                ENVIRONMENT "LSAN_OPTIONS=suppressions=${CML_LSAN_SUPP}:print_suppressions=0")
        endif()
    endif()
endif()

# Branch-coverage build + report, SQLite-style union semantics (a branch is
# covered when ANY test binary takes it):
#
#   cmake -S . -B build-coverage -DCMAKE_C_FLAGS="--coverage -O0"
#   cmake --build build-coverage -j
#   cmake --build build-coverage --target coverage
#     -> runs ctest SERIALLY (all 150+ binaries share one set of .gcda
#        counters; parallel runs corrupt them), then prints the merged
#        per-file line/branch table via tools/coverage_report.sh
add_custom_target(coverage
    COMMAND ${CMAKE_CTEST_COMMAND} -j1 --output-on-failure
    COMMAND ${CMAKE_SOURCE_DIR}/tools/coverage_report.sh ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

option(BUILD_EXAMPLES "Build example programs" ON)

function(cml_add_example target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} cml_static m dl)
    if(LLVM_BACKEND_FOUND)
        target_link_libraries(${target_name} ${LLVM_BACKEND_LIBS_LIST})
    endif()
endfunction()

if(BUILD_EXAMPLES)
    cml_add_example(experiment_demo examples/experiment_demo.c)
    cml_add_example(sweep_runner examples/sweep_runner.c)
    cml_add_example(hyperband_sweep examples/hyperband_sweep.c)
    cml_add_example(torch_c_example examples/tutorials/torch_c_example.c)
    cml_add_example(bench_torch_eager benchmarks/bench_torch_eager.c)
    cml_add_example(hello_cml examples/tutorials/hello_cml.c)
    cml_add_example(simple_xor examples/tutorials/simple_xor.c)
    cml_add_example(tensor_ops examples/tutorials/tensor_ops.c)
    cml_add_example(linear_regression examples/tutorials/linear_regression.c)
    cml_add_example(logistic_regression examples/tutorials/logistic_regression.c)
    cml_add_example(mlp_classifier examples/tutorials/mlp_classifier.c)
    cml_add_example(autoencoder examples/tutorials/autoencoder.c)
    cml_add_example(conv_net examples/tutorials/conv_net.c)
    cml_add_example(rnn_sequence examples/tutorials/rnn_sequence.c)
    cml_add_example(lstm_timeseries examples/tutorials/lstm_timeseries.c)
    cml_add_example(gru_classifier examples/tutorials/gru_classifier.c)
    cml_add_example(embedding examples/tutorials/embedding.c)
    cml_add_example(gan examples/tutorials/gan.c)
    cml_add_example(multi_task examples/tutorials/multi_task.c)
    cml_add_example(transformer examples/tutorials/transformer.c)
    cml_add_example(lr_scheduler examples/tutorials/lr_scheduler.c)
    cml_add_example(activations examples/tutorials/activations.c)
    cml_add_example(bench_forward examples/benchmarks/bench_forward.c)
    cml_add_example(bench_gemm examples/benchmarks/bench_gemm.c)
    cml_add_example(bench_cross_framework benchmarks/bench_cross_framework.c)
    cml_add_example(bench_torch_c benchmarks/bench_torch_c.c)
    cml_add_example(profile_overhead benchmarks/profile_overhead.c)
    cml_add_example(profile_mlp_conv benchmarks/profile_mlp_conv.c)
    cml_add_example(profile_overhead_detailed benchmarks/profile_overhead_detailed.c)
    cml_add_example(profile_train_conv benchmarks/profile_train_conv.c)
    cml_add_example(llama_inference examples/llama_inference.c)
    cml_add_example(autograd_example examples/demos/autograd_example.c)
    cml_add_example(auto_capture_example examples/demos/auto_capture_example.c)
    cml_add_example(comprehensive_fusion_example examples/demos/comprehensive_fusion_example.c)
    cml_add_example(dead_code_example examples/demos/dead_code_example.c)
    cml_add_example(early_stopping_lr_scheduler examples/demos/early_stopping_lr_scheduler.c)
    cml_add_example(export_graph examples/demos/export_graph.c)
    cml_add_example(mnist_example examples/demos/mnist_example.c)
    cml_add_example(print_kernels examples/demos/print_kernels.c)
    cml_add_example(training_loop_example examples/demos/training_loop_example.c)
    cml_add_example(resnet examples/demos/resnet.c)
    cml_add_example(bert examples/demos/bert.c)
    cml_add_example(gpt2 examples/demos/gpt2.c)
    cml_add_example(vit examples/demos/vit.c)
    cml_add_example(t5 examples/demos/t5.c)
    cml_add_example(unet examples/demos/unet.c)
    cml_add_example(clip examples/demos/clip.c)
    cml_add_example(convnext examples/demos/convnext.c)
    cml_add_example(inception examples/demos/inception.c)
    cml_add_example(efficientnet examples/demos/efficientnet.c)
    cml_add_example(whisper examples/demos/whisper.c)
    cml_add_example(stable_diffusion examples/demos/stable_diffusion.c)
    cml_add_example(yolov8 examples/demos/yolov8.c)
    cml_add_example(mask_rcnn examples/demos/mask_rcnn.c)
    cml_add_example(retinanet examples/demos/retinanet.c)
    cml_add_example(unet3d examples/demos/unet3d.c)
    cml_add_example(rnnt examples/demos/rnnt.c)
    cml_add_example(mlperf_resnet50 examples/mlperf/mlperf_resnet50.c)
endif()

install(TARGETS cml cml_static
    EXPORT CMLTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(BUILD_SHARED_LIBS)
    install(TARGETS cml_shared EXPORT CMLTargets LIBRARY DESTINATION lib)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMLConfig.cmake.in)
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMLConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CMLConfig.cmake
        INSTALL_DESTINATION lib/cmake/CML
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/CMLConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(EXPORT CMLTargets FILE CMLTargets.cmake NAMESPACE CML:: DESTINATION lib/cmake/CML)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CMLConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/CMLConfigVersion.cmake
        DESTINATION lib/cmake/CML
    )
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cml.pc.in)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cml.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/cml.pc
        @ONLY
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cml.pc DESTINATION lib/pkgconfig)
endif()

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h")

if(UNIX)
    set(VIZ_INSTALL_DIR share/cml)
elseif(WIN32)
    set(VIZ_INSTALL_DIR bin/viz)
endif()

install(FILES scripts/viz.py
    DESTINATION ${VIZ_INSTALL_DIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
# Optional server script: only installed when present so a missing file cannot
# break `cmake --install`.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/fastapi_server.py")
    install(FILES scripts/fastapi_server.py
        DESTINATION ${VIZ_INSTALL_DIR}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/viz-ui/dist")
    install(DIRECTORY viz-ui/dist DESTINATION ${VIZ_INSTALL_DIR}/viz-ui)
endif()

# ── Simulation & Fuzz Testing ─────────────────────────────────────────────────
# sim_test: seeded deterministic simulation + OOM sweep + gradient check
# BUILD_SIM=ON  (default ON, requires no extra deps)
# BUILD_FUZZ=ON (default OFF, requires clang -fsanitize=fuzzer or AFL)

option(BUILD_SIM  "Build the seeded simulation / OOM-sweep test harness" ON)
option(BUILD_FUZZ "Build libFuzzer / AFL fuzz entry points (requires clang)" OFF)

if(BUILD_SIM)
    add_executable(sim_test tools/sim_test.c)
    target_link_libraries(sim_test cml_static m)
    cml_link_all(sim_test)
    # Register as a quick smoke-run (single seed, no oom sweep, no grad check)
    add_test(NAME sim_test COMMAND sim_test --seed 0xC0FFEE --iters 32)
endif()

if(BUILD_FUZZ)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
        message(WARNING "BUILD_FUZZ requires clang; skipping fuzz targets")
    else()
        add_executable(fuzz_ir tools/fuzz_ir.c)
        target_link_libraries(fuzz_ir cml_static m)
        cml_link_all(fuzz_ir)
        target_compile_options(fuzz_ir PRIVATE -fsanitize=fuzzer,address,undefined)
        target_link_options(fuzz_ir    PRIVATE -fsanitize=fuzzer,address,undefined)

        # Minimal corpus seed so libFuzzer starts with a valid-ish input
        add_custom_command(TARGET fuzz_ir POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    ${CMAKE_BINARY_DIR}/corpus/fuzz_ir
            COMMAND ${CMAKE_COMMAND} -E echo_append ""
                    > ${CMAKE_BINARY_DIR}/corpus/fuzz_ir/seed0
        )

        message(STATUS "  fuzz_ir built — run: ./build/bin/fuzz_ir build/corpus/fuzz_ir/")
    endif()
endif()

message(STATUS "")
message(STATUS "C-ML ${PROJECT_VERSION} (${CMAKE_BUILD_TYPE})")
message(STATUS "  LLVM: ${LLVM_BACKEND_FOUND}  CUDA: ${CUDAToolkit_FOUND}  ROCm: ${hip_FOUND}  Metal: ${METAL_FOUND}")
message(STATUS "  Vulkan: ${ENABLE_VULKAN}  WebGPU: ${ENABLE_WEBGPU}  OpenCL: ${OpenCL_FOUND}")
message(STATUS "  NV Driver: ${ENABLE_NV_DRIVER}  AM Driver: ${ENABLE_AM_DRIVER}  NIR: ${ENABLE_NIR}  NAK: ${ENABLE_NAK}")
message(STATUS "  Tests: ${BUILD_TESTS}  Examples: ${BUILD_EXAMPLES}  Sim: ${BUILD_SIM}  Fuzz: ${BUILD_FUZZ}  Distributed: ${ENABLE_DISTRIBUTED}  IB: ${ENABLE_IB_TRANSPORT}")
message(STATUS "")
