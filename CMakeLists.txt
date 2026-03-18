cmake_minimum_required(VERSION 3.16)
project(C-ML VERSION 0.0.3 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -march=native is critical: without it the compiler emits SSE2 only
# and our AVX2 microkernels are dead code.
include(CheckCCompilerFlag)
check_c_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
    set(NATIVE_FLAG "-march=native")
else()
    set(NATIVE_FLAG "")
endif()

set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra ${NATIVE_FLAG}")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG ${NATIVE_FLAG}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG ${NATIVE_FLAG}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/include)

# --- LLVM Backend -----------------------------------------------------------

option(ENABLE_LLVM_BACKEND "Enable direct LLVM IR JIT backend" ON)

if(ENABLE_LLVM_BACKEND)
    find_program(LLVM_CONFIG_EXE NAMES llvm-config llvm-config-21 llvm-config-20 llvm-config-19)
    if(LLVM_CONFIG_EXE)
        execute_process(COMMAND ${LLVM_CONFIG_EXE} --cflags
            OUTPUT_VARIABLE LLVM_BACKEND_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_EXE} --ldflags
            OUTPUT_VARIABLE LLVM_BACKEND_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_EXE} --libs core orcjit native nvptx amdgpu
            OUTPUT_VARIABLE LLVM_BACKEND_LIBS_RAW OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_EXE} --system-libs
            OUTPUT_VARIABLE LLVM_BACKEND_SYSLIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_EXE} --includedir
            OUTPUT_VARIABLE LLVM_BACKEND_INCDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_EXE} --version
            OUTPUT_VARIABLE LLVM_BACKEND_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

        include(CheckIncludeFile)
        set(CMAKE_REQUIRED_INCLUDES ${LLVM_BACKEND_INCDIR})
        check_include_file("llvm-c/LLJIT.h" HAS_LLVM_LLJIT)

        if(HAS_LLVM_LLJIT)
            add_definitions(-DCML_HAS_LLVM_BACKEND)
            include_directories(${LLVM_BACKEND_INCDIR})
            separate_arguments(LLVM_BACKEND_LIBS_LIST UNIX_COMMAND
                "${LLVM_BACKEND_LDFLAGS} ${LLVM_BACKEND_LIBS_RAW} ${LLVM_BACKEND_SYSLIBS}")
            message(STATUS "LLVM backend enabled (LLVM ${LLVM_BACKEND_VERSION})")
            set(LLVM_BACKEND_FOUND TRUE)
        else()
            message(STATUS "LLVM backend: LLJIT header not found, disabling")
            set(LLVM_BACKEND_FOUND FALSE)
        endif()
    else()
        message(STATUS "llvm-config not found — LLVM backend disabled")
        set(LLVM_BACKEND_FOUND FALSE)
    endif()
else()
    message(STATUS "LLVM backend disabled")
    set(LLVM_BACKEND_FOUND FALSE)
endif()

# --- GPU SDK Detection ------------------------------------------------------

option(ENABLE_CUDA "Enable CUDA backend" ON)
if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
        add_definitions(-DCML_HAS_CUDA)
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
        # Don't link libcuda.so — we dlopen it. The stub causes hangs without a GPU.
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
        include_directories(${OpenCL_INCLUDE_DIRS})
        set(OPENCL_LIBS ${OpenCL_LIBRARIES})
    else()
        message(STATUS "OpenCL not found - OpenCL backend disabled")
        set(OPENCL_LIBS "")
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

# --- Sources -----------------------------------------------------------------

set(CORE_SOURCES
    src/cml.c
    src/core/logging.c
    src/core/config.c
    src/core/dataset.c
    src/core/augmentation.c
    src/core/model_architecture.c
    src/core/training_loop.c
    src/core/training_metrics.c
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
)

if(ENABLE_ONNX)
    list(APPEND CORE_SOURCES src/core/onnx.c src/core/onnx_ops.c)
endif()

set(ALLOC_SOURCES
    src/alloc/memory_management.c
    src/alloc/memory_pools.c
    src/alloc/graph_allocator.c
    src/alloc/tlsf_alloc.c
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
)

set(OPS_SOURCES
    src/ops/uops.c
    src/ops/winograd.c
    src/ops/ir/ir.c
    src/ops/ir/context.c
    src/ops/ir/execution.c
    src/ops/ir/backward.c
    src/ops/ir/export.c
    src/ops/ir/optimization.c
    src/ops/ir/graph_cache.c
    src/ops/ir/aot.c
    src/ops/ir/kernel_cache.c
    src/ops/ir/fusion_patterns.c
    src/ops/ir/dispatch.c
    src/ops/ir/decompose.c
    src/ops/ir/pattern_matcher.c
    src/ops/ir/hcq.c
    src/ops/ir/hcq_cuda.c
    src/ops/ir/hcq_opencl.c
    src/ops/ir/hcq_vulkan.c
    src/ops/ir/hcq_nv.c
    src/ops/ir/hcq_am.c
    src/ops/ir/beam_search.c
    src/ops/ir/trace.c
    src/ops/ir/tiny_jit.c
    src/ops/ir/beam_cuda_timing.c
    src/ops/ir/schedule.c
    src/ops/ir/schedule_v2.c
    src/ops/ir/linearize.c
    src/ops/ir/fused_codegen.c
    src/ops/ir/runtime_compiler.c
    src/ops/ir/graph_capture.c
    src/ops/ir/nir_compiler.c
    src/ops/ir/hcq_nir.c
    src/ops/ir/z3_verify.c
    src/ops/ir/compiler_viz.c
    src/ops/ir/gpu/cuda_backend.c
    src/ops/ir/gpu/rocm_backend.c
    src/ops/ir/gpu/wmma.c
    src/ops/ir/gpu/wgsl_codegen.c
    src/ops/ir/gpu/metal_codegen.c
    src/ops/ir/gpu/ptx_codegen.c
    src/ops/ir/gpu/nv_driver.c
    src/ops/ir/gpu/am_driver.c
    src/ops/ir/gpu/adreno_backend.c
    src/ops/ir/gpu/hexagon_backend.c
    src/ops/simd_utils.c
    src/ops/simd_math.c
    src/ops/simd_views.c
)

if(METAL_FOUND)
    list(APPEND OPS_SOURCES src/ops/ir/gpu/metal_backend.m)
endif()

list(APPEND OPS_SOURCES src/ops/ir/gpu/webgpu_backend.c)

if(ENABLE_VULKAN)
    list(APPEND OPS_SOURCES
        src/ops/ir/gpu/vulkan_backend.c
        src/ops/ir/gpu/spirv_codegen.c
    )
endif()

if(ENABLE_NIR)
    list(APPEND OPS_SOURCES src/ops/ir/nir_compiler.c src/ops/ir/hcq_nir.c)
endif()

list(APPEND OPS_SOURCES src/ops/ir/schedule_v2.c src/ops/ir/linearize.c)

if(LLVM_BACKEND_FOUND)
    list(APPEND OPS_SOURCES src/ops/ir/llvm/llvm_backend.c src/ops/ir/gpu/gpu_codegen.c)
endif()

set(TENSOR_SOURCES
    src/tensor/tensor.c
    src/tensor/tensor_views.c
    src/tensor/tensor_manipulation.c
    src/tensor/sparse_tensor.c
    src/tensor/image_dtype.c
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
    src/nn/layers/batchnorm2d.c
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
    src/nn/layers/batchnorm3d.c
    src/nn/layers/layernorm2d.c
    src/nn/layers/conv_transpose3d.c
    src/nn/layers/upsample.c
    src/nn/layers/pixel_shuffle.c
    src/nn/layers/flatten.c
    src/nn/layers/identity.c
    src/nn/layers/batchnorm1d.c
    src/nn/layers/prelu.c
    src/nn/llm_ops.c
    src/nn/llama.c
    src/nn/lora.c
    src/nn/qlora.c
    src/nn/paged_attention.c
    src/nn/serving.c
    src/nn/speculative.c
)

set(SYMBOLIC_SOURCES src/symbolic/symbolic.c)

set(OPTIM_SOURCES src/optim.c src/optim/lr_scheduler.c)

option(ENABLE_DISTRIBUTED "Enable distributed training" OFF)
option(ENABLE_NCCL "Enable NCCL backend (requires libnccl)" ON)
option(ENABLE_MPI "Enable MPI backend (requires libmpi)" ON)

if(ENABLE_DISTRIBUTED)
    add_definitions(-DCML_HAS_DISTRIBUTED)
    set(DISTRIBUTED_SOURCES
        src/distributed/distributed.c
        src/distributed/comm_backend.c
        src/distributed/gloo_backend.c
        src/distributed/ring_allreduce.c
        src/distributed/data_parallel.c
        src/distributed/pipeline_parallel.c
        src/distributed/tensor_parallel.c
    )
    if(ENABLE_NCCL)
        list(APPEND DISTRIBUTED_SOURCES src/distributed/nccl_backend.c)
        add_definitions(-DCML_HAS_NCCL)
    endif()
    if(ENABLE_MPI)
        list(APPEND DISTRIBUTED_SOURCES src/distributed/mpi_backend.c)
        add_definitions(-DCML_HAS_MPI)
    endif()
    message(STATUS "Distributed training enabled (NCCL: ${ENABLE_NCCL}, MPI: ${ENABLE_MPI})")
else()
    set(DISTRIBUTED_SOURCES src/distributed/tensor_parallel.c)
    message(STATUS "Distributed training disabled (tensor parallelism still available)")
endif()

set(DATASET_SOURCES
    src/datasets/datasets.c
    src/datasets/csv_parser.c
    src/datasets/idx_parser.c
    src/datasets/builtin_data.c
)

option(BUILD_MODEL_ZOO "Build model zoo" ON)
if(BUILD_MODEL_ZOO)
    set(ZOO_SOURCES src/zoo/zoo.c)
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
    ${SYMBOLIC_SOURCES}
    ${DISTRIBUTED_SOURCES}
    ${ZOO_SOURCES}
    ${DATASET_SOURCES}
)

# --- Helper to link all optional backends to a target ------------------------

function(cml_link_all target)
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
endfunction()

# --- Targets -----------------------------------------------------------------

add_executable(cml main.c ${LIBRARY_SOURCES})
set_target_properties(cml PROPERTIES
    OUTPUT_NAME "cml"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
cml_link_all(cml)

add_library(cml_static STATIC ${LIBRARY_SOURCES})
set_target_properties(cml_static PROPERTIES
    OUTPUT_NAME "cml"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_compile_definitions(cml_static PUBLIC CML_STATIC_DEFINE)
cml_link_all(cml_static)

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
endif()

# --- Tests -------------------------------------------------------------------

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    file(GLOB TEST_SOURCES "tests/*.c")
    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source} ${LIBRARY_SOURCES})
        cml_link_all(${test_name})
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# --- Examples ----------------------------------------------------------------

option(BUILD_EXAMPLES "Build example programs" ON)

function(cml_add_example target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} cml_static m dl)
    if(LLVM_BACKEND_FOUND)
        target_link_libraries(${target_name} ${LLVM_BACKEND_LIBS_LIST})
    endif()
endfunction()

if(BUILD_EXAMPLES)
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
endif()

# --- Install -----------------------------------------------------------------

install(TARGETS cml cml_static
    EXPORT CMLTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(BUILD_SHARED_LIBS)
    install(TARGETS cml_shared EXPORT CMLTargets LIBRARY DESTINATION lib)
endif()

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

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cml.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/cml.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cml.pc DESTINATION lib/pkgconfig)

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h")

if(UNIX)
    set(VIZ_INSTALL_DIR share/cml)
elseif(WIN32)
    set(VIZ_INSTALL_DIR bin/viz)
endif()

install(FILES scripts/viz.py scripts/fastapi_server.py
    DESTINATION ${VIZ_INSTALL_DIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/viz-ui/dist")
    install(DIRECTORY viz-ui/dist DESTINATION ${VIZ_INSTALL_DIR}/viz-ui)
endif()

# --- Summary -----------------------------------------------------------------

message(STATUS "")
message(STATUS "C-ML ${PROJECT_VERSION} (${CMAKE_BUILD_TYPE})")
message(STATUS "  LLVM: ${LLVM_BACKEND_FOUND}  CUDA: ${CUDAToolkit_FOUND}  ROCm: ${hip_FOUND}  Metal: ${METAL_FOUND}")
message(STATUS "  Vulkan: ${ENABLE_VULKAN}  WebGPU: ${ENABLE_WEBGPU}  OpenCL: ${OpenCL_FOUND}")
message(STATUS "  NV Driver: ${ENABLE_NV_DRIVER}  AM Driver: ${ENABLE_AM_DRIVER}  NIR: ${ENABLE_NIR}")
message(STATUS "  Tests: ${BUILD_TESTS}  Examples: ${BUILD_EXAMPLES}  Distributed: ${ENABLE_DISTRIBUTED}")
message(STATUS "")
