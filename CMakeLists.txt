cmake_minimum_required(VERSION 3.16)
project(C-ML VERSION 0.0.2 LANGUAGES C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# MLIR Support
option(ENABLE_MLIR "Enable MLIR backend for JIT compilation" ON)

if(ENABLE_MLIR)
    find_package(MLIR CONFIG)
    if(MLIR_FOUND)
        message(STATUS "MLIR found: ${MLIR_DIR}")
        add_definitions(-DCML_HAS_MLIR)

        # Get LLVM/MLIR include directories
        list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
        list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

        include(TableGen)
        include(AddLLVM)
        include(AddMLIR)
        include(HandleLLVMOptions)

        include_directories(${LLVM_INCLUDE_DIRS})
        include_directories(${MLIR_INCLUDE_DIRS})

        # MLIR libraries - use combined libraries plus C API
        set(MLIR_LIBS
            MLIR
            MLIRExecutionEngineShared
            LLVM
            # MLIR C API libraries (static)
            MLIRCAPIRegisterEverything
            MLIRCAPIIR
            MLIRCAPILinalg
            MLIRCAPIMath
            MLIRCAPITransforms
            MLIRCAPIExecutionEngine
            MLIRCAPIConversion
            MLIRCAPILLVM
            MLIRCAPISCF
            MLIRCAPIControlFlow
            MLIRCAPIInterfaces
            MLIRCAPIDebug
        )

        message(STATUS "MLIR support enabled")
    else()
        message(STATUS "MLIR not found - building without MLIR support")
        message(STATUS "To enable: cmake -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir ..")
    endif()
else()
    message(STATUS "MLIR support disabled")
endif()

# ============================================================================
# GPU SDK Detection (Optional)
# ============================================================================

# CUDA Toolkit Detection
option(ENABLE_CUDA "Enable CUDA backend" ON)
if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
        add_definitions(-DCML_HAS_CUDA)
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
        set(CUDA_LIBS CUDA::cudart CUDA::cuda_driver)
    else()
        message(STATUS "CUDA Toolkit not found - CUDA backend uses dynamic loading")
    endif()
endif()

# ROCm/HIP Detection
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

# Vulkan SDK Detection
option(ENABLE_VULKAN "Enable Vulkan backend" ON)
if(ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan found: ${Vulkan_VERSION}")
        add_definitions(-DCML_HAS_VULKAN)
        include_directories(${Vulkan_INCLUDE_DIRS})
        set(VULKAN_LIBS Vulkan::Vulkan)
    else()
        message(STATUS "Vulkan SDK not found - Vulkan backend uses dynamic loading")
    endif()
endif()

# Metal (macOS only)
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        message(STATUS "Metal framework found")
        add_definitions(-DCML_HAS_METAL)
        set(METAL_LIBS ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    endif()
endif()

# Source files
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
    src/core/cleanup.c
    src/core/error_stack.c
    src/core/computation_graph.c
    src/core/graph_context.c
)

set(ALLOC_SOURCES
    src/alloc/memory_management.c
    src/alloc/memory_pools.c
    src/alloc/graph_allocator.c
)

set(BACKEND_SOURCES
    src/backend/backend.c
    src/backend/backend_buffer.c
    src/backend/device.c
    src/backend/threadpool.c
    src/backend/profiling.c
    src/backend/blas.c
)

set(OPS_SOURCES
    src/ops/uops.c
    src/ops/ir/ir.c
    src/ops/ir/context.c
    src/ops/ir/execution.c
    src/ops/ir/backward.c
    src/ops/ir/export.c
    src/ops/ir/optimization.c
    src/ops/ir/graph_cache.c
    src/ops/simd_utils.c
    src/ops/simd_math.c
    src/ops/ir/mlir/mlir_dispatch.c
    src/ops/ir/mlir/mlir_kernel_cache.c
    src/ops/ir/mlir/backends/cuda_backend.c
    src/ops/ir/mlir/backends/rocm_backend.c
    src/ops/ir/mlir/backends/metal_backend.c
    src/ops/ir/mlir/backends/vulkan_backend.c
)

# MLIR sources (only included when MLIR is found)
if(MLIR_FOUND)
    set(MLIR_SOURCES
        src/ops/ir/mlir/mlir_context.c
        src/ops/ir/mlir/mlir_convert.c
        src/ops/ir/mlir/mlir_execute.c
        src/ops/ir/mlir/mlir_uops_builder.c
        src/ops/ir/mlir/mlir_utils.c
        src/ops/ir/mlir/mlir_codegen.c
        src/ops/ir/mlir/mlir_fusion.c
        src/ops/ir/mlir/mlir_serialization.c
        src/ops/ir/mlir/mlir_multi_backend.c
        src/ops/ir/mlir/mlir_cpp_bridge.cpp
    )
    list(APPEND OPS_SOURCES ${MLIR_SOURCES})
endif()

set(TENSOR_SOURCES
    src/tensor/tensor.c
    src/tensor/tensor_views.c
    src/tensor/tensor_manipulation.c
)

set(AUTOGRAD_SOURCES
    src/autograd/autograd.c
    src/autograd/forward_ops.c
    src/autograd/loss_functions.c
    src/autograd/checkpointing.c
)

set(NN_SOURCES
    src/nn.c
    src/nn/layers/linear.c
    src/nn/layers/activations.c
    src/nn/layers/dropout.c
    src/nn/layers/conv2d.c
    src/nn/layers/batchnorm2d.c
    src/nn/layers/layernorm.c
    src/nn/layers/pooling.c
    src/nn/layers/sequential.c
)

set(OPTIM_SOURCES
    src/optim.c
)

set(LIBRARY_SOURCES
    ${CORE_SOURCES}
    ${ALLOC_SOURCES}
    ${BACKEND_SOURCES}
    ${OPS_SOURCES}
    ${TENSOR_SOURCES}
    ${AUTOGRAD_SOURCES}
    ${NN_SOURCES}
    ${OPTIM_SOURCES}
)

# Main executable
add_executable(cml main.c ${LIBRARY_SOURCES})

# Set target properties
set_target_properties(cml PROPERTIES
    OUTPUT_NAME "cml"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Link libraries
target_link_libraries(cml m dl)  # dl for dynamic loading
if(MLIR_FOUND)
    target_link_libraries(cml ${MLIR_LIBS})
endif()
if(CUDA_LIBS)
    target_link_libraries(cml ${CUDA_LIBS})
endif()
if(ROCM_LIBS)
    target_link_libraries(cml ${ROCM_LIBS})
endif()
if(VULKAN_LIBS)
    target_link_libraries(cml ${VULKAN_LIBS})
endif()
if(METAL_LIBS)
    target_link_libraries(cml ${METAL_LIBS})
endif()

# Create static library
add_library(cml_static STATIC ${LIBRARY_SOURCES})
set_target_properties(cml_static PROPERTIES
    OUTPUT_NAME "cml"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_link_libraries(cml_static m dl)
if(MLIR_FOUND)
    target_link_libraries(cml_static ${MLIR_LIBS})
endif()
if(CUDA_LIBS)
    target_link_libraries(cml_static ${CUDA_LIBS})
endif()
if(ROCM_LIBS)
    target_link_libraries(cml_static ${ROCM_LIBS})
endif()
if(VULKAN_LIBS)
    target_link_libraries(cml_static ${VULKAN_LIBS})
endif()
if(METAL_LIBS)
    target_link_libraries(cml_static ${METAL_LIBS})
endif()

# Create shared library
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
if(BUILD_SHARED_LIBS)
    add_library(cml_shared SHARED ${LIBRARY_SOURCES})
    set_target_properties(cml_shared PROPERTIES
        OUTPUT_NAME "cml"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    target_link_libraries(cml_shared m dl)
    if(MLIR_FOUND)
        target_link_libraries(cml_shared ${MLIR_LIBS})
    endif()
    if(CUDA_LIBS)
        target_link_libraries(cml_shared ${CUDA_LIBS})
    endif()
    if(ROCM_LIBS)
        target_link_libraries(cml_shared ${ROCM_LIBS})
    endif()
    if(VULKAN_LIBS)
        target_link_libraries(cml_shared ${VULKAN_LIBS})
    endif()
    if(METAL_LIBS)
        target_link_libraries(cml_shared ${METAL_LIBS})
    endif()
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()

    # Find test files in test/ directory
    file(GLOB TEST_SOURCES "test/*.c")

    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source} ${LIBRARY_SOURCES})
        target_link_libraries(${test_name} m dl)
        if(MLIR_FOUND)
            target_link_libraries(${test_name} ${MLIR_LIBS})
        endif()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()

    # Dispatch layer tests
    add_executable(test_dispatch tests/test_dispatch.c ${LIBRARY_SOURCES})
    target_link_libraries(test_dispatch m dl)
    if(MLIR_FOUND)
        target_link_libraries(test_dispatch ${MLIR_LIBS})
    endif()
    add_test(NAME test_dispatch COMMAND test_dispatch)

    # Kernel cache tests
    add_executable(test_kernel_cache tests/test_kernel_cache.c ${LIBRARY_SOURCES})
    target_link_libraries(test_kernel_cache m dl)
    if(MLIR_FOUND)
        target_link_libraries(test_kernel_cache ${MLIR_LIBS})
    endif()
    add_test(NAME test_kernel_cache COMMAND test_kernel_cache)

    # Backend tests
    add_executable(test_backends tests/test_backends.c ${LIBRARY_SOURCES})
    target_link_libraries(test_backends m dl)
    if(MLIR_FOUND)
        target_link_libraries(test_backends ${MLIR_LIBS})
    endif()
    add_test(NAME test_backends COMMAND test_backends)

    # Backend benchmark
    add_executable(bench_backends tests/bench_backends.c ${LIBRARY_SOURCES})
    target_link_libraries(bench_backends m dl)
    if(MLIR_FOUND)
        target_link_libraries(bench_backends ${MLIR_LIBS})
    endif()
endif()

# Python bindings (placeholder for future)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    # Python binding implementation would go here
    message(STATUS "Python bindings not implemented yet")
endif()

# Documentation
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

        add_custom_target(docs ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
    endif()
endif()

# Install rules
install(TARGETS cml cml_static
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(BUILD_SHARED_LIBS)
    install(TARGETS cml_shared
        LIBRARY DESTINATION lib
    )
endif()

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Install visualization tools
if(UNIX)
    set(VIZ_INSTALL_DIR share/cml)
elseif(WIN32)
    set(VIZ_INSTALL_DIR bin/viz) # Or somewhere else appropriate
endif()

install(FILES scripts/viz.py scripts/fastapi_server.py
    DESTINATION ${VIZ_INSTALL_DIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install frontend if built
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/viz-ui/dist")
    install(DIRECTORY viz-ui/dist
        DESTINATION ${VIZ_INSTALL_DIR}/viz-ui
    )
else()
    message(WARNING "Visualization frontend not found at viz-ui/dist. Run 'npm run build' in viz-ui to include it in install.")
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "C-ML Configuration Summary")
message(STATUS "==========================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "")
message(STATUS "Backend Support:")
message(STATUS "  MLIR/LLVM JIT: ${MLIR_FOUND}")
message(STATUS "  CUDA: ${CUDAToolkit_FOUND}")
message(STATUS "  ROCm/HIP: ${hip_FOUND}")
message(STATUS "  Vulkan: ${Vulkan_FOUND}")
if(APPLE)
message(STATUS "  Metal: Available")
endif()
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "  Build Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Build documentation: ${BUILD_DOCS}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")

# Examples
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_executable(autograd_example examples/autograd_example.c)
    target_link_libraries(autograd_example cml_static m dl)
    if(MLIR_FOUND)
        target_link_libraries(autograd_example ${MLIR_LIBS})
    endif()

    add_executable(training_loop_example examples/training_loop_example.c)
    target_link_libraries(training_loop_example cml_static m dl)
    if(MLIR_FOUND)
        target_link_libraries(training_loop_example ${MLIR_LIBS})
    endif()

    add_executable(test_example examples/test.c)
    target_link_libraries(test_example cml_static m dl)
    if(MLIR_FOUND)
        target_link_libraries(test_example ${MLIR_LIBS})
    endif()

    add_executable(early_stopping_lr_scheduler examples/early_stopping_lr_scheduler.c)
    target_link_libraries(early_stopping_lr_scheduler cml_static m dl)
    if(MLIR_FOUND)
        target_link_libraries(early_stopping_lr_scheduler ${MLIR_LIBS})
    endif()

    add_executable(print_kernels examples/print_kernels.c)
    target_link_libraries(print_kernels cml_static m dl)
    if(MLIR_FOUND)
        target_link_libraries(print_kernels ${MLIR_LIBS})
    endif()

    add_executable(mnist_example examples/mnist_example.c)
    target_link_libraries(mnist_example cml_static m dl)
    if(MLIR_FOUND)
        target_link_libraries(mnist_example ${MLIR_LIBS})
    endif()

    add_executable(dead_code_example examples/dead_code_example.c)
    target_link_libraries(dead_code_example cml_static m dl)
    if(MLIR_FOUND)
        target_link_libraries(dead_code_example ${MLIR_LIBS})
    endif()

    add_executable(opcheck examples/opcheck.c)
    target_link_libraries(opcheck cml_static m dl)
    if(MLIR_FOUND)
        target_link_libraries(opcheck ${MLIR_LIBS})
    endif()
endif()
