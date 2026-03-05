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

# ============================================================================
# Direct LLVM Backend
# ============================================================================
option(ENABLE_LLVM_BACKEND "Enable direct LLVM IR JIT backend" ON)

if(ENABLE_LLVM_BACKEND)
    # Use llvm-config to find LLVM
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

        # Check for required header
        include(CheckIncludeFile)
        set(CMAKE_REQUIRED_INCLUDES ${LLVM_BACKEND_INCDIR})
        check_include_file("llvm-c/LLJIT.h" HAS_LLVM_LLJIT)

        if(HAS_LLVM_LLJIT)
            add_definitions(-DCML_HAS_LLVM_BACKEND)
            include_directories(${LLVM_BACKEND_INCDIR})

            # Parse the library flags into a list for CMake
            separate_arguments(LLVM_BACKEND_LIBS_LIST UNIX_COMMAND "${LLVM_BACKEND_LDFLAGS} ${LLVM_BACKEND_LIBS_RAW} ${LLVM_BACKEND_SYSLIBS}")

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
        # CUDA backend uses dlopen/dlsym — no link-time CUDA libs needed.
        # Linking the CUDA stub libcuda.so causes hangs on systems without a GPU driver.
        set(CUDA_LIBS "")
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
    src/core/quantization.c
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
    src/backend/opencl_backend.c
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
    src/ops/ir/aot.c
    src/ops/ir/kernel_cache.c
    src/ops/ir/fusion_patterns.c
    src/ops/ir/dispatch.c
    src/ops/ir/gpu/cuda_backend.c
    src/ops/ir/gpu/rocm_backend.c
    src/ops/simd_utils.c
    src/ops/simd_math.c
    src/ops/simd_views.c
)

# LLVM backend source (direct LLVM IR JIT)
if(LLVM_BACKEND_FOUND)
    list(APPEND OPS_SOURCES src/ops/ir/llvm/llvm_backend.c)
    list(APPEND OPS_SOURCES src/ops/ir/gpu/gpu_codegen.c)
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
)

set(OPTIM_SOURCES
    src/optim.c
    src/optim/lr_scheduler.c
)

# Distributed training
option(ENABLE_DISTRIBUTED "Enable distributed training" OFF)
option(ENABLE_NCCL "Enable NCCL backend (requires libnccl)" ON)
option(ENABLE_MPI "Enable MPI backend (requires libmpi)" ON)

if(ENABLE_DISTRIBUTED)
    add_definitions(-DCML_HAS_DISTRIBUTED)
    set(DISTRIBUTED_SOURCES
        src/distributed/distributed.c
        src/distributed/gloo_backend.c
        src/distributed/ring_allreduce.c
        src/distributed/data_parallel.c
        src/distributed/pipeline_parallel.c
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
    set(DISTRIBUTED_SOURCES "")
    message(STATUS "Distributed training disabled")
endif()

# Dataset Hub
set(DATASET_SOURCES
    src/datasets/datasets.c
    src/datasets/csv_parser.c
    src/datasets/idx_parser.c
    src/datasets/builtin_data.c
)

# Model Zoo
option(BUILD_MODEL_ZOO "Build model zoo" ON)

if(BUILD_MODEL_ZOO)
    set(ZOO_SOURCES
        src/zoo/zoo.c
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
    ${DISTRIBUTED_SOURCES}
    ${ZOO_SOURCES}
    ${DATASET_SOURCES}
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
if(ENABLE_DISTRIBUTED)
    target_link_libraries(cml pthread)
endif()
if(LLVM_BACKEND_FOUND)
    target_link_libraries(cml ${LLVM_BACKEND_LIBS_LIST})
endif()
if(CUDA_LIBS)
    target_link_libraries(cml ${CUDA_LIBS})
endif()
if(ROCM_LIBS)
    target_link_libraries(cml ${ROCM_LIBS})
endif()

# Create static library
add_library(cml_static STATIC ${LIBRARY_SOURCES})
set_target_properties(cml_static PROPERTIES
    OUTPUT_NAME "cml"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_link_libraries(cml_static m dl)
if(ENABLE_DISTRIBUTED)
    target_link_libraries(cml_static pthread)
endif()
if(LLVM_BACKEND_FOUND)
    target_link_libraries(cml_static ${LLVM_BACKEND_LIBS_LIST})
endif()
if(CUDA_LIBS)
    target_link_libraries(cml_static ${CUDA_LIBS})
endif()
if(ROCM_LIBS)
    target_link_libraries(cml_static ${ROCM_LIBS})
endif()
if(OPENCL_LIBS)
    target_link_libraries(cml_static ${OPENCL_LIBS})
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
    if(ENABLE_DISTRIBUTED)
        target_link_libraries(cml_shared pthread)
    endif()
    if(LLVM_BACKEND_FOUND)
        target_link_libraries(cml_shared ${LLVM_BACKEND_LIBS_LIST})
    endif()
    if(CUDA_LIBS)
        target_link_libraries(cml_shared ${CUDA_LIBS})
    endif()
    if(ROCM_LIBS)
        target_link_libraries(cml_shared ${ROCM_LIBS})
    endif()
    if(OPENCL_LIBS)
        target_link_libraries(cml_shared ${OPENCL_LIBS})
    endif()
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()

    # Find test files in tests/ directory
    file(GLOB TEST_SOURCES "tests/*.c")

    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source} ${LIBRARY_SOURCES})
        target_link_libraries(${test_name} m dl)
        if(ENABLE_DISTRIBUTED)
            target_link_libraries(${test_name} pthread)
        endif()
        if(LLVM_BACKEND_FOUND)
            target_link_libraries(${test_name} ${LLVM_BACKEND_LIBS_LIST})
        endif()
        if(CUDA_LIBS)
            target_link_libraries(${test_name} ${CUDA_LIBS})
        endif()
        if(OPENCL_LIBS)
            target_link_libraries(${test_name} ${OPENCL_LIBS})
        endif()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
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
message(STATUS "  LLVM JIT: ${LLVM_BACKEND_FOUND}")
message(STATUS "  CUDA: ${CUDAToolkit_FOUND}")
message(STATUS "  ROCm/HIP: ${hip_FOUND}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "  Build Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Build documentation: ${BUILD_DOCS}")
message(STATUS "  Distributed training: ${ENABLE_DISTRIBUTED}")
message(STATUS "  Model zoo: ${BUILD_MODEL_ZOO}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")

# Examples
option(BUILD_EXAMPLES "Build example programs" ON)

# Helper function to add an example with all needed link libraries
function(cml_add_example target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} cml_static m dl)
    if(LLVM_BACKEND_FOUND)
        target_link_libraries(${target_name} ${LLVM_BACKEND_LIBS_LIST})
    endif()
endfunction()

if(BUILD_EXAMPLES)
    # Tutorials
    cml_add_example(hello_cml examples/tutorials/hello_cml.c)
    cml_add_example(simple_xor examples/tutorials/simple_xor.c)
    cml_add_example(ex01_tensor_ops examples/tutorials/ex01_tensor_ops.c)
    cml_add_example(ex02_linear_regression examples/tutorials/ex02_linear_regression.c)
    cml_add_example(ex03_logistic_regression examples/tutorials/ex03_logistic_regression.c)
    cml_add_example(ex04_mlp_classifier examples/tutorials/ex04_mlp_classifier.c)
    cml_add_example(ex05_autoencoder examples/tutorials/ex05_autoencoder.c)
    cml_add_example(ex06_conv_net examples/tutorials/ex06_conv_net.c)
    cml_add_example(ex07_rnn_sequence examples/tutorials/ex07_rnn_sequence.c)
    cml_add_example(ex08_lstm_timeseries examples/tutorials/ex08_lstm_timeseries.c)
    cml_add_example(ex09_gru_classifier examples/tutorials/ex09_gru_classifier.c)
    cml_add_example(ex10_embedding examples/tutorials/ex10_embedding.c)
    cml_add_example(ex11_gan examples/tutorials/ex11_gan.c)
    cml_add_example(ex12_multi_task examples/tutorials/ex12_multi_task.c)
    cml_add_example(ex13_transformer examples/tutorials/ex13_transformer.c)
    cml_add_example(ex14_lr_scheduler examples/tutorials/ex14_lr_scheduler.c)
    cml_add_example(ex15_activations_showcase examples/tutorials/ex15_activations_showcase.c)

    # Benchmarks
    cml_add_example(bench_forward examples/benchmarks/bench_forward.c)
    cml_add_example(bench_gemm examples/benchmarks/bench_gemm.c)

    # Demos
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
