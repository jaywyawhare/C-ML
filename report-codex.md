# C-ML Codebase Audit Report (`report-codex.md`)

Date: 2026-04-21
Auditor: Codex
Repository: `/home/arrry/dev/personal/C-ML`

## 1) Scope and Method

This pass was done across tracked code files and build/test entrypoints.

- Total tracked code files scanned: `649`
- Distribution:
- `src`: 221
- `include`: 206
- `tests`: 103
- `examples`: 47
- `python`: 27
- `website`: 13
- `viz`: 13
- `scripts`: 6
- `benchmarks`: 6
- root build/run files: 7

Audit method used:

1. Repository-wide code inventory using `rg --files`.
2. Build graph validation from `CMakeLists.txt`.
3. Warning-focused rebuild (`build-audit`) with `-Wunused-function/-Wunused-variable`.
4. Runtime validation using `ctest` and executable sweep (`check_all.sh`).
5. Targeted failure triage for failing test/demo binaries.
6. Cross-check for dead/unused code in C, Python, and website modules.

## 2) How the Codebase Should Work (Expected Flow)

The repo is designed as a C11 ML stack with this runtime flow:

1. `cml.h` API drives tensor/autograd/nn/optimizer APIs.
2. Core execution goes through tensor ops (`src/tensor`, `src/ops`) and backend dispatch (`src/backend`).
3. Compiler/IR path (`src/ops/ir`) handles graph lowering, scheduling, optimization, codegen, and cache.
4. Higher layers (`src/nn`, `src/autograd`, `src/zoo`) consume this runtime.
5. Tests (`tests/*.c`) are all standalone executables linked against `cml_static`.
6. Python package (`python/cml`) expects built CFFI binding module (`cml._cml_lib`).
7. Website is a separate Vite React app under `website/`.

Build graph consistency check:

- `src/*.c` files in tree: `217`
- `src/*.c` files referenced in `CMakeLists.txt`: `217`
- Result: no orphan `src` implementation file found.

## 3) What Is Not Working (Confirmed)

### P0/P1 Functional breakages

1. `test_label_smoothing` is failing (`96/97` ctests pass, 1 fails).
- Failing test cases:
- `test_sparse_smooth_zero_delegates`
- `test_sparse_smooth_nonzero`
- File references:
- `/home/arrry/dev/personal/C-ML/tests/test_label_smoothing.c:236`
- `/home/arrry/dev/personal/C-ML/tests/test_label_smoothing.c:237`
- Implementation path under test:
- `/home/arrry/dev/personal/C-ML/src/autograd/loss_functions.c:689`

2. `dead_code_example` crashes with `SIGSEGV` (exit 139).
- Reproduced in `build/bin` and `build-audit/bin`.
- ASAN run pinpoints crash in:
- `/home/arrry/dev/personal/C-ML/src/autograd/autograd.c:737`
- Called from:
- `/home/arrry/dev/personal/C-ML/examples/demos/dead_code_example.c:75`
- Impact: the demo specifically intended to showcase dead-code/IR optimization currently crashes.

### Environment/packaging breakages

3. Python tests in `tests/*.py` fail at import due missing `torch` dependency.
- `ModuleNotFoundError: No module named 'torch'`
- Affects: `tests/test_unary.py`, `test_reductions.py`, `test_elementwise.py`, `test_losses.py`, `test_layers.py`, `test_tensor_ops.py`, `test_activations.py`

4. `python/tests/test_zoo.py` fails because CFFI module is not built.
- `ImportError: No module named 'cml._cml_lib'`
- Error points users to build step:
- `cd python && python3 cml/build_cffi.py`

5. Install script reference is broken in CMake.
- `CMakeLists.txt` installs `scripts/fastapi_server.py`
- File does not exist in repository.
- This can break `cmake --install` workflows.

## 4) Dead Code and Unused Code Findings

Compiler-backed dead/unused findings from `build-audit/build.log`:

### Unused functions

1. `/home/arrry/dev/personal/C-ML/tests/test_zoo_models.c:30`
- `has_nonzero` defined but not used

2. `/home/arrry/dev/personal/C-ML/tests/test_zoo_models.c:226`
- `test_t5_small` defined but not used

3. `/home/arrry/dev/personal/C-ML/tests/test_zoo_models.c:244`
- `test_resnet_param_count_reasonable` defined but not used

4. `/home/arrry/dev/personal/C-ML/tests/test_zoo_models.c:260`
- `test_gpt2_param_count_reasonable` defined but not used

### Unused variables/constants

1. `/home/arrry/dev/personal/C-ML/benchmarks/profile_mlp_conv.c:117`
- `t_transpose`, `t_biasadd1`, `t_relu` unused

2. `/home/arrry/dev/personal/C-ML/benchmarks/profile_mlp_conv.c:118`
- `t_matmul2`, `t_biasadd2` unused

3. `/home/arrry/dev/personal/C-ML/tests/test_graph_cache_integration.c:29`
- `h0` unused

4. `/home/arrry/dev/personal/C-ML/tests/test_graph_cache_integration.c:65`
- `cache` unused

5. `/home/arrry/dev/personal/C-ML/tests/test_numerical_ops.c:187`
- `exp` unused

6. `/home/arrry/dev/personal/C-ML/tests/test_tensor_ops_extra.c:14`
- `cpu_i32` unused

### Website dead module

1. `/home/arrry/dev/personal/C-ML/website/src/hooks/useLossLandscape.js`
- Present but not imported anywhere in `website/src`.
- This is dead code in frontend bundle source.

## 5) Runtime Health Summary

From executable sweep (`check_all.sh` on current `build/bin`):

- PASS: `125`
- FAIL: `2`
- TIMEOUT/SLOW (>5s): `21`
- SKIPPED: `2`

Failing binaries:

1. `dead_code_example` (exit 139)
2. `test_label_smoothing` (exit 1)

Note:

- `check_all.sh` hardcodes `build/bin` via `BIN_DIR="$(cd "$(dirname "$0")/build/bin" && pwd)"` and ignores externally passed `BIN_DIR` value.
- This is not dead code, but it is a script behavior defect for custom build directories.

## 6) Per-Subsystem Analysis

### `src/core`, `src/tensor`, `src/ops`, `src/backend`

- Core runtime builds successfully in audit build.
- No global source file orphaning in CMake source lists.
- Primary reliability risk currently is not compile failure, but specific runtime crash path in autograd graph JSON export.

### `src/autograd`

- Label smoothing sparse path behavior is inconsistent with tests.
- Export path in `autograd_export_json` can dereference null node state in current demo flow.

### `src/nn`, `src/zoo`, `src/distributed`

- Broad test surface passes.
- Dead code mostly appears in test coverage helper functions, not in core runtime module exports.

### `tests`

- C tests are generally healthy.
- One failing C test (`test_label_smoothing`).
- Several dead tests/helpers exist in `test_zoo_models.c` (defined but never executed).

### `examples`

- Most examples execute.
- `dead_code_example` is currently broken, which undermines the optimization demo path.

### `python`

- Python package import depends on generated binding module.
- Missing dependency (`torch`) blocks Python-based comparison tests.
- Current out-of-box test run is not self-contained.

### `website`

- Routing/component tree is coherent.
- One dead hook module (`useLossLandscape.js`) is not used.

## 7) Priority Fix Queue (Recommended)

1. Fix `dead_code_example` crash path (`autograd_export_json` null safety around node fusion metadata).
2. Fix sparse label smoothing implementation/test mismatch in `loss_functions.c`.
3. Remove or wire dead tests/functions in `tests/test_zoo_models.c`.
4. Remove or use dead website hook `useLossLandscape.js`.
5. Fix CMake install reference to non-existent `scripts/fastapi_server.py`.
6. Make Python test prerequisites explicit (`torch`, CFFI build step), or guard tests to skip cleanly when deps are missing.
7. Update `check_all.sh` to respect externally provided `BIN_DIR`.

## 8) File-by-File Coverage List

The following tracked code files were included in this audit scope (one-by-one inventory coverage):

- `CMakeLists.txt`
- `Makefile`
- `bench_tinygrad.py`
- `benchmarks/bench_all.py`
- `benchmarks/bench_cross_framework.c`
- `benchmarks/profile_mlp_conv.c`
- `benchmarks/profile_overhead.c`
- `benchmarks/profile_overhead_detailed.c`
- `benchmarks/profile_train_conv.c`
- `build.sh`
- `check_all.sh`
- `examples/benchmarks/bench_forward.c`
- `examples/benchmarks/bench_gemm.c`
- `examples/demos/auto_capture_example.c`
- `examples/demos/autograd_example.c`
- `examples/demos/bert.c`
- `examples/demos/clip.c`
- `examples/demos/comprehensive_fusion_example.c`
- `examples/demos/convnext.c`
- `examples/demos/dead_code_example.c`
- `examples/demos/early_stopping_lr_scheduler.c`
- `examples/demos/efficientnet.c`
- `examples/demos/export_graph.c`
- `examples/demos/gpt2.c`
- `examples/demos/inception.c`
- `examples/demos/mask_rcnn.c`
- `examples/demos/mnist_example.c`
- `examples/demos/print_kernels.c`
- `examples/demos/resnet.c`
- `examples/demos/retinanet.c`
- `examples/demos/rnnt.c`
- `examples/demos/stable_diffusion.c`
- `examples/demos/t5.c`
- `examples/demos/training_loop_example.c`
- `examples/demos/unet.c`
- `examples/demos/unet3d.c`
- `examples/demos/vit.c`
- `examples/demos/whisper.c`
- `examples/demos/yolov8.c`
- `examples/llama_inference.c`
- `examples/mlperf/mlperf_resnet50.c`
- `examples/tutorials/activations.c`
- `examples/tutorials/autoencoder.c`
- `examples/tutorials/conv_net.c`
- `examples/tutorials/embedding.c`
- `examples/tutorials/gan.c`
- `examples/tutorials/gru_classifier.c`
- `examples/tutorials/hello_cml.c`
- `examples/tutorials/linear_regression.c`
- `examples/tutorials/logistic_regression.c`
- `examples/tutorials/lr_scheduler.c`
- `examples/tutorials/lstm_timeseries.c`
- `examples/tutorials/mlp_classifier.c`
- `examples/tutorials/multi_task.c`
- `examples/tutorials/rnn_sequence.c`
- `examples/tutorials/simple_xor.c`
- `examples/tutorials/tensor_ops.c`
- `examples/tutorials/transformer.c`
- `include/alloc/graph_allocator.h`
- `include/alloc/memory_management.h`
- `include/alloc/memory_pools.h`
- `include/alloc/tlsf_alloc.h`
- `include/autograd/amp.h`
- `include/autograd/autograd.h`
- `include/autograd/checkpointing.h`
- `include/autograd/forward_ops.h`
- `include/autograd/loss_functions.h`
- `include/backend/backend.h`
- `include/backend/backend_buffer.h`
- `include/backend/blas.h`
- `include/backend/device.h`
- `include/backend/disk_backend.h`
- `include/backend/null_device.h`
- `include/backend/opencl_backend.h`
- `include/backend/profiling.h`
- `include/backend/remote_device.h`
- `include/backend/threadpool.h`
- `include/backend/thunder_executor.h`
- `include/backend/usb3_gpu.h`
- `include/backend/usb_device.h`
- `include/cml.h`
- `include/core/augmentation.h`
- `include/core/cleanup.h`
- `include/core/computation_graph.h`
- `include/core/config.h`
- `include/core/dataset.h`
- `include/core/error_codes.h`
- `include/core/error_stack.h`
- `include/core/export.h`
- `include/core/gguf.h`
- `include/core/gguf_quant.h`
- `include/core/graph_context.h`
- `include/core/hevc.h`
- `include/core/logging.h`
- `include/core/mlperf_logging.h`
- `include/core/model_architecture.h`
- `include/core/onnx.h`
- `include/core/protobuf_mini.h`
- `include/core/pth_loader.h`
- `include/core/quantization.h`
- `include/core/safetensors.h`
- `include/core/serialization.h`
- `include/core/threefry.h`
- `include/core/tinyfs.h`
- `include/core/training_loop.h`
- `include/core/training_metrics.h`
- `include/datasets/datasets.h`
- `include/datasets/loaders.h`
- `include/distributed/comm_backend.h`
- `include/distributed/data_parallel.h`
- `include/distributed/distributed.h`
- `include/distributed/ib_transport.h`
- `include/distributed/pipeline_parallel.h`
- `include/distributed/ring_allreduce.h`
- `include/distributed/tensor_parallel.h`
- `include/nn.h`
- `include/nn/layers.h`
- `include/nn/layers/activations.h`
- `include/nn/layers/batchnorm1d.h`
- `include/nn/layers/batchnorm2d.h`
- `include/nn/layers/batchnorm3d.h`
- `include/nn/layers/containers.h`
- `include/nn/layers/conv1d.h`
- `include/nn/layers/conv2d.h`
- `include/nn/layers/conv3d.h`
- `include/nn/layers/conv_transpose1d.h`
- `include/nn/layers/conv_transpose2d.h`
- `include/nn/layers/conv_transpose3d.h`
- `include/nn/layers/dropout.h`
- `include/nn/layers/embedding.h`
- `include/nn/layers/flatten.h`
- `include/nn/layers/groupnorm.h`
- `include/nn/layers/identity.h`
- `include/nn/layers/instancenorm.h`
- `include/nn/layers/layernorm.h`
- `include/nn/layers/layernorm2d.h`
- `include/nn/layers/linear.h`
- `include/nn/layers/pixel_shuffle.h`
- `include/nn/layers/pooling.h`
- `include/nn/layers/prelu.h`
- `include/nn/layers/rmsnorm.h`
- `include/nn/layers/rnn.h`
- `include/nn/layers/sequential.h`
- `include/nn/layers/transformer.h`
- `include/nn/layers/upsample.h`
- `include/nn/llama.h`
- `include/nn/llm_ops.h`
- `include/nn/lora.h`
- `include/nn/model_io.h`
- `include/nn/openai_api.h`
- `include/nn/paged_attention.h`
- `include/nn/qlora.h`
- `include/nn/serving.h`
- `include/nn/speculative.h`
- `include/nn/state.h`
- `include/ops/ir/aot.h`
- `include/ops/ir/backward.h`
- `include/ops/ir/beam_search.h`
- `include/ops/ir/compiler_viz.h`
- `include/ops/ir/context.h`
- `include/ops/ir/cpu_lazy_materialize.h`
- `include/ops/ir/cross_boundary_fusion.h`
- `include/ops/ir/decompose.h`
- `include/ops/ir/disk_cache.h`
- `include/ops/ir/dispatch.h`
- `include/ops/ir/execution.h`
- `include/ops/ir/export.h`
- `include/ops/ir/fused_codegen.h`
- `include/ops/ir/fusion_patterns.h`
- `include/ops/ir/gpu/adreno_backend.h`
- `include/ops/ir/gpu/am_driver.h`
- `include/ops/ir/gpu/am_mock.h`
- `include/ops/ir/gpu/amd_llvm.h`
- `include/ops/ir/gpu/amd_profiling.h`
- `include/ops/ir/gpu/amdgpu_kd.h`
- `include/ops/ir/gpu/amx.h`
- `include/ops/ir/gpu/cuda_backend.h`
- `include/ops/ir/gpu/cuda_graph_replay.h`
- `include/ops/ir/gpu/gpu_codegen.h`
- `include/ops/ir/gpu/hexagon_backend.h`
- `include/ops/ir/gpu/hexagon_sim.h`
- `include/ops/ir/gpu/metal_backend.h`
- `include/ops/ir/gpu/nak_backend.h`
- `include/ops/ir/gpu/nv_driver.h`
- `include/ops/ir/gpu/nv_mock.h`
- `include/ops/ir/gpu/nv_qmd.h`
- `include/ops/ir/gpu/opencl_ir_backend.h`
- `include/ops/ir/gpu/ptx_codegen.h`
- `include/ops/ir/gpu/rdna3_emu.h`
- `include/ops/ir/gpu/rocm_backend.h`
- `include/ops/ir/gpu/spirv_codegen.h`
- `include/ops/ir/gpu/vulkan_backend.h`
- `include/ops/ir/gpu/webgpu_backend.h`
- `include/ops/ir/gpu/wmma.h`
- `include/ops/ir/gpu/xmx.h`
- `include/ops/ir/graph_cache.h`
- `include/ops/ir/graph_capture.h`
- `include/ops/ir/hcq.h`
- `include/ops/ir/heuristic_opt.h`
- `include/ops/ir/intern.h`
- `include/ops/ir/internal.h`
- `include/ops/ir/ir.h`
- `include/ops/ir/kernel_cache.h`
- `include/ops/ir/late_passes.h`
- `include/ops/ir/linearize.h`
- `include/ops/ir/llvm/llvm_backend.h`
- `include/ops/ir/memory_planner.h`
- `include/ops/ir/multi_output.h`
- `include/ops/ir/nir_compiler.h`
- `include/ops/ir/opt_transforms.h`
- `include/ops/ir/optimization.h`
- `include/ops/ir/pattern_matcher.h`
- `include/ops/ir/process_replay.h`
- `include/ops/ir/rangeify.h`
- `include/ops/ir/runtime_compiler.h`
- `include/ops/ir/schedule.h`
- `include/ops/ir/schedule_allreduce.h`
- `include/ops/ir/schedule_indexing.h`
- `include/ops/ir/schedule_multi.h`
- `include/ops/ir/spec.h`
- `include/ops/ir/tc_opt.h`
- `include/ops/ir/tiny_jit.h`
- `include/ops/ir/trace.h`
- `include/ops/ir/tree_automaton.h`
- `include/ops/ir/validate.h`
- `include/ops/ir/viz_server.h`
- `include/ops/ir/z3_verify.h`
- `include/ops/simd_math.h`
- `include/ops/simd_utils.h`
- `include/ops/simd_views.h`
- `include/ops/uops.h`
- `include/ops/winograd.h`
- `include/optim.h`
- `include/optim/lr_scheduler.h`
- `include/symbolic/divandmod.h`
- `include/symbolic/symbolic.h`
- `include/tensor/dtype_vec.h`
- `include/tensor/image_dtype.h`
- `include/tensor/realize.h`
- `include/tensor/shape_tracker.h`
- `include/tensor/shard.h`
- `include/tensor/sparse_tensor.h`
- `include/tensor/tensor.h`
- `include/tensor/tensor_manipulation.h`
- `include/tensor/tensor_views.h`
- `include/torch_compat.h`
- `include/zoo/bert.h`
- `include/zoo/clip.h`
- `include/zoo/convnext.h`
- `include/zoo/efficientnet.h`
- `include/zoo/gpt2.h`
- `include/zoo/inception.h`
- `include/zoo/mask_rcnn.h`
- `include/zoo/resnet.h`
- `include/zoo/retinanet.h`
- `include/zoo/rnnt.h`
- `include/zoo/stable_diffusion.h`
- `include/zoo/t5.h`
- `include/zoo/unet.h`
- `include/zoo/unet3d.h`
- `include/zoo/vit.h`
- `include/zoo/whisper.h`
- `include/zoo/yolov8.h`
- `include/zoo/zoo.h`
- `main.c`
- `python/build.py`
- `python/cml/__init__.py`
- `python/cml/_cml_cffi.py`
- `python/cml/autograd.py`
- `python/cml/build_cffi.py`
- `python/cml/core.py`
- `python/cml/data.py`
- `python/cml/distributed.py`
- `python/cml/functional.py`
- `python/cml/losses.py`
- `python/cml/nn.py`
- `python/cml/optim.py`
- `python/cml/simple_api.py`
- `python/cml/tensor.py`
- `python/cml/tensor_ops.py`
- `python/cml/utils.py`
- `python/cml/viz/__init__.py`
- `python/cml/viz/server.py`
- `python/cml/zoo.py`
- `python/examples/classification.py`
- `python/examples/convolution.py`
- `python/examples/easy_api.py`
- `python/examples/hello_cml.py`
- `python/examples/neural_network.py`
- `python/setup.py`
- `python/test_cffi_complete.py`
- `python/tests/test_zoo.py`
- `scripts/check-main-branch.sh`
- `scripts/mlperf/run_bert.sh`
- `scripts/mlperf/run_resnet50.sh`
- `scripts/process_replay.sh`
- `scripts/setup_env.sh`
- `scripts/viz.py`
- `src/alloc/graph_allocator.c`
- `src/alloc/memory_management.c`
- `src/alloc/memory_pools.c`
- `src/alloc/tlsf_alloc.c`
- `src/autograd/amp.c`
- `src/autograd/autograd.c`
- `src/autograd/checkpointing.c`
- `src/autograd/forward_ops.c`
- `src/autograd/loss_functions.c`
- `src/backend/backend.c`
- `src/backend/backend_buffer.c`
- `src/backend/blas.c`
- `src/backend/device.c`
- `src/backend/disk_backend.c`
- `src/backend/null_device.c`
- `src/backend/opencl_backend.c`
- `src/backend/profiling.c`
- `src/backend/remote_device.c`
- `src/backend/threadpool.c`
- `src/backend/thunder_executor.c`
- `src/backend/usb3_gpu.c`
- `src/backend/usb_device.c`
- `src/cml.c`
- `src/core/augmentation.c`
- `src/core/cleanup.c`
- `src/core/computation_graph.c`
- `src/core/config.c`
- `src/core/dataset.c`
- `src/core/error_stack.c`
- `src/core/gguf.c`
- `src/core/gguf_quant.c`
- `src/core/graph_context.c`
- `src/core/hevc.c`
- `src/core/logging.c`
- `src/core/mlperf_logging.c`
- `src/core/model_architecture.c`
- `src/core/onnx.c`
- `src/core/onnx_ops.c`
- `src/core/protobuf_mini.c`
- `src/core/pth_loader.c`
- `src/core/quantization.c`
- `src/core/safetensors.c`
- `src/core/serialization.c`
- `src/core/threefry.c`
- `src/core/tinyfs.c`
- `src/core/training_loop.c`
- `src/core/training_metrics.c`
- `src/core/viz_assets.h`
- `src/datasets/builtin_data.c`
- `src/datasets/csv_parser.c`
- `src/datasets/datasets.c`
- `src/datasets/idx_parser.c`
- `src/datasets/loaders.c`
- `src/distributed/comm_backend.c`
- `src/distributed/data_parallel.c`
- `src/distributed/distributed.c`
- `src/distributed/gloo_backend.c`
- `src/distributed/ib_transport.c`
- `src/distributed/mpi_backend.c`
- `src/distributed/nccl_backend.c`
- `src/distributed/pipeline_parallel.c`
- `src/distributed/ring_allreduce.c`
- `src/distributed/tensor_parallel.c`
- `src/nn.c`
- `src/nn/layers/activations.c`
- `src/nn/layers/batchnorm1d.c`
- `src/nn/layers/batchnorm2d.c`
- `src/nn/layers/batchnorm3d.c`
- `src/nn/layers/containers.c`
- `src/nn/layers/conv1d.c`
- `src/nn/layers/conv2d.c`
- `src/nn/layers/conv3d.c`
- `src/nn/layers/conv_transpose1d.c`
- `src/nn/layers/conv_transpose2d.c`
- `src/nn/layers/conv_transpose3d.c`
- `src/nn/layers/dropout.c`
- `src/nn/layers/embedding.c`
- `src/nn/layers/flatten.c`
- `src/nn/layers/groupnorm.c`
- `src/nn/layers/identity.c`
- `src/nn/layers/instancenorm.c`
- `src/nn/layers/layernorm.c`
- `src/nn/layers/layernorm2d.c`
- `src/nn/layers/linear.c`
- `src/nn/layers/pixel_shuffle.c`
- `src/nn/layers/pooling.c`
- `src/nn/layers/prelu.c`
- `src/nn/layers/rmsnorm.c`
- `src/nn/layers/rnn.c`
- `src/nn/layers/sequential.c`
- `src/nn/layers/transformer.c`
- `src/nn/layers/upsample.c`
- `src/nn/llama.c`
- `src/nn/llm_ops.c`
- `src/nn/lora.c`
- `src/nn/model_io.c`
- `src/nn/openai_api.c`
- `src/nn/paged_attention.c`
- `src/nn/qlora.c`
- `src/nn/serving.c`
- `src/nn/speculative.c`
- `src/nn/state.c`
- `src/ops/ir/aot.c`
- `src/ops/ir/backward.c`
- `src/ops/ir/beam_cuda_timing.c`
- `src/ops/ir/beam_search.c`
- `src/ops/ir/compiler_viz.c`
- `src/ops/ir/context.c`
- `src/ops/ir/cpu_lazy_materialize.c`
- `src/ops/ir/cross_boundary_fusion.c`
- `src/ops/ir/decompose.c`
- `src/ops/ir/disk_cache.c`
- `src/ops/ir/dispatch.c`
- `src/ops/ir/execution.c`
- `src/ops/ir/export.c`
- `src/ops/ir/fused_codegen.c`
- `src/ops/ir/fusion_patterns.c`
- `src/ops/ir/gpu/adreno_backend.c`
- `src/ops/ir/gpu/am_driver.c`
- `src/ops/ir/gpu/am_mock.c`
- `src/ops/ir/gpu/amd_llvm.c`
- `src/ops/ir/gpu/amd_profiling.c`
- `src/ops/ir/gpu/amx.c`
- `src/ops/ir/gpu/cuda_backend.c`
- `src/ops/ir/gpu/cuda_graph_replay.c`
- `src/ops/ir/gpu/gpu_codegen.c`
- `src/ops/ir/gpu/hexagon_backend.c`
- `src/ops/ir/gpu/hexagon_sim.c`
- `src/ops/ir/gpu/metal_backend.m`
- `src/ops/ir/gpu/metal_codegen.m`
- `src/ops/ir/gpu/metal_mps.m`
- `src/ops/ir/gpu/nak_backend.c`
- `src/ops/ir/gpu/nv_driver.c`
- `src/ops/ir/gpu/nv_mock.c`
- `src/ops/ir/gpu/nv_qmd.c`
- `src/ops/ir/gpu/opencl_ir_backend.c`
- `src/ops/ir/gpu/ptx_codegen.c`
- `src/ops/ir/gpu/rdna3_emu.c`
- `src/ops/ir/gpu/rocm_backend.c`
- `src/ops/ir/gpu/spirv_codegen.c`
- `src/ops/ir/gpu/vulkan_backend.c`
- `src/ops/ir/gpu/webgpu_backend.c`
- `src/ops/ir/gpu/wgsl_codegen.c`
- `src/ops/ir/gpu/wmma.c`
- `src/ops/ir/gpu/xmx.c`
- `src/ops/ir/graph_cache.c`
- `src/ops/ir/graph_capture.c`
- `src/ops/ir/hcq.c`
- `src/ops/ir/hcq_am.c`
- `src/ops/ir/hcq_cuda.c`
- `src/ops/ir/hcq_nir.c`
- `src/ops/ir/hcq_nv.c`
- `src/ops/ir/hcq_opencl.c`
- `src/ops/ir/hcq_vulkan.c`
- `src/ops/ir/heuristic_opt.c`
- `src/ops/ir/intern.c`
- `src/ops/ir/ir.c`
- `src/ops/ir/kernel_cache.c`
- `src/ops/ir/late_passes.c`
- `src/ops/ir/linearize.c`
- `src/ops/ir/llvm/llvm_backend.c`
- `src/ops/ir/memory_planner.c`
- `src/ops/ir/multi_output.c`
- `src/ops/ir/nir_compiler.c`
- `src/ops/ir/opt_transforms.c`
- `src/ops/ir/optimization.c`
- `src/ops/ir/pattern_matcher.c`
- `src/ops/ir/process_replay.c`
- `src/ops/ir/rangeify.c`
- `src/ops/ir/runtime_compiler.c`
- `src/ops/ir/schedule.c`
- `src/ops/ir/schedule_allreduce.c`
- `src/ops/ir/schedule_indexing.c`
- `src/ops/ir/schedule_multi.c`
- `src/ops/ir/schedule_v2.c`
- `src/ops/ir/spec.c`
- `src/ops/ir/tc_opt.c`
- `src/ops/ir/tiny_jit.c`
- `src/ops/ir/trace.c`
- `src/ops/ir/tree_automaton.c`
- `src/ops/ir/validate.c`
- `src/ops/ir/viz_server.c`
- `src/ops/ir/z3_verify.c`
- `src/ops/simd_math.c`
- `src/ops/simd_utils.c`
- `src/ops/simd_views.c`
- `src/ops/uops.c`
- `src/ops/winograd.c`
- `src/optim.c`
- `src/optim/lr_scheduler.c`
- `src/symbolic/divandmod.c`
- `src/symbolic/symbolic.c`
- `src/tensor/dtype_vec.c`
- `src/tensor/image_dtype.c`
- `src/tensor/realize.c`
- `src/tensor/shape_tracker.c`
- `src/tensor/shard.c`
- `src/tensor/sparse_tensor.c`
- `src/tensor/tensor.c`
- `src/tensor/tensor_hash.c`
- `src/tensor/tensor_manipulation.c`
- `src/tensor/tensor_ops_extra.c`
- `src/tensor/tensor_views.c`
- `src/zoo/bert.c`
- `src/zoo/clip.c`
- `src/zoo/convnext.c`
- `src/zoo/efficientnet.c`
- `src/zoo/gpt2.c`
- `src/zoo/inception.c`
- `src/zoo/mask_rcnn.c`
- `src/zoo/resnet.c`
- `src/zoo/retinanet.c`
- `src/zoo/rnnt.c`
- `src/zoo/stable_diffusion.c`
- `src/zoo/t5.c`
- `src/zoo/unet.c`
- `src/zoo/unet3d.c`
- `src/zoo/vit.c`
- `src/zoo/whisper.c`
- `src/zoo/yolov8.c`
- `src/zoo/zoo.c`
- `test-leak.sh`
- `tests/bench_backends.c`
- `tests/conftest.py`
- `tests/grad_check.c`
- `tests/opcheck.c`
- `tests/test_activations.py`
- `tests/test_am_driver.c`
- `tests/test_amd_profiling.c`
- `tests/test_autograd.c`
- `tests/test_backends.c`
- `tests/test_beam_search.c`
- `tests/test_compiler_features.c`
- `tests/test_convergence.c`
- `tests/test_cross_boundary_fusion.c`
- `tests/test_cuda_graph_replay.c`
- `tests/test_disk_backend.c`
- `tests/test_disk_cache.c`
- `tests/test_dispatch.c`
- `tests/test_distributed.c`
- `tests/test_edge_cases.c`
- `tests/test_elementwise.py`
- `tests/test_fp8_fnuz.c`
- `tests/test_fused_codegen.c`
- `tests/test_fusion_schedule.c`
- `tests/test_fuzzer_ops.c`
- `tests/test_gguf_quant.c`
- `tests/test_gpu_codegen.c`
- `tests/test_gqa_flash.c`
- `tests/test_grad_check.c`
- `tests/test_graph_cache_integration.c`
- `tests/test_graph_capture.c`
- `tests/test_hcq.c`
- `tests/test_heuristic_opt.c`
- `tests/test_hevc.c`
- `tests/test_hexagon_sim.c`
- `tests/test_hw_support.c`
- `tests/test_ib_transport.c`
- `tests/test_image_dtype.c`
- `tests/test_intern.c`
- `tests/test_kernel_cache.c`
- `tests/test_label_smoothing.c`
- `tests/test_late_passes.c`
- `tests/test_layers.py`
- `tests/test_llama.c`
- `tests/test_llm_ops.c`
- `tests/test_loaders.c`
- `tests/test_lora.c`
- `tests/test_losses.py`
- `tests/test_memory.c`
- `tests/test_memory_planner.c`
- `tests/test_metal_backend.c`
- `tests/test_missing_ops.c`
- `tests/test_modlist_crash.c`
- `tests/test_multi_gpu.c`
- `tests/test_new_features.c`
- `tests/test_new_layers.c`
- `tests/test_nir_compiler.c`
- `tests/test_nn_layers.c`
- `tests/test_null_device.c`
- `tests/test_numerical_ops.c`
- `tests/test_nv_driver.c`
- `tests/test_onnx.c`
- `tests/test_openai_api.c`
- `tests/test_opencl_ir.c`
- `tests/test_opt_transforms.c`
- `tests/test_optim.c`
- `tests/test_paged_attention.c`
- `tests/test_pattern_matcher.c`
- `tests/test_process_replay.c`
- `tests/test_pth_loader.c`
- `tests/test_ptx_codegen.c`
- `tests/test_qlora.c`
- `tests/test_rangeify.c`
- `tests/test_reductions.py`
- `tests/test_remote_device.c`
- `tests/test_runtime_compiler.c`
- `tests/test_schedule.c`
- `tests/test_schedule_v2.c`
- `tests/test_serialization.c`
- `tests/test_serving.c`
- `tests/test_shard.c`
- `tests/test_spec.c`
- `tests/test_speculative.c`
- `tests/test_symbolic.c`
- `tests/test_tc_opt.c`
- `tests/test_tensor.c`
- `tests/test_tensor_assign.c`
- `tests/test_tensor_hash.c`
- `tests/test_tensor_ops.py`
- `tests/test_tensor_ops_extra.c`
- `tests/test_tensor_parallel.c`
- `tests/test_threefry.c`
- `tests/test_tlsf.c`
- `tests/test_torch_compat.c`
- `tests/test_trace.c`
- `tests/test_training_loop.c`
- `tests/test_tree_automaton.c`
- `tests/test_unary.py`
- `tests/test_usb3_gpu.c`
- `tests/test_vulkan_backend.c`
- `tests/test_webgpu_backend.c`
- `tests/test_winograd.c`
- `tests/test_wmma.c`
- `tests/test_zoo_models.c`
- `viz/assets/cytoscape-dagre.js`
- `viz/assets/cytoscape-elk.js`
- `viz/assets/cytoscape.min.js`
- `viz/assets/d3.min.js`
- `viz/assets/dagre.min.js`
- `viz/assets/elk.bundled.js`
- `viz/assets/highlight.min.js`
- `viz/assets/languages/c.min.js`
- `viz/fetch_assets.sh`
- `viz/serve.py`
- `viz/viz-charts.js`
- `viz/viz.js`
- `viz/worker.js`
- `website/src/App.jsx`
- `website/src/components/Architecture.jsx`
- `website/src/components/CTA.jsx`
- `website/src/components/CodeShowcase.jsx`
- `website/src/components/DocsPage.jsx`
- `website/src/components/Features.jsx`
- `website/src/components/Footer.jsx`
- `website/src/components/Hero.jsx`
- `website/src/components/Navbar.jsx`
- `website/src/components/Numbers.jsx`
- `website/src/hooks/useLossLandscape.js`
- `website/src/main.jsx`
- `website/vite.config.js`

## 9) Deep Dive Addendum: IR Generation + Training Loop

This section is a focused follow-up audit on IR generation and training loop behavior.

### 9.1 Validation performed

1. Fresh ASAN build created from current sources:
- build dir: `build-asan-audit`
- flags: `ENABLE_SANITIZERS=ON`, Debug, tests/examples enabled.

2. IR/training focused tests executed under ASAN (`ASAN_OPTIONS=detect_leaks=0`):
- `test_compiler_features`
- `test_dispatch`
- `test_fused_codegen`
- `test_graph_capture`
- `test_late_passes`
- `test_memory_planner`
- `test_opt_transforms`
- `test_process_replay`
- `test_rangeify`
- `test_runtime_compiler`
- `test_schedule`
- `test_schedule_v2`
- `test_spec`
- `test_trace`
- `test_training_loop`
- Result: all passed in this focused run.

3. IR/training examples executed under ASAN:
- `training_loop_example`: fails (heap-use-after-free)
- `dead_code_example`: fails (null dereference)
- `export_graph`: passes
- `auto_capture_example`: passes
- `comprehensive_fusion_example`: passes
- `print_kernels`: passes

### 9.2 Confirmed issues (high confidence)

#### A) Training loop demo has UAF due reset/free ordering

- File: `/home/arrry/dev/personal/C-ML/examples/demos/training_loop_example.c:196`
- Problem sequence in demo loop:
1. `cml_reset_ir_context();`
2. `tensor_free(loss);`
3. `tensor_free(outputs);`

Because IR reset tears down graph-owned output tensors, freeing `loss/outputs` immediately after can dereference freed memory.

Evidence:
- ASAN stack points to `training_loop_example.c:196` and subsequent `tensor_free(...)` at lines 199/200.

Impact:
- Example aborts and does not demonstrate training loop reliably.

#### B) Dead-code IR demo crashes in graph JSON export

- File: `/home/arrry/dev/personal/C-ML/src/autograd/autograd.c:744`
- In `autograd_export_json`, this loop assumes `node->inputs` is non-null when `num_inputs > 0`:
- `if (!node->inputs[j] || !node->inputs[j]->ir_node) ...`

But some nodes can violate that assumption, producing null dereference.

Evidence:
- ASAN crash from `dead_code_example`:
- callsite: `/home/arrry/dev/personal/C-ML/examples/demos/dead_code_example.c:75`
- crash: `/home/arrry/dev/personal/C-ML/src/autograd/autograd.c:744`

Impact:
- Dedicated dead-code/optimization demo is currently broken.

### 9.3 Training loop API review findings

#### C) Validation path does not switch model mode

- File: `/home/arrry/dev/personal/C-ML/src/core/training_loop.c`
- `cml_train_with_validation` performs validation forward using same model state with no explicit switch to eval mode.

Risk:
- Layers such as dropout/batchnorm may run in training behavior during validation, making validation metrics biased/noisy.

#### D) Validation path does not clear IR per validation batch

- File: `/home/arrry/dev/personal/C-ML/src/core/training_loop.c:741+`
- Training loop explicitly resets IR each train batch (`line 636`), but validation loop lacks corresponding reset before each validation forward.

Risk:
- Larger validation loops can accumulate graph state and memory footprint unnecessarily.

#### E) Validation path builds graph though no backward

- File: `/home/arrry/dev/personal/C-ML/src/core/training_loop.c:744-766`
- Validation uses full `module_forward` + loss construction without no-grad/eval guard.

Risk:
- Extra graph construction overhead during validation.

### 9.4 IR lifecycle/ownership observations

#### F) `cml_ir_free` aggressively frees node outputs and tensor refs

- File: `/home/arrry/dev/personal/C-ML/src/ops/ir/ir.c:756-809`
- Current behavior is internally consistent for tests, but user/demo code must avoid freeing graph tensors after `cml_reset_ir_context()` unless they were detached/realized correctly.

Practical implication:
- The API is easy to misuse; examples currently do misuse it (see issue A).

### 9.5 Focused recommendation order (IR + training)

1. Fix `training_loop_example` ordering bug (free tensors before IR reset, or stop manual frees for tensors already invalidated by reset).
2. Harden `autograd_export_json` against null/malformed `node->inputs` (guard `node->inputs` before indexing).
3. In `cml_train_with_validation`, switch model to eval mode during validation and restore mode after.
4. Reset IR context per validation batch or run validation in an explicit no-grad/eager path.
5. Document IR reset ownership rules near `cml_reset_ir_context` and example code.


• Completed a deeper focused audit on IR generation + training loop and updated the
  report with a new section:

  - Updated file: report-codex.md

  What I specifically validated:

  1. Fresh sanitizer rebuild (build-asan-audit) from current sources.
  2. IR/training-focused test subset under ASAN (detect_leaks=0): all passed.
  3. IR/training demos under ASAN: 2 real failures confirmed.

  Key confirmed issues:

  1. training_loop_example has use-after-free due reset/free order.

  - Reset happens before freeing tensors at examples/demos/
    training_loop_example.c:196 and frees occur at examples/demos/
    training_loop_example.c:199, examples/demos/training_loop_example.c:200.

  2. dead_code_example crashes in graph export because autograd_export_json assumes
     node->inputs is non-null.

  - Crash site: src/autograd/autograd.c:744
  - Caller: examples/demos/dead_code_example.c:75

  3. Training API quality gaps in validation path:

  - No explicit eval-mode switch in validation loop: src/core/training_loop.c:578
  - Train batches reset IR at src/core/training_loop.c:636, validation loop lacks
    equivalent reset.
  - Validation still builds normal graph path (no no-grad fast path).

  If you want, next I can fix these IR/training issues in code in priority order.
