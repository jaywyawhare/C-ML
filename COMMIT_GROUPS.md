# Git Commit Organization Guide

## Files to IGNORE (already in .gitignore or should be added)

These files should NOT be committed:
- `graph.json`, `graph_dead.json`, `graph_fused.json`, `graph_optimized.json`, `graph_unopt.json` (generated visualization outputs)
- `kernels.json` (generated kernel data)
- `model_architecture.json` (generated model architecture visualization)
- `training.json` (generated training data visualization)
- `callgrind.out` (profiling output)
- `build/` directory (build artifacts)
- `data/` directory (dataset files)
- `ideation/` directory (already in .gitignore)
- `python/cml.egg-info/` (Python build artifacts)
- `viz-ui/node_modules/` (Node.js dependencies)
- `viz-ui/dist/` (frontend build output)

---

## Commit Groups

### Commit 1: Core Project Configuration Files
**Purpose:** Essential project setup and build configuration

```
.gitignore
CMakeLists.txt
Makefile
README.md
QUICKSTART.md
LICENCE.md
mkdocs.yml
build.sh
build.bat
build.ps1
```

### Commit 2: Core Documentation
**Purpose:** Main documentation files

```
docs/index.md
docs/api_reference.md
docs/autograd.md
docs/getting_started.md
docs/graph_mode.md
docs/ir_graph_management.md
docs/kernel_studio.md
docs/kernel_studio_quickref.md
docs/nn_layers.md
docs/optimizations.md
docs/training.md
docs/MLIR_INTEGRATION_PLAN.md
docs/MLIR_STATUS.md
docs/extra.css
docs/light-mode.svg
docs/logo-switcher.js
docs/mlir/
```

### Commit 3: Core Header Files (Public API)
**Purpose:** Main public API headers

```
include/cml.h
include/nn.h
include/optim.h
include/tensor/tensor.h
include/tensor/tensor_manipulation.h
include/tensor/tensor_views.h
```

### Commit 4: Autograd Headers
**Purpose:** Automatic differentiation headers

```
include/autograd/autograd.h
include/autograd/checkpointing.h
include/autograd/forward_ops.h
```

### Commit 5: Neural Network Layer Headers
**Purpose:** NN layer definitions

```
include/nn/layers/activations.h
include/nn/layers/batchnorm2d.h
include/nn/layers/conv2d.h
include/nn/layers/dropout.h
include/nn/layers/layernorm.h
include/nn/layers/linear.h
include/nn/layers/pooling.h
include/nn/layers/sequential.h
```

### Commit 6: Core Module Headers
**Purpose:** Core functionality headers

```
include/core/
include/alloc/
include/backend/
include/ops/
```

### Commit 7: Core Source Files
**Purpose:** Main implementation files

```
src/cml.c
src/nn.c
src/optim.c
main.c
```

### Commit 8: Tensor Source Files
**Purpose:** Tensor implementation

```
src/tensor/
```

### Commit 9: Autograd Source Files
**Purpose:** Automatic differentiation implementation

```
src/autograd/autograd.c
src/autograd/checkpointing.c
src/autograd/forward_ops.c
src/autograd/loss_functions.c
```

### Commit 10: Neural Network Layer Source Files
**Purpose:** NN layer implementations

```
src/nn/layers/activations.c
src/nn/layers/batchnorm2d.c
src/nn/layers/conv2d.c
src/nn/layers/dropout.c
src/nn/layers/layernorm.c
src/nn/layers/linear.c
src/nn/layers/pooling.c
src/nn/layers/sequential.c
```

### Commit 11: Core Module Source Files
**Purpose:** Core functionality implementations

```
src/core/
src/alloc/
src/backend/
src/ops/
```

### Commit 12: Example Programs
**Purpose:** Example usage code

```
examples/README.md
examples/autograd_example.c
examples/auto_capture_example.c
examples/bench_forward.c
examples/bench_gemm.c
examples/comprehensive_fusion_example.c
examples/dead_code_example.c
examples/early_stopping_lr_scheduler.c
examples/export_graph.c
examples/hello_cml.c
examples/mlir_lowering_poc.c
examples/mnist_example.c
examples/opcheck.c
examples/print_kernels.c
examples/simple_xor.c
examples/test.c
examples/training_loop_example.c
examples/unified_api_example.c
```

### Commit 13: Test Files
**Purpose:** Test suite

```
tests/bench_backends.c
tests/grad_check.c
tests/test_backends.c
tests/test_dispatch.c
tests/test_kernel_cache.c
tests/test_activations.py
tests/test_elementwise.py
tests/test_layers.py
tests/test_losses.py
tests/test_reductions.py
tests/test_tensor_ops.py
tests/test_unary.py
tests/conftest.py
tests/mlir/
```

### Commit 14: Python Bindings
**Purpose:** Python interface and bindings

```
python/
```

### Commit 15: Scripts
**Purpose:** Utility scripts

```
scripts/fastapi_server.py
scripts/setup_env.sh
scripts/viz.py
```

### Commit 16: Visualization UI
**Purpose:** Frontend visualization interface

```
viz-ui/package.json
viz-ui/vite.config.js
viz-ui/index.html
viz-ui/src/
viz-ui/public/
```

### Commit 17: Additional Documentation
**Purpose:** Supplementary documentation

```
new_docs/
```

---

## Quick Reference: Files to Add by Category

### All at once (if preferred):
```bash
# Core files
git add .gitignore CMakeLists.txt Makefile README.md QUICKSTART.md LICENCE.md mkdocs.yml build.sh build.bat build.ps1

# Documentation
git add docs/

# Headers
git add include/

# Source files
git add src/ main.c

# Examples
git add examples/

# Tests
git add tests/

# Python
git add python/

# Scripts
git add scripts/

# UI
git add viz-ui/package.json viz-ui/vite.config.js viz-ui/index.html viz-ui/src/ viz-ui/public/

# Additional docs
git add new_docs/
```
