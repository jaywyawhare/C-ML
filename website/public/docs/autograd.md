# Autograd API

## Initialization

```c
#include "cml.h"  // or "autograd/autograd.h"

autograd_init();
// ...
autograd_shutdown();
```

## API Reference

### Initialization

- `autograd_init()` - Initialize autograd engine
- `autograd_shutdown()` - Shutdown autograd engine
- `autograd_get_engine()` - Get global autograd engine

### Gradient Mode

```c
// High-level API (cml_* prefix)
cml_no_grad();
cml_enable_grad();
bool enabled = cml_is_grad_enabled();

// Low-level API
autograd_set_grad_mode(true);
autograd_set_grad_mode(false);
bool enabled = autograd_is_grad_enabled();
autograd_no_grad_enter();
autograd_no_grad_exit();
```

### Backward Pass

```c
// High-level API
cml_backward(Tensor *tensor, Tensor *gradient, bool retain_graph, bool create_graph)
// or tensor_backward(Tensor *tensor, Tensor *gradient, bool retain_graph, bool create_graph)
```

**Parameters:**

- `tensor`: Output tensor to compute gradients from
- `gradient`: Optional gradient (NULL for scalar tensors)
- `retain_graph`: Keep computation graph after backward (for multiple backward passes)
- `create_graph`: Create graph for higher-order derivatives

```c
// Simple backward
cml_backward(loss, NULL, false, false);

// Retain graph for multiple backward passes
cml_backward(loss, NULL, true, false);

// Higher-order gradients
cml_backward(loss, NULL, false, true);
```

### Tensor Gradient Management

```c
// High-level API
bool requires = cml_requires_grad(tensor);
cml_set_requires_grad(tensor, true);
bool is_leaf = cml_is_leaf(tensor);
cml_zero_grad(tensor);
Tensor* detached = cml_detach(tensor);

// Low-level API
bool requires = tensor_requires_grad(tensor);
tensor_set_requires_grad(tensor, true);
bool is_leaf = tensor_is_leaf(tensor);
tensor_zero_grad(tensor);
Tensor* detached = tensor_detach(tensor);
tensor_accumulate_grad(tensor, new_grad);  // Accumulate gradients
```

### Advanced Features

```c
// Anomaly detection (debugging)
autograd_set_anomaly_detection(true);

autograd_print_graph(tensor);
```

## Usage Example

```c
autograd_init();

int shape[] = {1};
Tensor *x = tensor_ones(shape, 1, NULL);
Tensor *y = tensor_ones(shape, 1, NULL);
tensor_set_float(x, 0, 3.0f);
tensor_set_float(y, 0, 4.0f);

x->requires_grad = true;
y->requires_grad = true;

Tensor *z = tensor_mul(x, y);
tensor_backward(z, NULL, false, false);

printf("dz/dx = %.2f\n", tensor_get_float(x->grad, 0));
printf("dz/dy = %.2f\n", tensor_get_float(y->grad, 0));

tensor_free(x);
tensor_free(y);
tensor_free(z);
autograd_shutdown();
```
