# Graph Mode API

## Enable Lazy Mode

```c
#include "cml.h"

cml_init();
cml_enable_lazy_mode();

Tensor* a = tensor_zeros_2d(10, 20, dtype, device);
Tensor* b = tensor_ones_2d(10, 20, dtype, device);
Tensor* c = tensor_add(a, b);  // Creates graph node, doesn't execute

cml_execute_current_graph();
cml_disable_lazy_mode();
cml_cleanup();
```

## Explicit Graph Building

```c
CMLGraph_t ir = cml_ir_new(IR_TARGET_CUDA);
Tensor* inputs[] = {a, b};
cml_ir_add_uop(ir, UOP_ADD, inputs, 2, NULL);
cml_ir_execute(ir);
cml_ir_free(ir);
```
