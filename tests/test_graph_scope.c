/* Graph-view scoping: every IR node emitted inside a layer's forward should
 * carry that layer's module path, so the dashboard can collapse a decomposed
 * graph back into the layers it came from. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cml.h"
#include "ops/ir/internal.h"
#include "ops/ir/export.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"


int main(void) {
    /* Enable scoping *after* cml_init: setting VIZ beforehand makes cml_init
     * spawn the dashboard server and a browser, which a test must not do.
     * cml_ir_scope_enabled() probes the env lazily, so this still takes. */
    cml_init();
    setenv("VIZ", "1", 1);
    printf("=== graph scope ===\n");

    CHECK("scope tracking enabled under VIZ", cml_ir_scope_enabled());

    /* Stack behaviour, independent of any model. */
    CHECK("scope starts empty", cml_ir_scope_current() == NULL);
    cml_ir_scope_push("Sequential");
    CHECK("single push", cml_ir_scope_current() &&
                         strcmp(cml_ir_scope_current(), "Sequential") == 0);
    cml_ir_scope_push("Linear");
    CHECK("nested push joins with '/'", cml_ir_scope_current() &&
                                        strcmp(cml_ir_scope_current(), "Sequential/Linear") == 0);
    cml_ir_scope_pop();
    CHECK("pop restores parent", cml_ir_scope_current() &&
                                 strcmp(cml_ir_scope_current(), "Sequential") == 0);
    cml_ir_scope_pop();
    CHECK("pop to empty", cml_ir_scope_current() == NULL);
    cml_ir_scope_pop(); /* underflow must not corrupt state */
    CHECK("extra pop is harmless", cml_ir_scope_current() == NULL);

    /* A real forward pass should tag its nodes without any layer opting in. */
    cml_reset_ir_context();

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(8, 16, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));

    int xshape[] = {2, 8};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* x = cml_zeros(xshape, 2, &cfg);
    Tensor* out = module_forward((Module*)model, x);
    CHECK("forward produced output", out != NULL);

    CHECK("scope balanced after forward", cml_ir_scope_current() == NULL);

    char* json = cml_ir_export_graph_json(cml_ir_get_or_create_context());
    CHECK("graph json exported", json != NULL);
    if (json) {
        CHECK("nodes carry a scope field", strstr(json, "\"scope\"") != NULL);
        CHECK("scope names the container", strstr(json, "Sequential") != NULL);
        cml_free(json); /* project allocator, not libc malloc */
    }

    return TEST_SUMMARY();
}
