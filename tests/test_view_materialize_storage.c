/* A storage-sharing view (EXPAND, SLICE) materialized onto its own buffer must
 * drop its hold on the aliased block, or tensor_free releases only the block and
 * leaks the private buffer every step. Regression guard for that leak. */
#include "cml.h"
#include "ops/uops.h"
#include "ops/ir/context.h"
#include "tensor/realize.h"
#include "test_harness.h"

#include <stdio.h>

static int test_expand_drops_storage(void) {
    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* a    = cml_tensor_2d(src, 1, 4);
    tensor_ensure_executed(a);

    int shape[2]     = {3, 4};
    ExpandParams prm = {.new_shape = shape, .new_ndim = 2};
    Tensor* e        = uop_expand(a, &prm);
    int ok           = e != NULL && e->storage != NULL; /* starts as a shared view */

    tensor_ensure_executed(e);
    const float* d = (const float*)tensor_data_ptr(e);
    for (int r = 0; ok && d && r < 3; r++)
        for (int c = 0; c < 4; c++)
            ok = ok && d[r * 4 + c] == src[c];

    ok = ok && d && e->owns_data && e->storage == NULL;
    /* The base still holds its own reference, and only that one. */
    ok = ok && a->storage && a->storage->refs == 1;

    tensor_free(e);
    tensor_free(a);
    cml_ir_reset_global_context();
    return ok;
}

static int test_slice_drops_storage(void) {
    float src[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    Tensor* a    = cml_tensor_2d(src, 2, 4);
    tensor_ensure_executed(a);

    int start[2] = {0, 1}, end[2] = {2, 3}, step[2] = {1, 1};
    SliceParams prm = {.start = start, .end = end, .step = step, .num_dims = 2};
    Tensor* s       = uop_slice(a, &prm);
    int ok          = s != NULL;

    tensor_ensure_executed(s);
    const float* d = (const float*)tensor_data_ptr(s);
    /* rows 0..1, cols 1..2 */
    ok = ok && d && d[0] == 1 && d[1] == 2 && d[2] == 5 && d[3] == 6;
    ok = ok && (s->storage == NULL || !s->owns_data);

    tensor_free(s);
    tensor_free(a);
    cml_ir_reset_global_context();
    return ok;
}

int main(void) {
    cml_init();
    printf("=== view materialization drops shared storage ===\n");
    TEST(expand_drops_storage);
    TEST(slice_drops_storage);
    cml_cleanup();
    return TEST_SUMMARY();
}
