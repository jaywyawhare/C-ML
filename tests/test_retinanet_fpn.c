/* RetinaNet must score anchors over the whole feature pyramid, not just the
 * finest level P3. The shared classification subnet runs on P3, P4 and P5 and
 * the per-level maps are flattened to [B, A*num_classes, H*W] and concatenated,
 * so the forward output is 3-D ([B, A*num_classes, total_locations]) rather than
 * the old single-level 4-D [B, A*num_classes, H, W]. */
#include <stdlib.h>
#include "cml.h"
#include "zoo/retinanet.h"
#include "test_harness.h"

static int test_retinanet_multilevel_output(void) {
    RetinaNetConfig cfg = cml_zoo_retinanet_default_config();
    cfg.num_classes     = 4; /* keep the head small/fast */
    cfg.num_anchors     = 3;
    Module* net         = cml_zoo_retinanet_create(&cfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!net)
        return 0;

    TensorConfig tc = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int shape[4]  = {1, 3, 64, 64};
    Tensor* input = tensor_randn(shape, 4, &tc);
    int ok        = input != NULL;

    if (ok) {
        Tensor* out = module_forward(net, input);
        ok          = out != NULL;
        if (ok) {
            tensor_ensure_executed(out);
            /* concatenated across pyramid levels => 3-D, channel dim is the
             * shared head width A*num_classes, and the location axis sums P3+P4+P5
             * (strictly more than any single level alone). */
            ok = out->ndim == 3 && out->shape[0] == 1 &&
                 out->shape[1] == cfg.num_anchors * cfg.num_classes && out->shape[2] > 0;
            /* 64x64 input over strides 8/16/32 => 8*8 + 4*4 + 2*2 = 84 locations. */
            ok = ok && out->shape[2] == (8 * 8 + 4 * 4 + 2 * 2);
        }
        tensor_free(input);
    }
    module_free(net);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== RetinaNet multi-level FPN head ===\n");
    TEST(retinanet_multilevel_output);
    cml_cleanup();
    return TEST_SUMMARY();
}
