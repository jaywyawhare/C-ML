#include "zoo/yolov8.h"
#include "zoo/zoo.h"
#include "nn.h"
#include "nn/layers.h"
#include "nn/model_io.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

YOLOv8Config yolov8n_config(int num_classes) {
    YOLOv8Config cfg = {
        .num_classes = num_classes > 0 ? num_classes : 80,
        .input_size = 640,
        .conf_threshold = 0.25f,
        .nms_threshold = 0.45f,
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU
    };
    return cfg;
}

static void add_conv_bn_silu(Sequential* seq, int in_ch, int out_ch, int kernel,
                              int stride, DType dtype, DeviceType device) {
    int pad = kernel / 2;
    sequential_add(seq, (Module*)nn_conv2d(in_ch, out_ch, kernel, stride, pad, 1, false, dtype, device));
    sequential_add(seq, (Module*)nn_batchnorm2d(out_ch, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
}

static void add_bottleneck(Sequential* seq, int channels, bool shortcut,
                            DType dtype, DeviceType device) {
    int hidden = channels / 2;
    sequential_add(seq, (Module*)nn_conv2d(channels, hidden, 1, 1, 0, 1, false, dtype, device));
    sequential_add(seq, (Module*)nn_batchnorm2d(hidden, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
    sequential_add(seq, (Module*)nn_conv2d(hidden, channels, 3, 1, 1, 1, false, dtype, device));
    sequential_add(seq, (Module*)nn_batchnorm2d(channels, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
    (void)shortcut;
}

static void add_c2f_block(Sequential* seq, int in_ch, int out_ch, int n_bottlenecks,
                           DType dtype, DeviceType device) {
    sequential_add(seq, (Module*)nn_conv2d(in_ch, out_ch, 1, 1, 0, 1, false, dtype, device));
    sequential_add(seq, (Module*)nn_batchnorm2d(out_ch, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());

    for (int i = 0; i < n_bottlenecks; i++)
        add_bottleneck(seq, out_ch, true, dtype, device);

    sequential_add(seq, (Module*)nn_conv2d(out_ch, out_ch, 1, 1, 0, 1, false, dtype, device));
    sequential_add(seq, (Module*)nn_batchnorm2d(out_ch, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
}

static void add_sppf(Sequential* seq, int channels, DType dtype, DeviceType device) {
    int hidden = channels / 2;
    sequential_add(seq, (Module*)nn_conv2d(channels, hidden, 1, 1, 0, 1, false, dtype, device));
    sequential_add(seq, (Module*)nn_batchnorm2d(hidden, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
    sequential_add(seq, (Module*)nn_maxpool2d(5, 1, 2, 1, false));
    sequential_add(seq, (Module*)nn_maxpool2d(5, 1, 2, 1, false));
    sequential_add(seq, (Module*)nn_maxpool2d(5, 1, 2, 1, false));
    sequential_add(seq, (Module*)nn_conv2d(hidden, channels, 1, 1, 0, 1, false, dtype, device));
    sequential_add(seq, (Module*)nn_batchnorm2d(channels, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
}

static void add_detect_head(Sequential* seq, int in_ch, int num_classes, int reg_max,
                             DType dtype, DeviceType device) {
    int bbox_ch = 4 * reg_max;

    sequential_add(seq, (Module*)nn_conv2d(in_ch, in_ch, 3, 1, 1, 1, false, dtype, device));
    sequential_add(seq, (Module*)nn_batchnorm2d(in_ch, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
    sequential_add(seq, (Module*)nn_conv2d(in_ch, bbox_ch, 1, 1, 0, 1, true, dtype, device));

    sequential_add(seq, (Module*)nn_conv2d(in_ch, in_ch, 3, 1, 1, 1, false, dtype, device));
    sequential_add(seq, (Module*)nn_batchnorm2d(in_ch, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
    sequential_add(seq, (Module*)nn_conv2d(in_ch, num_classes, 1, 1, 0, 1, true, dtype, device));
}

Module* cml_zoo_yolov8n(const YOLOv8Config* config) {
    YOLOv8Config cfg = config ? *config : yolov8n_config(80);
    DType dt = cfg.dtype;
    DeviceType dev = cfg.device;

    Sequential* model = nn_sequential();

    /* Backbone: CSPDarknet-nano */
    add_conv_bn_silu(model, 3,  16, 3, 2, dt, dev);   /* P1/2   */
    add_conv_bn_silu(model, 16, 32, 3, 2, dt, dev);   /* P2/4   */
    add_c2f_block(model, 32, 32, 1, dt, dev);

    add_conv_bn_silu(model, 32, 64, 3, 2, dt, dev);   /* P3/8   */
    add_c2f_block(model, 64, 64, 2, dt, dev);

    add_conv_bn_silu(model, 64, 128, 3, 2, dt, dev);  /* P4/16  */
    add_c2f_block(model, 128, 128, 2, dt, dev);

    add_conv_bn_silu(model, 128, 256, 3, 2, dt, dev); /* P5/32  */
    add_c2f_block(model, 256, 256, 1, dt, dev);
    add_sppf(model, 256, dt, dev);

    /* Neck: PAN-FPN (top-down) */
    sequential_add(model, (Module*)nn_upsample(2.0f, NULL, 0, UPSAMPLE_NEAREST, false));
    add_c2f_block(model, 256, 128, 1, dt, dev);

    sequential_add(model, (Module*)nn_upsample(2.0f, NULL, 0, UPSAMPLE_NEAREST, false));
    add_c2f_block(model, 128, 64, 1, dt, dev);

    /* Neck: PAN-FPN (bottom-up) */
    add_conv_bn_silu(model, 64, 64, 3, 2, dt, dev);
    add_c2f_block(model, 64, 128, 1, dt, dev);

    add_conv_bn_silu(model, 128, 128, 3, 2, dt, dev);
    add_c2f_block(model, 128, 256, 1, dt, dev);

    /* Detection heads: P3, P4, P5 */
    int reg_max = 16;
    add_detect_head(model, 64,  cfg.num_classes, reg_max, dt, dev);
    add_detect_head(model, 128, cfg.num_classes, reg_max, dt, dev);
    add_detect_head(model, 256, cfg.num_classes, reg_max, dt, dev);

    LOG_INFO("Created YOLOv8-n: %d classes, %dx%d input", cfg.num_classes, cfg.input_size, cfg.input_size);
    return (Module*)model;
}
