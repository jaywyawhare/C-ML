#include "zoo/mask_rcnn.h"
#include "zoo/resnet.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_manipulation.h"
#include "core/logging.h"
#include <stdlib.h>

MaskRCNNConfig cml_zoo_mask_rcnn_default_config(void) {
    MaskRCNNConfig cfg = {
        .num_classes = 81,
        .num_anchors = 9,
        .fpn_channels = 256,
        .roi_output_size = 7,
        .mask_output_size = 14
    };
    return cfg;
}

typedef struct {
    Module base;
    Sequential* stem;
    Sequential* layer1;
    Sequential* layer2;
    Sequential* layer3;
    Sequential* layer4;
} Backbone;

static void build_bottleneck_layer(Sequential* stage, int num_blocks, int in_ch, int mid_ch,
                                    int out_ch, int first_stride, DType dtype, DeviceType device) {
    for (int i = 0; i < num_blocks; i++) {
        int inch = (i == 0) ? in_ch : out_ch;
        int stride = (i == 0) ? first_stride : 1;

        Sequential* blk = nn_sequential();
        sequential_add(blk, (Module*)nn_conv2d(inch, mid_ch, 1, 1, 0, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(mid_ch, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(blk, (Module*)nn_conv2d(mid_ch, mid_ch, 3, stride, 1, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(mid_ch, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(blk, (Module*)nn_conv2d(mid_ch, out_ch, 1, 1, 0, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(out_ch, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(stage, (Module*)blk);
    }
}

static Tensor* backbone_forward(Module* module, Tensor* input) {
    Backbone* bb = (Backbone*)module;
    if (!bb || !input) return NULL;

    Tensor* x = module_forward((Module*)bb->stem, input);
    if (!x) return NULL;
    x = module_forward((Module*)bb->layer1, x);
    if (!x) return NULL;
    x = module_forward((Module*)bb->layer2, x);
    if (!x) return NULL;
    x = module_forward((Module*)bb->layer3, x);
    if (!x) return NULL;
    return module_forward((Module*)bb->layer4, x);
}

static void backbone_free(Module* module) {
    Backbone* bb = (Backbone*)module;
    if (!bb) return;
    if (bb->stem) module_free((Module*)bb->stem);
    if (bb->layer1) module_free((Module*)bb->layer1);
    if (bb->layer2) module_free((Module*)bb->layer2);
    if (bb->layer3) module_free((Module*)bb->layer3);
    if (bb->layer4) module_free((Module*)bb->layer4);
    free(bb);
}

static Module* create_backbone(DType dtype, DeviceType device) {
    Backbone* bb = malloc(sizeof(Backbone));
    if (!bb) return NULL;

    if (module_init((Module*)bb, "ResNet50+FPN_Backbone", backbone_forward, backbone_free) != 0) {
        free(bb);
        return NULL;
    }

    bb->stem = nn_sequential();
    sequential_add(bb->stem, (Module*)nn_conv2d(3, 64, 7, 2, 3, 1, false, dtype, device));
    sequential_add(bb->stem, (Module*)nn_batchnorm2d(64, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(bb->stem, (Module*)nn_relu(false));
    sequential_add(bb->stem, (Module*)nn_maxpool2d(3, 2, 1, 1, false));

    bb->layer1 = nn_sequential();
    build_bottleneck_layer(bb->layer1, 3, 64, 64, 256, 1, dtype, device);

    bb->layer2 = nn_sequential();
    build_bottleneck_layer(bb->layer2, 4, 256, 128, 512, 2, dtype, device);

    bb->layer3 = nn_sequential();
    build_bottleneck_layer(bb->layer3, 6, 512, 256, 1024, 2, dtype, device);

    bb->layer4 = nn_sequential();
    build_bottleneck_layer(bb->layer4, 3, 1024, 512, 2048, 2, dtype, device);

    return (Module*)bb;
}

typedef struct {
    Module base;
    Module* lateral2;
    Module* lateral3;
    Module* lateral4;
    Module* lateral5;
    Module* smooth2;
    Module* smooth3;
    Module* smooth4;
    Module* smooth5;
} FPN;

static Tensor* fpn_forward(Module* module, Tensor* input) {
    (void)module;
    return input;
}

static void fpn_free(Module* module) {
    FPN* fpn = (FPN*)module;
    if (!fpn) return;
    if (fpn->lateral2) module_free(fpn->lateral2);
    if (fpn->lateral3) module_free(fpn->lateral3);
    if (fpn->lateral4) module_free(fpn->lateral4);
    if (fpn->lateral5) module_free(fpn->lateral5);
    if (fpn->smooth2) module_free(fpn->smooth2);
    if (fpn->smooth3) module_free(fpn->smooth3);
    if (fpn->smooth4) module_free(fpn->smooth4);
    if (fpn->smooth5) module_free(fpn->smooth5);
    free(fpn);
}

static Module* create_fpn(int fpn_ch, DType dtype, DeviceType device) {
    FPN* fpn = malloc(sizeof(FPN));
    if (!fpn) return NULL;

    if (module_init((Module*)fpn, "FPN", fpn_forward, fpn_free) != 0) {
        free(fpn);
        return NULL;
    }

    fpn->lateral2 = (Module*)nn_conv2d(256, fpn_ch, 1, 1, 0, 1, true, dtype, device);
    fpn->lateral3 = (Module*)nn_conv2d(512, fpn_ch, 1, 1, 0, 1, true, dtype, device);
    fpn->lateral4 = (Module*)nn_conv2d(1024, fpn_ch, 1, 1, 0, 1, true, dtype, device);
    fpn->lateral5 = (Module*)nn_conv2d(2048, fpn_ch, 1, 1, 0, 1, true, dtype, device);

    fpn->smooth2 = (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 1, 1, 1, true, dtype, device);
    fpn->smooth3 = (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 1, 1, 1, true, dtype, device);
    fpn->smooth4 = (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 1, 1, 1, true, dtype, device);
    fpn->smooth5 = (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 1, 1, 1, true, dtype, device);

    return (Module*)fpn;
}

typedef struct {
    Module base;
    Module* rpn_conv;
    Module* rpn_cls;
    Module* rpn_bbox;
} RPN;

static Tensor* rpn_forward(Module* module, Tensor* input) {
    RPN* rpn = (RPN*)module;
    if (!rpn || !input) return NULL;

    Tensor* x = module_forward(rpn->rpn_conv, input);
    if (!x) return NULL;
    return f_relu(x);
}

static void rpn_free(Module* module) {
    RPN* rpn = (RPN*)module;
    if (!rpn) return;
    if (rpn->rpn_conv) module_free(rpn->rpn_conv);
    if (rpn->rpn_cls) module_free(rpn->rpn_cls);
    if (rpn->rpn_bbox) module_free(rpn->rpn_bbox);
    free(rpn);
}

static Module* create_rpn(int fpn_ch, int num_anchors, DType dtype, DeviceType device) {
    RPN* rpn = malloc(sizeof(RPN));
    if (!rpn) return NULL;

    if (module_init((Module*)rpn, "RPN", rpn_forward, rpn_free) != 0) {
        free(rpn);
        return NULL;
    }

    rpn->rpn_conv = (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 1, 1, 1, true, dtype, device);
    rpn->rpn_cls = (Module*)nn_conv2d(fpn_ch, 2 * num_anchors, 1, 1, 0, 1, true, dtype, device);
    rpn->rpn_bbox = (Module*)nn_conv2d(fpn_ch, 4 * num_anchors, 1, 1, 0, 1, true, dtype, device);

    return (Module*)rpn;
}

typedef struct {
    Module base;
    Module* fc1;
    Module* fc2;
    Module* cls_score;
    Module* bbox_pred;
    int roi_output_size;
} ROIHead;

static Tensor* roi_head_forward(Module* module, Tensor* input) {
    ROIHead* head = (ROIHead*)module;
    if (!head || !input) return NULL;

    Tensor* x = module_forward(head->fc1, input);
    if (!x) return NULL;
    x = f_relu(x);

    x = module_forward(head->fc2, x);
    if (!x) return NULL;
    x = f_relu(x);

    return module_forward(head->cls_score, x);
}

static void roi_head_free(Module* module) {
    ROIHead* head = (ROIHead*)module;
    if (!head) return;
    if (head->fc1) module_free(head->fc1);
    if (head->fc2) module_free(head->fc2);
    if (head->cls_score) module_free(head->cls_score);
    if (head->bbox_pred) module_free(head->bbox_pred);
    free(head);
}

static Module* create_roi_head(int fpn_ch, int roi_size, int num_classes,
                                DType dtype, DeviceType device) {
    ROIHead* head = malloc(sizeof(ROIHead));
    if (!head) return NULL;

    if (module_init((Module*)head, "ROIHead", roi_head_forward, roi_head_free) != 0) {
        free(head);
        return NULL;
    }

    int fc_in = fpn_ch * roi_size * roi_size;
    head->roi_output_size = roi_size;
    head->fc1 = (Module*)nn_linear(fc_in, 1024, dtype, device, true);
    head->fc2 = (Module*)nn_linear(1024, 1024, dtype, device, true);
    head->cls_score = (Module*)nn_linear(1024, num_classes, dtype, device, true);
    head->bbox_pred = (Module*)nn_linear(1024, num_classes * 4, dtype, device, true);

    return (Module*)head;
}

typedef struct {
    Module base;
    Sequential* conv_layers;
    Module* deconv;
    Module* mask_pred;
} MaskHead;

static Tensor* mask_head_forward(Module* module, Tensor* input) {
    MaskHead* head = (MaskHead*)module;
    if (!head || !input) return NULL;

    Tensor* x = module_forward((Module*)head->conv_layers, input);
    if (!x) return NULL;

    x = module_forward(head->deconv, x);
    if (!x) return NULL;
    x = f_relu(x);

    return module_forward(head->mask_pred, x);
}

static void mask_head_free(Module* module) {
    MaskHead* head = (MaskHead*)module;
    if (!head) return;
    if (head->conv_layers) module_free((Module*)head->conv_layers);
    if (head->deconv) module_free(head->deconv);
    if (head->mask_pred) module_free(head->mask_pred);
    free(head);
}

static Module* create_mask_head(int fpn_ch, int num_classes, DType dtype, DeviceType device) {
    MaskHead* head = malloc(sizeof(MaskHead));
    if (!head) return NULL;

    if (module_init((Module*)head, "MaskHead", mask_head_forward, mask_head_free) != 0) {
        free(head);
        return NULL;
    }

    head->conv_layers = nn_sequential();
    for (int i = 0; i < 4; i++) {
        sequential_add(head->conv_layers, (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 1, 1, 1, true, dtype, device));
        sequential_add(head->conv_layers, (Module*)nn_relu(false));
    }

    head->deconv = (Module*)nn_conv_transpose2d(fpn_ch, fpn_ch, 2, 2, 0, 0, true, dtype, device);
    head->mask_pred = (Module*)nn_conv2d(fpn_ch, num_classes, 1, 1, 0, 1, true, dtype, device);

    return (Module*)head;
}

typedef struct {
    Module base;
    Module* backbone;
    Module* fpn;
    Module* rpn;
    Module* roi_head;
    Module* mask_head;
    int num_classes;
} MaskRCNN;

static Tensor* mask_rcnn_forward(Module* module, Tensor* input) {
    MaskRCNN* net = (MaskRCNN*)module;
    if (!net || !input) return NULL;

    Tensor* features = module_forward(net->backbone, input);
    if (!features) return NULL;

    Tensor* rpn_out = module_forward(net->rpn, features);
    if (!rpn_out) return NULL;

    return module_forward(net->roi_head, rpn_out);
}

static void mask_rcnn_free(Module* module) {
    MaskRCNN* net = (MaskRCNN*)module;
    if (!net) return;
    if (net->backbone) module_free(net->backbone);
    if (net->fpn) module_free(net->fpn);
    if (net->rpn) module_free(net->rpn);
    if (net->roi_head) module_free(net->roi_head);
    if (net->mask_head) module_free(net->mask_head);
    free(net);
}

Module* cml_zoo_mask_rcnn_create(const MaskRCNNConfig* cfg, DType dtype, DeviceType device) {
    MaskRCNNConfig c = cfg ? *cfg : cml_zoo_mask_rcnn_default_config();
    if (c.num_classes <= 0) c.num_classes = 81;
    if (c.num_anchors <= 0) c.num_anchors = 9;
    if (c.fpn_channels <= 0) c.fpn_channels = 256;
    if (c.roi_output_size <= 0) c.roi_output_size = 7;
    if (c.mask_output_size <= 0) c.mask_output_size = 14;

    MaskRCNN* net = malloc(sizeof(MaskRCNN));
    if (!net) return NULL;

    if (module_init((Module*)net, "MaskRCNN", mask_rcnn_forward, mask_rcnn_free) != 0) {
        free(net);
        return NULL;
    }

    net->num_classes = c.num_classes;
    net->backbone = create_backbone(dtype, device);
    net->fpn = create_fpn(c.fpn_channels, dtype, device);
    net->rpn = create_rpn(c.fpn_channels, c.num_anchors, dtype, device);
    net->roi_head = create_roi_head(c.fpn_channels, c.roi_output_size, c.num_classes, dtype, device);
    net->mask_head = create_mask_head(c.fpn_channels, c.num_classes, dtype, device);

    LOG_INFO("Created Mask R-CNN (%d classes, %d anchors, FPN %d, ROI %dx%d, mask %dx%d)",
             c.num_classes, c.num_anchors, c.fpn_channels,
             c.roi_output_size, c.roi_output_size,
             c.mask_output_size, c.mask_output_size);
    return (Module*)net;
}
