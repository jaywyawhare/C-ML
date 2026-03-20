#include "zoo/retinanet.h"
#include "zoo/resnet.h"
#include "nn/layers.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_manipulation.h"
#include "core/logging.h"
#include <stdlib.h>

RetinaNetConfig cml_zoo_retinanet_default_config(void) {
    RetinaNetConfig cfg = {
        .num_classes = 80,
        .num_anchors = 9,
        .fpn_channels = 256
    };
    return cfg;
}

typedef struct {
    Module base;
    Sequential* backbone_stem;
    Sequential* layer2;
    Sequential* layer3;
    Sequential* layer4;
} ResNet50Backbone;

static Tensor* backbone_forward(Module* module, Tensor* input) {
    ResNet50Backbone* bb = (ResNet50Backbone*)module;
    if (!bb || !input)
        return NULL;

    Tensor* x = module_forward((Module*)bb->backbone_stem, input);
    if (!x) return NULL;
    x = module_forward((Module*)bb->layer2, x);
    if (!x) return NULL;
    x = module_forward((Module*)bb->layer3, x);
    if (!x) return NULL;
    return module_forward((Module*)bb->layer4, x);
}

static void backbone_free(Module* module) {
    ResNet50Backbone* bb = (ResNet50Backbone*)module;
    if (!bb) return;
    if (bb->backbone_stem) module_free((Module*)bb->backbone_stem);
    if (bb->layer2) module_free((Module*)bb->layer2);
    if (bb->layer3) module_free((Module*)bb->layer3);
    if (bb->layer4) module_free((Module*)bb->layer4);
    free(bb);
}

typedef struct {
    Module base;
    Module* lateral3;
    Module* lateral4;
    Module* lateral5;
    Module* smooth3;
    Module* smooth4;
    Module* smooth5;
    Module* extra_p6;
    Module* extra_p7;
} FPN;

static Tensor* fpn_forward(Module* module, Tensor* input) {
    (void)module;
    return input;
}

static void fpn_free(Module* module) {
    FPN* fpn = (FPN*)module;
    if (!fpn) return;
    if (fpn->lateral3) module_free(fpn->lateral3);
    if (fpn->lateral4) module_free(fpn->lateral4);
    if (fpn->lateral5) module_free(fpn->lateral5);
    if (fpn->smooth3) module_free(fpn->smooth3);
    if (fpn->smooth4) module_free(fpn->smooth4);
    if (fpn->smooth5) module_free(fpn->smooth5);
    if (fpn->extra_p6) module_free(fpn->extra_p6);
    if (fpn->extra_p7) module_free(fpn->extra_p7);
    free(fpn);
}

static Module* create_fpn(int fpn_ch, DType dtype, DeviceType device) {
    FPN* fpn = malloc(sizeof(FPN));
    if (!fpn) return NULL;

    if (module_init((Module*)fpn, "FPN", fpn_forward, fpn_free) != 0) {
        free(fpn);
        return NULL;
    }

    fpn->lateral3 = (Module*)nn_conv2d(512, fpn_ch, 1, 1, 0, 1, true, dtype, device);
    fpn->lateral4 = (Module*)nn_conv2d(1024, fpn_ch, 1, 1, 0, 1, true, dtype, device);
    fpn->lateral5 = (Module*)nn_conv2d(2048, fpn_ch, 1, 1, 0, 1, true, dtype, device);

    fpn->smooth3 = (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 1, 1, 1, true, dtype, device);
    fpn->smooth4 = (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 1, 1, 1, true, dtype, device);
    fpn->smooth5 = (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 1, 1, 1, true, dtype, device);

    fpn->extra_p6 = (Module*)nn_conv2d(2048, fpn_ch, 3, 2, 1, 1, true, dtype, device);
    fpn->extra_p7 = (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 2, 1, 1, true, dtype, device);

    return (Module*)fpn;
}

static Module* create_subnet(int fpn_ch, int num_convs, int out_ch, DType dtype, DeviceType device) {
    Sequential* net = nn_sequential();
    if (!net) return NULL;

    for (int i = 0; i < num_convs; i++) {
        sequential_add(net, (Module*)nn_conv2d(fpn_ch, fpn_ch, 3, 1, 1, 1, true, dtype, device));
        sequential_add(net, (Module*)nn_relu(false));
    }

    sequential_add(net, (Module*)nn_conv2d(fpn_ch, out_ch, 3, 1, 1, 1, true, dtype, device));
    return (Module*)net;
}

typedef struct {
    Module base;
    Module* backbone;
    Module* fpn;
    Module* cls_subnet;
    Module* box_subnet;
    int num_classes;
    int num_anchors;
} RetinaNet;

static Tensor* retinanet_forward(Module* module, Tensor* input) {
    RetinaNet* net = (RetinaNet*)module;
    if (!net || !input)
        return NULL;

    Tensor* features = module_forward(net->backbone, input);
    if (!features)
        return NULL;

    Tensor* cls_out = module_forward(net->cls_subnet, features);
    if (!cls_out)
        return NULL;

    return cls_out;
}

static void retinanet_free(Module* module) {
    RetinaNet* net = (RetinaNet*)module;
    if (!net) return;
    if (net->backbone) module_free(net->backbone);
    if (net->fpn) module_free(net->fpn);
    if (net->cls_subnet) module_free(net->cls_subnet);
    if (net->box_subnet) module_free(net->box_subnet);
    free(net);
}

Module* cml_zoo_retinanet_create(const RetinaNetConfig* cfg, DType dtype, DeviceType device) {
    RetinaNetConfig c = cfg ? *cfg : cml_zoo_retinanet_default_config();
    if (c.num_classes <= 0) c.num_classes = 80;
    if (c.num_anchors <= 0) c.num_anchors = 9;
    if (c.fpn_channels <= 0) c.fpn_channels = 256;

    RetinaNet* net = malloc(sizeof(RetinaNet));
    if (!net) return NULL;

    if (module_init((Module*)net, "RetinaNet", retinanet_forward, retinanet_free) != 0) {
        free(net);
        return NULL;
    }

    net->num_classes = c.num_classes;
    net->num_anchors = c.num_anchors;

    ResNet50Backbone* bb = malloc(sizeof(ResNet50Backbone));
    if (!bb) { free(net); return NULL; }
    if (module_init((Module*)bb, "ResNet50Backbone", backbone_forward, backbone_free) != 0) {
        free(bb); free(net); return NULL;
    }

    bb->backbone_stem = nn_sequential();
    sequential_add(bb->backbone_stem, (Module*)nn_conv2d(3, 64, 7, 2, 3, 1, false, dtype, device));
    sequential_add(bb->backbone_stem, (Module*)nn_batchnorm2d(64, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(bb->backbone_stem, (Module*)nn_relu(false));
    sequential_add(bb->backbone_stem, (Module*)nn_maxpool2d(3, 2, 1, 1, false));

    bb->layer2 = nn_sequential();
    for (int i = 0; i < 3; i++) {
        int in_ch = (i == 0) ? 64 : 256;
        Sequential* blk = nn_sequential();
        sequential_add(blk, (Module*)nn_conv2d(in_ch, 64, 1, 1, 0, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(64, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(blk, (Module*)nn_conv2d(64, 64, 3, 1, 1, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(64, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(blk, (Module*)nn_conv2d(64, 256, 1, 1, 0, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(256, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(bb->layer2, (Module*)blk);
    }

    bb->layer3 = nn_sequential();
    for (int i = 0; i < 4; i++) {
        int in_ch = (i == 0) ? 256 : 512;
        int stride = (i == 0) ? 2 : 1;
        Sequential* blk = nn_sequential();
        sequential_add(blk, (Module*)nn_conv2d(in_ch, 128, 1, 1, 0, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(128, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(blk, (Module*)nn_conv2d(128, 128, 3, stride, 1, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(128, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(blk, (Module*)nn_conv2d(128, 512, 1, 1, 0, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(512, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(bb->layer3, (Module*)blk);
    }

    bb->layer4 = nn_sequential();
    for (int i = 0; i < 6; i++) {
        int in_ch = (i == 0) ? 512 : 1024;
        int stride = (i == 0) ? 2 : 1;
        Sequential* blk = nn_sequential();
        sequential_add(blk, (Module*)nn_conv2d(in_ch, 256, 1, 1, 0, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(256, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(blk, (Module*)nn_conv2d(256, 256, 3, stride, 1, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(256, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(blk, (Module*)nn_conv2d(256, 1024, 1, 1, 0, 1, false, dtype, device));
        sequential_add(blk, (Module*)nn_batchnorm2d(1024, 1e-5f, 0.1f, true, true, dtype, device));
        sequential_add(blk, (Module*)nn_relu(false));
        sequential_add(bb->layer4, (Module*)blk);
    }

    net->backbone = (Module*)bb;
    net->fpn = create_fpn(c.fpn_channels, dtype, device);
    net->cls_subnet = create_subnet(c.fpn_channels, 4, c.num_classes * c.num_anchors, dtype, device);
    net->box_subnet = create_subnet(c.fpn_channels, 4, 4 * c.num_anchors, dtype, device);

    LOG_INFO("Created RetinaNet (%d classes, %d anchors, FPN %d)",
             c.num_classes, c.num_anchors, c.fpn_channels);
    return (Module*)net;
}
