#!/bin/bash
# MLPerf Training v4.0 - ResNet-50 on ImageNet
# Target: 75.9% top-1 accuracy in minimum epochs
set -e
export CML_BACKEND=${CML_BACKEND:-cuda}
./build/bin/mlperf_resnet50 \
    --data-dir=${IMAGENET_DIR:-/data/imagenet} \
    --batch-size=${BS:-256} \
    --epochs=${EPOCHS:-90} \
    --lr=0.1 \
    --momentum=0.9 \
    --weight-decay=1e-4 \
    --target-accuracy=0.759
