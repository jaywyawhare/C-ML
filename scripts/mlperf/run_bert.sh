#!/bin/bash
# MLPerf Training v4.0 - BERT Pretraining
# Target: 0.72 masked LM accuracy
set -e
export CML_BACKEND=${CML_BACKEND:-cuda}
./build/bin/mlperf_resnet50 \
    --data-dir=${BERT_DATA_DIR:-/data/bert} \
    --batch-size=${BS:-64} \
    --epochs=${EPOCHS:-3} \
    --lr=1e-4 \
    --target-accuracy=0.72
