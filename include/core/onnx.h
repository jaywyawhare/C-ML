#ifndef CML_CORE_ONNX_H
#define CML_CORE_ONNX_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_ONNX_MAX_INPUTS  32
#define CML_ONNX_MAX_OUTPUTS 32
#define CML_ONNX_MAX_NODES   512
#define CML_ONNX_MAX_ATTRS   16

typedef enum {
    CML_ONNX_ATTR_INT = 0,
    CML_ONNX_ATTR_FLOAT,
    CML_ONNX_ATTR_STRING,
    CML_ONNX_ATTR_INTS,
    CML_ONNX_ATTR_FLOATS,
    CML_ONNX_ATTR_TENSOR,
} CMLONNXAttrType;

typedef struct {
    char name[64];
    CMLONNXAttrType type;
    union {
        int64_t i;
        float f;
        struct { char data[256]; size_t len; } s;
        struct { int64_t* data; int count; } ints;
        struct { float* data; int count; } floats;
        Tensor* tensor;
    } value;
} CMLONNXAttribute;

typedef struct {
    char op_type[64];
    char* inputs[CML_ONNX_MAX_INPUTS];
    int num_inputs;
    char* outputs[CML_ONNX_MAX_OUTPUTS];
    int num_outputs;
    CMLONNXAttribute attrs[CML_ONNX_MAX_ATTRS];
    int num_attrs;
    char name[128];
} CMLONNXNode;

typedef struct {
    char name[128];
    int shape[8];
    int ndim;
    DType dtype;
} CMLONNXTensorInfo;

typedef struct {
    char name[128];
    Tensor* tensor;
} CMLONNXInitializer;

typedef struct {
    CMLONNXNode* nodes;
    int num_nodes;
    CMLONNXTensorInfo* inputs;
    int num_inputs;
    CMLONNXTensorInfo* outputs;
    int num_outputs;
    CMLONNXInitializer* initializers;
    int num_initializers;
    char name[128];
} CMLONNXGraph;

typedef struct {
    int64_t ir_version;
    int64_t opset_version;
    char producer_name[128];
    char domain[64];
    CMLONNXGraph graph;
} CMLONNXModel;

CMLONNXModel* cml_onnx_load(const char* filepath);
CMLONNXModel* cml_onnx_load_buffer(const uint8_t* data, size_t length);
void cml_onnx_free(CMLONNXModel* model);
bool cml_onnx_op_supported(const char* op_type);
int cml_onnx_run(CMLONNXModel* model, Tensor** inputs, int num_inputs,
                 Tensor** outputs, int num_outputs);
int cml_onnx_list_supported_ops(const char*** ops_out, int* count_out);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_ONNX_H */
