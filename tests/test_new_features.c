#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "nn/layers.h"
#include "optim.h"
#include "nn.h"
#include "core/gguf.h"
#include "core/safetensors.h"
#include "core/serialization.h"

static int tests_passed = 0;
static int tests_total  = 0;

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

#define TEST(name)                                     \
    do {                                               \
        tests_total++;                                 \
        printf("  TEST: %s ... ", #name);              \
        if (test_##name()) {                           \
            tests_passed++;                            \
            printf("PASSED\n");                        \
        } else {                                       \
            printf("FAILED\n");                        \
        }                                              \
    } while (0)

#define APPROX(a, b) (fabsf((a) - (b)) < 1e-4f)

static int test_unfold_1d(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int shape[] = {5};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_unfold(t, 3, 1);
    tensor_ensure_executed(out);
    // [5] with kernel=3, stride=1 -> [3, 3] (3 windows of size 3)
    if (out->ndim != 2) return 0;
    if (out->shape[0] != 3 || out->shape[1] != 3) return 0;
    float* d = out->data;
    // Window 0: [1,2,3], Window 1: [2,3,4], Window 2: [3,4,5]
    if (!APPROX(d[0], 1.0f)) return 0;
    if (!APPROX(d[1], 2.0f)) return 0;
    if (!APPROX(d[2], 3.0f)) return 0;
    if (!APPROX(d[3], 2.0f)) return 0;
    if (!APPROX(d[4], 3.0f)) return 0;
    if (!APPROX(d[5], 4.0f)) return 0;
    if (!APPROX(d[6], 3.0f)) return 0;
    if (!APPROX(d[7], 4.0f)) return 0;
    if (!APPROX(d[8], 5.0f)) return 0;
    tensor_free(t);
    tensor_free(out);
    return 1;
}

static int test_unfold_stride(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int shape[] = {6};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_unfold(t, 2, 2);
    tensor_ensure_executed(out);
    // [6] with kernel=2, stride=2 -> [3, 2]
    if (out->ndim != 2) return 0;
    if (out->shape[0] != 3 || out->shape[1] != 2) return 0;
    float* d = out->data;
    if (!APPROX(d[0], 1.0f)) return 0;
    if (!APPROX(d[1], 2.0f)) return 0;
    if (!APPROX(d[2], 3.0f)) return 0;
    if (!APPROX(d[3], 4.0f)) return 0;
    if (!APPROX(d[4], 5.0f)) return 0;
    if (!APPROX(d[5], 6.0f)) return 0;
    tensor_free(t);
    tensor_free(out);
    return 1;
}

static int test_var_mean(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int shape[] = {2, 3};
    Tensor* t = tensor_from_data(data, shape, 2, &cpu_f32);
    Tensor *out_var, *out_mean;
    int dims[] = {1};
    ReduceParams params = {.dims = dims, .num_dims = 1, .keepdim = false};
    uop_var_mean(t, &params, &out_var, &out_mean);
    tensor_ensure_executed(out_var);
    tensor_ensure_executed(out_mean);
    // Row 0: mean=2, biased var=2/3; Row 1: mean=5, biased var=2/3
    float* mv = out_mean->data;
    float* vv = out_var->data;
    if (!APPROX(mv[0], 2.0f)) return 0;
    if (!APPROX(mv[1], 5.0f)) return 0;
    if (!APPROX(vv[0], 2.0f/3.0f)) return 0;
    if (!APPROX(vv[1], 2.0f/3.0f)) return 0;
    tensor_free(t);
    tensor_free(out_var);
    tensor_free(out_mean);
    return 1;
}

static int test_std_mean(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int shape[] = {2, 3};
    Tensor* t = tensor_from_data(data, shape, 2, &cpu_f32);
    Tensor *out_std, *out_mean;
    int dims[] = {1};
    ReduceParams params = {.dims = dims, .num_dims = 1, .keepdim = false};
    uop_std_mean(t, &params, &out_std, &out_mean);
    tensor_ensure_executed(out_std);
    tensor_ensure_executed(out_mean);
    float* ms = out_mean->data;
    float* ss = out_std->data;
    if (!APPROX(ms[0], 2.0f)) return 0;
    if (!APPROX(ms[1], 5.0f)) return 0;
    // biased std = sqrt(2/3) ≈ 0.8165
    if (!APPROX(ss[0], sqrtf(2.0f/3.0f))) return 0;
    if (!APPROX(ss[1], sqrtf(2.0f/3.0f))) return 0;
    tensor_free(t);
    tensor_free(out_std);
    tensor_free(out_mean);
    return 1;
}

static int test_conv_transpose1d(void) {
    ConvTranspose1d* layer = nn_conv_transpose1d(1, 1, 3, 1, 0, 0, false, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    // Set kernel weights to all 1.0
    Tensor* w = layer->weight->tensor;
    tensor_ensure_executed(w);
    for (size_t i = 0; i < w->numel; i++)
        tensor_set_float(w, i, 1.0f);
    // Input: [1, 1, 3] (batch=1, channels=1, length=3)
    float input_data[] = {1.0f, 2.0f, 3.0f};
    int input_shape[] = {1, 1, 3};
    Tensor* input = tensor_from_data(input_data, input_shape, 3, &cpu_f32);
    Tensor* output = module_forward((Module*)layer, input);
    if (!output) { tensor_free(input); module_free((Module*)layer); return 0; }
    tensor_ensure_executed(output);
    // ConvTranspose1d with kernel=3, stride=1, no padding: output length = 3 + 3 - 1 = 5
    if (output->shape[2] != 5) { tensor_free(input); tensor_free(output); module_free((Module*)layer); return 0; }
    tensor_free(input);
    tensor_free(output);
    module_free((Module*)layer);
    return 1;
}

static int test_batchnorm3d(void) {
    BatchNorm3d* layer = nn_batchnorm3d(2, 1e-5f, 0.1f, true, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    module_set_training((Module*)layer, true);
    // Input: [1, 2, 2, 2, 2] (N=1, C=2, D=2, H=2, W=2)
    int input_shape[] = {1, 2, 2, 2, 2};
    Tensor* input = tensor_ones(input_shape, 5, &cpu_f32);
    // Set channel 1 to 2.0
    for (size_t i = 8; i < 16; i++)
        tensor_set_float(input, i, 2.0f);
    Tensor* output = module_forward((Module*)layer, input);
    if (!output) { tensor_free(input); module_free((Module*)layer); return 0; }
    tensor_ensure_executed(output);
    // After batchnorm with all-same values per channel, output should be ~0 (normalized)
    float v = tensor_get_float(output, 0);
    if (fabsf(v) > 0.1f) { tensor_free(input); tensor_free(output); module_free((Module*)layer); return 0; }
    tensor_free(input);
    tensor_free(output);
    module_free((Module*)layer);
    return 1;
}

static int test_layernorm2d(void) {
    LayerNorm2d* layer = nn_layernorm2d(3, 1e-5f, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    // Input: [1, 3, 2, 2] (N=1, C=3, H=2, W=2)
    int input_shape[] = {1, 3, 2, 2};
    Tensor* input = tensor_ones(input_shape, 4, &cpu_f32);
    Tensor* output = module_forward((Module*)layer, input);
    if (!output) { tensor_free(input); module_free((Module*)layer); return 0; }
    tensor_ensure_executed(output);
    // All same values -> normalized to 0 (with affine weight=1, bias=0)
    float v = tensor_get_float(output, 0);
    if (fabsf(v) > 0.1f) { tensor_free(input); tensor_free(output); module_free((Module*)layer); return 0; }
    tensor_free(input);
    tensor_free(output);
    module_free((Module*)layer);
    return 1;
}

static int test_dtype_int16(void) {
    TensorConfig cfg = {.dtype = DTYPE_INT16, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int shape[] = {4};
    Tensor* t = tensor_ones(shape, 1, &cfg);
    if (!t) return 0;
    if (cml_dtype_size(DTYPE_INT16) != 2) return 0;
    tensor_set_float(t, 0, 42.0f);
    float v = tensor_get_float(t, 0);
    if (!APPROX(v, 42.0f)) return 0;
    float v1 = tensor_get_float(t, 1);
    if (!APPROX(v1, 1.0f)) return 0;
    tensor_free(t);
    return 1;
}

static int test_dtype_uint16(void) {
    TensorConfig cfg = {.dtype = DTYPE_UINT16, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int shape[] = {4};
    Tensor* t = tensor_full(shape, 1, &cfg, 100.0f);
    if (!t) return 0;
    float v = tensor_get_float(t, 0);
    if (!APPROX(v, 100.0f)) return 0;
    tensor_free(t);
    return 1;
}

static int test_dtype_uint32(void) {
    TensorConfig cfg = {.dtype = DTYPE_UINT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int shape[] = {3};
    Tensor* t = tensor_ones(shape, 1, &cfg);
    if (!t) return 0;
    if (cml_dtype_size(DTYPE_UINT32) != 4) return 0;
    tensor_set_float(t, 0, 999.0f);
    float v = tensor_get_float(t, 0);
    if (!APPROX(v, 999.0f)) return 0;
    tensor_free(t);
    return 1;
}

static int test_dtype_uint64(void) {
    TensorConfig cfg = {.dtype = DTYPE_UINT64, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int shape[] = {2};
    Tensor* t = tensor_full(shape, 1, &cfg, 12345.0f);
    if (!t) return 0;
    if (cml_dtype_size(DTYPE_UINT64) != 8) return 0;
    float v = tensor_get_float(t, 0);
    if (!APPROX(v, 12345.0f)) return 0;
    tensor_free(t);
    return 1;
}

static int test_muon_optimizer(void) {
    // Create a simple parameter
    int shape[] = {2, 3};
    Tensor* weight = tensor_ones(shape, 2, &cpu_f32);
    weight->requires_grad = true;
    weight->grad = tensor_ones(shape, 2, &cpu_f32);

    Parameter* param = (Parameter*)malloc(sizeof(Parameter));
    if (!param) { tensor_free(weight); return 0; }
    param->tensor = weight;
    param->name = strdup("test_weight");
    param->requires_grad = true;

    Parameter* params[] = {param};
    Optimizer* opt = optim_muon(params, 1, 0.02f, 0.95f, 0.0f, true);
    if (!opt) { free(param); tensor_free(weight); return 0; }

    // Step should not crash
    optimizer_step(opt);

    // Check weight was updated (should differ from all-ones now)
    tensor_ensure_executed(weight);
    float v = tensor_get_float(weight, 0);
    int changed = !APPROX(v, 1.0f);

    // Cleanup: optimizer_free frees state but not the parameters themselves
    optimizer_free(opt);
    // Free grad first, then weight (grad is a separate tensor)
    if (weight->grad) {
        Tensor* g = weight->grad;
        weight->grad = NULL;
        tensor_free(g);
    }
    tensor_free(weight);
    free(param->name);
    free(param);
    return changed;
}

static int test_tensor_cast(void) {
    float data[] = {1.5f, 2.7f, 3.1f, 4.9f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* t_int = tensor_cast(t, DTYPE_INT32);
    if (!t_int) { tensor_free(t); return 0; }
    if (t_int->dtype != DTYPE_INT32) { tensor_free(t); tensor_free(t_int); return 0; }
    float v = tensor_get_float(t_int, 0);
    if (!APPROX(v, 1.0f)) { tensor_free(t); tensor_free(t_int); return 0; }
    v = tensor_get_float(t_int, 3);
    if (!APPROX(v, 4.0f)) { tensor_free(t); tensor_free(t_int); return 0; }
    tensor_free(t);
    tensor_free(t_int);
    return 1;
}

static int test_tensor_contiguous(void) {
    int shape[] = {2, 3};
    Tensor* t = tensor_ones(shape, 2, &cpu_f32);
    if (!t) return 0;
    Tensor* c = tensor_contiguous(t);
    if (!c) { tensor_free(t); return 0; }
    if (!c->is_contiguous) { tensor_free(t); tensor_free(c); return 0; }
    float v = tensor_get_float(c, 0);
    if (!APPROX(v, 1.0f)) { tensor_free(t); tensor_free(c); return 0; }
    tensor_free(t);
    tensor_free(c);
    return 1;
}

static int test_tensor_from_blob(void) {
    float data[] = {10.0f, 20.0f, 30.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_blob(data, shape, 1, &cpu_f32);
    if (!t) return 0;
    if (t->owns_data) { tensor_free(t); return 0; }
    float v = tensor_get_float(t, 1);
    if (!APPROX(v, 20.0f)) { tensor_free(t); return 0; }
    tensor_free(t);
    // data should still be valid since tensor doesn't own it
    if (!APPROX(data[2], 30.0f)) return 0;
    return 1;
}

static int test_randperm(void) {
    Tensor* t = tensor_randperm(5, &cpu_f32);
    if (!t) return 0;
    if (t->numel != 5) { tensor_free(t); return 0; }
    // Check all values 0-4 are present
    int found[5] = {0};
    for (int i = 0; i < 5; i++) {
        int v = (int)tensor_get_float(t, i);
        if (v < 0 || v >= 5) { tensor_free(t); return 0; }
        found[v] = 1;
    }
    for (int i = 0; i < 5; i++) {
        if (!found[i]) { tensor_free(t); return 0; }
    }
    tensor_free(t);
    return 1;
}

static int test_tensor_dot(void) {
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f, 6.0f};
    int shape[] = {3};
    Tensor* a = tensor_from_data(a_data, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(b_data, shape, 1, &cpu_f32);
    Tensor* result = tensor_dot(a, b);
    if (!result) { tensor_free(a); tensor_free(b); return 0; }
    tensor_ensure_executed(result);
    float v = tensor_get_float(result, 0);
    // 1*4 + 2*5 + 3*6 = 32
    if (!APPROX(v, 32.0f)) { tensor_free(a); tensor_free(b); tensor_free(result); return 0; }
    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_interpolate_nearest(void) {
    // [1, 1, 2, 2] -> [1, 1, 4, 4]
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {1, 1, 2, 2};
    Tensor* t = tensor_from_data(data, shape, 4, &cpu_f32);
    int out_size[] = {4, 4};
    Tensor* out = tensor_interpolate(t, out_size, 2, INTERP_NEAREST);
    if (!out) { tensor_free(t); return 0; }
    if (out->shape[2] != 4 || out->shape[3] != 4) { tensor_free(t); tensor_free(out); return 0; }
    // Top-left quadrant should be 1.0
    float v = tensor_get_float(out, 0);
    if (!APPROX(v, 1.0f)) { tensor_free(t); tensor_free(out); return 0; }
    tensor_free(t);
    tensor_free(out);
    return 1;
}

static int test_maxpool1d(void) {
    MaxPool1d* pool = nn_maxpool1d(3, 1, 0, 1, false);
    if (!pool) return 0;
    float data[] = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};
    int shape[] = {1, 1, 5};
    Tensor* input = tensor_from_data(data, shape, 3, &cpu_f32);
    Tensor* output = module_forward((Module*)pool, input);
    if (!output) { tensor_free(input); module_free((Module*)pool); return 0; }
    tensor_ensure_executed(output);
    // kernel=3, stride=1: output length = 5 - 3 + 1 = 3
    if (output->shape[2] != 3) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    float* o = output->data;
    // max([1,3,2])=3, max([3,2,5])=5, max([2,5,4])=5
    if (!APPROX(o[0], 3.0f)) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    if (!APPROX(o[1], 5.0f)) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    if (!APPROX(o[2], 5.0f)) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    tensor_free(input);
    tensor_free(output);
    module_free((Module*)pool);
    return 1;
}

static int test_avgpool1d(void) {
    AvgPool1d* pool = nn_avgpool1d(2, 2, 0, false, false);
    if (!pool) return 0;
    float data[] = {1.0f, 3.0f, 5.0f, 7.0f};
    int shape[] = {1, 1, 4};
    Tensor* input = tensor_from_data(data, shape, 3, &cpu_f32);
    Tensor* output = module_forward((Module*)pool, input);
    if (!output) { tensor_free(input); module_free((Module*)pool); return 0; }
    tensor_ensure_executed(output);
    // kernel=2, stride=2: [avg(1,3)=2, avg(5,7)=6]
    if (output->shape[2] != 2) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    float* o = output->data;
    if (!APPROX(o[0], 2.0f)) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    if (!APPROX(o[1], 6.0f)) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    tensor_free(input);
    tensor_free(output);
    module_free((Module*)pool);
    return 1;
}

static int test_adaptive_avgpool2d(void) {
    AdaptiveAvgPool2d* pool = nn_adaptive_avgpool2d(1, 1);
    if (!pool) return 0;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {1, 1, 2, 2};
    Tensor* input = tensor_from_data(data, shape, 4, &cpu_f32);
    Tensor* output = module_forward((Module*)pool, input);
    if (!output) { tensor_free(input); module_free((Module*)pool); return 0; }
    tensor_ensure_executed(output);
    // Global average: (1+2+3+4)/4 = 2.5
    if (output->shape[2] != 1 || output->shape[3] != 1) {
        tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0;
    }
    float v = tensor_get_float(output, 0);
    if (!APPROX(v, 2.5f)) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    tensor_free(input);
    tensor_free(output);
    module_free((Module*)pool);
    return 1;
}

static int test_adaptive_avgpool1d(void) {
    AdaptiveAvgPool1d* pool = nn_adaptive_avgpool1d(2);
    if (!pool) return 0;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {1, 1, 4};
    Tensor* input = tensor_from_data(data, shape, 3, &cpu_f32);
    Tensor* output = module_forward((Module*)pool, input);
    if (!output) { tensor_free(input); module_free((Module*)pool); return 0; }
    tensor_ensure_executed(output);
    // Adaptive to size 2: avg(1,2)=1.5, avg(3,4)=3.5
    if (output->shape[2] != 2) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    float* o = output->data;
    if (!APPROX(o[0], 1.5f)) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    if (!APPROX(o[1], 3.5f)) { tensor_free(input); tensor_free(output); module_free((Module*)pool); return 0; }
    tensor_free(input);
    tensor_free(output);
    module_free((Module*)pool);
    return 1;
}

static int test_nadam_optimizer(void) {
    int shape[] = {2, 3};
    Tensor* weight = tensor_ones(shape, 2, &cpu_f32);
    weight->requires_grad = true;
    weight->grad = tensor_ones(shape, 2, &cpu_f32);

    Parameter* param = (Parameter*)malloc(sizeof(Parameter));
    if (!param) { tensor_free(weight); return 0; }
    param->tensor = weight;
    param->name = strdup("w");
    param->requires_grad = true;

    Parameter* params[] = {param};
    Optimizer* opt = optim_nadam(params, 1, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!opt) { free(param->name); free(param); tensor_free(weight); return 0; }

    optimizer_step(opt);
    tensor_ensure_executed(weight);
    float v = tensor_get_float(weight, 0);
    int changed = !APPROX(v, 1.0f);

    optimizer_free(opt);
    if (weight->grad) { Tensor* g = weight->grad; weight->grad = NULL; tensor_free(g); }
    tensor_free(weight);
    free(param->name);
    free(param);
    return changed;
}

static int test_adamax_optimizer(void) {
    int shape[] = {2, 3};
    Tensor* weight = tensor_ones(shape, 2, &cpu_f32);
    weight->requires_grad = true;
    weight->grad = tensor_ones(shape, 2, &cpu_f32);

    Parameter* param = (Parameter*)malloc(sizeof(Parameter));
    if (!param) { tensor_free(weight); return 0; }
    param->tensor = weight;
    param->name = strdup("w");
    param->requires_grad = true;

    Parameter* params[] = {param};
    Optimizer* opt = optim_adamax(params, 1, 0.002f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!opt) { free(param->name); free(param); tensor_free(weight); return 0; }

    optimizer_step(opt);
    tensor_ensure_executed(weight);
    float v = tensor_get_float(weight, 0);
    int changed = !APPROX(v, 1.0f);

    optimizer_free(opt);
    if (weight->grad) { Tensor* g = weight->grad; weight->grad = NULL; tensor_free(g); }
    tensor_free(weight);
    free(param->name);
    free(param);
    return changed;
}

static int test_maxpool3d(void) {
    // Input: [1, 1, 4, 4, 4] — fill with sequential values
    int shape[] = {1, 1, 4, 4, 4};
    Tensor* input = tensor_empty(shape, 5, &cpu_f32);
    tensor_ensure_executed(input);
    for (int i = 0; i < 64; i++) tensor_set_float(input, i, (float)i);

    MaxPool3d* pool = nn_maxpool3d(2, 2, 0, 1, false);
    Tensor* out = module_forward((Module*)pool, input);
    if (!out) { module_free((Module*)pool); tensor_free(input); return 0; }
    tensor_ensure_executed(out);

    // Output: [1, 1, 2, 2, 2]
    int ok = (out->ndim == 5 && out->shape[2] == 2 && out->shape[3] == 2 && out->shape[4] == 2);
    // max of first 2x2x2 block (indices 0,1,4,5,16,17,20,21) = 21
    if (ok) {
        float v = tensor_get_float(out, 0);
        ok = (v == 21.0f);
    }

    tensor_free(out);
    module_free((Module*)pool);
    tensor_free(input);
    return ok;
}

static int test_avgpool3d(void) {
    int shape[] = {1, 1, 2, 2, 2};
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor* input = tensor_from_data(data, shape, 5, &cpu_f32);
    tensor_ensure_executed(input);

    AvgPool3d* pool = nn_avgpool3d(2, 2, 0, false, true);
    Tensor* out = module_forward((Module*)pool, input);
    if (!out) { module_free((Module*)pool); tensor_free(input); return 0; }
    tensor_ensure_executed(out);

    // Global avg of [1..8] = 4.5
    int ok = (out->shape[2] == 1 && out->shape[3] == 1 && out->shape[4] == 1);
    if (ok) ok = fabsf(tensor_get_float(out, 0) - 4.5f) < 0.01f;

    tensor_free(out);
    module_free((Module*)pool);
    tensor_free(input);
    return ok;
}

static int test_adaptive_maxpool2d(void) {
    int shape[] = {1, 1, 4, 4};
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    Tensor* input = tensor_from_data(data, shape, 4, &cpu_f32);
    tensor_ensure_executed(input);

    AdaptiveMaxPool2d* pool = nn_adaptive_maxpool2d(2, 2);
    Tensor* out = module_forward((Module*)pool, input);
    if (!out) { module_free((Module*)pool); tensor_free(input); return 0; }
    tensor_ensure_executed(out);

    // 4x4 -> 2x2: windows are [0:2,0:2], [0:2,2:4], [2:4,0:2], [2:4,2:4]
    // max of {1,2,5,6}=6, {3,4,7,8}=8, {9,10,13,14}=14, {11,12,15,16}=16
    int ok = (out->shape[2] == 2 && out->shape[3] == 2);
    if (ok) ok = (tensor_get_float(out, 0) == 6.0f &&
                  tensor_get_float(out, 1) == 8.0f &&
                  tensor_get_float(out, 2) == 14.0f &&
                  tensor_get_float(out, 3) == 16.0f);

    tensor_free(out);
    module_free((Module*)pool);
    tensor_free(input);
    return ok;
}

static int test_adaptive_maxpool1d(void) {
    int shape[] = {1, 1, 4};
    float data[] = {3, 1, 4, 2};
    Tensor* input = tensor_from_data(data, shape, 3, &cpu_f32);
    tensor_ensure_executed(input);

    AdaptiveMaxPool1d* pool = nn_adaptive_maxpool1d(2);
    Tensor* out = module_forward((Module*)pool, input);
    if (!out) { module_free((Module*)pool); tensor_free(input); return 0; }
    tensor_ensure_executed(out);

    // 4 -> 2: windows [0:2]={3,1}->3, [2:4]={4,2}->4
    int ok = (out->shape[2] == 2);
    if (ok) ok = (tensor_get_float(out, 0) == 3.0f && tensor_get_float(out, 1) == 4.0f);

    tensor_free(out);
    module_free((Module*)pool);
    tensor_free(input);
    return ok;
}

static int test_scatter_reduce(void) {
    // self = [0, 0, 0, 0, 0], scatter_reduce with sum
    int shape[] = {5};
    float self_data[] = {0, 0, 0, 0, 0};
    Tensor* self = tensor_from_data(self_data, shape, 1, &cpu_f32);

    int idx_shape[] = {4};
    float idx_data[] = {0, 1, 0, 2};  // indices
    Tensor* index = tensor_from_data(idx_data, idx_shape, 1, &cpu_f32);

    float src_data[] = {1, 2, 3, 4};
    Tensor* src = tensor_from_data(src_data, idx_shape, 1, &cpu_f32);

    tensor_ensure_executed(self);
    tensor_ensure_executed(index);
    tensor_ensure_executed(src);

    Tensor* out = tensor_scatter_reduce(self, 0, index, src, SCATTER_REDUCE_SUM);
    if (!out) { tensor_free(self); tensor_free(index); tensor_free(src); return 0; }
    tensor_ensure_executed(out);

    // idx 0: 0 + 1 + 3 = 4, idx 1: 0 + 2 = 2, idx 2: 0 + 4 = 4
    int ok = (fabsf(tensor_get_float(out, 0) - 4.0f) < 0.01f &&
              fabsf(tensor_get_float(out, 1) - 2.0f) < 0.01f &&
              fabsf(tensor_get_float(out, 2) - 4.0f) < 0.01f);

    tensor_free(out);
    tensor_free(self);
    tensor_free(index);
    tensor_free(src);
    return ok;
}

static int test_bitcast(void) {
    // Create a float32 tensor and bitcast to int32 (same size)
    int shape[] = {2};
    float data[] = {1.0f, 2.0f};
    Tensor* a = tensor_from_data(data, shape, 1, &cpu_f32);
    tensor_ensure_executed(a);

    Tensor* out = tensor_bitcast(a, DTYPE_INT32);
    if (!out) { tensor_free(a); return 0; }
    tensor_ensure_executed(out);

    // Bitcast back should give original
    Tensor* back = tensor_bitcast(out, DTYPE_FLOAT32);
    if (!back) { tensor_free(out); tensor_free(a); return 0; }
    tensor_ensure_executed(back);

    int ok = (fabsf(tensor_get_float(back, 0) - 1.0f) < 0.001f &&
              fabsf(tensor_get_float(back, 1) - 2.0f) < 0.001f);

    tensor_free(back);
    tensor_free(out);
    tensor_free(a);
    return ok;
}

static int test_qr(void) {
    // 3x2 matrix: [[1,2],[3,4],[5,6]]
    int shape[] = {3, 2};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor* a = tensor_from_data(data, shape, 2, &cpu_f32);
    tensor_ensure_executed(a);

    QRResult qr = tensor_qr(a);
    if (!qr.Q || !qr.R) { tensor_free(a); return 0; }
    tensor_ensure_executed(qr.Q);
    tensor_ensure_executed(qr.R);

    // Q should be [3, 2], R should be [2, 2]
    int ok = (qr.Q->shape[0] == 3 && qr.Q->shape[1] == 2 &&
              qr.R->shape[0] == 2 && qr.R->shape[1] == 2);

    // Q^T * Q should be approximately identity
    if (ok) {
        float qtq00 = 0, qtq01 = 0, qtq11 = 0;
        for (int i = 0; i < 3; i++) {
            float q0 = tensor_get_float(qr.Q, i * 2 + 0);
            float q1 = tensor_get_float(qr.Q, i * 2 + 1);
            qtq00 += q0 * q0;
            qtq01 += q0 * q1;
            qtq11 += q1 * q1;
        }
        ok = (fabsf(qtq00 - 1.0f) < 0.01f && fabsf(qtq11 - 1.0f) < 0.01f && fabsf(qtq01) < 0.01f);
    }

    // R should be upper triangular (R[1][0] ≈ 0)
    if (ok) {
        float r10 = tensor_get_float(qr.R, 1 * 2 + 0);
        ok = (fabsf(r10) < 0.01f);
    }

    tensor_free(qr.Q);
    tensor_free(qr.R);
    tensor_free(a);
    return ok;
}

static int test_svd(void) {
    // 2x3 matrix
    int shape[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor* a = tensor_from_data(data, shape, 2, &cpu_f32);
    tensor_ensure_executed(a);

    SVDResult svd = tensor_svd(a);
    if (!svd.U || !svd.S || !svd.Vt) { tensor_free(a); return 0; }
    tensor_ensure_executed(svd.U);
    tensor_ensure_executed(svd.S);
    tensor_ensure_executed(svd.Vt);

    // U: [2, 2], S: [2], Vt: [2, 3]
    int ok = (svd.U->shape[0] == 2 && svd.U->shape[1] == 2 &&
              svd.S->shape[0] == 2 &&
              svd.Vt->shape[0] == 2 && svd.Vt->shape[1] == 3);

    // Singular values should be positive and descending
    if (ok) {
        float s0 = tensor_get_float(svd.S, 0);
        float s1 = tensor_get_float(svd.S, 1);
        ok = (s0 > s1 && s1 > 0.0f);
    }

    // Verify U * diag(S) * Vt ≈ A
    if (ok) {
        for (int i = 0; i < 2 && ok; i++) {
            for (int j = 0; j < 3 && ok; j++) {
                float val = 0;
                for (int k = 0; k < 2; k++) {
                    val += tensor_get_float(svd.U, i * 2 + k) *
                           tensor_get_float(svd.S, k) *
                           tensor_get_float(svd.Vt, k * 3 + j);
                }
                float expected = data[i * 3 + j];
                if (fabsf(val - expected) > 0.1f) ok = 0;
            }
        }
    }

    tensor_free(svd.U);
    tensor_free(svd.S);
    tensor_free(svd.Vt);
    tensor_free(a);
    return ok;
}

static int test_fp8_e4m3(void) {
    int shape[] = {3};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT8_E4M3, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_empty(shape, 1, &cfg);
    tensor_ensure_executed(t);

    tensor_set_float(t, 0, 1.0f);
    tensor_set_float(t, 1, 2.0f);
    tensor_set_float(t, 2, -1.0f);

    float v0 = tensor_get_float(t, 0);
    float v1 = tensor_get_float(t, 1);
    float v2 = tensor_get_float(t, 2);

    int ok = (fabsf(v0 - 1.0f) < 0.2f && fabsf(v1 - 2.0f) < 0.5f && v2 < 0.0f);
    // Check that dtype size is 1 byte
    ok = ok && (cml_dtype_size(DTYPE_FLOAT8_E4M3) == 1);

    tensor_free(t);
    return ok;
}

static int test_fp8_e5m2(void) {
    int shape[] = {3};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT8_E5M2, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_empty(shape, 1, &cfg);
    tensor_ensure_executed(t);

    tensor_set_float(t, 0, 1.0f);
    tensor_set_float(t, 1, 4.0f);
    tensor_set_float(t, 2, -0.5f);

    float v0 = tensor_get_float(t, 0);
    float v1 = tensor_get_float(t, 1);
    float v2 = tensor_get_float(t, 2);

    int ok = (fabsf(v0 - 1.0f) < 0.3f && fabsf(v1 - 4.0f) < 1.0f && v2 < 0.0f);
    ok = ok && (cml_dtype_size(DTYPE_FLOAT8_E5M2) == 1);

    tensor_free(t);
    return ok;
}

static int test_transformer_encoder(void) {
    TransformerEncoder* enc = nn_transformer_encoder(8, 2, 16, 0.0f, 2, DTYPE_FLOAT32, DEVICE_CPU);
    if (!enc) return 0;

    // Input: [1, 4, 8] (batch=1, seq=4, d_model=8)
    int shape[] = {1, 4, 8};
    Tensor* input = tensor_ones(shape, 3, &cpu_f32);
    tensor_ensure_executed(input);

    Tensor* out = module_forward((Module*)enc, input);
    int ok = (out != NULL);
    if (ok) {
        tensor_ensure_executed(out);
        ok = (out->ndim == 3 && out->shape[0] == 1 && out->shape[1] == 4 && out->shape[2] == 8);
        tensor_free(out);
    }

    module_free((Module*)enc);
    tensor_free(input);
    return ok;
}

static int test_transformer_decoder_layer(void) {
    TransformerDecoderLayer* layer = nn_transformer_decoder_layer(8, 2, 16, 0.0f, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;

    int tgt_shape[] = {1, 3, 8};
    int mem_shape[] = {1, 5, 8};
    Tensor* tgt = tensor_ones(tgt_shape, 3, &cpu_f32);
    Tensor* memory = tensor_ones(mem_shape, 3, &cpu_f32);
    tensor_ensure_executed(tgt);
    tensor_ensure_executed(memory);

    Tensor* out = transformer_decoder_layer_forward(layer, tgt, memory, NULL, NULL);
    int ok = (out != NULL);
    if (ok) {
        tensor_ensure_executed(out);
        ok = (out->ndim == 3 && out->shape[0] == 1 && out->shape[1] == 3 && out->shape[2] == 8);
        tensor_free(out);
    }

    module_free((Module*)layer);
    tensor_free(tgt);
    tensor_free(memory);
    return ok;
}

static int test_transformer_decoder(void) {
    TransformerDecoder* dec = nn_transformer_decoder(8, 2, 16, 0.0f, 2, DTYPE_FLOAT32, DEVICE_CPU);
    if (!dec) return 0;

    int shape[] = {1, 3, 8};
    Tensor* input = tensor_ones(shape, 3, &cpu_f32);
    tensor_ensure_executed(input);

    Tensor* out = module_forward((Module*)dec, input);
    int ok = (out != NULL);
    if (ok) {
        tensor_ensure_executed(out);
        ok = (out->ndim == 3 && out->shape[0] == 1 && out->shape[1] == 3 && out->shape[2] == 8);
        tensor_free(out);
    }

    module_free((Module*)dec);
    tensor_free(input);
    return ok;
}

static int test_gguf_serialization(void) {
    const char* path = "/tmp/test_cml.gguf";

    // Write
    {
        GGUFContext* ctx = gguf_open_write(path);
        if (!ctx) return 0;

        int shape[] = {2, 3};
        float data[] = {1, 2, 3, 4, 5, 6};
        Tensor* t = tensor_from_data(data, shape, 2, &cpu_f32);
        tensor_ensure_executed(t);

        gguf_write_tensor(ctx, "test_weight", t);
        gguf_close(ctx);
        tensor_free(t);
    }

    // Read
    {
        GGUFContext* ctx = gguf_open_read(path);
        if (!ctx) return 0;

        int ok = (gguf_get_num_tensors(ctx) == 1);
        if (ok) {
            const char* name = gguf_get_tensor_name(ctx, 0);
            ok = (name && strcmp(name, "test_weight") == 0);
        }

        Tensor* t = gguf_read_tensor(ctx, "test_weight");
        if (!t) { gguf_close(ctx); return 0; }
        tensor_ensure_executed(t);

        int ok2 = (t->shape[0] == 2 && t->shape[1] == 3);
        ok2 = ok2 && APPROX(tensor_get_float(t, 0), 1.0f);
        ok2 = ok2 && APPROX(tensor_get_float(t, 5), 6.0f);

        tensor_free(t);
        gguf_close(ctx);
        remove(path);
        return ok && ok2;
    }
}

static int test_safetensors_serialization(void) {
    const char* path = "/tmp/test_cml.safetensors";

    // Write
    {
        SafeTensorsContext* ctx = safetensors_open_write(path);
        if (!ctx) return 0;

        int shape[] = {3};
        float data[] = {10, 20, 30};
        Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
        tensor_ensure_executed(t);

        safetensors_write_tensor(ctx, "bias", t);
        safetensors_close(ctx);
        tensor_free(t);
    }

    // Read
    {
        SafeTensorsContext* ctx = safetensors_open_read(path);
        if (!ctx) return 0;

        int ok = (safetensors_get_num_tensors(ctx) == 1);

        Tensor* t = safetensors_read_tensor(ctx, "bias");
        if (!t) { safetensors_close(ctx); return 0; }
        tensor_ensure_executed(t);

        ok = ok && (t->shape[0] == 3);
        ok = ok && APPROX(tensor_get_float(t, 0), 10.0f);
        ok = ok && APPROX(tensor_get_float(t, 2), 30.0f);

        tensor_free(t);
        safetensors_close(ctx);
        remove(path);
        return ok;
    }
}

static int test_tensor_from_url_api(void) {
    // Test that the function exists and handles NULL gracefully
    Tensor* t = tensor_from_url(NULL);
    int ok = (t == NULL);  // Should return NULL for NULL input

    // Also test with a non-existent URL (should fail gracefully)
    t = tensor_from_url("http://localhost:99999/nonexistent.tensor");
    ok = ok && (t == NULL);

    return ok;
}

int main(void) {
    printf("=== New Features Tests ===\n");

    TEST(unfold_1d);
    TEST(unfold_stride);
    TEST(var_mean);
    TEST(std_mean);
    TEST(conv_transpose1d);
    TEST(batchnorm3d);
    TEST(layernorm2d);
    TEST(dtype_int16);
    TEST(dtype_uint16);
    TEST(dtype_uint32);
    TEST(dtype_uint64);
    TEST(muon_optimizer);
    TEST(tensor_cast);
    TEST(tensor_contiguous);
    TEST(tensor_from_blob);
    TEST(randperm);
    TEST(tensor_dot);
    TEST(interpolate_nearest);
    TEST(maxpool1d);
    TEST(avgpool1d);
    TEST(adaptive_avgpool2d);
    TEST(adaptive_avgpool1d);
    TEST(nadam_optimizer);
    TEST(adamax_optimizer);
    TEST(maxpool3d);
    TEST(avgpool3d);
    TEST(adaptive_maxpool2d);
    TEST(adaptive_maxpool1d);
    TEST(scatter_reduce);
    TEST(bitcast);
    TEST(qr);
    TEST(svd);
    TEST(fp8_e4m3);
    TEST(fp8_e5m2);
    TEST(transformer_encoder);
    TEST(transformer_decoder_layer);
    TEST(transformer_decoder);
    TEST(gguf_serialization);
    TEST(safetensors_serialization);
    TEST(tensor_from_url_api);

    printf("\n%d/%d tests passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
