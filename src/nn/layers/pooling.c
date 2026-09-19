#include "nn/layers/pooling.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "alloc/cml_allocator.h"

static int pool2d_out_dim(int in_size, int kernel, int stride, int padding, int dilation,
                          bool ceil_mode) {
    int numer = in_size + 2 * padding - dilation * (kernel - 1) - 1;
    int out   = numer / stride + 1;
    if (ceil_mode && (numer % stride) != 0)
        out++;
    return out;
}

static Tensor* maxpool2d_forward(Module* module, Tensor* input) {
    MaxPool2d* pool = (MaxPool2d*)module;

    if (!pool || !input)
        return NULL;
    if (input->ndim != 4) {
        LOG_ERROR("MaxPool2d expects 4D input [batch, channels, height, width], got %dD",
                  input->ndim);
        return NULL;
    }

    int in_height = input->shape[2];
    int in_width  = input->shape[3];

    int kernel_h   = pool->kernel_size[0];
    int kernel_w   = pool->kernel_size[1];
    int stride_h   = pool->stride[0];
    int stride_w   = pool->stride[1];
    int padding_h  = pool->padding[0];
    int padding_w  = pool->padding[1];
    int dilation_h = pool->dilation[0];
    int dilation_w = pool->dilation[1];
    if (dilation_h != 1 || dilation_w != 1) {
        LOG_WARNING("MaxPool2d lazy path currently requires dilation=1");
        return NULL;
    }

    int out_height =
        pool2d_out_dim(in_height, kernel_h, stride_h, padding_h, dilation_h, pool->ceil_mode);
    int out_width =
        pool2d_out_dim(in_width, kernel_w, stride_w, padding_w, dilation_w, pool->ceil_mode);
    if (out_height <= 0 || out_width <= 0)
        return NULL;

    Pool2DParams params = {
        .kernel_size       = {kernel_h, kernel_w},
        .stride            = {stride_h, stride_w},
        .padding           = {padding_h, padding_w},
        .dilation          = {dilation_h, dilation_w},
        .ceil_mode         = pool->ceil_mode,
        .count_include_pad = false,
    };
    return uop_maxpool2d(input, &params);
}

static void maxpool2d_free(Module* module) { cml_free(module); }

MaxPool2d* nn_maxpool2d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode) {
    MaxPool2d* pool = cml_malloc(sizeof(MaxPool2d));
    if (!pool)
        return NULL;

    if (module_init((Module*)pool, "MaxPool2d", maxpool2d_forward, maxpool2d_free) != 0) {
        cml_free(pool);
        return NULL;
    }

    pool->kernel_size[0] = kernel_size;
    pool->kernel_size[1] = kernel_size;
    pool->stride[0]      = stride > 0 ? stride : kernel_size;
    pool->stride[1]      = stride > 0 ? stride : kernel_size;
    pool->padding[0]     = padding;
    pool->padding[1]     = padding;
    pool->dilation[0]    = dilation;
    pool->dilation[1]    = dilation;
    pool->ceil_mode      = ceil_mode;

    return pool;
}

static Tensor* avgpool2d_forward(Module* module, Tensor* input) {
    AvgPool2d* pool = (AvgPool2d*)module;

    if (!pool || !input)
        return NULL;
    if (input->ndim != 4) {
        LOG_ERROR("AvgPool2d expects 4D input [batch, channels, height, width], got %dD",
                  input->ndim);
        return NULL;
    }

    int in_height = input->shape[2];
    int in_width  = input->shape[3];

    int kernel_h   = pool->kernel_size[0];
    int kernel_w   = pool->kernel_size[1];
    int stride_h   = pool->stride[0];
    int stride_w   = pool->stride[1];
    int padding_h  = pool->padding[0];
    int padding_w  = pool->padding[1];
    int out_height = pool2d_out_dim(in_height, kernel_h, stride_h, padding_h, 1, pool->ceil_mode);
    int out_width  = pool2d_out_dim(in_width, kernel_w, stride_w, padding_w, 1, pool->ceil_mode);
    if (out_height <= 0 || out_width <= 0)
        return NULL;

    Pool2DParams params = {
        .kernel_size       = {kernel_h, kernel_w},
        .stride            = {stride_h, stride_w},
        .padding           = {padding_h, padding_w},
        .dilation          = {1, 1},
        .ceil_mode         = pool->ceil_mode,
        .count_include_pad = pool->count_include_pad,
    };
    return uop_avgpool2d(input, &params);
}

static void avgpool2d_free(Module* module) { cml_free(module); }

AvgPool2d* nn_avgpool2d(int kernel_size, int stride, int padding, bool ceil_mode,
                        bool count_include_pad) {
    AvgPool2d* pool = cml_malloc(sizeof(AvgPool2d));
    if (!pool)
        return NULL;

    if (module_init((Module*)pool, "AvgPool2d", avgpool2d_forward, avgpool2d_free) != 0) {
        cml_free(pool);
        return NULL;
    }

    pool->kernel_size[0]    = kernel_size;
    pool->kernel_size[1]    = kernel_size;
    pool->stride[0]         = stride > 0 ? stride : kernel_size;
    pool->stride[1]         = stride > 0 ? stride : kernel_size;
    pool->padding[0]        = padding;
    pool->padding[1]        = padding;
    pool->ceil_mode         = ceil_mode;
    pool->count_include_pad = count_include_pad;

    return pool;
}

/* Allocate a materialised output tensor of `shape`/`ndim` matching `input`'s
 * dtype and device, handing back its data pointer. */
static Tensor* pool_alloc_output(Tensor* input, const int* shape, int ndim, float** out_data) {
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty((int*)shape, ndim, &config);
    if (!output)
        return NULL;
    tensor_ensure_executed(output);
    *out_data = (float*)output->data;
    return output;
}

static Tensor* maxpool3d_forward(Module* module, Tensor* input) {
    MaxPool3d* pool = (MaxPool3d*)module;
    if (!input || input->ndim != 5)
        return NULL; // [N, C, D, H, W]

    int N = input->shape[0], C = input->shape[1];
    int D = input->shape[2], H = input->shape[3], W = input->shape[4];

    /* Lazy, composed from the tested 2D pool — no new UOP. Max is separable, so
     * a 3D max-pool == pool (H,W) then pool D. Each axis is placed in the 2D
     * pool's spatial slot via reshape (no permute needed); backward is automatic
     * from the reshape/pool grad rules. (dilation=1 path; eager fallback below.) */
    if (pool->dilation[0] == 1 && pool->dilation[1] == 1 && pool->dilation[2] == 1) {
        int s1[4]        = {N * C * D, 1, H, W};
        ReshapeParams r1 = {.new_shape = s1, .new_ndim = 4};
        Tensor* x1       = uop_reshape(input, &r1);
        if (!x1)
            return NULL;
        Pool2DParams pHW = {.kernel_size       = {pool->kernel_size[1], pool->kernel_size[2]},
                            .stride            = {pool->stride[1], pool->stride[2]},
                            .padding           = {pool->padding[1], pool->padding[2]},
                            .dilation          = {1, 1},
                            .ceil_mode         = pool->ceil_mode,
                            .count_include_pad = false};
        Tensor* y1       = uop_maxpool2d(x1, &pHW); /* [N*C*D, 1, oh, ow] */
        if (!y1 || y1->ndim != 4)
            return NULL;
        int oh = y1->shape[2], ow = y1->shape[3];

        int s2[4]        = {N * C, 1, D, oh * ow};
        ReshapeParams r2 = {.new_shape = s2, .new_ndim = 4};
        Tensor* x2       = uop_reshape(y1, &r2);
        if (!x2)
            return NULL;
        Pool2DParams pD = {.kernel_size       = {pool->kernel_size[0], 1},
                           .stride            = {pool->stride[0], 1},
                           .padding           = {pool->padding[0], 0},
                           .dilation          = {1, 1},
                           .ceil_mode         = pool->ceil_mode,
                           .count_include_pad = false};
        Tensor* y2      = uop_maxpool2d(x2, &pD); /* [N*C, 1, od, oh*ow] */
        if (!y2 || y2->ndim != 4)
            return NULL;
        int od = y2->shape[2];

        int s3[5]        = {N, C, od, oh, ow};
        ReshapeParams r3 = {.new_shape = s3, .new_ndim = 5};
        return uop_reshape(y2, &r3);
    }

    /* Eager fallback (dilation != 1). */
    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data)
        return NULL;

    int out_d = (D + 2 * pool->padding[0] - pool->dilation[0] * (pool->kernel_size[0] - 1) - 1) /
                    pool->stride[0] +
                1;
    int out_h = (H + 2 * pool->padding[1] - pool->dilation[1] * (pool->kernel_size[1] - 1) - 1) /
                    pool->stride[1] +
                1;
    int out_w = (W + 2 * pool->padding[2] - pool->dilation[2] * (pool->kernel_size[2] - 1) - 1) /
                    pool->stride[2] +
                1;

    if (pool->ceil_mode) {
        if ((D + 2 * pool->padding[0] - pool->dilation[0] * (pool->kernel_size[0] - 1) - 1) %
                pool->stride[0] !=
            0)
            out_d++;
        if ((H + 2 * pool->padding[1] - pool->dilation[1] * (pool->kernel_size[1] - 1) - 1) %
                pool->stride[1] !=
            0)
            out_h++;
        if ((W + 2 * pool->padding[2] - pool->dilation[2] * (pool->kernel_size[2] - 1) - 1) %
                pool->stride[2] !=
            0)
            out_w++;
    }

    int out_shape[] = {N, C, out_d, out_h, out_w};
    float* out_data;
    Tensor* output = pool_alloc_output(input, out_shape, 5, &out_data);
    if (!output)
        return NULL;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int od = 0; od < out_d; od++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        float max_val = -FLT_MAX;
                        for (int kd = 0; kd < pool->kernel_size[0]; kd++) {
                            int id =
                                od * pool->stride[0] - pool->padding[0] + kd * pool->dilation[0];
                            if (id < 0 || id >= D)
                                continue;
                            for (int kh = 0; kh < pool->kernel_size[1]; kh++) {
                                int ih = oh * pool->stride[1] - pool->padding[1] +
                                         kh * pool->dilation[1];
                                if (ih < 0 || ih >= H)
                                    continue;
                                for (int kw = 0; kw < pool->kernel_size[2]; kw++) {
                                    int iw = ow * pool->stride[2] - pool->padding[2] +
                                             kw * pool->dilation[2];
                                    if (iw < 0 || iw >= W)
                                        continue;
                                    float v = in_data[n * C * D * H * W + c * D * H * W +
                                                      id * H * W + ih * W + iw];
                                    if (v > max_val)
                                        max_val = v;
                                }
                            }
                        }
                        out_data[n * C * out_d * out_h * out_w + c * out_d * out_h * out_w +
                                 od * out_h * out_w + oh * out_w + ow] = max_val;
                    }
                }
            }
        }
    }
    return output;
}

static void maxpool3d_free(Module* module) { cml_free(module); }

MaxPool3d* nn_maxpool3d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode) {
    MaxPool3d* pool = cml_malloc(sizeof(MaxPool3d));
    if (!pool)
        return NULL;
    if (module_init((Module*)pool, "MaxPool3d", maxpool3d_forward, maxpool3d_free) != 0) {
        cml_free(pool);
        return NULL;
    }
    for (int i = 0; i < 3; i++) {
        pool->kernel_size[i] = kernel_size;
        pool->stride[i]      = stride > 0 ? stride : kernel_size;
        pool->padding[i]     = padding;
        pool->dilation[i]    = dilation > 0 ? dilation : 1;
    }
    pool->ceil_mode = ceil_mode;
    return pool;
}

static Tensor* avgpool3d_forward(Module* module, Tensor* input) {
    AvgPool3d* pool = (AvgPool3d*)module;
    if (!input || input->ndim != 5)
        return NULL; // [N, C, D, H, W]

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data)
        return NULL;

    int N = input->shape[0], C = input->shape[1];
    int D = input->shape[2], H = input->shape[3], W = input->shape[4];

    int out_d = (D + 2 * pool->padding[0] - pool->kernel_size[0]) / pool->stride[0] + 1;
    int out_h = (H + 2 * pool->padding[1] - pool->kernel_size[1]) / pool->stride[1] + 1;
    int out_w = (W + 2 * pool->padding[2] - pool->kernel_size[2]) / pool->stride[2] + 1;

    if (pool->ceil_mode) {
        if ((D + 2 * pool->padding[0] - pool->kernel_size[0]) % pool->stride[0] != 0)
            out_d++;
        if ((H + 2 * pool->padding[1] - pool->kernel_size[1]) % pool->stride[1] != 0)
            out_h++;
        if ((W + 2 * pool->padding[2] - pool->kernel_size[2]) % pool->stride[2] != 0)
            out_w++;
    }

    int out_shape[] = {N, C, out_d, out_h, out_w};
    float* out_data;
    Tensor* output = pool_alloc_output(input, out_shape, 5, &out_data);
    if (!output)
        return NULL;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int od = 0; od < out_d; od++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        float sum = 0.0f;
                        int count = 0;
                        for (int kd = 0; kd < pool->kernel_size[0]; kd++) {
                            int id = od * pool->stride[0] - pool->padding[0] + kd;
                            if (id < 0 || id >= D) {
                                if (pool->count_include_pad)
                                    count++;
                                continue;
                            }
                            for (int kh = 0; kh < pool->kernel_size[1]; kh++) {
                                int ih = oh * pool->stride[1] - pool->padding[1] + kh;
                                if (ih < 0 || ih >= H) {
                                    if (pool->count_include_pad)
                                        count++;
                                    continue;
                                }
                                for (int kw = 0; kw < pool->kernel_size[2]; kw++) {
                                    int iw = ow * pool->stride[2] - pool->padding[2] + kw;
                                    if (iw < 0 || iw >= W) {
                                        if (pool->count_include_pad)
                                            count++;
                                        continue;
                                    }
                                    sum += in_data[n * C * D * H * W + c * D * H * W + id * H * W +
                                                   ih * W + iw];
                                    count++;
                                }
                            }
                        }
                        out_data[n * C * out_d * out_h * out_w + c * out_d * out_h * out_w +
                                 od * out_h * out_w + oh * out_w + ow] =
                            count > 0 ? sum / count : 0.0f;
                    }
                }
            }
        }
    }
    return output;
}

static void avgpool3d_free(Module* module) { cml_free(module); }

AvgPool3d* nn_avgpool3d(int kernel_size, int stride, int padding, bool ceil_mode,
                        bool count_include_pad) {
    AvgPool3d* pool = cml_malloc(sizeof(AvgPool3d));
    if (!pool)
        return NULL;
    if (module_init((Module*)pool, "AvgPool3d", avgpool3d_forward, avgpool3d_free) != 0) {
        cml_free(pool);
        return NULL;
    }
    for (int i = 0; i < 3; i++) {
        pool->kernel_size[i] = kernel_size;
        pool->stride[i]      = stride > 0 ? stride : kernel_size;
        pool->padding[i]     = padding;
    }
    pool->ceil_mode         = ceil_mode;
    pool->count_include_pad = count_include_pad;
    return pool;
}

static Tensor* maxpool1d_forward(Module* module, Tensor* input) {
    MaxPool1d* pool = (MaxPool1d*)module;
    if (!input || input->ndim != 3)
        return NULL; // [N, C, L]
    int N = input->shape[0], C = input->shape[1], L = input->shape[2];

    /* Lazy path: 1D pooling == 2D pooling over a singleton height dim, reusing
     * the (differentiable) uop_maxpool2d. The 2D lazy path needs dilation=1. */
    if (pool->dilation == 1) {
        int s4[4]        = {N, C, 1, L};
        ReshapeParams rp = {.new_shape = s4, .new_ndim = 4};
        Tensor* x4       = uop_reshape(input, &rp);
        if (!x4)
            return NULL;
        Pool2DParams p2 = {
            .kernel_size       = {1, pool->kernel_size},
            .stride            = {1, pool->stride},
            .padding           = {0, pool->padding},
            .dilation          = {1, 1},
            .ceil_mode         = pool->ceil_mode,
            .count_include_pad = false,
        };
        Tensor* y4 = uop_maxpool2d(x4, &p2);
        if (!y4 || y4->ndim != 4)
            return NULL;
        int s3[3]         = {y4->shape[0], y4->shape[1], y4->shape[3]};
        ReshapeParams rp2 = {.new_shape = s3, .new_ndim = 3};
        return uop_reshape(y4, &rp2);
    }

    /* Eager fallback for dilation != 1 (no lazy dilated-pool UOP yet). */
    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data)
        return NULL;
    int out_l;
    if (pool->ceil_mode)
        out_l = (int)ceilf(
            (float)(L + 2 * pool->padding - pool->dilation * (pool->kernel_size - 1) - 1) /
                pool->stride +
            1);
    else
        out_l =
            (L + 2 * pool->padding - pool->dilation * (pool->kernel_size - 1) - 1) / pool->stride +
            1;

    int out_shape[] = {N, C, out_l};
    float* out_data;
    Tensor* output = pool_alloc_output(input, out_shape, 3, &out_data);
    if (!output)
        return NULL;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ol = 0; ol < out_l; ol++) {
                float max_val = -FLT_MAX;
                for (int k = 0; k < pool->kernel_size; k++) {
                    int il = ol * pool->stride - pool->padding + k * pool->dilation;
                    if (il >= 0 && il < L) {
                        float v = in_data[n * C * L + c * L + il];
                        if (v > max_val)
                            max_val = v;
                    }
                }
                out_data[n * C * out_l + c * out_l + ol] = max_val;
            }
        }
    }
    return output;
}

static void maxpool1d_free(Module* module) { cml_free(module); }

MaxPool1d* nn_maxpool1d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode) {
    MaxPool1d* pool = cml_malloc(sizeof(MaxPool1d));
    if (!pool)
        return NULL;
    if (module_init((Module*)pool, "MaxPool1d", maxpool1d_forward, maxpool1d_free) != 0) {
        cml_free(pool);
        return NULL;
    }
    pool->kernel_size = kernel_size;
    pool->stride      = stride > 0 ? stride : kernel_size;
    pool->padding     = padding;
    pool->dilation    = dilation > 0 ? dilation : 1;
    pool->ceil_mode   = ceil_mode;
    return pool;
}

static Tensor* avgpool1d_forward(Module* module, Tensor* input) {
    AvgPool1d* pool = (AvgPool1d*)module;
    if (!input || input->ndim != 3)
        return NULL; // [N, C, L]
    int N = input->shape[0], C = input->shape[1], L = input->shape[2];

    /* Lazy: 1D avg-pool == 2D avg-pool over a singleton height dim, reusing the
     * (differentiable) uop_avgpool2d (which honors count_include_pad). */
    int s4[4]        = {N, C, 1, L};
    ReshapeParams rp = {.new_shape = s4, .new_ndim = 4};
    Tensor* x4       = uop_reshape(input, &rp);
    if (!x4)
        return NULL;
    Pool2DParams p2 = {
        .kernel_size       = {1, pool->kernel_size},
        .stride            = {1, pool->stride},
        .padding           = {0, pool->padding},
        .dilation          = {1, 1},
        .ceil_mode         = pool->ceil_mode,
        .count_include_pad = pool->count_include_pad,
    };
    Tensor* y4 = uop_avgpool2d(x4, &p2);
    if (!y4 || y4->ndim != 4)
        return NULL;
    int s3[3]         = {y4->shape[0], y4->shape[1], y4->shape[3]};
    ReshapeParams rp2 = {.new_shape = s3, .new_ndim = 3};
    return uop_reshape(y4, &rp2);
}

static void avgpool1d_free(Module* module) { cml_free(module); }

AvgPool1d* nn_avgpool1d(int kernel_size, int stride, int padding, bool ceil_mode,
                        bool count_include_pad) {
    AvgPool1d* pool = cml_malloc(sizeof(AvgPool1d));
    if (!pool)
        return NULL;
    if (module_init((Module*)pool, "AvgPool1d", avgpool1d_forward, avgpool1d_free) != 0) {
        cml_free(pool);
        return NULL;
    }
    pool->kernel_size       = kernel_size;
    pool->stride            = stride > 0 ? stride : kernel_size;
    pool->padding           = padding;
    pool->ceil_mode         = ceil_mode;
    pool->count_include_pad = count_include_pad;
    return pool;
}

/* Adaptive 2-D pooling over [N, C, H, W]: output cell `o` reduces the input
 * window [floor(o*in/out), ceil((o+1)*in/out)). `take_max` selects max-pooling
 * over average-pooling -- the only difference between the two layers. */
static Tensor* adaptive_pool2d(Tensor* input, const int* output_size, bool take_max) {
    if (!input || input->ndim != 4)
        return NULL;

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data)
        return NULL;

    int N = input->shape[0], C = input->shape[1];
    int in_h = input->shape[2], in_w = input->shape[3];
    int out_h = output_size[0], out_w = output_size[1];

    int out_shape[]     = {N, C, out_h, out_w};
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 4, &config);
    if (!output)
        return NULL;
    tensor_ensure_executed(output);
    float* out_data = (float*)output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float* plane = in_data + (size_t)(n * C + c) * in_h * in_w;
            float* out_plane   = out_data + (size_t)(n * C + c) * out_h * out_w;
            for (int oh = 0; oh < out_h; oh++) {
                int h_start = (int)floorf((float)oh * in_h / out_h);
                int h_end   = (int)ceilf((float)(oh + 1) * in_h / out_h);
                for (int ow = 0; ow < out_w; ow++) {
                    int w_start = (int)floorf((float)ow * in_w / out_w);
                    int w_end   = (int)ceilf((float)(ow + 1) * in_w / out_w);

                    float acc = take_max ? -FLT_MAX : 0.0f;
                    int count = 0;
                    for (int h = h_start; h < h_end; h++) {
                        for (int w = w_start; w < w_end; w++) {
                            float v = plane[h * in_w + w];
                            if (take_max) {
                                if (v > acc)
                                    acc = v;
                            } else {
                                acc += v;
                            }
                            count++;
                        }
                    }
                    out_plane[oh * out_w + ow] =
                        take_max ? (count > 0 ? acc : 0.0f) : (count > 0 ? acc / count : 0.0f);
                }
            }
        }
    }
    return output;
}

static Tensor* adaptive_avgpool2d_forward(Module* module, Tensor* input) {
    return adaptive_pool2d(input, ((AdaptiveAvgPool2d*)module)->output_size, false);
}

static void adaptive_avgpool2d_free(Module* module) { cml_free(module); }

AdaptiveAvgPool2d* nn_adaptive_avgpool2d(int output_h, int output_w) {
    AdaptiveAvgPool2d* pool = cml_malloc(sizeof(AdaptiveAvgPool2d));
    if (!pool)
        return NULL;
    if (module_init((Module*)pool, "AdaptiveAvgPool2d", adaptive_avgpool2d_forward,
                    adaptive_avgpool2d_free) != 0) {
        cml_free(pool);
        return NULL;
    }
    pool->output_size[0] = output_h;
    pool->output_size[1] = output_w;
    return pool;
}

/* Adaptive 1-D pooling over [N, C, L]: output cell `o` reduces the input window
 * [floor(o*L/out), ceil((o+1)*L/out)). `take_max` selects max over average. */
static Tensor* adaptive_pool1d(Tensor* input, int out_l, bool take_max) {
    if (!input || input->ndim != 3)
        return NULL;

    tensor_ensure_executed(input);
    float* in_data = (float*)input->data;
    if (!in_data)
        return NULL;

    int N = input->shape[0], C = input->shape[1], L = input->shape[2];
    int out_shape[] = {N, C, out_l};
    float* out_data;
    Tensor* output = pool_alloc_output(input, out_shape, 3, &out_data);
    if (!output)
        return NULL;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float* row = in_data + (size_t)(n * C + c) * L;
            float* out_row   = out_data + (size_t)(n * C + c) * out_l;
            for (int ol = 0; ol < out_l; ol++) {
                int start = (int)floorf((float)ol * L / out_l);
                int end   = (int)ceilf((float)(ol + 1) * L / out_l);
                float acc = take_max ? -FLT_MAX : 0.0f;
                for (int i = start; i < end; i++) {
                    if (take_max) {
                        if (row[i] > acc)
                            acc = row[i];
                    } else {
                        acc += row[i];
                    }
                }
                out_row[ol] = take_max ? acc : acc / (float)(end - start);
            }
        }
    }
    return output;
}

static Tensor* adaptive_avgpool1d_forward(Module* module, Tensor* input) {
    return adaptive_pool1d(input, ((AdaptiveAvgPool1d*)module)->output_size, false);
}

static void adaptive_avgpool1d_free(Module* module) { cml_free(module); }

AdaptiveAvgPool1d* nn_adaptive_avgpool1d(int output_size) {
    AdaptiveAvgPool1d* pool = cml_malloc(sizeof(AdaptiveAvgPool1d));
    if (!pool)
        return NULL;
    if (module_init((Module*)pool, "AdaptiveAvgPool1d", adaptive_avgpool1d_forward,
                    adaptive_avgpool1d_free) != 0) {
        cml_free(pool);
        return NULL;
    }
    pool->output_size = output_size;
    return pool;
}

static Tensor* adaptive_maxpool2d_forward(Module* module, Tensor* input) {
    return adaptive_pool2d(input, ((AdaptiveMaxPool2d*)module)->output_size, true);
}

static void adaptive_maxpool2d_free(Module* module) { cml_free(module); }

AdaptiveMaxPool2d* nn_adaptive_maxpool2d(int output_h, int output_w) {
    AdaptiveMaxPool2d* pool = cml_malloc(sizeof(AdaptiveMaxPool2d));
    if (!pool)
        return NULL;
    if (module_init((Module*)pool, "AdaptiveMaxPool2d", adaptive_maxpool2d_forward,
                    adaptive_maxpool2d_free) != 0) {
        cml_free(pool);
        return NULL;
    }
    pool->output_size[0] = output_h;
    pool->output_size[1] = output_w;
    return pool;
}

static Tensor* adaptive_maxpool1d_forward(Module* module, Tensor* input) {
    return adaptive_pool1d(input, ((AdaptiveMaxPool1d*)module)->output_size, true);
}

static void adaptive_maxpool1d_free(Module* module) { cml_free(module); }

AdaptiveMaxPool1d* nn_adaptive_maxpool1d(int output_size) {
    AdaptiveMaxPool1d* pool = cml_malloc(sizeof(AdaptiveMaxPool1d));
    if (!pool)
        return NULL;
    if (module_init((Module*)pool, "AdaptiveMaxPool1d", adaptive_maxpool1d_forward,
                    adaptive_maxpool1d_free) != 0) {
        cml_free(pool);
        return NULL;
    }
    pool->output_size = output_size;
    return pool;
}
