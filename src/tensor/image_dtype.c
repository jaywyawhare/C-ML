#include "tensor/image_dtype.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

int cml_image_dtype_channels(CMLImageFormat format) {
    switch (format) {
    case CML_IMAGE_R32F:
    case CML_IMAGE_R16F:
        return 1;
    case CML_IMAGE_RG32F:
    case CML_IMAGE_RG16F:
        return 2;
    case CML_IMAGE_RGBA8:
    case CML_IMAGE_RGBA16F:
    case CML_IMAGE_RGBA32F:
        return 4;
    default:
        return 0;
    }
}

int cml_image_dtype_bpp(CMLImageFormat format) {
    switch (format) {
    case CML_IMAGE_RGBA8:    return 4;
    case CML_IMAGE_R16F:     return 2;
    case CML_IMAGE_RG16F:    return 4;
    case CML_IMAGE_RGBA16F:  return 8;
    case CML_IMAGE_R32F:     return 4;
    case CML_IMAGE_RG32F:    return 8;
    case CML_IMAGE_RGBA32F:  return 16;
    default:                 return 0;
    }
}

const char* cml_image_dtype_name(CMLImageFormat format) {
    switch (format) {
    case CML_IMAGE_NONE:     return "none";
    case CML_IMAGE_RGBA8:    return "rgba8";
    case CML_IMAGE_RGBA16F:  return "rgba16f";
    case CML_IMAGE_RGBA32F:  return "rgba32f";
    case CML_IMAGE_R32F:     return "r32f";
    case CML_IMAGE_RG32F:    return "rg32f";
    case CML_IMAGE_R16F:     return "r16f";
    case CML_IMAGE_RG16F:    return "rg16f";
    default:                 return "unknown";
    }
}

bool cml_image_dtype_compatible(const int* shape, int ndim, CMLImageFormat format) {
    if (!shape || ndim < 1 || format == CML_IMAGE_NONE) return false;

    int channels = cml_image_dtype_channels(format);
    if (channels == 0) return false;

    /* Last dim must be divisible by channels */
    int last_dim = shape[ndim - 1];
    if (last_dim % channels != 0) return false;

    /* Total elements must fit in 2D texture */
    size_t total = 1;
    for (int i = 0; i < ndim; i++) total *= (size_t)shape[i];
    size_t pixels = total / (size_t)channels;

    /* Max texture dimensions (conservative GPU limit) */
    size_t max_dim = 16384;
    if (pixels > max_dim * max_dim) return false;

    return true;
}

CMLImageFormat cml_image_dtype_select(const int* shape, int ndim) {
    if (!shape || ndim < 1) return CML_IMAGE_NONE;

    int last_dim = shape[ndim - 1];

    /* Prefer RGBA32F for float data with channels divisible by 4 */
    if (last_dim % 4 == 0) return CML_IMAGE_RGBA32F;
    if (last_dim % 2 == 0) return CML_IMAGE_RG32F;
    return CML_IMAGE_R32F;
}

void cml_image_dtype_dims(const int* shape, int ndim, CMLImageFormat format,
                           int* out_width, int* out_height) {
    if (!shape || ndim < 1 || !out_width || !out_height) return;

    int channels = cml_image_dtype_channels(format);
    if (channels == 0) { *out_width = 0; *out_height = 0; return; }

    size_t total = 1;
    for (int i = 0; i < ndim; i++) total *= (size_t)shape[i];
    size_t pixels = total / (size_t)channels;

    /* Find dimensions close to square root */
    int w = (int)ceil(sqrt((double)pixels));
    while (pixels % (size_t)w != 0 && w > 1) w--;
    int h = (int)(pixels / (size_t)w);

    *out_width = w;
    *out_height = h;
}

CMLImageTensor* cml_image_tensor_create(Tensor* tensor, CMLImageFormat format) {
    if (!tensor) return NULL;

    if (!cml_image_dtype_compatible(tensor->shape, tensor->ndim, format))
        return NULL;

    CMLImageTensor* img = (CMLImageTensor*)calloc(1, sizeof(CMLImageTensor));
    if (!img) return NULL;

    img->tensor = tensor;
    img->format = format;
    img->channels = cml_image_dtype_channels(format);

    /* Store original shape */
    img->base_ndim = tensor->ndim;
    for (int i = 0; i < tensor->ndim && i < 4; i++)
        img->base_shape[i] = tensor->shape[i];

    /* Compute image dimensions */
    cml_image_dtype_dims(tensor->shape, tensor->ndim, format,
                          &img->width, &img->height);

    return img;
}

Tensor* cml_image_tensor_to_regular(CMLImageTensor* img) {
    if (!img || !img->tensor) return NULL;
    /* The underlying tensor is already a regular tensor */
    return img->tensor;
}

void cml_image_tensor_free(CMLImageTensor* img) {
    if (!img) return;
    /* Note: does NOT free the underlying tensor */
    free(img);
}

size_t cml_image_tensor_memory(const CMLImageTensor* img) {
    if (!img) return 0;
    return (size_t)img->width * (size_t)img->height * (size_t)cml_image_dtype_bpp(img->format);
}

void cml_image_tensor_print(const CMLImageTensor* img) {
    if (!img) {
        printf("ImageTensor: NULL\n");
        return;
    }
    printf("ImageTensor: format=%s, %dx%d (%d ch), mem=%zu bytes\n",
           cml_image_dtype_name(img->format),
           img->width, img->height, img->channels,
           cml_image_tensor_memory(img));
    printf("  Base shape: [");
    for (int i = 0; i < img->base_ndim; i++)
        printf("%d%s", img->base_shape[i], i < img->base_ndim - 1 ? ", " : "");
    printf("]\n");
}
