#ifndef CML_IMAGE_DTYPE_H
#define CML_IMAGE_DTYPE_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CML_IMAGE_NONE = 0,     /* Not an image tensor */
    CML_IMAGE_RGBA8,         /* 4x uint8 per pixel */
    CML_IMAGE_RGBA16F,       /* 4x float16 per pixel */
    CML_IMAGE_RGBA32F,       /* 4x float32 per pixel */
    CML_IMAGE_R32F,          /* 1x float32 per pixel (single channel) */
    CML_IMAGE_RG32F,         /* 2x float32 per pixel */
    CML_IMAGE_R16F,          /* 1x float16 per pixel */
    CML_IMAGE_RG16F,         /* 2x float16 per pixel */
} CMLImageFormat;

typedef struct CMLImageTensor {
    Tensor* tensor;          /* Underlying tensor data */
    CMLImageFormat format;   /* Image format */
    int width;               /* Texture width */
    int height;              /* Texture height */
    int channels;            /* Number of channels (1, 2, or 4) */
    int base_shape[4];       /* Original tensor shape before image conversion */
    int base_ndim;           /* Original number of dimensions */
    void* texture_handle;    /* GPU texture object (backend-specific) */
    bool is_bound;           /* Whether bound to GPU texture */
} CMLImageTensor;

bool cml_image_dtype_compatible(const int* shape, int ndim, CMLImageFormat format);
CMLImageFormat cml_image_dtype_select(const int* shape, int ndim);
CMLImageTensor* cml_image_tensor_create(Tensor* tensor, CMLImageFormat format);
Tensor* cml_image_tensor_to_regular(CMLImageTensor* img);

/* Does NOT free underlying tensor */
void cml_image_tensor_free(CMLImageTensor* img);

void cml_image_dtype_dims(const int* shape, int ndim, CMLImageFormat format,
                           int* out_width, int* out_height);
int cml_image_dtype_bpp(CMLImageFormat format);
int cml_image_dtype_channels(CMLImageFormat format);
const char* cml_image_dtype_name(CMLImageFormat format);
size_t cml_image_tensor_memory(const CMLImageTensor* img);
void cml_image_tensor_print(const CMLImageTensor* img);

#ifdef __cplusplus
}
#endif

#endif /* CML_IMAGE_DTYPE_H */
