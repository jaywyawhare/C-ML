/*
 * Portable view operations: transpose, gather/scatter, strided copy, permute.
 *
 * The hand-rolled 4x4/8x8 SSE/AVX transpose micro-kernels, AVX2 gather, and
 * vector copy/fill intrinsics were removed.  These cache-blocked scalar loops
 * auto-vectorize at -O3; the JIT backend emits shape-specialized permute/gather
 * kernels for the compiled path.
 */
#include "ops/simd_views.h"
#include <string.h>
#include <stdlib.h>
#include "alloc/cml_allocator.h"

/* Cache-blocked scalar transpose. */
void simd_transpose_2d_f32(const float* src, float* dst, int rows, int cols) {
    const int BLOCK = 32;
    for (int i = 0; i < rows; i += BLOCK) {
        for (int j = 0; j < cols; j += BLOCK) {
            int i_end = (i + BLOCK < rows) ? i + BLOCK : rows;
            int j_end = (j + BLOCK < cols) ? j + BLOCK : cols;
            for (int ii = i; ii < i_end; ii++)
                for (int jj = j; jj < j_end; jj++)
                    dst[jj * rows + ii] = src[ii * cols + jj];
        }
    }
}

void simd_transpose_inplace_f32(float* data, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            float tmp       = data[i * n + j];
            data[i * n + j] = data[j * n + i];
            data[j * n + i] = tmp;
        }
}

void simd_transpose_batched_f32(const float* src, float* dst, int batch, int rows, int cols) {
    size_t src_stride = (size_t)rows * cols;
    size_t dst_stride = (size_t)cols * rows;
    for (int b = 0; b < batch; b++)
        simd_transpose_2d_f32(src + b * src_stride, dst + b * dst_stride, rows, cols);
}

void simd_gather_f32(const float* src, const int32_t* indices, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = src[indices[i]];
}

void simd_scatter_f32(const float* src, const int32_t* indices, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[indices[i]] = src[i];
}

void simd_scatter_add_f32(const float* src, const int32_t* indices, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[indices[i]] += src[i];
}

void simd_copy_f32(const float* src, float* dst, size_t n) { memcpy(dst, src, n * sizeof(float)); }

void simd_strided_copy_f32(const float* src, float* dst, size_t n, size_t src_stride,
                           size_t dst_stride) {
    if (src_stride == 1 && dst_stride == 1) {
        simd_copy_f32(src, dst, n);
        return;
    }
    for (size_t i = 0; i < n; i++)
        dst[i * dst_stride] = src[i * src_stride];
}

void simd_fill_f32(float* dst, float value, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = value;
}

void simd_broadcast_copy_f32(const float* src, size_t src_n, float* dst, size_t dst_n) {
    if (src_n == 0 || dst_n == 0)
        return;
    if (src_n == 1) {
        simd_fill_f32(dst, src[0], dst_n);
        return;
    }
    simd_copy_f32(src, dst, src_n);
    size_t copied = src_n;
    while (copied < dst_n) {
        size_t to_copy = copied;
        if (copied + to_copy > dst_n)
            to_copy = dst_n - copied;
        memcpy(dst + copied, dst, to_copy * sizeof(float));
        copied += to_copy;
    }
}

void simd_permute_nd_f32(const float* src, float* dst, const int* shape, const size_t* strides,
                         const int* perm, int ndim, size_t numel) {
    int* dst_shape      = (int*)cml_malloc(ndim * sizeof(int));
    size_t* dst_strides = (size_t*)cml_malloc(ndim * sizeof(size_t));
    if (!dst_shape || !dst_strides) {
        cml_free(dst_shape);
        cml_free(dst_strides);
        memcpy(dst, src, numel * sizeof(float));
        return;
    }

    for (int i = 0; i < ndim; i++)
        dst_shape[i] = shape[perm[i]];

    dst_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];

    int* coords = (int*)cml_calloc(ndim, sizeof(int));
    if (!coords) {
        cml_free(dst_shape);
        cml_free(dst_strides);
        memcpy(dst, src, numel * sizeof(float));
        return;
    }

    for (size_t idx = 0; idx < numel; idx++) {
        size_t src_offset = 0;
        for (int d = 0; d < ndim; d++)
            src_offset += coords[d] * strides[d];

        size_t dst_offset = 0;
        for (int d = 0; d < ndim; d++)
            dst_offset += coords[perm[d]] * dst_strides[d];

        dst[dst_offset] = src[src_offset];

        for (int d = ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < shape[d])
                break;
            coords[d] = 0;
        }
    }

    cml_free(coords);
    cml_free(dst_shape);
    cml_free(dst_strides);
}
