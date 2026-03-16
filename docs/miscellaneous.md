# Miscellaneous APIs

Additional C-ML subsystems covering sparse tensors, image dtypes, data augmentation, Winograd convolution, symbolic computation, SIMD vectorization, disk-backed storage, and CMake/pkg-config integration.

## Table of Contents

- [Sparse Tensors](#sparse-tensors)
- [Image Dtype](#image-dtype)
- [Data Augmentation](#data-augmentation)
- [Winograd Convolution](#winograd-convolution)
- [Symbolic Computation](#symbolic-computation)
- [SIMD Vectorization](#simd-vectorization)
- [Disk Backend](#disk-backend)
- [CMake Integration](#cmake-integration)

______________________________________________________________________

## Sparse Tensors

Sparse COO (Coordinate) format stores only non-zero entries as `(indices, values)` pairs, dramatically reducing memory for tensors where most elements are zero.

```c
#include "tensor/sparse_tensor.h"
```

### Types

```c
typedef enum {
    TENSOR_FORMAT_DENSE,
    TENSOR_FORMAT_SPARSE_COO
} TensorFormat;

typedef struct SparseCOOData {
    Tensor* indices;    // [nnz, ndim] - integer coordinates of non-zeros
    Tensor* values;     // [nnz] - values at those coordinates
    int* dense_shape;
    int dense_ndim;
    int nnz;            // Number of non-zero elements
} SparseCOOData;
```

### Functions

```c
// Create sparse tensor from explicit indices and values
SparseCOOData* sparse_coo_tensor(Tensor* indices, Tensor* values,
                                  const int* dense_shape, int dense_ndim);

// Convert dense tensor to sparse (drops zeros)
SparseCOOData* sparse_from_dense(Tensor* dense);

// Convert sparse back to dense
Tensor* sparse_to_dense(SparseCOOData* sparse, const TensorConfig* config);

// Sparse-dense matrix multiply: C = A * B where A is sparse [M,K], B is dense [K,N]
Tensor* sparse_matmul(SparseCOOData* sparse, Tensor* dense);

// Sort indices and sum duplicate entries
SparseCOOData* sparse_coalesce(SparseCOOData* sparse);

// Free sparse tensor
void sparse_free(SparseCOOData* sparse);
```

### Example

```c
// Create a sparse 1000x1000 matrix with 3 non-zero entries
int idx_data[] = {0, 0,  1, 5,  999, 999};   // row,col pairs
float val_data[] = {1.0f, 2.0f, 3.0f};
int dense_shape[] = {1000, 1000};

Tensor* indices = cml_tensor_2d((float*)idx_data, 3, 2);
Tensor* values  = cml_tensor_1d(val_data, 3);

SparseCOOData* sp = sparse_coo_tensor(indices, values, dense_shape, 2);
Tensor* dense_B   = cml_ones_2d(1000, 64);
Tensor* result    = sparse_matmul(sp, dense_B);    // [1000, 64]

sparse_free(sp);
```

______________________________________________________________________

## Image Dtype

Image dtypes map tensor data to GPU texture memory formats, enabling faster access through hardware texture units. This is particularly useful for convolution workloads on GPUs.

```c
#include "tensor/image_dtype.h"
```

### Formats

```c
typedef enum {
    CML_IMAGE_NONE = 0,     // Not an image tensor
    CML_IMAGE_RGBA8,         // 4x uint8 per pixel
    CML_IMAGE_RGBA16F,       // 4x float16 per pixel
    CML_IMAGE_RGBA32F,       // 4x float32 per pixel
    CML_IMAGE_R32F,          // 1x float32 (single channel)
    CML_IMAGE_RG32F,         // 2x float32
    CML_IMAGE_R16F,          // 1x float16
    CML_IMAGE_RG16F,         // 2x float16
} CMLImageFormat;
```

### Functions

```c
// Check if a tensor shape is suitable for a given image format
bool cml_image_dtype_compatible(const int* shape, int ndim, CMLImageFormat format);

// Auto-select the best image format for a tensor shape
CMLImageFormat cml_image_dtype_select(const int* shape, int ndim);

// Create image tensor from a regular tensor
CMLImageTensor* cml_image_tensor_create(Tensor* tensor, CMLImageFormat format);

// Convert back to regular tensor
Tensor* cml_image_tensor_to_regular(CMLImageTensor* img);

// Free image tensor (does NOT free the underlying tensor)
void cml_image_tensor_free(CMLImageTensor* img);

// Query helpers
void        cml_image_dtype_dims(const int* shape, int ndim, CMLImageFormat format,
                                  int* out_width, int* out_height);
int         cml_image_dtype_bpp(CMLImageFormat format);       // Bytes per pixel
int         cml_image_dtype_channels(CMLImageFormat format);  // Channel count
const char* cml_image_dtype_name(CMLImageFormat format);      // Format name string
size_t      cml_image_tensor_memory(const CMLImageTensor* img);
void        cml_image_tensor_print(const CMLImageTensor* img);
```

### Example

```c
Tensor* conv_weight = cml_ones((int[]){64, 3, 3, 3}, 4, NULL);

CMLImageFormat fmt = cml_image_dtype_select(conv_weight->shape, conv_weight->ndim);
if (fmt != CML_IMAGE_NONE) {
    CMLImageTensor* img = cml_image_tensor_create(conv_weight, fmt);
    cml_image_tensor_print(img);
    // Use img for GPU texture-backed operations ...
    cml_image_tensor_free(img);
}
```

______________________________________________________________________

## Data Augmentation

Image augmentation pipeline for training data, operating on tensors in `[batch, channels, height, width]` layout.

```c
#include "core/augmentation.h"
```

### Configuration

```c
typedef struct AugmentationConfig {
    bool  random_crop;
    int   crop_size[2];                // [height, width]

    bool  random_horizontal_flip;
    float horizontal_flip_prob;        // 0.0 to 1.0
    bool  random_vertical_flip;
    float vertical_flip_prob;

    bool  random_rotation;
    float rotation_angle_min;          // Degrees
    float rotation_angle_max;

    bool  color_jitter;
    float brightness, contrast, saturation, hue;

    bool   normalize;
    float* mean;                       // Per-channel means
    float* std;                        // Per-channel stds
    int    num_channels;
} AugmentationConfig;

AugmentationConfig* augmentation_config_create(void);  // Defaults (all disabled)
void                augmentation_config_free(AugmentationConfig* config);
```

### Individual Transforms

```c
Tensor* augment_random_crop(Tensor* input, int crop_height, int crop_width);
Tensor* augment_random_horizontal_flip(Tensor* input, float prob);
Tensor* augment_random_vertical_flip(Tensor* input, float prob);
Tensor* augment_random_rotation(Tensor* input, float angle_min, float angle_max);
Tensor* augment_color_jitter(Tensor* input, float brightness, float contrast,
                             float saturation, float hue);
Tensor* augment_normalize(Tensor* input, float* mean, float* std, int num_channels);
```

### Pipeline

```c
// Apply all enabled transforms from a config in one call
Tensor* augment_apply(Tensor* input, AugmentationConfig* config);
```

### Example

```c
AugmentationConfig* aug = augmentation_config_create();
aug->random_horizontal_flip = true;
aug->horizontal_flip_prob   = 0.5f;
aug->random_crop            = true;
aug->crop_size[0] = 224;
aug->crop_size[1] = 224;
aug->normalize              = true;
float mean[] = {0.485f, 0.456f, 0.406f};
float std[]  = {0.229f, 0.224f, 0.225f};
aug->mean = mean;
aug->std  = std;
aug->num_channels = 3;

Tensor* augmented = augment_apply(batch, aug);
augmentation_config_free(aug);
```

______________________________________________________________________

## Winograd Convolution

Winograd-domain transforms accelerate 3x3 convolutions by reducing the number of multiplications. Two variants are supported:

| Variant           | Output Tile | Transform Tile | Multiplication Savings |
| ----------------- | ----------- | -------------- | ---------------------- |
| `WINOGRAD_F2x2_3x3` | 2x2      | 4x4            | ~2.25x fewer           |
| `WINOGRAD_F4x4_3x3` | 4x4      | 6x6            | ~4x fewer              |

```c
#include "ops/winograd.h"
```

### Types

```c
typedef enum {
    WINOGRAD_F2x2_3x3 = 0,
    WINOGRAD_F4x4_3x3 = 1,
} WinogradVariant;

typedef struct {
    WinogradVariant variant;
    int tile_size;      // 4 or 6
    int output_tile;    // 2 or 4
    int kernel_size;    // 3
} WinogradConfig;
```

### Functions

```c
// Check if Winograd applies (requires 3x3 kernel, stride 1, dilation 1)
bool winograd_applicable(int kernel_h, int kernel_w,
                         int stride_h, int stride_w,
                         int dilation_h, int dilation_w);

// Auto-select the best variant based on spatial dimensions
WinogradConfig winograd_select_variant(int height, int width);

// Pre-transform weights to Winograd domain (do once, reuse across batches)
int winograd_transform_weight(const float* weight, int out_channels, int in_channels,
                              const WinogradConfig* config, float* transformed);

// Full Winograd convolution (supports grouped convolution)
int winograd_conv2d(const float* input, const float* weight, const float* bias,
                    float* output, int batch, int in_channels, int out_channels,
                    int height, int width, int padding_h, int padding_w,
                    int groups, const WinogradConfig* config);
```

### Example

```c
WinogradConfig cfg = winograd_select_variant(224, 224);

// Pre-transform weights once
float* w_transformed = malloc(winograd_transformed_size);
winograd_transform_weight(weight, 64, 3, &cfg, w_transformed);

// Run convolution
winograd_conv2d(input, weight, bias, output,
                /*batch=*/1, /*in_ch=*/3, /*out_ch=*/64,
                /*h=*/224, /*w=*/224, /*pad_h=*/1, /*pad_w=*/1,
                /*groups=*/1, &cfg);
```

______________________________________________________________________

## Symbolic Computation

Symbolic shapes represent tensor dimensions whose values are only known at runtime. Expressions carry bounds `[vmin, vmax]` and support arithmetic, simplification, and evaluation. All expression nodes are reference-counted.

```c
#include "symbolic/symbolic.h"
```

### Expression Creation

```c
SymExpr* sym_const(int64_t value);                          // Constant
SymExpr* sym_var(const char* name, int64_t vmin, int64_t vmax); // Variable with bounds
```

### Arithmetic (constant-folds at construction time)

```c
SymExpr* sym_add(SymExpr* a, SymExpr* b);
SymExpr* sym_mul(SymExpr* a, SymExpr* b);
SymExpr* sym_div(SymExpr* a, SymExpr* b);       // Integer division
SymExpr* sym_mod(SymExpr* a, SymExpr* b);
SymExpr* sym_min_expr(SymExpr* a, SymExpr* b);
SymExpr* sym_max_expr(SymExpr* a, SymExpr* b);
```

### Bounds Inference and Evaluation

```c
int64_t sym_expr_min(const SymExpr* e);   // Minimum possible value
int64_t sym_expr_max(const SymExpr* e);   // Maximum possible value

// Substitute variable values and compute result
int64_t sym_eval(const SymExpr* e, const char** var_names,
                 const int64_t* values, int num_vars);
```

### Simplification and Debug

```c
SymExpr* sym_simplify(SymExpr* e);   // Identity/constant folding (returns new expr)
int sym_expr_to_string(const SymExpr* e, char* buf, int buf_size);
```

### Memory Management

```c
void sym_expr_retain(SymExpr* e);
void sym_expr_release(SymExpr* e);
```

### Symbolic Shapes

```c
SymDim   sym_dim_concrete(int value);
SymDim   sym_dim_symbolic(SymExpr* expr);
void     sym_dim_release(SymDim* dim);

SymShape* sym_shape_from_concrete(const int* dims, int ndim);
SymShape* sym_shape_broadcast(const SymShape* a, const SymShape* b);
int       sym_shape_eval(const SymShape* shape, const char** var_names,
                         const int64_t* values, int num_vars, int* out_dims);
int       sym_shape_to_string(const SymShape* shape, char* buf, int buf_size);
void      sym_shape_retain(SymShape* shape);
void      sym_shape_release(SymShape* shape);
```

### Example

```c
// Represent a dynamic batch dimension: batch in [1, 128]
SymExpr* batch = sym_var("batch", 1, 128);
SymExpr* seq   = sym_var("seq", 1, 2048);

// Compute output shape: batch * seq
SymExpr* total = sym_mul(batch, seq);
printf("min elements = %ld, max elements = %ld\n",
       sym_expr_min(total), sym_expr_max(total));  // 1, 262144

// Evaluate with concrete values
const char* names[] = {"batch", "seq"};
int64_t vals[] = {32, 512};
int64_t result = sym_eval(total, names, vals, 2);  // 16384

sym_expr_release(total);
sym_expr_release(seq);
sym_expr_release(batch);
```

______________________________________________________________________

## SIMD Vectorization

C-ML provides three SIMD headers for low-level vectorized operations. These are used internally by the tensor engine but are also available for direct use.

Platform support: SSE/SSE4, AVX/AVX2, AVX-512, and ARM NEON, with runtime detection and scalar fallbacks.

### Runtime Capability Detection

```c
#include "ops/simd_math.h"

typedef struct {
    bool has_sse, has_sse4, has_avx, has_avx2, has_avx512;
    bool has_neon;
    bool has_sleef;   // SLEEF library available for high-accuracy transcendentals
} CMLSimdCaps;

CMLSimdCaps        cml_detect_simd_caps(void);
const CMLSimdCaps* cml_get_simd_caps(void);   // Cached (lazy init)
void               cml_print_simd_caps(void);
```

### Reduction Operations (`simd_utils.h`)

```c
#include "ops/simd_utils.h"

float simd_sum_float(const float* data, size_t count);
float simd_sum_float_strided(const float* data, size_t count, size_t stride);
float simd_max_float(const float* data, size_t count);
float simd_min_float(const float* data, size_t count);
```

### Vectorized Math (`simd_math.h`)

Unary operations (all follow the signature `void fn(const float* in, float* out, size_t n)`):

| Function            | Operation               |
| ------------------- | ----------------------- |
| `simd_exp_f32`      | exp(x)                  |
| `simd_log_f32`      | ln(x)                   |
| `simd_sqrt_f32`     | sqrt(x)                 |
| `simd_rsqrt_f32`    | 1/sqrt(x)               |
| `simd_recip_f32`    | 1/x                     |
| `simd_abs_f32`      | |x|                     |
| `simd_sin_f32`      | sin(x)                  |
| `simd_cos_f32`      | cos(x)                  |
| `simd_tan_f32`      | tan(x)                  |
| `simd_tanh_f32`     | tanh(x)                 |
| `simd_sigmoid_f32`  | 1/(1+exp(-x))           |
| `simd_neg_f32`      | -x                      |

Binary operations (all follow `void fn(const float* a, const float* b, float* out, size_t n)`):

`simd_add_f32`, `simd_sub_f32`, `simd_mul_f32`, `simd_div_f32`, `simd_pow_f32`, `simd_min_f32`, `simd_max_f32`, `simd_cmplt_f32`, `simd_cmpgt_f32`

Ternary: `simd_where_f32(cond, a, b, out, n)` -- conditional select.

Broadcast helpers: `simd_add_scalar_f32`, `simd_mul_scalar_f32`, `simd_add_broadcast_f32`, `simd_mul_broadcast_f32`, `simd_max_broadcast_f32`.

Parallel variants (multi-threaded for large arrays): `simd_add_f32_parallel`, `simd_mul_f32_parallel`, `simd_exp_f32_parallel`, `simd_sum_f32_parallel`. Control the threshold with `simd_set_parallel_threshold(size_t n)` (default: 10000).

### View Operations (`simd_views.h`)

```c
#include "ops/simd_views.h"

// Transpose
void simd_transpose_2d_f32(const float* src, float* dst, int rows, int cols);
void simd_transpose_inplace_f32(float* data, int n);          // Square in-place
void simd_transpose_batched_f32(const float* src, float* dst, int batch, int rows, int cols);

// Gather / Scatter
void simd_gather_f32(const float* src, const int32_t* indices, float* out, size_t n);
void simd_scatter_f32(const float* src, const int32_t* indices, float* dst, size_t n);
void simd_scatter_add_f32(const float* src, const int32_t* indices, float* dst, size_t n);

// Copy variants
void simd_strided_copy_f32(const float* src, float* dst, size_t n,
                           size_t src_stride, size_t dst_stride);
void simd_copy_f32(const float* src, float* dst, size_t n);
void simd_fill_f32(float* dst, float value, size_t n);
void simd_broadcast_copy_f32(const float* src, size_t src_n, float* dst, size_t dst_n);

// N-D permute
void simd_permute_nd_f32(const float* src, float* dst, const int* shape,
                         const size_t* strides, const int* perm, int ndim, size_t numel);
```

______________________________________________________________________

## Disk Backend

The disk backend enables tensor storage on disk via synchronous I/O, memory-mapped I/O, or asynchronous I/O (io_uring on Linux). This lets you work with datasets larger than available RAM.

```c
#include "backend/disk_backend.h"
```

### I/O Modes

```c
typedef enum {
    CML_DISK_SYNC = 0,    // Synchronous read/write
    CML_DISK_MMAP,         // Memory-mapped I/O
    CML_DISK_ASYNC,        // Async I/O (io_uring on Linux, fallback to sync)
} CMLDiskIOMode;
```

### Backend Lifecycle

```c
CMLDiskBackend* cml_disk_backend_create(const char* base_path, CMLDiskIOMode mode);
void            cml_disk_backend_free(CMLDiskBackend* backend);
```

### Save and Load

```c
// Save tensor to disk under a name
int cml_disk_save_tensor(CMLDiskBackend* backend, const char* name, Tensor* tensor);

// Load tensor fully into memory
Tensor* cml_disk_load_tensor(CMLDiskBackend* backend, const char* name);
```

### Memory-Mapped Access

```c
// Memory-map a tensor (lazy, zero-copy)
CMLDiskTensor* cml_disk_mmap_tensor(CMLDiskBackend* backend, const char* name);

// Read a sub-range from a mapped tensor
int cml_disk_tensor_read(CMLDiskTensor* dt, void* buffer, size_t offset, size_t size);

// Unmap and free
void cml_disk_tensor_unmap(CMLDiskTensor* dt);
void cml_disk_tensor_free(CMLDiskTensor* dt);

// Copy mapped tensor into a regular tensor
Tensor* cml_disk_tensor_to_tensor(CMLDiskTensor* dt);
```

### Async I/O

```c
int cml_disk_async_read(CMLDiskBackend* backend, const char* name,
                         void* buffer, size_t size);
int cml_disk_wait(CMLDiskBackend* backend);  // Wait for all pending reads
```

### Statistics

```c
void cml_disk_backend_stats(const CMLDiskBackend* backend,
                             uint64_t* bytes_read, uint64_t* bytes_written,
                             uint64_t* num_reads, uint64_t* num_writes);
void cml_disk_backend_print(const CMLDiskBackend* backend);
```

### Example

```c
CMLDiskBackend* disk = cml_disk_backend_create("/data/tensors", CML_DISK_MMAP);

// Save a large tensor
Tensor* embeddings = cml_ones((int[]){1000000, 768}, 2, NULL);
cml_disk_save_tensor(disk, "embeddings", embeddings);
tensor_free(embeddings);

// Later, memory-map it (zero-copy, lazy)
CMLDiskTensor* dt = cml_disk_mmap_tensor(disk, "embeddings");
Tensor* loaded = cml_disk_tensor_to_tensor(dt);

cml_disk_tensor_free(dt);
cml_disk_backend_free(disk);
```

______________________________________________________________________

## CMake Integration

C-ML installs CMake package config files and a pkg-config `.pc` file so downstream projects can find and link against CML.

### Using `find_package` (CMake)

After installing CML (e.g., `cmake --install build`), use it in your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.14)
project(my_app LANGUAGES C)

find_package(CML REQUIRED)

add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE CML::cml)
```

`find_package(CML)` imports the `CML::cml` target, which carries the correct include paths and link flags. No manual `-I` or `-L` flags are needed.

If CML is installed to a non-standard prefix, point CMake to it:

```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/cml/install ..
```

### Using pkg-config

The installed `cml.pc` file contains:

```
Name: CML
Description: C Machine Learning Library
Version: <version>
Libs: -L${libdir} -lcml
Cflags: -I${includedir}
```

Use it from a Makefile or build script:

```makefile
CFLAGS  += $(shell pkg-config --cflags cml)
LDFLAGS += $(shell pkg-config --libs cml)
```

Or query it directly:

```bash
pkg-config --cflags cml   # -I/usr/local/include
pkg-config --libs cml     # -L/usr/local/lib -lcml
```

If CML is installed to a non-standard prefix, set `PKG_CONFIG_PATH`:

```bash
export PKG_CONFIG_PATH=/path/to/cml/install/lib/pkgconfig:$PKG_CONFIG_PATH
```
