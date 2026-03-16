# Model I/O Formats

C-ML supports multiple serialization formats for saving, loading, and exchanging model weights and tensors. This document covers each format, its API, usage examples, and trade-offs.

---

## Table of Contents

1. [Native Serialization](#native-serialization)
2. [Model I/O Utilities (Checkpointing)](#model-io-utilities)
3. [GGUF Format](#gguf-format)
4. [GGUF Quantization](#gguf-quantization)
5. [SafeTensors Format](#safetensors-format)
6. [ONNX Runtime](#onnx-runtime)
7. [PyTorch .pth Loader](#pytorch-pth-loader)
8. [Quantization (int8 / uint8 / NF4)](#quantization)
9. [Architecture Export](#architecture-export)

---

## Native Serialization

**Header:** `include/core/serialization.h`

The built-in binary format for saving and loading modules, individual tensors, and optimizer state. Use this when you are training and checkpointing entirely within C-ML and do not need interoperability with other frameworks.

### Key API

| Function | Description |
|---|---|
| `module_save(Module*, const char*)` | Save module weights to a file path |
| `module_load(Module*, const char*)` | Load module weights from a file path |
| `module_save_stream(Module*, FILE*)` | Save module weights to an open `FILE*` stream |
| `module_load_stream(Module*, FILE*)` | Load module weights from a `FILE*` stream |
| `tensor_write_file(Tensor*, const char*)` | Save a single tensor (dtype, shape, device, data) |
| `tensor_read_file(const char*)` | Load a single tensor from file; returns `Tensor*` |
| `tensor_write_stream(Tensor*, FILE*)` | Write a tensor to a stream |
| `tensor_read_stream(FILE*)` | Read a tensor from a stream |
| `optimizer_save(Optimizer*, const char*)` | Save optimizer state (momentum buffers, etc.) |
| `optimizer_load(Optimizer*, const char*)` | Restore optimizer state |
| `module_named_parameters(Module*, NamedParameter**, int*)` | Get named parameter list |
| `module_named_parameters_free(NamedParameter*, int)` | Free the named parameter array |

### Example

```c
#include "core/serialization.h"

module_save(model, "weights.bin");
module_load(model, "weights.bin");

optimizer_save(opt, "optimizer.bin");
optimizer_load(opt, "optimizer.bin");

tensor_write_file(my_tensor, "embedding.bin");
Tensor* loaded = tensor_read_file("embedding.bin");
```

### Notes

- This is a C-ML-internal binary format; it is not compatible with external tools.
- Stream variants (`*_stream`) let you multiplex several objects into one file or pipe data over a network socket.
- All functions return 0 on success and a negative value on failure.

---

## Model I/O Utilities

**Header:** `include/nn/model_io.h`

Higher-level convenience functions that bundle a model and its training state into a single checkpoint file.

### Key API

| Function | Description |
|---|---|
| `model_save(Module*, const char*)` | Save model weights |
| `model_load(Module*, const char*)` | Load model weights |
| `model_save_checkpoint(Module*, Optimizer*, int epoch, float loss, const char*)` | Save a full training checkpoint (weights + optimizer + epoch + loss) |
| `model_load_checkpoint(Module*, Optimizer*, int* epoch, float* loss, const char*)` | Restore a full training checkpoint |

### Example

```c
#include "nn/model_io.h"

model_save(model, "model.bin");
model_load(model, "model.bin");

model_save_checkpoint(model, optimizer, epoch, loss, "checkpoint_ep10.bin");

int resumed_epoch;
float resumed_loss;
model_load_checkpoint(model, optimizer, &resumed_epoch, &resumed_loss, "checkpoint_ep10.bin");
```

### Notes

- `model_save_checkpoint` / `model_load_checkpoint` are the recommended way to persist training runs because they capture everything needed to resume.
- The format is C-ML-internal and not intended for cross-framework exchange.

---

## GGUF Format

**Header:** `include/core/gguf.h`

GGUF (GPT-Generated Unified Format) is the standard format used by llama.cpp and the broader GGML ecosystem. Use GGUF when you need to load models distributed in `.gguf` files (e.g., quantized LLaMA models from Hugging Face) or when you want to export C-ML models for use with llama.cpp.

### Supported Tensor Types

| Enum | Type |
|---|---|
| `GGUF_TENSOR_F32` | 32-bit float |
| `GGUF_TENSOR_F16` | 16-bit float |
| `GGUF_TENSOR_Q4_0` | 4-bit quantized (variant 0) |
| `GGUF_TENSOR_Q4_1` | 4-bit quantized (variant 1) |
| `GGUF_TENSOR_Q8_0` | 8-bit quantized |
| `GGUF_TENSOR_Q4_K` | 4-bit K-quant |
| `GGUF_TENSOR_Q5_K` | 5-bit K-quant |
| `GGUF_TENSOR_Q6_K` | 6-bit K-quant |
| `GGUF_TENSOR_I8` / `I16` / `I32` | Integer types |

### Key API

| Function | Description |
|---|---|
| `gguf_open_read(const char*)` | Open a `.gguf` file for reading; returns `GGUFContext*` |
| `gguf_open_write(const char*)` | Create a new `.gguf` file for writing |
| `gguf_close(GGUFContext*)` | Close and free context |
| `gguf_get_num_tensors(GGUFContext*)` | Number of tensors in the file |
| `gguf_get_tensor_name(GGUFContext*, int)` | Get tensor name by index |
| `gguf_read_tensor(GGUFContext*, const char*)` | Read a tensor by name |
| `gguf_write_tensor(GGUFContext*, const char*, Tensor*)` | Write a tensor |
| `module_save_gguf(Module*, const char*)` | Save an entire module to GGUF |
| `module_load_gguf(Module*, const char*)` | Load an entire module from GGUF |

### Example

```c
#include "core/gguf.h"

module_load_gguf(model, "llama-7b-q4_0.gguf");

GGUFContext* ctx = gguf_open_read("model.gguf");
int n = gguf_get_num_tensors(ctx);
for (int i = 0; i < n; i++) {
    const char* name = gguf_get_tensor_name(ctx, i);
    Tensor* t = gguf_read_tensor(ctx, name);
    // ... use tensor ...
}
gguf_close(ctx);

module_save_gguf(model, "exported.gguf");
```

### Notes

- The magic number is `0x46475547` (`"GGUF"` in little-endian).
- Quantized tensor types are automatically dequantized on read via `gguf_quant.h` (see next section).
- Metadata types (strings, arrays, etc.) are defined in the `GGUFMetadataType` enum.

---

## GGUF Quantization

**Header:** `include/core/gguf_quant.h`

Dequantization kernels for GGUF quantized tensor types. These are used internally when loading quantized GGUF files, but can also be called directly if you need to dequantize raw quantized buffers.

### Supported Quantization Formats

| Format | Block Size | Description |
|---|---|---|
| Q4_0 | 32 | 4-bit, delta-only |
| Q4_1 | 32 | 4-bit, delta + min |
| Q8_0 | 32 | 8-bit, delta-only |
| Q4_K | 256 | 4-bit K-quant with sub-block scales |
| Q5_K | 256 | 5-bit K-quant with high-bit plane |
| Q6_K | 256 | 6-bit K-quant |

### Key API

| Function | Description |
|---|---|
| `gguf_type_is_quantized(GGUFTensorType)` | Returns `true` if the type is a quantized format |
| `gguf_quant_block_size(GGUFTensorType)` | Elements per quantization block (e.g., 32 or 256) |
| `gguf_quant_type_size(GGUFTensorType)` | Byte size of one block |
| `gguf_dequantize(GGUFTensorType, const void*, float*, size_t)` | Dequantize a buffer to float32 |

### Example

```c
#include "core/gguf_quant.h"

if (gguf_type_is_quantized(GGUF_TENSOR_Q4_0)) {
    int block_sz = gguf_quant_block_size(GGUF_TENSOR_Q4_0); // 32

    float* output = malloc(numel * sizeof(float));
    gguf_dequantize(GGUF_TENSOR_Q4_0, raw_data, output, numel);
}
```

### Notes

- The `numel` argument to `gguf_dequantize` must be a multiple of the block size.
- Block structures (`BlockQ4_0`, `BlockQ4_K`, etc.) match the llama.cpp memory layout exactly.
- Dequantization always produces float32 output.

---

## SafeTensors Format

**Header:** `include/core/safetensors.h`

SafeTensors is the format popularized by Hugging Face. It stores tensors with a JSON header followed by raw tensor data. Use this format for interoperability with Hugging Face models and any tooling that reads `.safetensors` files.

### Key API

| Function | Description |
|---|---|
| `safetensors_open_read(const char*)` | Open a `.safetensors` file; returns `SafeTensorsContext*` |
| `safetensors_open_write(const char*)` | Create a new `.safetensors` file |
| `safetensors_close(SafeTensorsContext*)` | Close and free context |
| `safetensors_get_num_tensors(SafeTensorsContext*)` | Number of tensors in the file |
| `safetensors_get_tensor_name(SafeTensorsContext*, int)` | Get tensor name by index |
| `safetensors_read_tensor(SafeTensorsContext*, const char*)` | Read a tensor by name |
| `safetensors_write_tensor(SafeTensorsContext*, const char*, Tensor*)` | Write a tensor |
| `module_save_safetensors(Module*, const char*)` | Save a module to SafeTensors |
| `module_load_safetensors(Module*, const char*)` | Load a module from SafeTensors |

### Example

```c
#include "core/safetensors.h"

module_load_safetensors(model, "model.safetensors");

SafeTensorsContext* ctx = safetensors_open_read("model.safetensors");
int n = safetensors_get_num_tensors(ctx);
for (int i = 0; i < n; i++) {
    const char* name = safetensors_get_tensor_name(ctx, i);
    Tensor* t = safetensors_read_tensor(ctx, name);
    // ...
}
safetensors_close(ctx);

module_save_safetensors(model, "exported.safetensors");
```

### Notes

- SafeTensors is designed for safe, fast loading -- no arbitrary code execution (unlike pickle-based formats).
- The API mirrors the GGUF API for consistency.
- Compatible with the Hugging Face `safetensors` Python library.

---

## ONNX Runtime

**Header:** `include/core/onnx.h`

Load and execute ONNX (Open Neural Network Exchange) models. C-ML parses the protobuf graph, maps ONNX operators to internal UOps, and executes the graph. Use this to run models exported from PyTorch, TensorFlow, or any ONNX-producing framework.

### Limits

| Constant | Value |
|---|---|
| `CML_ONNX_MAX_INPUTS` | 32 |
| `CML_ONNX_MAX_OUTPUTS` | 32 |
| `CML_ONNX_MAX_NODES` | 512 |
| `CML_ONNX_MAX_ATTRS` | 16 |

### Key API

| Function | Description |
|---|---|
| `cml_onnx_load(const char*)` | Load an ONNX model from a `.onnx` file |
| `cml_onnx_load_buffer(const uint8_t*, size_t)` | Load from an in-memory buffer |
| `cml_onnx_free(CMLONNXModel*)` | Free the model |
| `cml_onnx_op_supported(const char*)` | Check if a specific ONNX op is supported |
| `cml_onnx_run(CMLONNXModel*, Tensor**, int, Tensor**, int)` | Run inference |
| `cml_onnx_list_supported_ops(const char***, int*)` | List all supported operators |

### Example

```c
#include "core/onnx.h"

CMLONNXModel* onnx = cml_onnx_load("resnet50.onnx");

if (!cml_onnx_op_supported("Conv")) {
    printf("Conv not supported!\n");
}

Tensor* inputs[] = { input_tensor };
Tensor* outputs[1];
cml_onnx_run(onnx, inputs, 1, outputs, 1);

// Use outputs[0] ...

cml_onnx_free(onnx);
```

### Notes

- ONNX support is inference-only; you cannot train through an ONNX graph.
- Not all ONNX operators are supported. Call `cml_onnx_list_supported_ops` or `cml_onnx_op_supported` to check coverage.
- The graph is limited to `CML_ONNX_MAX_NODES` (512) nodes. Models with more nodes will fail to load.
- Attribute name strings are capped at 64 characters; tensor info names at 128 characters.

---

## PyTorch .pth Loader

**Header:** `include/core/pth_loader.h`

Load PyTorch state dictionaries saved with `torch.save()`. Handles the modern ZIP-based format (pickle protocol + raw tensor storage). Use this to import pre-trained PyTorch models directly without converting to an intermediate format.

### Key API

| Function | Description |
|---|---|
| `cml_pth_load(const char*)` | Parse a `.pth` / `.pt` file; returns `CMLPthStateDict*` |
| `cml_pth_free(CMLPthStateDict*)` | Free the state dict |
| `cml_pth_get_tensor(const CMLPthStateDict*, const char*)` | Get a tensor by key |
| `cml_pth_num_entries(const CMLPthStateDict*)` | Number of entries |
| `cml_pth_get_key(const CMLPthStateDict*, int)` | Key name by index |
| `cml_pth_has_key(const CMLPthStateDict*, const char*)` | Check if a key exists |
| `cml_pth_list_keys(const CMLPthStateDict*, int*)` | List all keys |
| `cml_pth_load_into_module(const CMLPthStateDict*, Module*)` | Load state dict into a C-ML Module |
| `cml_pth_print(const CMLPthStateDict*)` | Print a summary of the state dict |
| `cml_pth_total_params(const CMLPthStateDict*)` | Total parameter count |
| `cml_pth_total_bytes(const CMLPthStateDict*)` | Total size in bytes |

### Example

```c
#include "core/pth_loader.h"

CMLPthStateDict* sd = cml_pth_load("resnet50.pth");

cml_pth_print(sd);
printf("Total params: %zu\n", cml_pth_total_params(sd));

Tensor* conv1_weight = cml_pth_get_tensor(sd, "conv1.weight");

cml_pth_load_into_module(sd, model);

cml_pth_free(sd);
```

### Notes

- Supports float32, float16, bfloat16, and integer tensor dtypes.
- Key length is capped at `CML_PTH_MAX_KEY_LEN` (256 characters).
- The loader handles the ZIP+pickle format; very old PyTorch files using raw pickle without ZIP may not be supported.
- The `original_dtype` field on each `CMLPthEntry` records the dtype before any conversion, which is useful for tracking precision changes.

---

## Quantization

**Header:** `include/core/quantization.h`

General-purpose quantization utilities for compressing float32 tensors to lower-precision integer representations. These are independent of any file format and can be combined with any serialization method.

### Supported Schemes

| Scheme | Function Pair | Description |
|---|---|---|
| int8 (symmetric/asymmetric) | `cml_quantize_int8` / `cml_dequantize_int8` | Per-tensor affine quantization to signed 8-bit |
| uint8 (symmetric/asymmetric) | `cml_quantize_uint8` / `cml_dequantize_uint8` | Per-tensor affine quantization to unsigned 8-bit |
| NF4 (4-bit) | `cml_quantize_nf4` / `cml_dequantize_nf4` | Normal Float 4-bit, block-wise quantization (QLoRA-style) |

### Key API

| Function | Description |
|---|---|
| `cml_quantize_compute_params(Tensor*, bool symmetric)` | Compute scale and zero_point for a tensor |
| `cml_quantize_int8(Tensor*, const QuantParams*, QuantParams*)` | Quantize float32 to int8 |
| `cml_dequantize_int8(Tensor*, const QuantParams*)` | Dequantize int8 back to float32 |
| `cml_quantize_uint8(Tensor*, const QuantParams*, QuantParams*)` | Quantize float32 to uint8 |
| `cml_dequantize_uint8(Tensor*, const QuantParams*)` | Dequantize uint8 back to float32 |
| `cml_quantize_nf4(Tensor*, int block_size, float**, int*)` | Quantize float32 to NF4 (4-bit packed into uint8) |
| `cml_dequantize_nf4(Tensor*, const float*, int, int, size_t)` | Dequantize NF4 back to float32 |

### Example

```c
#include "core/quantization.h"

QuantParams params;
Tensor* quantized = cml_quantize_int8(float_tensor, NULL, &params);
// params.scale and params.zero_point are now set

Tensor* restored = cml_dequantize_int8(quantized, &params);

float* scales;
int num_scales;
Tensor* nf4 = cml_quantize_nf4(float_tensor, 64, &scales, &num_scales);

size_t original_numel = 4096;
Tensor* restored_nf4 = cml_dequantize_nf4(nf4, scales, num_scales, 64, original_numel);
free(scales);
```

### Notes

- Passing `NULL` for `params` in the quantize functions triggers automatic parameter computation.
- Symmetric quantization (`zero_point = 0`) is slightly faster but may waste range for asymmetric distributions.
- NF4 packs two 4-bit values per uint8. The resulting tensor has `numel / 2` elements.
- The `CML_NF4_TABLE` global contains the 16 values optimized for normally distributed weights.

---

## Architecture Export

**Header:** `include/core/model_architecture.h`

Extract and export a model's architecture as structured JSON for visualization and documentation purposes. This is not a weight serialization format -- it captures layer types, shapes, parameter counts, and connectivity.

### Key API

| Function | Description |
|---|---|
| `model_architecture_create()` | Allocate a new `ModelArchitecture` struct |
| `model_architecture_extract(Module*, ModelArchitecture*)` | Walk the module tree and populate layer info |
| `model_architecture_export_json(const ModelArchitecture*, const char*)` | Write architecture to a JSON file |
| `model_architecture_free(ModelArchitecture*)` | Free the architecture struct |

### Example

```c
#include "core/model_architecture.h"

ModelArchitecture* arch = model_architecture_create();
model_architecture_extract(model, arch);

printf("Layers: %zu, Total params: %d\n", arch->num_layers, arch->total_params);

model_architecture_export_json(arch, "architecture.json");
model_architecture_free(arch);
```

### Notes

- The `LayerInfo` struct captures type, dimensions, kernel size, stride, padding, bias presence, and parameter count for each layer.
- Each layer can carry a `details` string with additional JSON-encoded information.
- Handles Sequential, Linear, Conv2d, and other standard module types.
- This is for introspection and visualization only -- it does not save or load weights.

---

## Format Comparison

| Format | Read | Write | Weights | Optimizer State | Interop | Quantized |
|---|---|---|---|---|---|---|
| Native (`serialization.h`) | Yes | Yes | Yes | Yes | No | No |
| Model I/O (`model_io.h`) | Yes | Yes | Yes | Yes (checkpoint) | No | No |
| GGUF (`gguf.h`) | Yes | Yes | Yes | No | llama.cpp | Yes |
| SafeTensors (`safetensors.h`) | Yes | Yes | Yes | No | Hugging Face | No |
| ONNX (`onnx.h`) | Yes | No | Yes | No | Universal | No |
| PyTorch .pth (`pth_loader.h`) | Yes | No | Yes | No | PyTorch | No |
| Quantization (`quantization.h`) | N/A | N/A | Transform | N/A | N/A | Yes |
| Architecture (`model_architecture.h`) | N/A | JSON | No | No | JSON | No |
