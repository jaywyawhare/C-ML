/*
 * Per-tensor affine quantization:
 *   quantized = clamp(round(value / scale) + zero_point, -128, 127)
 *   dequantized = (quantized - zero_point) * scale
 */

#ifndef CML_CORE_QUANTIZATION_H
#define CML_CORE_QUANTIZATION_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QuantParams {
    float scale;        // Scale factor
    int32_t zero_point; // Zero point offset
} QuantParams;

QuantParams cml_quantize_compute_params(Tensor* tensor, bool symmetric);
Tensor* cml_quantize_int8(Tensor* tensor, const QuantParams* params, QuantParams* out_params);
Tensor* cml_dequantize_int8(Tensor* tensor, const QuantParams* params);
Tensor* cml_quantize_uint8(Tensor* tensor, const QuantParams* params, QuantParams* out_params);
Tensor* cml_dequantize_uint8(Tensor* tensor, const QuantParams* params);

/*
 * Weight-only affine-int8 quantization for matmul.
 *
 * cml_quantize_weight_int8 quantizes a 2D [K,N] float weight into an int8
 * tensor that carries its affine params (quant_type = CML_QUANT_AFFINE_INT8,
 * quant_scale/quant_zero_point set, int8 data in ->data).  When such a tensor is
 * the right-hand operand of a matmul, the executor dispatches to the fast
 * integer-weight path below instead of dequantizing to f32 first.
 */
Tensor* cml_quantize_weight_int8(Tensor* weight, bool symmetric);

/*
 * y[M,N] = scale * ( x[M,K] @ w[K,N] - zero_point * rowsum(x) )
 * where w is int8 and x/y are float32.  Returns 0 on success, -1 on bad args.
 */
int cml_qmatmul_affine_int8(const float* x, const int8_t* w, float scale, int32_t zero_point,
                            float* y, int M, int K, int N);

/*
 * Weight-only packed 4-bit quantization for matmul (AUDIT #19 follow-up).
 *
 * Both produce a tensor that keeps its original f32 shape/dtype while the
 * compressed payload lives in ->quant_data (GGUF convention), so the matmul
 * executor can dispatch to the fused GEMM below without a dequant round-trip.
 *
 *   cml_quantize_weight_int4 — per-tensor symmetric affine: q in [-8,7],
 *     scale = absmax/8, zero_point 0.  Payload is ceil(K*N/2) bytes, two
 *     signed nibbles per byte (high nibble = even flat index).
 *   cml_quantize_weight_nf4 — block-wise NF4 with per-block absmax scales.
 *     Payload layout: [num_scales floats][packed nibbles], same nibble order;
 *     block size recorded on the tensor (quant_block_size).
 */
Tensor* cml_quantize_weight_int4(Tensor* weight);
Tensor* cml_quantize_weight_nf4(Tensor* weight, int block_size);

/*
 * y[M,N] = scale * ( x[M,K] @ unpack(w_packed)[K,N] - zero_point * rowsum(x) )
 * with signed 4-bit w.  Returns 0 on success, -1 on bad args.
 */
int cml_qmatmul_affine_int4(const float* x, const uint8_t* w_packed, float scale,
                            int32_t zero_point, float* y, int M, int K, int N);

/*
 * y[M,N] = x[M,K] @ dequant_nf4(w_packed, scales, block_size)[K,N]
 * with f32 activations.  num_scales must equal ceil(K*N / block_size).
 * Returns 0 on success, -1 on bad args.
 */
int cml_qmatmul_nf4(const float* x, const uint8_t* w_packed, const float* scales, int num_scales,
                    int block_size, float* y, int M, int K, int N);

/* NF4 (Normal Float 4-bit) lookup table - 16 values optimal for normal distribution */
extern const float CML_NF4_TABLE[16];

/*
 * Each uint8 stores two NF4 values (high nibble + low nibble).
 * Block size determines granularity of scale factors.
 */
Tensor* cml_quantize_nf4(Tensor* tensor, int block_size, float** out_scales, int* out_num_scales);

Tensor* cml_dequantize_nf4(Tensor* nf4_tensor, const float* scales, int num_scales, int block_size,
                           size_t original_numel);

/* ---- Quantization-aware training (QAT) primitives ---- */
/*
 * Observers calibrate a per-tensor float range across successive update()
 * calls so a training loop can track activation/weight ranges while the
 * weights keep moving:
 *
 *   CML_QAT_OBS_MINMAX             plain running min/max; the range only
 *                                  widens until cml_qat_observer_reset()
 *   CML_QAT_OBS_MOVING_AVG_MINMAX  momentum EMA over each batch's min/max,
 *                                  torch MovingAverageMinMax semantics:
 *                                  run = (1 - momentum)*run + momentum*batch
 *
 * cml_qat_observer_params derives symmetric per-tensor int8 params from the
 * calibrated range (scale = absmax/127, zero_point 0), matching
 * cml_quantize_compute_params(tensor, symmetric=true).
 */
typedef enum {
    CML_QAT_OBS_MINMAX = 0,
    CML_QAT_OBS_MOVING_AVG_MINMAX,
} CmlQatObserverMode;

typedef struct QatObserver {
    CmlQatObserverMode mode;
    float momentum;    // EMA weight of the new batch (MOVING_AVG mode only)
    float running_min; // Calibrated range; invalid before first update
    float running_max;
    bool initialized;   // False until the first update()
    size_t num_updates; // update() calls since create/reset
} QatObserver;

/* momentum is the EMA weight of each new batch, in (0, 1] (ignored for
 * CML_QAT_OBS_MINMAX). Returns NULL on invalid args or allocation failure. */
QatObserver* cml_qat_observer_create(CmlQatObserverMode mode, float momentum);
void cml_qat_observer_free(QatObserver* obs);
void cml_qat_observer_reset(QatObserver* obs);
/* Running calibration over one tensor. Returns 0 on success, -1 on bad args. */
int cml_qat_observer_update(QatObserver* obs, Tensor* tensor);
/* Symmetric int8 params for the calibrated range (scale >= 1e-10, zp = 0).
 * Callers must not invoke this before the first successful update(). */
QuantParams cml_qat_observer_params(const QatObserver* obs);

/*
 * Fake quantization: dequant(quant(t)) with per-tensor params — the exact
 * round-to-grid forward value, composed from existing DIV/ADD/ROUND/CLAMP/
 * SUB/MUL uops.
 *
 * Gradients do NOT flow through that composite literally: ROUND's derivative
 * is zero almost everywhere and the graph autodiff would correctly kill all
 * gradient flow at it. Instead the correction term is detached and re-added,
 *
 *     out = t + stop_gradient(fake_quant(t) - t),
 *
 * i.e. exact forward values with an identity gradient w.r.t. t — the
 * straight-through estimator. This keeps autodiff.c untouched (a global
 * identity VJP on ROUND would silently change every other round/floor user).
 */
Tensor* cml_qat_fake_quant(Tensor* tensor, const QuantParams* params);

/* Convenience wrapper: update the observer with `tensor`, derive symmetric
 * int8 params from it, and fake-quantize. Returns NULL on bad args. */
Tensor* cml_qat_fake_quant_observed(Tensor* tensor, QatObserver* obs);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_QUANTIZATION_H
