#ifndef CML_NN_PACKED_SEQUENCE_H
#define CML_NN_PACKED_SEQUENCE_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Packed variable-length sequence batch (as torch's PackedSequence): time steps
 * concatenated in length-sorted order with per-step active counts.
 */
typedef struct {
    Tensor* data;          /* [sum(lengths), F] rows grouped by timestep */
    int* batch_sizes;      /* [max_len] active sequences per timestep */
    int max_len;           /* number of timesteps */
    int* sorted_indices;   /* [B] sorted-order position -> original batch index */
    int* unsorted_indices; /* [B] original batch index -> sorted-order position */
    int batch_size;        /* B */
    int feature_size;      /* F */
} CMLPackedSequence;

/**
 * Pack a padded batch, dropping padding time steps (as torch's
 * pack_padded_sequence).
 * @param input (Tensor*) [T,B,F], or [B,T,F] if batch_first
 * @param lengths (const int*) valid length per sequence, size B
 * @param batch_size (int) B
 * @param batch_first (bool) input layout is [B,T,F] if true
 * @param enforce_sorted (bool) assume lengths already descending if true
 * @return (CMLPackedSequence*) packed batch, or NULL on error
 */
CMLPackedSequence* cml_pack_padded_sequence(Tensor* input, const int* lengths, int batch_size,
                                            bool batch_first, bool enforce_sorted);

/**
 * Pad a packed batch back to a dense tensor (as torch's pad_packed_sequence).
 * @param packed (const CMLPackedSequence*)
 * @param batch_first (bool) produce [B,T,F] if true, else [T,B,F]
 * @param padding_value (float) fill for padded positions
 * @param total_length (int) output time dimension, or <=0 for packed->max_len
 * @param out_lengths (int*) receives recovered lengths [B], may be NULL
 * @return (Tensor*) padded dense tensor, or NULL on error
 */
Tensor* cml_pad_packed_sequence(const CMLPackedSequence* packed, bool batch_first,
                                float padding_value, int total_length, int* out_lengths);

/**
 * Free a packed sequence and its owned buffers.
 * @param p (CMLPackedSequence*)
 * @return (void)
 */
void cml_packed_sequence_free(CMLPackedSequence* p);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_PACKED_SEQUENCE_H */
