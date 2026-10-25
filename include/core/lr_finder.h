#ifndef CML_CORE_LR_FINDER_H
#define CML_CORE_LR_FINDER_H

#include "nn.h"
#include "optim.h"
#include "core/dataset.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Result of a learning-rate range test: parallel lrs/losses arrays of `count` points
 * plus a suggested LR (the steepest loss descent). */
typedef struct CMLLRFindResult {
    float* lrs;
    float* losses;
    int count;
    float suggested_lr;
} CMLLRFindResult;

/**
 * Learning-rate range test (Smith): run @p num_steps training steps with the LR swept
 * geometrically from @p lr_min to @p lr_max, recording the loss at each. The model
 * weights and the optimizer LR are snapshotted and restored, so the model is unchanged
 * on return. The suggested LR is the point of steepest loss descent. Returns NULL on a
 * bad argument or allocation failure; the caller frees with cml_lr_find_result_free.
 * @param model (Module*)
 * @param loader (DataLoader*)
 * @param opt (Optimizer*)
 * @param loss_fn (Tensor* (*)(Tensor*, Tensor*))
 * @param lr_min (float)
 * @param lr_max (float)
 * @param num_steps (int)
 * @return (CMLLRFindResult*)
 */
CMLLRFindResult* cml_lr_range_find(Module* model, DataLoader* loader, Optimizer* opt,
                                   Tensor* (*loss_fn)(Tensor*, Tensor*), float lr_min, float lr_max,
                                   int num_steps);

/**
 * Free a learning-rate range-test result.
 * @param r (CMLLRFindResult*)
 * @return (void)
 */
void cml_lr_find_result_free(CMLLRFindResult* r);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_LR_FINDER_H */
