#ifndef CML_CORE_INTERPRET_H
#define CML_CORE_INTERPRET_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gradient-based input attribution. Both routines run the model differentiably with
 * respect to the input and return a new tensor shaped like the input (caller frees). */

/**
 * Saliency map: |d output[target_class] / d input| for each input element.
 * @param model (Module*)
 * @param input (Tensor*)
 * @param target_class (int)
 * @return (Tensor*)
 */
Tensor* cml_saliency(Module* model, Tensor* input, int target_class);

/**
 * Integrated gradients: attribution of output[target_class] to each input element,
 * IG = (input - baseline) * mean over `steps` points of the gradient along the straight
 * path from baseline to input. A NULL baseline is treated as zeros. By the completeness
 * axiom the attributions sum to output[target](input) - output[target](baseline).
 * @param model (Module*)
 * @param input (Tensor*)
 * @param baseline (Tensor*)
 * @param target_class (int)
 * @param steps (int)
 * @return (Tensor*)
 */
Tensor* cml_integrated_gradients(Module* model, Tensor* input, Tensor* baseline, int target_class,
                                 int steps);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_INTERPRET_H */
