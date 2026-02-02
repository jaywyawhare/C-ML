/**
 * @file optimization.h
 * @brief IR optimization passes
 */

#ifndef CML_OPS_IR_OPTIMIZATION_H
#define CML_OPS_IR_OPTIMIZATION_H

#include "ops/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Optimize IR
 */
int cml_ir_optimize(CMLIR_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_OPTIMIZATION_H
