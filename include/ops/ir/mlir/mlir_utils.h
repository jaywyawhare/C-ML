#ifndef CML_OPS_IR_MLIR_UTILS_H
#define CML_OPS_IR_MLIR_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get MLIR version string
 * @return Version string (e.g., "18.1.0"), or "Not Available" if no MLIR
 */
const char* cml_mlir_version(void);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_MLIR_UTILS_H
