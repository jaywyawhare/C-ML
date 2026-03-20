#ifndef CML_OPS_IR_GPU_AMX_H
#define CML_OPS_IR_GPU_AMX_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

bool cml_amx_available(void);

int cml_amx_matmul_f32(const float* A, const float* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc);
int cml_amx_matmul_f16(const void* A, const void* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_GPU_AMX_H */
