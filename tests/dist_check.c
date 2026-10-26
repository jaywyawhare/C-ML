/* Standalone cross-host distributed smoke test: GLOO all-reduce of a rank-valued vector.
 * Each rank fills a [4] tensor with (rank+1); SUM all-reduce must yield sum_{r} (r+1). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cml.h"
#include "distributed/distributed.h"
#include "tensor/tensor.h"

int main(void) {
    cml_init();
    if (cml_dist_init(DIST_BACKEND_GLOO, -1, -1) != 0) {
        fprintf(stderr, "dist init failed\n");
        return 2;
    }
    int rank = cml_dist_get_rank();
    int ws   = cml_dist_get_world_size();
    printf("[rank %d/%d] up\n", rank, ws);
    fflush(stdout);
    float v       = (float)(rank + 1);
    float data[4] = {v, v, v, v};
    Tensor* t     = cml_tensor_1d(data, 4);
    if (cml_dist_allreduce(t, DIST_REDUCE_SUM) != 0) {
        fprintf(stderr, "[rank %d] allreduce failed\n", rank);
        return 2;
    }
    float* d     = (float*)tensor_data_ptr(t);
    float expect = 0.0f;
    for (int r = 0; r < ws; r++)
        expect += (float)(r + 1);
    int ok = d && fabsf(d[0] - expect) < 1e-4f && fabsf(d[3] - expect) < 1e-4f;
    printf("[rank %d] allreduce sum = %.1f (expect %.1f) %s\n", rank, d ? d[0] : -1.0, expect,
           ok ? "PASS" : "FAIL");
    fflush(stdout);
    cml_dist_barrier();
    tensor_free(t);
    cml_dist_destroy();
    cml_cleanup();
    return ok ? 0 : 1;
}
