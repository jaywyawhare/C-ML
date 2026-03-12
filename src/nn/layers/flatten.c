#include "nn/layers/flatten.h"
#include "nn.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>

static Tensor* flatten_forward(Module* module, Tensor* input) {
    Flatten* fl = (Flatten*)module;
    if (!fl || !input) return NULL;

    int start = fl->start_dim;
    int end   = fl->end_dim;

    if (start < 0) start += input->ndim;
    if (end < 0)   end   += input->ndim;

    if (start < 0 || start >= input->ndim || end < 0 || end >= input->ndim || start > end) {
        LOG_ERROR("Flatten: invalid start_dim=%d, end_dim=%d for %dD input",
                  fl->start_dim, fl->end_dim, input->ndim);
        return NULL;
    }

    int new_ndim = input->ndim - (end - start);
    int new_shape[16];
    int idx = 0;

    for (int i = 0; i < start; i++)
        new_shape[idx++] = input->shape[i];

    int flat_size = 1;
    for (int i = start; i <= end; i++)
        flat_size *= input->shape[i];
    new_shape[idx++] = flat_size;

    for (int i = end + 1; i < input->ndim; i++)
        new_shape[idx++] = input->shape[i];

    ReshapeParams params = { .new_shape = new_shape, .new_ndim = new_ndim };
    return uop_reshape(input, &params);
}

static void flatten_free(Module* module) {
    free(module);
}

Flatten* nn_flatten(int start_dim, int end_dim) {
    Flatten* fl = malloc(sizeof(Flatten));
    if (!fl) return NULL;

    if (module_init((Module*)fl, "Flatten", flatten_forward, flatten_free) != 0) {
        free(fl);
        return NULL;
    }

    fl->start_dim = start_dim;
    fl->end_dim   = end_dim;
    return fl;
}
