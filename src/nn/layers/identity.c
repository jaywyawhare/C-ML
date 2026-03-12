#include "nn/layers/identity.h"
#include "nn.h"
#include <stdlib.h>

static Tensor* identity_forward(Module* module, Tensor* input) {
    (void)module;
    return input;
}

static void identity_free(Module* module) {
    free(module);
}

Identity* nn_identity(void) {
    Identity* id = malloc(sizeof(Identity));
    if (!id) return NULL;

    if (module_init((Module*)id, "Identity", identity_forward, identity_free) != 0) {
        free(id);
        return NULL;
    }
    return id;
}
