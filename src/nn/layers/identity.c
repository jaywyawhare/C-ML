#include "nn/layers/identity.h"
#include "nn.h"
#include <stdlib.h>
#include "alloc/cml_allocator.h"

/** torch.nn.Identity forward: returns the input tensor unchanged. */
static Tensor* identity_forward(Module* module, Tensor* input) {
    (void)module;
    return input;
}

/** Free the Identity module (no owned parameters). */
static void identity_free(Module* module) { cml_free(module); }

/** Construct an Identity layer. Returns NULL on allocation/init failure. */
Identity* nn_identity(void) {
    Identity* id = cml_malloc(sizeof(Identity));
    if (!id)
        return NULL;

    if (module_init((Module*)id, "Identity", identity_forward, identity_free) != 0) {
        cml_free(id);
        return NULL;
    }
    return id;
}
