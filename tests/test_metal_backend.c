#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ops/ir/gpu/metal_backend.h"

int main(void) {
    printf("=== Metal Backend Tests ===\n");

#ifdef CML_HAS_METAL

    /* Test 1: Check Metal availability */
    printf("  test_metal_available...");
    bool available = cml_metal_available();
    printf(" available=%s", available ? "true" : "false");
    printf(" PASS\n");

    if (available) {
        /* Test 2: Backend create */
        printf("  test_metal_backend_create...");
        CMLMetalBackend* backend = cml_metal_backend_create();
        assert(backend != NULL);
        assert(backend->initialized == false);
        printf(" PASS\n");

        /* Test 3: Backend init */
        printf("  test_metal_backend_init...");
        int ret = cml_metal_backend_init(backend);
        assert(ret == 0);
        assert(backend->initialized == true);
        assert(backend->device != NULL);
        assert(backend->command_queue != NULL);
        assert(strlen(backend->device_name) > 0);
        printf(" (device: %s) PASS\n", backend->device_name);

        /* Test 4: Buffer alloc and free */
        printf("  test_metal_buffer_alloc_free...");
        size_t buf_size = 1024 * sizeof(float);
        void* buffer = cml_metal_alloc(backend, buf_size);
        assert(buffer != NULL);

        /* Upload some data */
        float* host_data = (float*)calloc(1024, sizeof(float));
        assert(host_data != NULL);
        for (int i = 0; i < 1024; i++) {
            host_data[i] = (float)i;
        }
        ret = cml_metal_upload(backend, buffer, host_data, buf_size);
        assert(ret == 0);

        /* Download and verify */
        float* host_out = (float*)calloc(1024, sizeof(float));
        assert(host_out != NULL);
        ret = cml_metal_download(backend, host_out, buffer, buf_size);
        assert(ret == 0);
        for (int i = 0; i < 1024; i++) {
            assert(host_out[i] == (float)i);
        }

        free(host_data);
        free(host_out);
        cml_metal_free(backend, buffer);
        printf(" PASS\n");

        /* Test 5: Backend free */
        printf("  test_metal_backend_free...");
        cml_metal_backend_free(backend);
        printf(" PASS\n");

    } else {
        printf("  Metal device not found, skipping device-dependent tests.\n");
    }

    printf("All Metal backend tests passed.\n");

#else
    printf("Metal not available, skipping tests\n");
#endif

    return 0;
}
