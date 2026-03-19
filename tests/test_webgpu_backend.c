#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ops/ir/gpu/webgpu_backend.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"

static void test_wgsl_generate_add(void) {
    printf("  test_wgsl_generate_add...");

    /* Create a mock IRNode for UOP_ADD */
    struct IRNode node;
    memset(&node, 0, sizeof(node));
    node.type = UOP_ADD;

    char* wgsl = cml_wgsl_generate(&node);
    assert(wgsl != NULL);

    /* Verify the generated WGSL contains expected tokens */
    assert(strstr(wgsl, "@compute") != NULL);
    assert(strstr(wgsl, "@workgroup_size") != NULL);
    assert(strstr(wgsl, "var<storage") != NULL);
    assert(strstr(wgsl, "@binding(0)") != NULL);
    assert(strstr(wgsl, "@binding(1)") != NULL);
    assert(strstr(wgsl, "result") != NULL);

    printf(" (generated %zu bytes) ", strlen(wgsl));
    free(wgsl);
    printf("PASS\n");
}

static void test_wgsl_generate_mul(void) {
    printf("  test_wgsl_generate_mul...");

    struct IRNode node;
    memset(&node, 0, sizeof(node));
    node.type = UOP_MUL;

    char* wgsl = cml_wgsl_generate(&node);
    assert(wgsl != NULL);

    /* Should still produce valid WGSL with compute shader structure */
    assert(strstr(wgsl, "@compute") != NULL);
    assert(strstr(wgsl, "@workgroup_size") != NULL);

    free(wgsl);
    printf(" PASS\n");
}

static void test_wgsl_generate_exp(void) {
    printf("  test_wgsl_generate_exp...");

    struct IRNode node;
    memset(&node, 0, sizeof(node));
    node.type = UOP_EXP;

    char* wgsl = cml_wgsl_generate(&node);
    assert(wgsl != NULL);

    /* Unary op should still produce valid WGSL */
    assert(strstr(wgsl, "@compute") != NULL);
    assert(strstr(wgsl, "@workgroup_size") != NULL);

    free(wgsl);
    printf(" PASS\n");
}

static void test_webgpu_availability(void) {
    printf("  test_webgpu_availability...");
    /* Just call, verify it returns without crashing */
    bool available = cml_webgpu_available();
    printf(" available=%s", available ? "true" : "false");
    printf(" PASS\n");
}

#ifdef CML_HAS_WEBGPU
static void test_webgpu_backend_lifecycle(void) {
    printf("  test_webgpu_backend_lifecycle...");

    CMLWebGPUBackend* backend = cml_webgpu_backend_create();
    assert(backend != NULL);
    assert(backend->initialized == false);

    int ret = cml_webgpu_backend_init(backend);
    if (ret == 0) {
        assert(backend->initialized == true);
        assert(backend->device != NULL);
        assert(backend->queue != NULL);

        /* Test buffer alloc/free */
        void* buf = cml_webgpu_alloc(backend, 256 * sizeof(float));
        assert(buf != NULL);
        cml_webgpu_free(backend, buf);
    } else {
        printf(" (init failed, no WebGPU device) ");
    }

    cml_webgpu_backend_free(backend);
    printf(" PASS\n");
}
#endif

int main(void) {
    printf("WebGPU Backend Tests\n");

    /* WGSL codegen tests do not require a WebGPU device */
    test_wgsl_generate_add();
    test_wgsl_generate_mul();
    test_wgsl_generate_exp();
    test_webgpu_availability();

#ifdef CML_HAS_WEBGPU
    test_webgpu_backend_lifecycle();
#else
    printf("  WebGPU device tests skipped (CML_HAS_WEBGPU not defined)\n");
#endif

    printf("All WebGPU backend tests passed.\n");
    return 0;
}
