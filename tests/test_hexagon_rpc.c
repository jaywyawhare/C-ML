/* Exercises the Hexagon backend's full FastRPC session lifecycle against
 * libfastrpc_mock.so (built by CMake from tools/fastrpc_mock.c):
 * open -> per-node invoke with method IDs derived from UOp types -> close.
 * Requires no DSP hardware; point CML_DSP_RPC_LIB at the mock library. */
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <dlfcn.h>

#include "cml.h"
#include "ops/ir/gpu/hexagon_backend.h"
#include "test_harness.h"

/* Mock library journal accessors (resolved via dlsym on the SAME instance
 * the backend dlopen'd — see fastrpc_mock.c). */
typedef int (*fn_int)(void);
static fn_int p_mock_journal_len;
static fn_int p_mock_open_handles;
static fn_int p_mock_reset;

int cml_hexagon_execute(CMLHexagonBackend* backend, CMLGraph_t ir);

static char g_lib_path[512];

static bool set_lib_path(void) {
    /* CMake compiles in the mock library's location. */
#ifdef FASTRPC_MOCK_LIB
    snprintf(g_lib_path, sizeof(g_lib_path), "%s", FASTRPC_MOCK_LIB);
    setenv("CML_DSP_RPC_LIB", g_lib_path, 1);
    return true;
#else
    return false;
#endif
}

static bool resolve_mock_api(void) {
    if (!set_lib_path()) return false;
    /* Attach to (or preload) the mock instance; the backend's own dlopen of
     * the same path then returns this identical handle, so journal state is
     * shared. */
    void* h = dlopen(g_lib_path, RTLD_LAZY);
    if (!h) return false;
    p_mock_journal_len  = (fn_int)dlsym(h, "fastrpc_mock_journal_len");
    p_mock_open_handles = (fn_int)dlsym(h, "fastrpc_mock_open_handles");
    p_mock_reset        = (fn_int)dlsym(h, "fastrpc_mock_reset");
    return p_mock_journal_len && p_mock_open_handles && p_mock_reset;
}

static bool test_session_lifecycle(void) {
    if (!set_lib_path() || !resolve_mock_api()) return false;
    p_mock_reset();

    CMLHexagonBackend* b = cml_hexagon_backend_create();
    if (!b) return false;
    if (cml_hexagon_backend_init(b) != 0) {
        cml_hexagon_backend_free(b);
        return false;
    }

    /* tiny graph: two ops */
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    int shape[] = {2, 2};
    Tensor* a = tensor_randn(shape, 2, &cfg);
    Tensor* s1 = tensor_relu(a);
    Tensor* s2 = tensor_neg(s1);
    if (!a || !s1 || !s2) return false;
    tensor_ensure_executed(s2);

    p_mock_reset();
    int rc = cml_hexagon_execute(b, tensor_get_ir_context(s2));

    bool ok = (rc == 0);
    /* journal: open + 2 invokes + close, no leaked sessions */
    ok = ok && (p_mock_journal_len() == 4);
    ok = ok && (p_mock_open_handles() == 0);

    tensor_free(s2);
    tensor_free(s1);
    tensor_free(a);
    cml_hexagon_backend_free(b);
    return ok;
}

#ifndef RUN_TEST
#define RUN_TEST(t) TEST(t)
#endif

int main(void) {
    printf("=== hexagon fastrpc mock ===\n");
    TEST(session_lifecycle);
    return TEST_SUMMARY();
}
