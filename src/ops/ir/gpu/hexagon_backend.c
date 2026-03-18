#include "ops/ir/gpu/hexagon_backend.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>

typedef int (*remote_handle_open_fn)(const char* name, uint32_t* handle);
typedef int (*remote_handle_invoke_fn)(uint32_t handle, uint32_t method_id, void* args, int nargs);
typedef int (*remote_handle_close_fn)(uint32_t handle);

static void* s_dsp_lib = NULL;
static remote_handle_open_fn   fn_remote_handle_open   = NULL;
static remote_handle_invoke_fn fn_remote_handle_invoke  = NULL;
static remote_handle_close_fn  fn_remote_handle_close   = NULL;

static void* try_dlopen_dsp(void) {
    void* h = dlopen("libcdsprpc.so", RTLD_LAZY);
    if (h) {
        LOG_DEBUG("Opened libcdsprpc.so (CDSP RPC)");
        return h;
    }
    h = dlopen("libadsprpc.so", RTLD_LAZY);
    if (h) {
        LOG_DEBUG("Opened libadsprpc.so (ADSP RPC)");
        return h;
    }
    return NULL;
}

bool cml_hexagon_available(void) {
    void* h = try_dlopen_dsp();
    if (h) {
        dlclose(h);
        return true;
    }
    return false;
}

CMLHexagonBackend* cml_hexagon_backend_create(void) {
    CMLHexagonBackend* b = (CMLHexagonBackend*)calloc(1, sizeof(CMLHexagonBackend));
    if (!b) {
        LOG_ERROR("Failed to allocate CMLHexagonBackend");
    }
    return b;
}

int cml_hexagon_backend_init(CMLHexagonBackend* backend) {
    if (!backend) return -1;

    if (backend->initialized) {
        LOG_WARNING("Hexagon backend already initialized");
        return 0;
    }

    void* lib = try_dlopen_dsp();
    if (!lib) {
        LOG_ERROR("Hexagon DSP RPC library not available");
        return -1;
    }

    fn_remote_handle_open = (remote_handle_open_fn)dlsym(lib, "remote_handle_open");
    if (!fn_remote_handle_open) {
        LOG_ERROR("Failed to load remote_handle_open: %s", dlerror());
        goto fail;
    }

    fn_remote_handle_invoke = (remote_handle_invoke_fn)dlsym(lib, "remote_handle_invoke");
    if (!fn_remote_handle_invoke) {
        LOG_ERROR("Failed to load remote_handle_invoke: %s", dlerror());
        goto fail;
    }

    fn_remote_handle_close = (remote_handle_close_fn)dlsym(lib, "remote_handle_close");
    if (!fn_remote_handle_close) {
        LOG_ERROR("Failed to load remote_handle_close: %s", dlerror());
        goto fail;
    }

    uint32_t test_handle = 0;
    int rc = fn_remote_handle_open("cml_hexagon_skel", &test_handle);
    if (rc == 0) {
        LOG_INFO("Hexagon DSP session opened successfully");
        fn_remote_handle_close(test_handle);
        backend->dsp_version = 68; /* Default to V68; real detection would query properties */
    } else {
        LOG_WARNING("Hexagon DSP available but session open failed (rc=%d), assuming V68", rc);
        backend->dsp_version = 68;
    }

    backend->hvx_length = 128;
    backend->has_hmx = (backend->dsp_version >= 73);
    backend->handle = lib;
    backend->initialized = true;
    s_dsp_lib = lib;

    LOG_INFO("Hexagon backend initialized: DSP V%d, HVX %d-byte, HMX=%s",
             backend->dsp_version, backend->hvx_length,
             backend->has_hmx ? "yes" : "no");
    return 0;

fail:
    dlclose(lib);
    fn_remote_handle_open = NULL;
    fn_remote_handle_invoke = NULL;
    fn_remote_handle_close = NULL;
    return -1;
}

void cml_hexagon_backend_free(CMLHexagonBackend* backend) {
    if (!backend) return;

    if (backend->handle) {
        dlclose(backend->handle);
        backend->handle = NULL;
        s_dsp_lib = NULL;
        fn_remote_handle_open = NULL;
        fn_remote_handle_invoke = NULL;
        fn_remote_handle_close = NULL;
    }

    backend->initialized = false;
    free(backend);
}

int cml_hexagon_execute(CMLHexagonBackend* backend, CMLGraph_t ir) {
    if (!backend || !backend->initialized) {
        LOG_ERROR("Hexagon backend not initialized");
        return -1;
    }
    if (!ir) {
        LOG_ERROR("Hexagon execute: NULL IR graph");
        return -1;
    }
    if (!fn_remote_handle_open || !fn_remote_handle_invoke || !fn_remote_handle_close) {
        LOG_ERROR("Hexagon execute: FastRPC function pointers not loaded");
        return -1;
    }

    uint32_t session = 0;
    int rc = fn_remote_handle_open("cml_hexagon_skel", &session);
    if (rc != 0) {
        LOG_ERROR("Hexagon execute: failed to open DSP session (rc=%d)", rc);
        return -1;
    }

    LOG_DEBUG("Hexagon execute: DSP session opened (handle=0x%x)", session);

    struct IRNode* node = ir->head;
    int node_idx = 0;
    int status = 0;

    while (node) {
        const char* op_name = uop_type_to_string(node->type);
        LOG_DEBUG("Hexagon execute: node %d, op=%s, inputs=%d",
                  node_idx, op_name, node->num_inputs);

        /*
         * Invoke the remote procedure for this operation.
         * Method ID is derived from UOp type. The args buffer would contain
         * serialized tensor data and parameters. In a full implementation,
         * tensor data is mapped into DSP-accessible shared memory via ION
         * buffers, and method_id maps to the skel function on the DSP.
         */
        uint32_t method_id = (uint32_t)node->type;
        rc = fn_remote_handle_invoke(session, method_id, node->params, node->num_inputs);
        if (rc != 0) {
            LOG_ERROR("Hexagon execute: remote invoke failed for node %d (op=%s, rc=%d)",
                      node_idx, op_name, rc);
            status = -1;
            break;
        }

        LOG_DEBUG("Hexagon execute: node %d completed on DSP", node_idx);
        node = node->next;
        node_idx++;
    }

    fn_remote_handle_close(session);
    LOG_DEBUG("Hexagon execute: DSP session closed (%d nodes processed)", node_idx);

    return status;
}
