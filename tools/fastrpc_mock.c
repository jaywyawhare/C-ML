/* Mock Qualcomm FastRPC transport (libcdsprpc.so / libadsprpc.so stand-in).
 *
 * Implements the three entry points the Hexagon backend dlsym's, so the
 * full session lifecycle (open -> per-node invoke -> close) can be exercised
 * on a machine without a DSP. The mock records every call; tests read back
 * the journal through fastrpc_mock_* accessors.
 *
 * Build as a shared library named libfastrpc_mock.so and point
 * CML_DSP_RPC_LIB at it.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MOCK_JOURNAL_CAP 256

typedef struct {
    int event;              /* 0=open, 1=invoke, 2=close */
    uint32_t handle;
    uint32_t method_id;
    char name[64];
} FastRpcMockEntry;

static FastRpcMockEntry g_journal[MOCK_JOURNAL_CAP];
static int g_journal_len = 0;
static int g_next_handle = 1;
static int g_open_handles = 0;

static void record(int event, uint32_t handle, uint32_t method_id, const char* name) {
    if (g_journal_len >= MOCK_JOURNAL_CAP)
        return;
    FastRpcMockEntry* e = &g_journal[g_journal_len++];
    e->event = event;
    e->handle = handle;
    e->method_id = method_id;
    if (name) {
        strncpy(e->name, name, sizeof(e->name) - 1);
        e->name[sizeof(e->name) - 1] = '\0';
    } else {
        e->name[0] = '\0';
    }
}

int remote_handle_open(const char* name, uint32_t* handle) {
    if (!name || !handle)
        return -1;
    /* Refuse an obviously-invalid skel name so error paths get exercised. */
    if (name[0] == '\0')
        return -2;
    *handle = (uint32_t)g_next_handle++;
    g_open_handles++;
    record(0, *handle, 0, name);
    return 0;
}

int remote_handle_invoke(uint32_t handle, uint32_t method_id, void* args, int nargs) {
    (void)args;
    if (g_open_handles <= 0)
        return -1; /* invoke on closed session */
    if (nargs < 0 || nargs > 1024)
        return -2;
    record(1, handle, method_id, NULL);
    return 0;
}

int remote_handle_close(uint32_t handle) {
    record(2, handle, 0, NULL);
    if (g_open_handles > 0)
        g_open_handles--;
    return 0;
}

/* --- test accessors ------------------------------------------------------ */

int fastrpc_mock_journal_len(void) { return g_journal_len; }

const FastRpcMockEntry* fastrpc_mock_entry(int i) {
    if (i < 0 || i >= g_journal_len)
        return NULL;
    return &g_journal[i];
}

void fastrpc_mock_reset(void) {
    g_journal_len = 0;
    g_next_handle = 1;
    g_open_handles = 0;
}

int fastrpc_mock_open_handles(void) { return g_open_handles; }
