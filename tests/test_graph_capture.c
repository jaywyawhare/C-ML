#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops/ir/graph_capture.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "test_harness.h"

#define RUN_TEST(test)                                                                             \
    do {                                                                                           \
        tests_run++;                                                                               \
        printf("  [%d] %-50s ", tests_run, #test);                                                 \
        if (test()) {                                                                              \
            tests_passed++;                                                                        \
            printf("PASS\n");                                                                      \
        } else {                                                                                   \
            printf("FAIL\n");                                                                      \
        }                                                                                          \
    } while (0)

static int test_create_destroy(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    if (!g)
        return 0;
    cml_graph_capture_free(g);
    return 1;
}

static int test_initial_state(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    if (!g)
        return 0;
    int ok = cml_graph_capture_state(g) == CML_CAPTURE_IDLE;
    ok     = ok && cml_graph_capture_num_nodes(g) == 0;
    cml_graph_capture_free(g);
    return ok;
}

static int test_begin_recording(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    if (!g)
        return 0;
    int ret = cml_graph_capture_begin(g);
    int ok  = (ret == 0) && (cml_graph_capture_state(g) == CML_CAPTURE_RECORDING);
    cml_graph_capture_free(g);
    return ok;
}

static int test_record_node(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    cml_graph_capture_begin(g);
    size_t grid[3]  = {256, 1, 1};
    size_t block[3] = {256, 1, 1};
    int ret         = cml_graph_capture_record(g, UOP_ADD, NULL, grid, block, NULL, 0, 0);
    int ok          = (ret == 0) && (cml_graph_capture_num_nodes(g) == 1);
    cml_graph_capture_free(g);
    return ok;
}

static int test_record_multiple(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    cml_graph_capture_begin(g);
    size_t grid[3]  = {256, 1, 1};
    size_t block[3] = {256, 1, 1};
    cml_graph_capture_record(g, UOP_ADD, NULL, grid, block, NULL, 0, 0);
    cml_graph_capture_record(g, UOP_MUL, NULL, grid, block, NULL, 0, 0);
    cml_graph_capture_record(g, UOP_NEG, NULL, grid, block, NULL, 0, 0);
    int ok = cml_graph_capture_num_nodes(g) == 3;
    cml_graph_capture_free(g);
    return ok;
}

static int test_end_recording(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    cml_graph_capture_begin(g);
    size_t grid[3]  = {256, 1, 1};
    size_t block[3] = {256, 1, 1};
    cml_graph_capture_record(g, UOP_ADD, NULL, grid, block, NULL, 0, 0);
    int ret = cml_graph_capture_end(g);
    int ok  = (ret == 0) && (cml_graph_capture_state(g) == CML_CAPTURE_READY);
    cml_graph_capture_free(g);
    return ok;
}

static int test_replay(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    cml_graph_capture_begin(g);
    size_t grid[3]  = {256, 1, 1};
    size_t block[3] = {256, 1, 1};
    cml_graph_capture_record(g, UOP_ADD, NULL, grid, block, NULL, 0, 0);
    cml_graph_capture_end(g);
    int ret = cml_graph_capture_replay(g);
    int ok  = (ret == 0);
    cml_graph_capture_free(g);
    return ok;
}

static int test_replay_count(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    cml_graph_capture_begin(g);
    size_t grid[3]  = {256, 1, 1};
    size_t block[3] = {256, 1, 1};
    cml_graph_capture_record(g, UOP_ADD, NULL, grid, block, NULL, 0, 0);
    cml_graph_capture_end(g);
    cml_graph_capture_replay(g);
    cml_graph_capture_replay(g);
    cml_graph_capture_replay(g);
    int count  = 0;
    double avg = 0;
    cml_graph_capture_stats(g, &count, &avg);
    int ok = (count == 3);
    cml_graph_capture_free(g);
    return ok;
}

static int test_cannot_record_when_idle(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    size_t grid[3]      = {256, 1, 1};
    size_t block[3]     = {256, 1, 1};
    int ret             = cml_graph_capture_record(g, UOP_ADD, NULL, grid, block, NULL, 0, 0);
    int ok              = (ret == -1);
    cml_graph_capture_free(g);
    return ok;
}

static int test_cannot_replay_when_not_ready(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    int ret             = cml_graph_capture_replay(g);
    int ok              = (ret == -1);
    cml_graph_capture_free(g);
    return ok;
}

static int test_reset(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    cml_graph_capture_begin(g);
    size_t grid[3]  = {256, 1, 1};
    size_t block[3] = {256, 1, 1};
    cml_graph_capture_record(g, UOP_ADD, NULL, grid, block, NULL, 0, 0);
    cml_graph_capture_end(g);
    cml_graph_capture_reset(g);
    int ok = (cml_graph_capture_state(g) == CML_CAPTURE_IDLE);
    ok     = ok && (cml_graph_capture_num_nodes(g) == 0);
    cml_graph_capture_free(g);
    return ok;
}

static int test_bind_input(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    int ret             = cml_graph_capture_bind_input(g, 0, NULL);
    int ok              = (ret == 0);
    ret                 = cml_graph_capture_bind_input(g, 2, NULL);
    ok                  = ok && (ret == 0);
    cml_graph_capture_free(g);
    return ok;
}

static int test_bind_output(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    int ret             = cml_graph_capture_bind_output(g, 0, NULL);
    int ok              = (ret == 0);
    cml_graph_capture_free(g);
    return ok;
}

static int test_print_no_crash(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    cml_graph_capture_begin(g);
    size_t grid[3]  = {256, 1, 1};
    size_t block[3] = {256, 1, 1};
    cml_graph_capture_record(g, UOP_ADD, NULL, grid, block, NULL, 0, 0);
    cml_graph_capture_end(g);
    cml_graph_capture_print(g);
    cml_graph_capture_free(g);
    return 1;
}

static int test_free_null(void) {
    cml_graph_capture_free(NULL);
    return 1;
}

static int test_record_with_args(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    cml_graph_capture_begin(g);
    size_t grid[3]  = {256, 1, 1};
    size_t block[3] = {256, 1, 1};
    void* args[2]   = {(void*)0x1234, (void*)0x5678};
    int ret         = cml_graph_capture_record(g, UOP_ADD, NULL, grid, block, args, 2, 128);
    int ok          = (ret == 0) && (cml_graph_capture_num_nodes(g) == 1);
    cml_graph_capture_free(g);
    return ok;
}

/* Replay must call the dispatch callback and substitute freshly-bound input
 * buffers into the captured argument list, so one capture runs against new
 * inputs. The mock dispatch records the pointer it saw for arg 0. */
static void* g_seen_arg0    = NULL;
static int g_dispatch_calls = 0;
static int mock_dispatch(const struct CMLCapturedNode* node, void* user) {
    (void)user;
    g_dispatch_calls++;
    if (node && node->kernel_args && node->num_args > 0)
        g_seen_arg0 = node->kernel_args[0];
    return 0;
}

static int test_replay_input_substitution(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    cml_graph_capture_begin(g);
    size_t grid[3]  = {1, 1, 1};
    size_t block[3] = {1, 1, 1};
    void* args[1]   = {(void*)0xDEAD}; /* placeholder captured arg */
    cml_graph_capture_record(g, UOP_ADD, NULL, grid, block, args, 1, 0);
    cml_graph_capture_end(g);

    cml_graph_capture_set_dispatch(g, mock_dispatch, NULL);
    /* node 0, arg 0 is input 0 */
    cml_graph_capture_map_input_arg(g, 0, 0, 0);

    Tensor a;
    memset(&a, 0, sizeof(a));
    int abuf = 0;
    a.data   = &abuf;
    Tensor b;
    memset(&b, 0, sizeof(b));
    int bbuf = 0;
    b.data   = &bbuf;

    g_dispatch_calls = 0;
    cml_graph_capture_bind_input(g, 0, &a);
    cml_graph_capture_replay(g);
    int ok = (g_dispatch_calls == 1) && (g_seen_arg0 == &abuf);

    /* Re-bind a different input tensor and replay the same capture. */
    cml_graph_capture_bind_input(g, 0, &b);
    cml_graph_capture_replay(g);
    ok = ok && (g_dispatch_calls == 2) && (g_seen_arg0 == &bbuf);

    cml_graph_capture_free(g);
    return ok;
}

int main(void) {
    printf("Graph Capture Tests\n");

    RUN_TEST(test_create_destroy);
    RUN_TEST(test_initial_state);
    RUN_TEST(test_begin_recording);
    RUN_TEST(test_record_node);
    RUN_TEST(test_record_multiple);
    RUN_TEST(test_end_recording);
    RUN_TEST(test_replay);
    RUN_TEST(test_replay_count);
    RUN_TEST(test_cannot_record_when_idle);
    RUN_TEST(test_cannot_replay_when_not_ready);
    RUN_TEST(test_reset);
    RUN_TEST(test_bind_input);
    RUN_TEST(test_bind_output);
    RUN_TEST(test_print_no_crash);
    RUN_TEST(test_free_null);
    RUN_TEST(test_record_with_args);
    RUN_TEST(test_replay_input_substitution);

    return TEST_SUMMARY();
}
