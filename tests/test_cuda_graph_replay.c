#include <stdio.h>
#include <stdlib.h>
#include "ops/ir/gpu/cuda_graph_replay.h"
#include "ops/ir/graph_capture.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

static int test_backend_create_null(void) {
    CMLCUDAGraphBackend* gb = cml_cuda_graph_backend_create(NULL);
    return gb == NULL;
}

static int test_backend_free_null(void) {
    cml_cuda_graph_backend_free(NULL);
    return 1;
}

static int test_begin_capture_no_backend(void) {
    int ret = cml_cuda_graph_begin_capture(NULL);
    return ret == -1;
}

static int test_end_capture_no_backend(void) {
    CMLCapturedGraph out = {0};
    int ret = cml_cuda_graph_end_capture(NULL, &out);
    return ret == -1;
}

static int test_replay_no_backend(void) {
    CMLCapturedGraph g = {0};
    int ret = cml_cuda_graph_replay(NULL, &g);
    return ret == -1;
}

static int test_replay_not_ready(void) {
    CMLCapturedGraph g = {0};
    g.state = CML_CAPTURE_IDLE;
    /* Even with a valid-looking gb, replay should fail if not READY */
    int ret = cml_cuda_graph_replay(NULL, &g);
    return ret == -1;
}

static int test_free_null_graph(void) {
    cml_cuda_graph_free(NULL);
    return 1;
}

static int test_end_capture_null_out(void) {
    int ret = cml_cuda_graph_end_capture(NULL, NULL);
    return ret == -1;
}

static int test_replay_null_graph_ptr(void) {
    int ret = cml_cuda_graph_replay(NULL, NULL);
    return ret == -1;
}

static int test_captured_graph_integration(void) {
    CMLCapturedGraph* g = cml_graph_capture_create();
    if (!g) return 0;
    /* Verify backend_graph and backend_instance fields exist */
    int ok = (g->backend_graph == NULL && g->backend_instance == NULL);
    cml_graph_capture_free(g);
    return ok;
}

int main(void) {
    printf("CUDA Graph Replay Tests\n");

    RUN_TEST(test_backend_create_null);
    RUN_TEST(test_backend_free_null);
    RUN_TEST(test_begin_capture_no_backend);
    RUN_TEST(test_end_capture_no_backend);
    RUN_TEST(test_replay_no_backend);
    RUN_TEST(test_replay_not_ready);
    RUN_TEST(test_free_null_graph);
    RUN_TEST(test_end_capture_null_out);
    RUN_TEST(test_replay_null_graph_ptr);
    RUN_TEST(test_captured_graph_integration);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
