#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "distributed/ib_transport.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { printf("PASS\n"); tests_passed++; } \
    else { printf("FAIL\n"); } \
} while(0)

static bool test_availability_check(void) {
    bool a = cml_ib_available();
    bool b = cml_ib_available();
    return a == b;
}

static bool test_create_invalid_params(void) {
    if (cml_ib_create(-1, 2) != NULL) return false;
    if (cml_ib_create(0, 0) != NULL) return false;
    if (cml_ib_create(3, 2) != NULL) return false;
    return true;
}

static bool test_free_null(void) {
    cml_ib_free(NULL);
    return true;
}

static bool test_send_not_connected(void) {
    CMLIBTransport dummy;
    memset(&dummy, 0, sizeof(dummy));
    dummy.connected = false;
    char buf[16] = {0};
    if (cml_ib_send(&dummy, 0, buf, sizeof(buf)) == 0) return false;
    return true;
}

static bool test_recv_not_connected(void) {
    CMLIBTransport dummy;
    memset(&dummy, 0, sizeof(dummy));
    dummy.connected = false;
    char buf[16] = {0};
    if (cml_ib_recv(&dummy, 0, buf, sizeof(buf)) == 0) return false;
    return true;
}

static bool test_barrier_not_connected(void) {
    CMLIBTransport dummy;
    memset(&dummy, 0, sizeof(dummy));
    dummy.connected = false;
    if (cml_ib_barrier(&dummy) == 0) return false;
    return true;
}

static bool test_allreduce_null(void) {
    if (cml_ib_allreduce(NULL, NULL, 0, 0) == 0) return false;
    return true;
}

static bool test_register_null(void) {
    if (cml_ib_register_memory(NULL, NULL, 0) != NULL) return false;
    return true;
}

static bool test_deregister_null(void) {
    cml_ib_deregister_memory(NULL, NULL);
    return true;
}

static bool test_connect_invalid(void) {
    if (cml_ib_connect(NULL, NULL, 0) == 0) return false;
    return true;
}

int main(void) {
    printf("=== InfiniBand RDMA Transport Tests ===\n");

    TEST(availability_check);
    TEST(create_invalid_params);
    TEST(free_null);
    TEST(send_not_connected);
    TEST(recv_not_connected);
    TEST(barrier_not_connected);
    TEST(allreduce_null);
    TEST(register_null);
    TEST(deregister_null);
    TEST(connect_invalid);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
