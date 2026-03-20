#include "backend/remote_device.h"
#include "backend/device.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TEST_PORT 19876

static void* server_thread(void* arg) {
    CMLRemoteServer* srv = (CMLRemoteServer*)arg;
    cml_remote_server_run(srv);
    return NULL;
}

static void test_connect_disconnect(void) {
    printf("  test_connect_disconnect...");

    CMLRemoteServer* srv = cml_remote_server_create(TEST_PORT);
    assert(srv != NULL);

    pthread_t tid;
    pthread_create(&tid, NULL, server_thread, srv);
    usleep(100000);

    CMLRemoteDevice* dev = cml_remote_connect("127.0.0.1", TEST_PORT);
    assert(dev != NULL);
    assert(cml_remote_is_connected(dev));
    assert(dev->session_id != 0);

    cml_remote_disconnect(dev);

    cml_remote_server_stop(srv);
    pthread_join(tid, NULL);
    cml_remote_server_free(srv);

    printf(" PASSED\n");
}

static void test_alloc_free(void) {
    printf("  test_alloc_free...");

    CMLRemoteServer* srv = cml_remote_server_create(TEST_PORT + 1);
    assert(srv != NULL);

    pthread_t tid;
    pthread_create(&tid, NULL, server_thread, srv);
    usleep(100000);

    CMLRemoteDevice* dev = cml_remote_connect("127.0.0.1", TEST_PORT + 1);
    assert(dev != NULL);

    uint64_t h = cml_remote_alloc(dev, 4096);
    assert(h != 0);

    cml_remote_free(dev, h);

    uint64_t handles[8];
    for (int i = 0; i < 8; i++) {
        handles[i] = cml_remote_alloc(dev, 1024);
        assert(handles[i] != 0);
    }
    for (int i = 0; i < 8; i++) {
        cml_remote_free(dev, handles[i]);
    }

    cml_remote_disconnect(dev);
    cml_remote_server_stop(srv);
    pthread_join(tid, NULL);
    cml_remote_server_free(srv);

    printf(" PASSED\n");
}

static void test_upload_download(void) {
    printf("  test_upload_download...");

    CMLRemoteServer* srv = cml_remote_server_create(TEST_PORT + 2);
    assert(srv != NULL);

    pthread_t tid;
    pthread_create(&tid, NULL, server_thread, srv);
    usleep(100000);

    CMLRemoteDevice* dev = cml_remote_connect("127.0.0.1", TEST_PORT + 2);
    assert(dev != NULL);

    size_t n = 256 * sizeof(float);
    uint64_t h = cml_remote_alloc(dev, n);
    assert(h != 0);

    float* send_buf = (float*)malloc(n);
    float* recv_buf = (float*)malloc(n);
    for (int i = 0; i < 256; i++) send_buf[i] = (float)i * 1.5f;

    assert(cml_remote_upload(dev, h, send_buf, n) == 0);
    assert(cml_remote_download(dev, h, recv_buf, n) == 0);

    for (int i = 0; i < 256; i++) {
        assert(recv_buf[i] == send_buf[i]);
    }

    free(send_buf);
    free(recv_buf);
    cml_remote_free(dev, h);
    cml_remote_disconnect(dev);
    cml_remote_server_stop(srv);
    pthread_join(tid, NULL);
    cml_remote_server_free(srv);

    printf(" PASSED\n");
}

static void test_null_args(void) {
    printf("  test_null_args...");

    assert(cml_remote_connect(NULL, 1234) == NULL);
    assert(cml_remote_connect("localhost", 0) == NULL);
    assert(!cml_remote_is_connected(NULL));

    cml_remote_disconnect(NULL);
    assert(cml_remote_alloc(NULL, 100) == 0);
    assert(cml_remote_upload(NULL, 1, NULL, 0) == -1);
    assert(cml_remote_download(NULL, 1, NULL, 0) == -1);

    assert(cml_remote_server_create(0) == NULL);
    assert(cml_remote_server_run(NULL) == -1);

    printf(" PASSED\n");
}

int main(void) {
    device_init();
    printf("Running remote device tests:\n");

    test_null_args();
    test_connect_disconnect();
    test_alloc_free();
    test_upload_download();

    printf("All remote device tests passed.\n");
    device_cleanup();
    return 0;
}
