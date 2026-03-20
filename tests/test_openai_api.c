#include "nn/openai_api.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define TEST_PORT 19900

static void* server_thread(void* arg) {
    CMLOpenAIServer* srv = (CMLOpenAIServer*)arg;
    cml_openai_server_run(srv);
    return NULL;
}

static int http_get(int port, const char* path, char* resp_buf, size_t buf_size) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }

    char req[512];
    int req_len = snprintf(req, sizeof(req),
                           "GET %s HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                           path);
    send(fd, req, (size_t)req_len, 0);

    size_t total = 0;
    while (total < buf_size - 1) {
        ssize_t n = recv(fd, resp_buf + total, buf_size - 1 - total, 0);
        if (n <= 0) break;
        total += (size_t)n;
    }
    resp_buf[total] = '\0';
    close(fd);
    return (int)total;
}

static void test_create_free(void) {
    printf("  test_create_free...");

    CMLOpenAIServer* srv = cml_openai_server_create(8080);
    assert(srv != NULL);
    assert(srv->port == 8080);
    assert(srv->max_tokens == 256);

    cml_openai_server_free(srv);

    assert(cml_openai_server_create(0) == NULL);

    printf(" PASSED\n");
}

static void test_health_endpoint(void) {
    printf("  test_health_endpoint...");

    CMLOpenAIServer* srv = cml_openai_server_create(TEST_PORT);
    assert(srv != NULL);

    pthread_t tid;
    pthread_create(&tid, NULL, server_thread, srv);
    usleep(100000);

    char resp[4096];
    int n = http_get(TEST_PORT, "/health", resp, sizeof(resp));
    assert(n > 0);
    assert(strstr(resp, "200 OK") != NULL);
    assert(strstr(resp, "\"status\":\"ok\"") != NULL);

    cml_openai_server_stop(srv);
    pthread_join(tid, NULL);
    cml_openai_server_free(srv);

    printf(" PASSED\n");
}

static void test_models_endpoint(void) {
    printf("  test_models_endpoint...");

    CMLOpenAIServer* srv = cml_openai_server_create(TEST_PORT + 1);
    assert(srv != NULL);

    pthread_t tid;
    pthread_create(&tid, NULL, server_thread, srv);
    usleep(100000);

    char resp[4096];
    int n = http_get(TEST_PORT + 1, "/v1/models", resp, sizeof(resp));
    assert(n > 0);
    assert(strstr(resp, "200 OK") != NULL);
    assert(strstr(resp, "\"object\":\"list\"") != NULL);
    assert(strstr(resp, "\"object\":\"model\"") != NULL);

    cml_openai_server_stop(srv);
    pthread_join(tid, NULL);
    cml_openai_server_free(srv);

    printf(" PASSED\n");
}

static void test_404(void) {
    printf("  test_404...");

    CMLOpenAIServer* srv = cml_openai_server_create(TEST_PORT + 2);
    assert(srv != NULL);

    pthread_t tid;
    pthread_create(&tid, NULL, server_thread, srv);
    usleep(100000);

    char resp[4096];
    int n = http_get(TEST_PORT + 2, "/nonexistent", resp, sizeof(resp));
    assert(n > 0);
    assert(strstr(resp, "404") != NULL);

    cml_openai_server_stop(srv);
    pthread_join(tid, NULL);
    cml_openai_server_free(srv);

    printf(" PASSED\n");
}

int main(void) {
    printf("Running OpenAI API tests:\n");

    test_create_free();
    test_health_endpoint();
    test_models_endpoint();
    test_404();

    printf("All OpenAI API tests passed.\n");
    return 0;
}
