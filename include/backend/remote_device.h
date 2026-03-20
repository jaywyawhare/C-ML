#ifndef CML_BACKEND_REMOTE_DEVICE_H
#define CML_BACKEND_REMOTE_DEVICE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_REMOTE_OP_ALLOC    1
#define CML_REMOTE_OP_FREE     2
#define CML_REMOTE_OP_UPLOAD   3
#define CML_REMOTE_OP_DOWNLOAD 4
#define CML_REMOTE_OP_EXECUTE  5
#define CML_REMOTE_OP_PING     6

#define CML_REMOTE_STATUS_OK    0
#define CML_REMOTE_STATUS_ERROR 1

typedef struct CMLRemoteDevice {
    char host[256];
    int port;
    int sock_fd;
    bool connected;
    uint64_t session_id;
} CMLRemoteDevice;

CMLRemoteDevice* cml_remote_connect(const char* host, int port);
void cml_remote_disconnect(CMLRemoteDevice* dev);
bool cml_remote_is_connected(CMLRemoteDevice* dev);

uint64_t cml_remote_alloc(CMLRemoteDevice* dev, size_t size);
void cml_remote_free(CMLRemoteDevice* dev, uint64_t handle);
int cml_remote_upload(CMLRemoteDevice* dev, uint64_t handle, const void* data, size_t n);
int cml_remote_download(CMLRemoteDevice* dev, uint64_t handle, void* data, size_t n);

int cml_remote_execute(CMLRemoteDevice* dev, const char* kernel_source,
                       uint64_t* buffer_handles, int num_buffers,
                       uint32_t grid[3], uint32_t block[3]);

typedef struct CMLRemoteServer {
    int listen_fd;
    int port;
    bool running;
} CMLRemoteServer;

CMLRemoteServer* cml_remote_server_create(int port);
int cml_remote_server_run(CMLRemoteServer* srv);
void cml_remote_server_stop(CMLRemoteServer* srv);
void cml_remote_server_free(CMLRemoteServer* srv);

#ifdef __cplusplus
}
#endif

#endif /* CML_BACKEND_REMOTE_DEVICE_H */
